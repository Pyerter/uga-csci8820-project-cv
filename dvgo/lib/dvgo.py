from email.policy import strict
import os, time
import torch
import numpy as np
from torch.utils.cpp_extension import load
from torch import nn
import torch.nn.functional as F
from tqdm import trange, tqdm

from .voxel_grid import MaskGrid, DenseGrid
from .alphas_to_weights import AlphasToWeights
from .masked_adam import MaskedAdam
from .segment_coo import segment_coo
from lib import voxel_grid


def raw_to_alpha(density, shift, interval):
    n_pts = density.size(0)
    if n_pts == 0:
        return torch.empty_like(density), torch.empty_like(density)

    # Compute exp_d and alpha directly using PyTorch operations
    exp_d = torch.exp(density + shift.cuda())
    alpha = 1 - torch.pow(1 + exp_d, -interval)

    return exp_d, alpha


def raw_to_alpha_backward(exp_d, grad_back, interval):
    n_pts = exp_d.size(0)
    if n_pts == 0:
        return torch.empty_like(exp_d)

    # Calculate gradients using PyTorch operations
    grad = (
        torch.clamp(exp_d, max=1e10)
        * torch.pow(1 + exp_d, -interval - 1)
        * interval
        * grad_back
    )

    return grad


def infer_t_minmax(rays_o, rays_d, xyz_min, xyz_max, near, far):
    # Compute intersections for each dimension
    avoid_zero = rays_d.abs().clamp(min=1e-6)
    t1 = (xyz_min - rays_o) / avoid_zero
    t2 = (xyz_max - rays_o) / avoid_zero

    t_min = torch.min(t1, t2).max(dim=1)[0]
    t_max = torch.max(t1, t2).min(dim=1)[0]

    # Clamp the t values between near and far
    t_min = t_min.clamp(min=near, max=far)
    t_max = t_max.clamp(min=near, max=far)

    return t_min, t_max


def infer_n_samples(rays_d, t_min, t_max, stepdist):
    ray_lengths = torch.norm(rays_d, dim=1)
    n_samples = ((t_max - t_min) * ray_lengths / stepdist).ceil().to(torch.int64)
    n_samples = torch.max(
        n_samples, torch.tensor(1, device=n_samples.device)
    )  # Ensure at least one sample
    return n_samples


def infer_ray_start_dir(rays_o, rays_d, t_min):
    ray_norms = torch.norm(rays_d, dim=1, keepdim=True)
    rays_start = rays_o + rays_d * t_min.unsqueeze(1)
    rays_dir = rays_d / ray_norms
    return rays_start, rays_dir


def sample_points_on_rays(rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist):
    n_rays = rays_o.size(0)

    # Compute ray-bbox intersection
    t_min, t_max = infer_t_minmax(rays_o, rays_d, xyz_min, xyz_max, near, far)

    # Calculate number of samples along each ray
    N_steps = infer_n_samples(rays_d, t_min, t_max, stepdist)
    total_len = N_steps.sum().item()

    # Create tensors for ray and step indices
    ray_id = torch.repeat_interleave(
        torch.arange(n_rays, device=rays_o.device), N_steps
    )
    step_id = torch.cat([torch.arange(0, n, device=rays_o.device) for n in N_steps])

    # Calculate start points and directions for each ray
    rays_start, rays_dir = infer_ray_start_dir(rays_o, rays_d, t_min)

    # Calculate points along rays
    dists = step_id * stepdist
    rays_pts = rays_start[ray_id] + rays_dir[ray_id] * dists.unsqueeze(1)

    # Check if points are out of bounds
    mask_outbbox = (rays_pts < xyz_min) | (rays_pts > xyz_max)
    mask_outbbox = mask_outbbox.any(dim=1)

    return rays_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max


mse2psnr = lambda x: -10.0 * torch.log10(x)


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top : top + BS]
        top += BS


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks):
    print("get_training_rays: start")
    assert len(np.unique(HW.cpu(), axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks), -1).cpu(), axis=0)) == 1
    assert (
        len(rgb_tr) == len(train_poses)
        and len(rgb_tr) == len(Ks)
        and len(rgb_tr) == len(HW)
    )
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
        )
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print("get_training_rays: finish (eps time:", eps_time, "sec)")
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks):
    print("get_training_rays_flatten: start")
    assert (
        len(rgb_tr_ori) == len(train_poses)
        and len(rgb_tr_ori) == len(Ks)
        and len(rgb_tr_ori) == len(HW)
    )
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
        )
        n = H * W
        rgb_tr[top : top + n].copy_(img.flatten(0, 1))
        rays_o_tr[top : top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
        rays_d_tr[top : top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
        viewdirs_tr[top : top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print("get_training_rays_flatten: finish (eps time:", eps_time, "sec)")
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(
    rgb_tr_ori,
    train_poses,
    HW,
    Ks,
    model,
    render_kwargs,
):
    print("get_training_rays_in_maskcache_sampling: start")
    assert (
        len(rgb_tr_ori) == len(train_poses)
        and len(rgb_tr_ori) == len(Ks)
        and len(rgb_tr_ori) == len(HW)
    )
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        print("img.shape", img.shape)
        print("H, W", H, W)
        H = 800
        W = 800
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
        )
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i : i + CHUNK] = model.hit_coarse_geo(
                rays_o=rays_o[i : i + CHUNK],
                rays_d=rays_d[i : i + CHUNK],
                **render_kwargs,
            ).to(DEVICE)
        n = mask.sum()
        rgb_tr[top : top + n].copy_(img[mask][:, :3])
        rays_o_tr[top : top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top : top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top : top + n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print("get_training_rays_in_maskcache_sampling: ratio", top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print(
        "get_training_rays_in_maskcache_sampling: finish (eps time:", eps_time, "sec)"
    )
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def load_checkpoint(model, optimizer, ckpt_path):
    ckpt = torch.load(ckpt_path)
    start = ckpt["global_step"]
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, start


def load_existing_model(args, model_attrs, train_attrs, reload_ckpt_path, device):
    model_class = DirectVoxGO
    ckpt = torch.load(reload_ckpt_path)
    model = model_class(**ckpt["model_kwargs"], model_attrs=model_attrs, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    optimizer = create_optimizer_or_freeze_model(model, train_attrs, global_step=0)
    model, optimizer, start = load_checkpoint(
        model, optimizer, reload_ckpt_path
    )
    return model, optimizer, start


def create_optimizer_or_freeze_model(model, train_attrs, global_step):
    decay_steps = train_attrs["lrate_decay"] * 1000
    decay_factor = 0.1 ** (global_step / decay_steps)

    param_group = []
    for k in train_attrs.keys():
        if not k.startswith("lrate_"):
            continue
        k = k[len("lrate_") :]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f"create_optimizer_or_freeze_model: param {k} not exist")
            continue

        lr = train_attrs[f"lrate_{k}"] * decay_factor
        if lr > 0:
            print(f"create_optimizer_or_freeze_model: param {k} lr {lr}")
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append(
                {
                    "params": param,
                    "lr": lr,
                    "skip_zero_grad": (k in train_attrs["skip_zero_grad_fields"]),
                }
            )
        else:
            print(f"create_optimizer_or_freeze_model: param {k} freeze")
            param.requires_grad = False
    return MaskedAdam(param_group)


def create_new_model(
    model_attrs, train_attrs: dict, xyz_min, xyz_max, stage, coarse_ckpt_path, device
):
    num_voxels = model_attrs["num_voxels"]
    if len(train_attrs["pg_scale"]):
        num_voxels = int(num_voxels / (2 ** len(train_attrs["pg_scale"])))

    print(f"scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m")
    model = DirectVoxGO(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        model_attrs=model_attrs,
    )
    model = model.to(device)
    optimizer = create_optimizer_or_freeze_model(model, train_attrs, global_step=0)
    return model, optimizer


def scene_rep_reconstruction(
    args,
    xyz_min,
    xyz_max,
    data_dict,
    stage,
    exp_path,
    model_attrs,
    train_attrs,
    coarse_ckpt_path=None,
    **kwargs,
):
    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if abs(model_attrs["world_bound_scale"] - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (model_attrs["world_bound_scale"] - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k]
        for k in [
            "HW",
            "Ks",
            "near",
            "far",
            "i_train",
            "i_val",
            "i_test",
            "poses",
            "render_poses",
            "images",
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(exp_path, f"{stage}_last.tar")
    if os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f"scene_rep_reconstruction ({stage}): train from scratch")
        model, optimizer = create_new_model(
            model_attrs, train_attrs, xyz_min, xyz_max, stage, coarse_ckpt_path, device
        )
        start = 0
        if model_attrs["maskout_near_cam_vox"]:
            model.maskout_near_cam_vox(poses[i_train, :3, 3], near)
    else:
        print(f"scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}")
        model, optimizer, start = load_existing_model(
            args, model_attrs, train_attrs, reload_ckpt_path, device=device
        )

    # init rendering setup
    render_kwargs = {
        "near": data_dict["near"],
        "far": data_dict["far"],
        "bg": 1,
        "stepsize": model_attrs["stepsize"],
    }

    # init batch rays sampler
    def gather_training_rays():
        rgb_tr_ori = images[i_train].to(device)

        if train_attrs["ray_sampler"] == "in_maskcache":
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = (
                get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train],
                    Ks=Ks[i_train],
                    model=model,
                    render_kwargs=render_kwargs,
                )
            )
        elif train_attrs["ray_sampler"] == "flatten":
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train],
                Ks=Ks[i_train],
            )
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train],
                Ks=Ks[i_train],
            )
        index_generator = batch_indices_generator(len(rgb_tr), train_attrs["N_rand"])
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = (
        gather_training_rays()
    )

    # view-count-based learning rate
    if train_attrs["pervoxel_lr"]:

        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr,
                rays_d_tr=rays_d_tr,
                imsz=imsz,
                near=near,
                far=far,
                stepsize=model_attrs["stepsize"],
                downrate=train_attrs["pervoxel_lr_downrate"],
            )
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False

        per_voxel_init()

    if train_attrs["maskout_lt_nviews"] > 0:
        model.update_occupancy_cache_lt_nviews(
            rays_o_tr, rays_d_tr, imsz, render_kwargs, train_attrs["maskout_lt_nviews"]
        )

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1 + start, 1 + train_attrs["N_iters"]):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in train_attrs["pg_scale"]:
            n_rest_scales = (
                len(train_attrs["pg_scale"])
                - train_attrs["pg_scale"].index(global_step)
                - 1
            )
            cur_voxels = int(model_attrs["num_voxels"] / (2**n_rest_scales))
            model.scale_volume_grid(cur_voxels)
            optimizer = create_optimizer_or_freeze_model(
                model, train_attrs, global_step=0
            )
            model.act_shift -= train_attrs["decay_after_scale"]
            torch.cuda.empty_cache()

        # random sample rays
        if train_attrs["ray_sampler"] in ["flatten", "in_maskcache"]:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif train_attrs["ray_sampler"] == "random":
            sel_b = torch.randint(rgb_tr.shape[0], [train_attrs["N_rand"]])
            sel_r = torch.randint(rgb_tr.shape[1], [train_attrs["N_rand"]])
            sel_c = torch.randint(rgb_tr.shape[2], [train_attrs["N_rand"]])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        # volume rendering
        render_result = model(
            rays_o.cuda(),
            rays_d.cuda(),
            viewdirs.cuda(),
            global_step=global_step,
            is_train=True,
            **render_kwargs,
        )

        # gradient descent step

        optimizer.zero_grad(set_to_none=True)
        print(render_result["rgb_marched"].shape, target.shape)
        target = target[:, :3]
        loss = train_attrs["weight_main"] * F.mse_loss(
            render_result["rgb_marched"], target
        )
        psnr = mse2psnr(loss.detach())
        if train_attrs["weight_entropy_last"] > 0:
            pout = render_result["alphainv_last"].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(
                pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)
            ).mean()
            loss += train_attrs["weight_entropy_last"] * entropy_last_loss
        if train_attrs["weight_rgbper"] > 0:
            rgbper = (
                (render_result["raw_rgb"] - target[render_result["ray_id"]])
                .pow(2)
                .sum(-1)
            )
            rgbper_loss = (rgbper * render_result["weights"].detach()).sum() / len(
                rays_o
            )
            loss += train_attrs["weight_rgbper"] * rgbper_loss
        loss.backward()

        if (
            global_step < train_attrs["tv_before"]
            and global_step > train_attrs["tv_after"]
            and global_step % train_attrs["tv_every"] == 0
        ):
            if train_attrs["weight_tv_density"] > 0:
                model.density_total_variation_add_grad(
                    train_attrs["weight_tv_density"] / len(rays_o),
                    global_step < train_attrs["tv_dense_before"],
                )
            if train_attrs["weight_tv_k0"] > 0:
                model.k0_total_variation_add_grad(
                    train_attrs["weight_tv_k0"] / len(rays_o),
                    global_step < train_attrs["tv_dense_before"],
                )

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = train_attrs["lrate_decay"] * 1000
        decay_factor = 0.1 ** (1 / decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = param_group["lr"] * decay_factor

        # check log & save
        if global_step % 5000 == 0:
            eps_time = time.time() - time0
            eps_time_str = (
                f"{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}"
            )
            tqdm.write(
                f"scene_rep_reconstruction ({stage}): iter {global_step:6d} / "
                f"Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / "
                f"Eps: {eps_time_str}"
            )
            psnr_lst = []

        if global_step % 100_000 == 0:
            path = os.path.join(exp_path, f"{stage}_{global_step:06d}.tar")
            torch.save(
                {
                    "global_step": global_step,
                    "model_kwargs": model.get_kwargs(),  # Assuming model.get_kwargs() returns a dict
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print(f"scene_rep_reconstruction ({stage}): saved checkpoints at", path)

        if global_step != -1:
            torch.save(
                {
                    "global_step": global_step,
                    "model_kwargs": model.get_kwargs(),  # Assuming model.get_kwargs() returns a dict
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                last_ckpt_path,
            )
            print(
                f"scene_rep_reconstruction ({stage}): saved checkpoints at",
                last_ckpt_path,
            )


def get_rays(H, W, K, c2w, mode="center"):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device),
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == "lefttop":
        pass
    elif mode == "center":
        i, j = i + 0.5, j + 0.5
    elif mode == "random":
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, mode="center"):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d, viewdirs


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        exp, alpha = raw_to_alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return (
            raw_to_alpha_backward(exp, grad_back.contiguous(), interval),
            None,
            None,
        )


class DirectVoxGO(torch.nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        model_attrs,
        num_voxels=1024000,
        num_voxels_base=1024000,
        alpha_init=None,
        mask_cache_path=None,
        mask_cache_thres=1e-3,
        mask_cache_world_size=None,
        fast_color_thres=0,
        density_type="DenseGrid",
        density_config={},
        k0_config={},
        rgbnet_dim=0,
        rgbnet_direct=False,
        rgbnet_full_implicit=False,
        rgbnet_depth=3,
        rgbnet_width=128,
        viewbase_pe=4,
        **kwargs,
    ):
        super(DirectVoxGO, self).__init__()
        print("dvgo: xyz_min          ", xyz_min)
        print("dvgo: xyz_max          ", xyz_max)
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = (
            (self.xyz_max - self.xyz_min).prod() / self.num_voxels_base
        ).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = model_attrs["alpha_init"]
        self.register_buffer(
            "act_shift",
            torch.FloatTensor([np.log(1 / (1 - model_attrs["alpha_init"]) - 1)]),
        )
        print("dvgo: set density bias shift to", self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = DenseGrid(
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )

        # init color representation
        self.rgbnet_kwargs = {
            "rgbnet_dim": rgbnet_dim,
            "rgbnet_direct": rgbnet_direct,
            "rgbnet_full_implicit": rgbnet_full_implicit,
            "rgbnet_depth": rgbnet_depth,
            "rgbnet_width": rgbnet_width,
            "viewbase_pe": viewbase_pe,
        }
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = DenseGrid(
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = DenseGrid(
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer(
                "viewfreq", torch.FloatTensor([(2**i) for i in range(viewbase_pe)])
            )
            dim0 = 3 + 3 * viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim - 3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width),
                nn.ReLU(inplace=True),
                *[
                    nn.Sequential(
                        nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)
                    )
                    for _ in range(rgbnet_depth - 2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print("dvgo: feature voxel grid", self.k0)
            print("dvgo: mlp", self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskGrid(
                path=mask_cache_path, mask_cache_thres=mask_cache_thres
            ).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]
                    ),
                ),
                -1,
            )
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = MaskGrid(
            path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max
        )

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(
            1 / 3
        ) + 1e-6
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print("dvgo: xyz_min          ", self.xyz_min)
        print("dvgo: xyz_max          ", self.xyz_max)
        print("dvgo: voxel_size      ", self.voxel_size)
        print("dvgo: world_size      ", self.world_size)
        print("dvgo: voxel_size_base ", self.voxel_size_base)
        print("dvgo: voxel_size_ratio", self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min.cpu().numpy(),
            "xyz_max": self.xyz_max.cpu().numpy(),
            "num_voxels": self.num_voxels,
            "num_voxels_base": self.num_voxels_base,
            "alpha_init": self.alpha_init,
            "voxel_size_ratio": self.voxel_size_ratio,
            "mask_cache_path": self.mask_cache_path,
            "mask_cache_thres": self.mask_cache_thres,
            "mask_cache_world_size": list(self.mask_cache.mask.shape),
            "fast_color_thres": self.fast_color_thres,
            "density_type": self.density_type,
            "density_config": self.density_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ),
            -1,
        )
        nearest_dist = torch.stack(
            [
                (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
                for co in cam_o.split(100)  # for memory saving
            ]
        ).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print("dvgo: scale_volume_grid start")
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print(
            "dvgo: scale_volume_grid scale world_size from",
            ori_world_size.tolist(),
            "to",
            self.world_size.tolist(),
        )

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.xyz_min[0], self.xyz_max[0], self.world_size[0]
                    ),
                    torch.linspace(
                        self.xyz_min[1], self.xyz_max[1], self.world_size[1]
                    ),
                    torch.linspace(
                        self.xyz_min[2], self.xyz_max[2], self.world_size[2]
                    ),
                ),
                -1,
            )
            self_alpha = F.max_pool3d(
                self.activate_density(self.density.get_dense_grid()),
                kernel_size=3,
                padding=1,
                stride=1,
            )[0, 0]
            self.mask_cache = MaskGrid(
                path=None,
                mask=self.mask_cache(self_grid_xyz)
                & (self_alpha > self.fast_color_thres),
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )

        print("dvgo: scale_volume_grid finish")

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]
                ),
                torch.linspace(
                    self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]
                ),
                torch.linspace(
                    self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]
                ),
            ),
            -1,
        )
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(
            cache_grid_alpha, kernel_size=3, padding=1, stride=1
        )[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres

    def voxel_count_views(
        self,
        rays_o_tr,
        rays_d_tr,
        imsz,
        near,
        far,
        stepsize,
        downrate=1,
    ):
        print("dvgo: voxel_count_views start")
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = (
            int(np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        )
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = DenseGrid(
                1,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
            )
            rays_o_ = (
                rays_o_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)
            )
            rays_d_ = (
                rays_d_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)
            )

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                rays_o = rays_o.cuda()
                vec = torch.where(
                    rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d
                ).cuda()
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng.cuda()
                interpx = (
                    t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True).cuda()
                )
                rays_pts = (
                    rays_o[..., None, :].cuda()
                    + rays_d[..., None, :].cuda() * interpx[..., None].cuda()
                )
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += ones.grid.grad > 1
        eps_time = time.time() - eps_time
        print("dvgo: voxel_count_views finish (eps time:", eps_time, "sec)")
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(
            shape
        )

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Check whether the rays hit the solved coarse geometry or not"""
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = sample_points_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist
        )[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = (
            sample_points_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist
            )
        )
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert (
            len(rays_o.shape) == 2 and rays_o.shape[-1] == 3
        ), "Only suuport point queries in [N, 3] format"

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs
        )
        interval = render_kwargs["stepsize"] * self.voxel_size_ratio

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        print(f"at line 1080 ray id = ", ray_id)
        # compute accumulated transmittance
        weights, alphainv_last = AlphasToWeights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        print(f"at line 1091 ray id = ", ray_id)
        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat(
                [viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1
            )
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3]),
            reduce="sum",
        )
        rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs["bg"]
        ret_dict.update(
            {
                "alphainv_last": alphainv_last,
                "weights": weights,
                "rgb_marched": rgb_marched,
                "raw_alpha": alpha,
                "raw_rgb": rgb,
                "ray_id": ray_id,
            }
        )

        if render_kwargs.get("render_depth", False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce="sum",
                )
            ret_dict.update({"depth": depth})

        return ret_dict
