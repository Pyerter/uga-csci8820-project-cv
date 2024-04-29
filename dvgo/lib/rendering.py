import os
import torch
import numpy as np
from tqdm import tqdm
import imageio

from .dvgo import get_rays_of_a_view
from .evaluation import rgb_ssim, rgb_lpips


def to8b(image):
    """Convert image to 8-bit format."""
    return (255 * np.clip(image, 0, 1)).astype(np.uint8)


@torch.no_grad()
def render_viewpoints(
    model,
    render_poses,
    HW,
    Ks,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
    render_video_flipy=False,
    render_video_rot90=0,
):
    """
    Render images from provided viewpoints and evaluate against ground truths if provided.

    Args:
        model: The rendering model.
        render_poses: List of camera poses for rendering.
        HW: List of tuples (height, width) for each image.
        Ks: List of intrinsic matrices for each image.
        render_kwargs: Additional keyword arguments for the rendering function.
        gt_imgs: List of ground truth images for evaluation.
        savedir: Directory to save rendered images.
        dump_images: Flag to enable saving of images.
        render_factor: Factor to scale down the image dimensions and intrinsics.
        render_video_flipy: Flag to vertically flip the rendered videos.
        render_video_rot90: Number of times to rotate the rendered videos by 90 degrees.
        eval_ssim: Flag to enable SSIM evaluation.
        eval_lpips_alex: Flag to enable LPIPS evaluation using AlexNet.
        eval_lpips_vgg: Flag to enable LPIPS evaluation using VGG.

    Returns:
        Tuple of arrays containing rendered RGB images, depth maps, and background masks.
    """
    assert (
        len(render_poses) == len(HW) == len(Ks)
    ), "Mismatch in number of poses, sizes, and intrinsics."

    if render_factor != 0:
        HW = (np.array(HW) / render_factor).astype(int)
        Ks = np.array(Ks)
        Ks[:, :2, :3] /= render_factor

    rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i, c2w in enumerate(tqdm(render_poses, desc="Rendering images")):
        H, W = HW[i]
        K = Ks[i]
        c2w_tensor = torch.Tensor(c2w).to(model.device)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, c2w_tensor)
        rays_o, rays_d, viewdirs = (
            rays_o.flatten(0, -2),
            rays_d.flatten(0, -2),
            viewdirs.flatten(0, -2),
        )

        render_result = model.render_chunked(
            rays_o,
            rays_d,
            viewdirs,
            keys=["rgb_marched", "depth", "alphainv_last"],
            **render_kwargs,
        )
        rgb, depth, bgmap = (
            render_result["rgb_marched"].cpu().numpy(),
            render_result["depth"].cpu().numpy(),
            render_result["alphainv_last"].cpu().numpy(),
        )

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

        if gt_imgs is not None:
            psnrs.append(-10.0 * np.log10(np.mean(np.square(rgb - gt_imgs[i]))))
            ssims.append(rgb_ssim(rgb, gt_imgs[i], max_val=1))

    # Post-processing: flip and rotate images if specified
    process_video_effects(rgbs, depths, bgmaps, render_video_flipy, render_video_rot90)

    # Optionally save images to disk
    save_images(rgbs, savedir)

    # Log evaluation results
    log_evaluation_results(psnrs, ssims)

    return np.array(rgbs), np.array(depths), np.array(bgmaps)


def process_video_effects(rgbs, depths, bgmaps, flipy, rot90):
    """Apply video effects such as flipping and rotation."""
    if flipy:
        rgbs, depths, bgmaps = [np.flip(arr, axis=0) for arr in (rgbs, depths, bgmaps)]
    if rot90:
        rgbs, depths, bgmaps = [
            np.rot90(arr, k=rot90, axes=(0, 1)) for arr in (rgbs, depths, bgmaps)
        ]


def save_images(rgbs, savedir):
    """Save rendered images to disk."""
    os.makedirs(savedir, exist_ok=True)
    for i, rgb in enumerate(tqdm(rgbs, desc="Saving images")):
        filename = os.path.join(savedir, f"{i:03d}.png")
        imageio.imwrite(filename, to8b(rgb))


def log_evaluation_results(psnrs, ssims):
    """Log evaluation metrics to console."""
    print(f"Testing PSNR: {np.mean(psnrs):.2f} (avg)")
    print(f"Testing SSIM: {np.mean(ssims):.2f} (avg)")
