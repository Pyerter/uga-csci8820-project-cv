import time
import torch
import numpy as np
from .dvgo import get_rays_of_a_view


# Decorator to disable gradient calculations
@torch.no_grad()
def compute_bounding_box_coarse(model_class, model_attrs, model_path, thresh=0.0001):
    """Compute the bounding box of active geometry from a pre-trained model.

    Args:
        model_class: The class of the model to load.
        model_path: Path to the pre-trained model's checkpoint file.
        thresh: Threshold for considering a density value to be part of the geometry.

    Returns:
        A tuple of minimum and maximum coordinates of the bounding box.
    """
    print("compute_bbox_by_coarse_geo: start")
    start_time = time.time()

    # Load the model from the checkpoint
    # open file from model_path
    model_file = open(model_path, "rb")
    ckpt = torch.load(model_file)
    model = model_class(**ckpt["model_kwargs"], model_attrs=model_attrs)
    model.load_state_dict(ckpt["model_state_dict"])
    model_file.close()

    # Create a grid of interpolated points within the model's defined world space
    interp = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, model.world_size[0]),
            torch.linspace(0, 1, model.world_size[1]),
            torch.linspace(0, 1, model.world_size[2]),
        ),
        -1,
    )
    # Compute the world coordinates of the dense grid points
    dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp

    # Compute the density at each grid point and activate it
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)

    # Filter out active voxels based on the threshold
    mask = alpha > thresh
    active_xyz = dense_xyz[mask]

    # Calculate the minimum and maximum active coordinates
    xyz_min = dense_xyz.amin(0)
    xyz_max = dense_xyz.amax(0)

    print("compute_bbox_by_coarse_geo: xyz_min", xyz_min)
    print("compute_bbox_by_coarse_geo: xyz_max", xyz_max)

    elapsed_time = time.time() - start_time
    print("compute_bbox_by_coarse_geo: finish (eps time:", elapsed_time, "secs)")
    return model.xyz_min, model.xyz_max


def compute_bounded_bounding_box_frustrum_cam(HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
        )
        pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    print(f"xyz_min: {xyz_min}, xyz_max: {xyz_max}")
    return xyz_min, xyz_max


def compute_bounding_box_frustrum_cam(HW, Ks, poses, i_train, near, far):
    """Compute the bounding box using camera frustrum for a given dataset."""
    print("compute_bbox_by_cam_frustrm: start")

    xyz_min, xyz_max = compute_bounded_bounding_box_frustrum_cam(
        HW, Ks, poses, i_train, near, far
    )

    print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
    print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
    print("compute_bbox_by_cam_frustrm: finish")

    return xyz_min, xyz_max
