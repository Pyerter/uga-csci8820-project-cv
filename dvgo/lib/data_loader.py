import os
import json
import torch
import numpy as np
import imageio

# Helper functions for transformation matrices


def translate_z(t, device):
    """Creates a translation matrix for translating along the z-axis."""
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rotate_y(phi, device):
    """Creates a rotation matrix around the y-axis by phi degrees."""
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
    ).float()


def rotate_z(theta, device):
    """Creates a rotation matrix around the z-axis by theta degrees."""
    return torch.Tensor(
        [
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
    ).float()


def camera_to_world_spherical(theta, phi, radius, device):
    """Generates a camera-to-world transformation matrix for a spherical coordinate."""
    c2w = translate_z(radius, device)
    c2w = rotate_y(phi / 180.0 * np.pi, device) @ c2w
    c2w = rotate_z((theta / 180.0 * np.pi).cpu(), device) @ c2w
    c2w = torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def data_loader(
    datadir,
    device,
    skip_test=1,
):
    """Loads data from disk and processes it into training, validation, and test splits.

    Args:
        datadir: Directory where metadata is located.
        skip_test: Skips test data loading if set to more than 1.
        device: Device to load tensors (e.g., 'cpu' or 'cuda:0').

    Returns:
        A dictionary containing loaded and processed data.
    """
    # Load metadata for splits
    splits = ["train", "val", "test"]
    metas = {
        s: json.load(open(os.path.join(datadir, f"transforms_{s}.json"), "r"))
        for s in splits
    }

    all_imgs, all_poses = [], []
    counts = [0]

    for s in splits:
        images = []
        poses = []
        skip = 1 if s == "train" or skip_test == 0 else skip_test

        # Load images and corresponding poses
        for frame in metas[s]["frames"][::skip]:
            fname = os.path.join(datadir, frame["file_path"] + ".png")
            images.append(imageio.imread(fname))
            poses.append(
                torch.tensor(
                    frame["transform_matrix"], dtype=torch.float32, device=device
                )
            )

        # Convert list of images to a single numpy array before converting to tensor
        images = (
            np.array(images, dtype=np.float32) / 255.0
        )  # Normalization included here
        images = torch.tensor(images, device=device)  # Conversion to tensor
        poses = torch.stack(poses)
        counts.append(counts[-1] + len(images))
        all_imgs.append(images)
        all_poses.append(poses)

    i_split = [torch.arange(counts[i], counts[i + 1], device=device) for i in range(3)]

    # Concatenate all images and poses
    images = torch.cat(all_imgs, 0)
    poses = torch.cat(all_poses, 0)

    # Calculate intrinsic camera matrix
    H, W = images.shape[2], images.shape[3]
    camera_angle_x = metas["train"]["camera_angle_x"]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Generate synthetic render poses
    render_poses = torch.stack(
        [
            camera_to_world_spherical(angle, -30.0, 4.0, device=device)
            for angle in torch.linspace(-180, 180, 160 + 1)[:-1]
        ],
        0,
    )

    near, far = 2.0, 6.0  # Define clipping planes

    K = torch.tensor(
        [[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )  # Assemble camera matrix

    Ks = K[None].expand(len(poses), -1, -1)

    data_dict = {
        "hwf": [H, W, focal],
        "HW": torch.tensor([im.shape[1:3] for im in images], device=device),
        "Ks": Ks,
        "near": near,
        "far": far,
        "i_train": i_split[0],
        "i_val": i_split[1],
        "i_test": i_split[2],
        "poses": poses,
        "render_poses": render_poses,
        "images": images,
    }
    print(f"Loaded blender {images.shape, render_poses.shape, [H, W, focal], datadir}")
    return data_dict
