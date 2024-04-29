import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load


def clamp(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)


def total_variation_add_grad(param, grad, wx, wy, wz, dense_mode):
    sz_i, sz_j, sz_k = param.shape[-3], param.shape[-2], param.shape[-1]
    wx, wy, wz = wx / 6, wy / 6, wz / 6

    padded = torch.nn.functional.pad(param, (1, 1, 1, 1, 1, 1), mode="replicate")
    grad_to_add = torch.zeros_like(param)

    # Calculate differences and clamp
    if dense_mode or torch.any(grad != 0):
        diff_k_minus = clamp(padded[..., :-2] - param, -1.0, 1.0)
        diff_k_plus = clamp(padded[..., 2:] - param, -1.0, 1.0)
        diff_j_minus = clamp(padded[..., :-2, 1:-1] - param, -1.0, 1.0)
        diff_j_plus = clamp(padded[..., 2:, 1:-1] - param, -1.0, 1.0)
        diff_i_minus = clamp(padded[..., 1:-1, :-2] - param, -1.0, 1.0)
        diff_i_plus = clamp(padded[..., 1:-1, 2:] - param, -1.0, 1.0)

        # Weighted sum of clamped differences
        grad_to_add += wz * (diff_k_minus + diff_k_plus)
        grad_to_add += wy * (diff_j_minus + diff_j_plus)
        grad_to_add += wx * (diff_i_minus + diff_i_plus)

    # Add computed gradients to the input gradient tensor
    grad += grad_to_add


def maskcache_lookup(world, xyz, xyz2ijk_scale, xyz2ijk_shift):
    # Compute the indices in the world tensor for each point in xyz
    ijk = torch.round(xyz * xyz2ijk_scale + xyz2ijk_shift).to(torch.int64)
    n_pts = xyz.size(0)

    # Initialize output tensor
    out = torch.zeros(n_pts, dtype=torch.bool, device=world.device)

    # Check boundaries and index world tensor
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < world.size(0))
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < world.size(1))
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < world.size(2))
    )

    valid_indices = ijk[valid]
    flat_index = (
        valid_indices[:, 0] * world.size(1) * world.size(2)
        + valid_indices[:, 1] * world.size(2)
        + valid_indices[:, 2]
    )
    out[valid] = world.view(-1)[flat_index]  # Use flattened indexing

    return out


class DenseGrid(nn.Module):
    """A dense grid module for volume rendering that supports operations like trilinear interpolation."""

    def __init__(self, channels, world_size, xyz_min, xyz_max):
        super().__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        # print(f"world size shape: {world_size.shape}")

        # self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        """Perform a forward pass using grid sampling with bilinear interpolation."""
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        # Normalize and flip coordinates to match grid sampling requirements
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode="bilinear", align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        return out.squeeze(-1) if self.channels == 1 else out

    def scale_volume_grid(self, new_world_size):
        """Scale the volume grid to a new size using trilinear interpolation."""
        mode = "trilinear" if self.channels > 1 else "bilinear"
        self.grid = nn.Parameter(
            F.interpolate(
                self.grid.data, size=new_world_size, mode=mode, align_corners=True
            )
        )

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        """Add gradient updates calculated from a total variation loss directly to the grid's gradient."""
        total_variation_add_grad(self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        """Return the current state of the dense grid."""
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        """In-place subtraction from the grid."""
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}"


class MaskGrid(nn.Module):
    """A module for handling mask grids that define free space and occupied space."""

    def __init__(
        self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None
    ):
        super().__init__()
        if path:
            state = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(
                state["model_state_dict"]["density.grid"],
                kernel_size=3,
                padding=1,
                stride=1,
            )
            alpha = 1 - torch.exp(
                -F.softplus(density + state["model_state_dict"]["act_shift"])
                * state["model_kwargs"]["voxel_size_ratio"]
            )
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min, xyz_max = torch.Tensor(
                state["model_kwargs"]["xyz_min"]
            ), torch.Tensor(state["model_kwargs"]["xyz_max"])
        else:
            mask = mask.bool()
            xyz_min, xyz_max = torch.Tensor(xyz_min), torch.Tensor(xyz_max)

        self.register_buffer("mask", mask)
        xyz_len = xyz_max - xyz_min
        mid = torch.Tensor(list(mask.shape)) - 1
        mid = mid.cuda()
        self.register_buffer("xyz2ijk_scale", mid / xyz_len)
        self.register_buffer("xyz2ijk_shift", -xyz_min * self.xyz2ijk_scale)

    def forward(self, xyz):
        """Forward pass for querying the mask grid."""
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        return mask.reshape(shape)
