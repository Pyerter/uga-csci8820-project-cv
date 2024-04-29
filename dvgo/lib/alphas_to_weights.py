import os, time
import torch
import numpy as np
from torch.utils.cpp_extension import load
from torch import nn
import torch.nn.functional as F
from tqdm import trange, tqdm


def alpha_to_weight(alpha, ray_id, n_rays):
    n_pts = alpha.shape[0]

    # Initialize tensors
    weight = torch.zeros_like(alpha)
    T = torch.ones_like(alpha)
    alphainv_last = torch.ones(n_rays, device=alpha.device, dtype=alpha.dtype)
    i_start = torch.zeros(n_rays + 1, dtype=torch.int64, device=alpha.device)
    i_end = torch.zeros(n_rays + 1, dtype=torch.int64, device=alpha.device)

    # Setting up start and end indices for each ray
    counts = torch.bincount(ray_id, minlength=n_rays)
    i_end[:-1] = torch.cumsum(counts, dim=0)
    i_start[1:] = i_end[:-1].clone()
    i_end[-1] = n_pts  # Ensure the last index is set correctly to total points

    # Ensure i_end for the last ray doesn't exceed n_pts
    i_end = torch.clamp(i_end, max=n_pts)

    # Process each ray
    for i in range(n_rays):
        start = i_start[i].item()
        end = i_end[i].item()

        T_cum = 1.0
        j = start
        for j in range(start, end):
            T[j] = T_cum
            weight[j] = T_cum * alpha[j]
            T_cum *= 1 - alpha[j]
            if T_cum < 1e-3:
                break
        i_end[i] = j + 1 if j < end else end
        alphainv_last[i] = T_cum

    return weight, T, alphainv_last, i_start, i_end


def alpha_to_weight_backward(
    alpha, weight, T, alphainv_last, i_start, i_end, grad_weights, grad_last
):
    grad = torch.zeros_like(alpha)

    print(f"i_start: {len(i_start) - 1}")
    for i_ray in range(len(i_start) - 1):
        start = i_start[i_ray].item()
        end = i_end[i_ray].item()

        # Reverse accumulation of gradients
        back_cum = grad_last[i_ray] * alphainv_last[i_ray]
        for i in range(end - 1, start - 1, -1):
            grad[i] = grad_weights[i] * T[i] - back_cum / (1 - alpha[i] + 1e-10)
            back_cum += grad_weights[i] * weight[i]

    return grad


class AlphasToWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = alpha_to_weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = alpha_to_weight_backward(
            alpha,
            weights,
            T,
            alphainv_last,
            i_start,
            i_end,
            grad_weights,
            grad_last,
        )
        return grad, None, None
