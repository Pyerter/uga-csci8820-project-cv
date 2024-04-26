from .tensoRF import TensoRFBase

import torch
import numpy as np

def render_octree_trilinear(rays, model: TensoRFBase, chunk=4096, numb_samples=-1, white_bg=True, training=False, device='cuda'):

    images, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    numb_rays = rays.shape[0]
    for chunk_index in range(numb_rays // chunk + int(numb_rays % chunk > 0)):
        ray_chunks = rays[chunk_index * chunk:(chunk_index + 1) * chunk].to(device)
    
        image_map, depth_map = model(ray_chunks=ray_chunks, training=training, white_bg=white_bg, numb_samples=numb_samples)

        images.append(image_map)
        depth_maps.append(depth_map)
    
    return torch.cat(images), None, torch.cat(depth_maps), None, None