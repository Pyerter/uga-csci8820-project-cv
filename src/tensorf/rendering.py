from .tensoRF import TensoRFBase
from ..util.model_util import visualize_depth_numpy, rgb_ssim, rgb_lpips

import os
import sys
import tqdm
import imageio
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

@torch.no_grad()
def render_model(dataset, model, renderer, export_folder, checkpoint_path, number_visible = 5, save_prefix = '', number_samples = -1, white_bg = False, compute_extra_metrics = True, device = 'cuda'):
    # Create lists for evaluation metrics
    PSNRs, img_maps, depth_maps, SSIMs = [], [], [], []
    lpips_alex, lpips_vgg = [], []

    os.makedirs(checkpoint_path(export_folder), exist_ok=True)

    try: tqdm._instances.clear()
    except: pass

    near, far = dataset.near, dataset.far
    eval_interval = 1 if number_visible < 0 else max(dataset.rays.shape[0] // number_visible, 1)
    iterations = dataset.rays.shape[0]
    indeces = list(range(0, dataset.rays.shape[0], eval_interval))
    print(f'Beginning evaluation!')
    for index, samples in tqdm.tqdm(enumerate(dataset.rays[0::eval_interval]), file=sys.stdout):
        height, width = dataset.height, dataset.width
        rays = samples.view(-1, samples.shape[-1])
        #print(f'Stopping short!')
        #return

        img_map, _, depth_map, _, _ = renderer(rays, model, chunk=dataset.batch_size, numb_samples=number_samples, white_bg=white_bg, device=device, training=False)
        img_map = img_map.clamp(0.0, 1.0)
        img_map, depth_map = img_map.reshape(height, width, 3).cpu(), depth_map.reshape(height, width).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), [near, far])
        if len(dataset.images):
            gt_img = dataset.images[indeces[index]].view(height, width, 3)
            loss = torch.mean((img_map - gt_img) ** 2) # Mean squared error
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(img_map, gt_img, 1)
                lpips_a = rgb_lpips(gt_img.numpy(), img_map.numpy(), 'alex', model.device)
                lpips_v = rgb_lpips(gt_img.numpy(), img_map.numpy(), 'vgg', model.device)
                SSIMs.append(ssim)
                lpips_alex.append(lpips_a)
                lpips_vgg.append(lpips_v)

        img_map = (img_map.numpy() * 255).astype('uint8')
        img_maps.append(img_map)
        depth_maps.append(depth_map)

        imageio.imwrite(checkpoint_path(f'{export_folder}/{save_prefix}{index:03d}.png'), img_map)
        img_map = np.concatenate((img_map, depth_map), axis=1)
        imageio.imwrite(checkpoint_path(f'{export_folder}/rgbd/{save_prefix}{index:03d}.png'), img_map)
        break

    imageio.mimwrite(checkpoint_path(f'{export_folder}/{save_prefix}video.mp4'), np.stack(img_maps), fps=30, quality=10)
    imageio.mimwrite(checkpoint_path(f'{export_folder}/{save_prefix}depth-video.mp4'), np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        metrics = [psnr]
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(SSIMs))
            lpips_a = np.mean(np.asarray(lpips_alex))
            lpips_v = np.mean(np.asarray(lpips_vgg))
            metrics += [ssim, lpips_a, lpips_v]
        np.savetxt(checkpoint_path(f"{export_folder}/{save_prefix}mean{'s' if compute_extra_metrics else ''}.txt"), np.asarray(metrics))

    return PSNRs

