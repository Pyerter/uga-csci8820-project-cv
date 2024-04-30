import torch
from torch.nn import Module

import numpy as np
import cv2
import scipy
import lpips

def voxel_number_to_resolution(numb_voxels, bounding_box):
    pos_min, pos_max = bounding_box
    dim = len(pos_min)
    diff = (pos_max - pos_min)
    voxel_size = (diff.prod() / numb_voxels).pow(1 / dim)
    return (diff / voxel_size).long().tolist()

class RandomSampler():
    def __init__(self, total, batch_size):
        self.total = total
        self.batch_size = batch_size
        self.current = total
        self.indeces = None

    def next_ids(self):
        self.current += self.batch_size
        if self.current + self.batch_size > self.total:
            self.indeces = torch.LongTensor(np.random.permutation(self.total))
            self.current = 0
        return self.indeces[self.current : self.current + self.batch_size]
    
class IterativeSampler():
    def __init__(self, total, batch_size):
        self.total = total
        self.batch_size = batch_size
        self.current = total
        self.indeces = torch.LongTensor(np.arange(0, self.total))

    def next_ids(self):
        self.current += self.batch_size
        if self.current + self.batch_size > self.total:
            self.current = 0
        return self.indeces[self.current : self.current + self.batch_size]
    
class TVLoss(Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        height_x = x.size()[2]
        width_x = x.size()[3]
        count_height = self._tensor_size(x[:,:,1:,:])
        count_width = self._tensor_size(x[:,:,:,1:])
        count_width = max(count_width, 1)
        height_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :height_x-1, :]),2).sum()
        width_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :width_x-1]),2).sum()
        return self.TVLoss_weight * 2 * (height_tv / count_height + width_tv / count_width) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
def calculate_number_samples(resolution, step_ratio=0.5):
    return int(np.linalg.norm(resolution) / step_ratio)

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Reference: https://github.com/apchenstu/TensoRF/blob/main/utils.py#L11 
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi)/(ma - mi + 1e-8) # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

# Structural similarity index measure
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    """
    Reference: https://github.com/apchenstu/TensoRF/blob/main/utils.py#L89
    and originally: https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    """
    Reference: https://github.com/apchenstu/TensoRF/blob/main/utils.py#L72 
    """
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()