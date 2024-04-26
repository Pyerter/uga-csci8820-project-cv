import torch
from torch.nn import Module

import numpy as np

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