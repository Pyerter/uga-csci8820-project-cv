import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..utility import norm_path_from_base, norm_path
from ..util.ray_util import get_rays, get_ray_directions

DATA_PATH = 'data/nerf_synthetic'
DATA_FOLDERS = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']

# Translation matrix by t
translate_t = lambda t : torch.tensor(np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32))

# Rotation matrix around angle phi
rotation_phi = lambda phi : torch.tensor(np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32))

# Rotation matrix around angle theta
rotation_theta = lambda theta : torch.tensor(np.array([
    [np.cos(theta),0,-np.sin(theta),0],
    [0,1,0,0],
    [np.sin(theta),0, np.cos(theta),0],
    [0,0,0,1],
], dtype=np.float32))

# Generate pose matrix from spherical coordinates (theta, phi, radius)
def pose_spherical(theta, phi, radius):
    cam_to_world = translate_t(radius) # Get translation of radius
    cam_to_world = rotation_phi(phi * (np.pi / 180.0)) @ cam_to_world # Rotation by phi (in radians) (azimuthal)
    cam_to_world = rotation_theta(theta * (np.pi / 180.0)) @ cam_to_world # Rotate by theta (in radians) (polar)
    cam_to_world = torch.tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32)) @ cam_to_world # Final rotation to match camera
    return cam_to_world

def load_synthetic(item_name = DATA_FOLDERS[0], test_skip = 1, half_resolution = False):
    print(f'----- Loading synthetic dataset -----')
    # Get base dir of synethetic dataset
    base_dir = norm_path_from_base(DATA_PATH)
    print(f'Fetching dataset from folder: {base_dir}')
    # 3 dataset splits
    splits = ['train', 'test', 'val']
    meta_data = {}
    # Get meta data for each split
    for split in splits:
        with open(os.path.join(base_dir, norm_path(f'{item_name}/transforms_{split}.json')), 'r') as f:
            meta_data[split] = json.load(f)

    transform = T.ToTensor()

    #print(meta_data)
    for split in splits:
        meta = meta_data[split]
        for frame in meta['frames'][::]:
            frame_path = os.path.join(base_dir, norm_path(f"{item_name}/{frame['file_path']}.png"))
            with Image.open(frame_path) as current_image:
                # Get height, width from image
                height, width = current_image.height, current_image.width
            break
        break
    # Camera angle
    camera_angle_x = float(meta['camera_angle_x'])
    # Focal length
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    # directions = get ray directions
    directions = get_ray_directions(height, width, [focal, focal])
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    #print(f'Directions shape: {directions.shape}')
    intrinsics = torch.tensor([[focal,0,width/2], [0,focal,height/2], [0,0,1]]).float()

    synth_to_opencv = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    #print(f'Synthetic to opencv matrix shape: {synth_to_opencv.shape}')

    # Collect all images, poses, and counts
    images, poses, rays, counts = [], [], [], [0]
    split_sizes = []
    debugged_ray = False
    for split in splits:
        # Get current metadata
        meta = meta_data[split]
        current_images = []
        current_poses = []
        current_rays = []
        # Skip specific indeces in data
        # Collect and read image and poses
        for frame in meta['frames']:
            frame_path = os.path.join(base_dir, norm_path(f"{item_name}/{frame['file_path']}.png"))
            with Image.open(frame_path) as img_file:
                img = transform(img_file) # -> (4, h, w)
            # If we want to downsample:
            # img = img.resize(new_shape, Image.LANCZOS)
            img = img.view(4, -1).permute(1, 0) # -> (4, h * w) -> (h * w, 4)
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:]) # Blend alpha into RGBs so that it's (h * w, 3)
            current_images.append(img) # imageio.imread(frame_path)
            pose = torch.FloatTensor(np.array(frame['transform_matrix']) @ synth_to_opencv)
            current_poses.append(np.array(pose)[np.newaxis, :, :])
            ray_origin, ray_direction = get_rays(directions, pose)
            current_rays.append(np.array(torch.cat([ray_origin, ray_direction], 1))[np.newaxis, :, :])
        # Store as RGBA np array
        current_images = (np.array(current_images) / 255.0).astype(np.float32) 
        # Store as np array
        current_poses = np.array(current_poses).astype(np.float32)
        # Store as np array
        current_rays = np.array(current_rays).astype(np.float32)

        # Append new start index of next split to end of counts
        counts.append(counts[-1] + current_images.shape[0])
        images.append(current_images)
        poses.append(current_poses)
        rays.append(current_rays)
        split_sizes.append(len(current_poses))

    # Create numbered lists containing each index in each split
    index_splits = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    print(f'Index split starts: {[split[0] for split in index_splits]}')

    # Squish lists along first axis so the dataset is one list instead of 3 splits
    poses = np.squeeze(np.concatenate(poses, axis=0), axis=1)
    images = np.concatenate(images, axis=0)
    rays = np.squeeze(np.concatenate(rays, axis=0), axis=1)
    print(f'Before stacking -> Images shape: {images.shape}, Poses shape: {poses.shape}, Rays shape: {rays.shape}')
    # And convert from shapes (length, height * width, 3) and (length, height * width, 6)
    # To (length * width * height, 3) and (length * height * width, 6)
    images = np.reshape(images, (-1, images.shape[-1]))
    rays = np.reshape(rays, (-1, rays.shape[-1]))
    print(f'After stacking -> Images shape: {images.shape}, Poses shape: {poses.shape}, Rays shape: {rays.shape}')
    # Finally convert to torch tensors
    poses = torch.FloatTensor(poses)
    images = torch.FloatTensor(images)
    rays = torch.FloatTensor(rays)

    # Get the render poses around a spherical view
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 41)[:-1]], 0)

    # Half the resolution if necessary
    if half_resolution:
        images = torch.nn.functional.interpolate(torch.from_numpy(images), size=[400, 400], mode='area').numpy()
        width = width // 2
        height = height // 2
        focal = focal / 2.0

    print(f'----- Finished Loading synthetic dataset -----')

    return images, poses, rays, render_poses, directions, intrinsics, [height, width, focal], index_splits, split_sizes

class SyntheticSet(Dataset):
    def __init__(self, item_name = DATA_FOLDERS[0], item_index = -1, split='train', test_skip = 1, half_resolution = False):
        self.split = split
        images, poses, rays, render_poses, directions, intrinsics, [height, width, focal], index_splits, split_sizes = load_synthetic(item_name if item_index < 0 or item_index >= len(DATA_FOLDERS) else DATA_FOLDERS[item_index], test_skip, half_resolution)
        # Main data
        self.images = images
        self.poses = poses
        self.rays = rays
        self.render_poses = render_poses
        self.directions = directions
        self.instrinsics = intrinsics
        self.split_sizes = split_sizes

        # Parameters
        self.batch_size = 4096
        self.number_samples = 1e6
        self.step_ratio = 0.5

        # Basic values
        self.height = height
        self.width = width
        self.focal = focal

        # Training split indeces
        self.index_splits = index_splits
        self.train_indeces = index_splits[0]
        self.test_indeces = index_splits[1]
        self.val_indeces = index_splits[2]

        # Scene variables
        self.white_bg = True
        self.hemi_r = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        self.near = self.hemi_r - 1.0
        self.far = self.hemi_r + 1.0
        self.scene_bounding_box = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.center = torch.mean(self.scene_bounding_box, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bounding_box[1] - self.center).float().view(1, 1, 3)
        

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx, split = None): # implement getitem from Dataset, idx stands for index
        if split is None: split = self.split
        split_indeces = self.train_indeces if split == 'train' else self.test_indeces if split == 'test' else self.val_indeces
        if split == 'train':
            sample = {'rays': self.rays[split_indeces[idx]],
                      'rgbs': self.images[split_indeces[idx]]}
            
        else:
            sample = {'rays': self.rays[split_indeces[idx]],
                      'rgbs': self.images[split_indeces[idx]]}
            # Missing masks here?

        return sample
    
    def grab_sample_set(self, start, stop = -1, split=None):
        if split is None: split = self.split
        split_indeces = self.train_indeces if split == 'train' else self.test_indeces if split == 'test' else self.val_indeces
        offset = split_indeces[0] * self.rays_in_image()
        if stop == -1: indeces = torch.IntTensor([start + offset])
        else: indeces = torch.IntTensor(np.arange(0, stop - start) + (start + offset))
        if split == 'train':
            sample = {'rays': self.rays[indeces],
                      'rgbs': self.images[indeces]}
            
        else:
            sample = {'rays': self.rays[indeces],
                      'rgbs': self.images[indeces]}
            # Missing masks here?

        return sample
    
    def numb_poses(self):
        return len(self.poses)
    
    def rays_in_image(self):
        return self.rays.shape[0] // len(self.poses)
    
    def number_split_images(self, split=None):
        if split is None: split = self.split
        split_indeces = self.train_indeces if split == 'train' else self.test_indeces if split == 'test' else self.val_indeces
        return len(split_indeces) // (self.rays.shape[0] // len(self.poses))
    
    def count_in_split(self, split=None):      
        if split is None: split = self.split
        for i, spl in enumerate(['train', 'test', 'val']):
            if split == spl: return self.split_sizes[i]
        return 0
    
    def rays_in_split(self, split=None):
        return self.count_in_split(split) * self.rays_in_image()

