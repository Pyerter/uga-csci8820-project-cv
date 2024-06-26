from .data_loading.data_synthetic import SyntheticSet, DATA_FOLDERS
from .tensorf.tensoRFCP import TensoRFCP
from .tensorf.tensoRFVM import TensoRFVM, TensoRFVMSplit
from .util.model_util import calculate_number_samples, voxel_number_to_resolution, RandomSampler, TVLoss
from .tensorf.rendering import render_octree_trilinear, render_model
from .utility import norm_path_from_base

import sys
import datetime
import os

import numpy as np
from tqdm import tqdm
import torch

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
parser = argparse.ArgumentParser()

class ConfigParser():
    def __init__(self):
        self.args = None
        self.initialized = False

conf = ConfigParser()

def pre_train():
    if conf.initialized:
        return
    print(f'Clearing cuda cache')
    torch.cuda.empty_cache()
    print(f'Parsing command line arguments')
    parser.add_argument('--ckpt_folder', help='The folder containing the desired checkpoint destination.')
    parser.add_argument('--render_test', default=False, type=bool, help='Render the test frames when training.')
    parser.add_argument('--iterations', default=30000, type=int, help='Number of iterations in training.')
    parser.add_argument('--only_test', default=False, help='Make sure that, instead of training, the model only runs the test splits, assuming --ckpt_holder is set.')
    parser.add_argument('--decomp_mode', default='cp', help='cp or vm, used to determine the model definition for decomposing the feature matrix.')
    parser.add_argument('--extrametrics', default=False, type=bool, help='If lpips of alexnet and vgg should be included in the metrics for rendering.')
    parser.add_argument('--test_samples', default=-1, type=int, help='Number of frames to render when testing.')
    parser.add_argument('--data_folder', default=0, type=int, help='The index of the data folder being used from the blender dataset.')
    conf.args = parser.parse_args()
    conf.initialized = True

def get_model(bb, resolution, device):
    if conf.args.decomp_mode == 'vm':
        return TensoRFVM(bb, resolution, device)
    elif conf.args.decomp_mode == 'vms':
        return TensoRFVMSplit(bb, resolution, device)
    return TensoRFCP(bb, resolution, device)

def get_dataset(split = 'train'):
    return SyntheticSet(DATA_FOLDERS[conf.args.data_folder], split=split)

def get_checkpoint_folder(checkpoint_folder_name, subfolder_name = None):
    if subfolder_name is None:
        return norm_path_from_base(f'checkpoints/{checkpoint_folder_name}')
    return norm_path_from_base(f'checkpoints/{checkpoint_folder_name}/{subfolder_name}')

def test_on_synthetic(checkpoint_folder_name = None, checkpoint_name = 'tensorf_model', dataset = None, model = None):
    pre_train()
    print(f'Asserting checkpoint folder...')
    if checkpoint_folder_name is None:
        if conf.args.ckpt_folder is not None:
            checkpoint_folder_name = conf.args.ckpt_folder
        else:
            print(f'No checkpoint folder provided for testing!')
            return
    print(f'Checkpoint folder: {checkpoint_folder_name}')
    checkpoint_path = lambda subpath = None: get_checkpoint_folder(checkpoint_folder_name, subpath)
    if model is None:
        ckpt_path = checkpoint_path(f'{checkpoint_name}.th')
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint path does not exist at location: {ckpt_path}')
            return

    if dataset is None:
        dataset = get_dataset('test')
    else:
        print(f'Testing dataset provided, no need to load.')
    white_bg = dataset.white_bg

    if model is None:
        print(f'Creating and loading model')
        scene_bb = dataset.scene_bounding_box
        resolution = [100, 100, 100]#voxel_number_to_resolution(2097156, scene_bb) # 128^3 = 2,097,156 voxels in 128 cubic grid
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = get_model(scene_bb, resolution, device)
        model.load(checkpoint)
        print(f'Done loading model!')
    else:
        print(f'No need to load already given model.')
    
    print(f'Evaluating!')
    os.makedirs(checkpoint_path('test_images'), exist_ok=True)
    render_model(dataset, model, render_octree_trilinear, 'test_images', checkpoint_path, number_visible=-1, number_samples=-1, white_bg=white_bg, compute_extra_metrics=conf.args.extrametrics, device=device, render_iterations=conf.args.test_samples)


def train_on_synthetic(checkpoint_name = 'tensorf_model'):
    pre_train()
    if conf.args.only_test:
        test_on_synthetic()
        return

    iterations = conf.args.iterations

    train_dataset = get_dataset('train')

    # Initialize and create checkpoint folders for this run
    checkpoint_folder_name = f'{checkpoint_name}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    checkpoint_folder = lambda subfolder = None: get_checkpoint_folder(checkpoint_folder_name, subfolder)
    os.makedirs(checkpoint_folder(), exist_ok=True)
    os.makedirs(checkpoint_folder('imgs_vis'), exist_ok=True)
    os.makedirs(checkpoint_folder('imgs_rgba'), exist_ok=True)
    os.makedirs(checkpoint_folder('rgba'), exist_ok=True)

    scene_bb = train_dataset.scene_bounding_box
    resolution = voxel_number_to_resolution(2097156, scene_bb) # 128^3 = 2,097,156 voxels in 128 cubic grid
    resolution_current = resolution
    number_samples = min(train_dataset.number_samples, int(np.linalg.norm(resolution)/train_dataset.step_ratio))
    upsample_list = [2000,3000,4000,5500,7000]
    update_alpha_mask_list = [2000, 4000]

    print(f'Creating model...')
    model = get_model(scene_bb, resolution, device)
    print(f'Created model!')

    lr_initial = 0.02
    lr_basis = 0.001
    lr_decay_target_ratio = 0.1
    lr_factor = lr_decay_target_ratio**(1 / iterations)
    lr_upsample_reset = 1

    optimizer_variables = model.get_optional_parameter_groups(lr_initial, lr_basis)
    optimizer = torch.optim.Adam(optimizer_variables, betas=(0.9, 0.99))

    #linear in logrithmic space
    voxel_init = 100**3
    numb_voxels_list = (torch.round(torch.exp(torch.linspace(np.log(voxel_init), np.log(voxel_init), len(upsample_list)+1))).long()).tolist()[1:]

    TV_weight_density = 0.1
    TV_weight_appearance = 0.01
    tv_reg = TVLoss()
    L1_reg_weight = 0.0
    ortho_reg_weight = 0.0

    print(f'Finished loading model parameters')

    PSNRs, PSNRs_test = [], [0]
    rays = train_dataset.rays
    images = train_dataset.images
    filter = False
    if filter:
        print(f'Filtering rays...')
        rays, images = model.filter_rays(rays, images, bb_only=True)
        print(f'Filtered rays!')

    sampler = RandomSampler(train_dataset.rays_in_split(), train_dataset.batch_size)
    print(f'Created sampler!')

    refresh_rate = 1000
    progress_bar = tqdm(range(iterations),  miniters=refresh_rate, file=sys.stdout)
    print(f'Beginning iteration of model training')
    for iteration in progress_bar:
        # Get sample in iteration
        ray_indeces = sampler.next_ids()
        # Get rays and images in sample
        rays_train, images_train = rays[ray_indeces], images[ray_indeces].to(device)

        # Render with a forward pass via the model using octree trilinear rendering
        img_map, alpha_map, depth_map, weights, uncertainty = render_octree_trilinear(rays_train, model, chunk=train_dataset.batch_size, white_bg=train_dataset.white_bg, device=device, training=True)

        # Mean squared error
        loss = torch.mean((img_map - images_train) ** 2)

        if ortho_reg_weight > 0 and model.can_compute_vector_component_diffs():
            loss_reg = model.vector_component_diffs()
            loss += ortho_reg_weight * loss_reg
            #summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)

        if L1_reg_weight > 0:
            loss_reg_L1 = model.density_L1()
            loss += L1_reg_weight * loss_reg_L1
            #summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = model.TV_loss_density(tv_reg) * TV_weight_density
            loss += loss_tv
            #summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)

        if TV_weight_appearance > 0:
            TV_weight_appearance *= lr_factor
            loss_tv = model.TV_loss_appearance(tv_reg) * TV_weight_appearance
            loss += loss_tv
            #summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        if iteration % refresh_rate == 0:
            progress_bar.set_description(f'Iteration {iteration:05d}:'
                                         + f' train_PSNR ({float(np.mean(PSNRs)):.2f})'
                                         + f' test_PSNR ({float(np.mean(PSNRs_test)):.2f})'
                                         + f' MSE ({loss:.6f})')
            PSNRs = []

        if iteration in update_alpha_mask_list:
            if resolution_current[0] * resolution_current[1] * resolution_current[2] < 256**3:
                resolution_mask = resolution_current
            new_bb = model.update_alpha_mask(resolution_mask)
            if iteration == update_alpha_mask_list[0]:
                model.shrink(new_bb)
                L1_reg_weight = 0
            

        if iteration in upsample_list:
            # Get next voxel count in list
            number_voxels = numb_voxels_list.pop(0)
            # Get new resolution
            resolution_current = voxel_number_to_resolution(number_voxels, model.bb)
            # Update samples to match voxels
            number_samples = min(number_samples, calculate_number_samples(resolution_current, model.step_ratio))
            # Upsample with the new resolution
            model.upsample_volume_grid(resolution_current)

            if lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = lr_decay_target_ratio ** (iteration / iterations)
            optimizer_variables = model.get_optional_parameter_groups(lr_initial * lr_scale, lr_basis * lr_scale)
            optimizer = torch.optim.Adam(optimizer_variables, betas=(0.9, 0.99))

    # Save the checkpoint here
    model.save(checkpoint_folder(f'{checkpoint_name}.th'))

    # TODO: Run tests and update PSNRs_test here
    if conf.args.render_test:
        train_dataset.split = 'test'
        test_on_synthetic(checkpoint_folder_name, dataset = train_dataset, model = model)
    #os.makedirs(checkpoint_folder('imgs_test_all'), exist_ok=True)
    #PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/', N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    #summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
    #print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    print(f'Yay!')
    
    

if __name__ == "__main__":
    train_on_synthetic()