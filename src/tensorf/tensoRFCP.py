import torch
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList, Linear, Module

from .tensoRF import TensoRFBase

# Represents the class for TensoRF-CP module
class TensoRFCP(TensoRFBase):
    def __init__(self, bb, grid_size, device, **kargs):
        super(TensoRFCP, self).__init__(bb, grid_size, device, **kargs)

        self.init_svd_volume(self.grid_size[0], device) # Initialize singular value decomposition (if implemented)
        
    # Initialize the singular value decomposition volume (SVD volume)
    def init_svd_volume(self, res, device):
        # Initialize density SVD component
        self.density_line = self.init_single_svd_component(self.numb_density_comps[0], self.grid_size, 0.2, device)
        # Initialize appearance SVD component
        self.appearance_line = self.init_single_svd_component(self.numb_appearance_comps[0], self.grid_size, 0.2, device)
        # Initialize basis matrix for appearance
        self.appearance_basis = Linear(self.numb_appearance_comps[0], self.appearance_dims, bias=False).to(device)

    # Initialize parameters for given components based on grid size and scale
    def init_single_svd_component(self, numb_component, grid_size, scale, device):
        line_coefficients = []
        # Iterate over vector operations order
        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i] # get vector dim
            # Append randomly initialized parameters to the line coefficients list
            line_coefficients.append(Parameter(scale * torch.randn(1, numb_component, grid_size[vec_id], 1)))
        # Convert coefficient list into parameters
        return ParameterList(line_coefficients).to(device)
    
    def get_optional_parameter_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        gradient_vars = [
            {'params': self.density_line, 'lr': lr_init_spatial},
            {'params': self.appearance_line, 'lr': lr_init_spatial},
            {'params': self.appearance_basis.parameters(), 'lr': lr_init_network}
        ]
        if isinstance(self.render_module, Module):
            gradient_vars += [{'params': self.render_module.parameters(), 'lr': lr_init_network}]
        return gradient_vars
    
    def _compute_line_coefficients(self, positions, line):
        # Create coordinate line tensor
        coordinate_line = torch.stack(tuple([positions[..., vec_id] for vec_id in self.vec_operations_order]))
        # Stack an extra dimension representing the line segment along each axis
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        # Sample density coefficients along the coordinate line using bilinear interpolation
        line_coefficient_points = None
        for i in range(3):
            current_coefficient_points = F.grid_sample(line[i], coordinate_line[[i]], align_corners=True).view(-1, *positions.shape[:1])
            line_coefficient_points = current_coefficient_points if line_coefficient_points is None else line_coefficient_points * current_coefficient_points

        return line_coefficient_points

    def compute_density_features(self, positions):
        # Compute sigma feature from density coefficients
        return torch.sum(self._compute_line_coefficients(positions, self.density_line), dim=0)
    
    def compute_appearance_features(self, positions):
        # Given density coefficients, apply the basis matrix to them and return the
        # transposition of the resulting tensor
        return self.appearance_basis(self._compute_line_coefficients(positions, self.appearance_line).T)
    
    # Create parameters for the line coefficients up sampled to a target resolution
    @torch.no_grad()
    def up_sample_vector(self, density_line_coefficients, appearance_line_coefficients, resolution_target):
        # iterate over vectors
        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i] # get vector dim
            # Create parameter for interpolating to upsample at target resolution for density and appearance
            density_line_coefficients[i] = Parameter(F.interpolate(density_line_coefficients[i].data, size=(resolution_target[vec_id], 1), mode='bilinear', align_corners=True))
            appearance_line_coefficients[i] = Parameter(F.interpolate(appearance_line_coefficients[i].data, size=(resolution_target[vec_id], 1), mode='bilinear', align_corners=True))
        return density_line_coefficients, appearance_line_coefficients
    
    # Up sample to a target resolution and update the step size
    @torch.no_grad()
    def up_sample_volume_grid(self, resolution_target):
        # Calculate up sampled vector
        self.density_line, self.appearance_line = self.up_sample_vector(self.density_line, self.appearance_line, resolution_target)
        # Update step size
        self.update_step_size(resolution_target)
        # Print an update message
        print(f'Upsampling to target resolution: {resolution_target}')

    def shrink(self, new_bb):
        print(f'----- Shrinking...')
        pos_min, pos_max = new_bb # get min/max from bounding box
        # Compute top left and bottom right corners based on old bounding box
        top_left, bottom_right = (pos_min - self.bb[0]) / self.units, (pos_max - self.bb[0]) / self.units
        # Round those values
        top_left, bottom_right = torch.round(top_left).long(), torch.round(bottom_right).long() + 1
        bottom_right = torch.stack([bottom_right, self.grid_size]).amin(0) # clamp bottom right corner by grid size

        # Iterate over vectors
        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i] # get vector dim
            # Create new parameters for the density and appearance lines
            self.density_line[i] = Parameter(self.density_line[i].data[...,top_left[vec_id]:bottom_right[vec_id], :])
            self.appearance_line[i] = Parameter(self.appearance_line[i].data[...,top_left[vec_id]:bottom_right[vec_id], :])

        # Check if the bounding box needs to be corrected
        if not torch.all(self.alpha_mask.grid_size == self.grid_size):
            # Calculated the corrected corners with respect to self.grid_size
            top_left_corrected, bottom_right_corrected = top_left / (self.grid_size - 1), bottom_right / (self.grid_size - 1)
            # Create blank tesor
            bb_corrected = torch.zeros_like(new_bb)
            # Fill tensor according to the corrected bounding box corners
            bb_corrected[0] = (1 - top_left_corrected) * self.bb[0] + top_left_corrected * self.bb[1]
            bb_corrected[1] = (1 - bottom_right_corrected) * self.bb[0] + bottom_right_corrected * self.bb[1]
            print(f'    Corrected axis-align bounding box from {new_bb} to {bb_corrected}')
            new_bb = bb_corrected
        
        new_size = bottom_right - top_left
        self.bb = new_bb
        self.update_step_size((new_size[0], new_size[1], new_size[2]))
        print(f'----- Finished shrinking.')

    def density_L1(self):
        total = 0
        for i in range(len(self.density_line)):
            total += torch.mean(torch.abs(self.density_line[i]))
        return total
    
    def _TV_loss_line(self, reg, line):
        total = 0
        for i in range(len(line)):
            total += reg(line[i]) * 1e-3
        return total
    
    def TV_loss_density(self, reg):
        return self._TV_loss_line(reg, self.density_line)
    
    def TV_loss_appearance(self, reg):
        return self._TV_loss_line(reg, self.appearance_line)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, appearance_line_coef, resolution_target):

        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(resolution_target[vec_id], 1), mode='bilinear', align_corners=True))
            appearance_line_coef[i] = torch.nn.Parameter(
                F.interpolate(appearance_line_coef[i].data, size=(resolution_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, appearance_line_coef
    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.appearance_line = self.up_sampling_Vector(self.density_line, self.appearance_line, res_target)
        self.update_step_size(res_target)
        print(f'upsamping to {res_target}')
        
