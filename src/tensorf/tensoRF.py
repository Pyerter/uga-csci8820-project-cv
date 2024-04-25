import numpy as np
import torch.nn.functional as F
from torch.nn import Module, Linear, Sequential, ReLU
from torch import LongTensor
import torch

def positional_encodings(positions, frequencies):
        frequency_bands = (2**torch.arange(frequencies).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * frequency_bands).reshape(positions.shape[:-1] + (frequencies * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw_to_alpha_compute(sigma, distances):
    alpha = 1.0 - torch.exp(-sigma * distances) # Calculate alpha values on sigma and distances: e^(-sigma*dists)
    # Cumulative product calculation:
    # Get transmittance of light (1 - alpha) along each point in alpha mask
    # Calculate transmittance across a sequence of points (cumulative product)
    t = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1), -1)
    # Calculate weights based on transmittance and absorbtion, getting the weight of each ray that will
    # help determine the final rgb color
    weights = alpha * t[:, :-1]
    return alpha, weights, t[:, -1:]

# Forward function directly passing RGB values forward
def RGBRender(positions, view_directions, features):
    return features # Directly return simply the RGB features

# Placeholder
# TODO: Implement eval_sh_bases - spherical harmonics functions
def eval_sh_bases(some, stuff):
    pass

# Forward function using spherical harmonics based on the view directions
def SHRender(positions, view_directions, features):
    return RGBRender(positions, view_directions, features)
    #sh_mult = eval_sh_bases(2, view_directions)[:, None] # Calculate spherical harmonics
    #out = features.view(-1, 3, sh_mult.shape[-1])
    #out = torch.relu(torch.sum(sh_mult * out, dim=-1) + 0.5)
    #return out

class AlphaGridMask(Module):
    def __init__(self, device, bb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device # save device
        self.bb = bb.to(self.device) # get axis aligned bounding box
        self.bb_size = self.bb[1] - self.bb[0] # get bounding box size
        self.inverse_grid_size = 2.0 / self.bb_size # get the inberse grid size, normalizing coords
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:]) # volume data
        self.grid_size = LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume[-3]]).to(self.device) # grid sizes

    # Get the alpha values at given positions
    def sample_alpha(self, positions):
        positions = self.normalize_coords(positions) # normalize coords
        # Get grid sample
        alpha_values = F.grid_sample(self.alpha_volume, positions.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_values

    # Normalize coords using the bounding box and inverse grid size
    def normalize_coords(self, positions):
        return (positions - self.bb[0]) * self.inverse_grid_size - 1


# Represents a torch module that functions, by default, as a multi layer
# perceptron with positional encodings.
class RenderModule(Module):
    def __init__(self, in_channels, view_posenc = 6, pos_posenc = -1, feature_posenc = -1, feature_count = 128):
        # In multilayer perceptron count is equal to in channels plus additional positional encoding
        # channels, calculated as 3 + 2 * encoding * 3, simplified to 3 * (1 + 2 * view_posenc).
        # One channel is a regular 3D grid, One channel is the geometry grid, one channel is the appearance grid:
        # G, G_g, G_c

        # Channels = in_channels + 3 + 2 * view_posenc * 3 = in_channels + 3 + 6 * view_posenc
        # if pos_posenc: + 3 + 2 * pos_posenc * 3 = + 3 + 6 * pos_posenc
        # if feature_posenc: + 2 * feature_posenc * in_channels
        self.in_mlp_count = 3 + 6 * view_posenc + in_channels # In channels + view positional encodings
        if pos_posenc > 0: self.in_mlp_count += 3 + 6 * pos_posenc # + any position positional encodings
        if feature_posenc > 0: self.in_mlp_count += 2 * feature_posenc * in_channels # + any feature positional encodings

        # Cache the view, position, and feature positional encodings
        self.view_posenc = view_posenc
        self.pos_posenc = pos_posenc
        self.feature_posenc = feature_posenc

        layer1 = Linear(self.in_mlp_count, feature_count)
        layer2 = Linear(feature_count, feature_count)
        layer3 = Linear(feature_count, 3)

        self.mlp = Sequential(layer1, ReLU(inplace=True), layer2, ReLU(inplace=True), layer3) # Create mlp
        torch.nn.init.constant_(self.mlp[-1].bias, 0) # Initialize bias terms to 0 in last dimension

    def forward(self, positions, view_directions, features):
        in_data = [features, view_directions] # Input data list

        # If position positional encodings
        if self.pos_posenc > 0:
            # Append position encodings to the data
            in_data += [positional_encodings(positions, self.pos_posenc)]
        # If feature positional encodings
        if self.feature_posenc > 0:
            # Append feature encodings to the data
            in_data += [positional_encodings(features, self.feature_posenc)]
        # If view positional encodings (which should be always)
        if self.view_posenc > 0:
            # Append view encodings to the data
            in_data += [positional_encodings(view_directions, self.view_posenc)]
        
        # Concatenate all of the data along the last dimension
        mlp_in = torch.cat(in_data, dim=-1)
        # Pass through the mlp
        out = self.mlp(mlp_in)
        out = torch.sigmoid(out)
        return out # Return the predicted RGB values for each point in the volume
        


# The base class for both TensoRF-CP and TensoRF-VM.
# Both classes have many similarities, so it's easy to base them off
# of a single class.
class TensoRFBase(Module):
    def __init__(self, bb, grid_size, device, **kwargs):

        self.bb = bb # The axis-aligned bounding box
        self.grid_size = grid_size # The size of the tensor grid
        self.device = device # Device that performs tensor operations

        self.numb_density_comps  = 8 # Number of components used for the density representation
        self.numb_appearance_comps = 24 # Number of components used for the appearance representation
        self.appearance_dims = 27 # Dimensionality of the appearance
        self.numb_features = 128 # Number of features

        self.shading_mode = 'MLP_PE' # Shading mode for rendering, such as MLP (multi-layer perceptron),
        # SH (spherical harmonics), or RGB (yes, it's red green blue)

        self.alpha_mask = None # Optional alpha mask for the tensor - binary mask for which parts of volume to render
        self.alpha_mask_threshold = 0.001 # Threshold for the alpha mask values

        self.near_far_ray_dists = [2.0, 6.0] # The near and far distances for ray tracing
        self.density_shift = -10 # ? Shift parameter for density computation - used to adjust density computations 
        # negative shifts will darken the image and positive shifts will brighten the image
        self.distance_scale = 25 # Scalar value for distance computation
        self.ray_march_weight_threshold = 0.0001 # ? Threshold for the ray marching weight
        
        self.pos_posenc = 6 # ? Parameter for positional positional encoding
        self.view_posenc = 6 # Parameter for view positional encoding
        self.feature_posenc = 6 # Parameter for feature positional encoding
        self.step_ratio = 2.0 # ? Ratio used for step size computation

        self.feature_density_act_funct = 'softplus' # The activation function for feature to density conversions

        # Mode definitions
        self.mat_dim_pairs = [[0, 1], [0, 2], [1, 2]] # Perform matrix operations on these pairs
        self.vec_operations_order = [2, 1, 0] # Order that vector dimensions will be operated on
        self.comp_weights = [1, 1, 1] # The weights across the components

        self.update_step_size(self.grid_size) # Update the step size

        self.init_svd_volume(self.grid_size[0], device) # Initialize singular value decomposition (if implemented)

        # Select the function that takes a feature vector and viewing direction to compute a color

        # Use the multilayer perceptron module
        if self.shading_mode.startswith('MLP'):
            if self.shading_mode == 'MLP_PE':
                self.feature_posenc = -1 # Ensure features are not used, only view and positions
            if self.shading_mode == 'MLP_Feature':
                self.pos_posenc = -1 # Ensure positions are not used, only view and features
            if self.shading_mode == 'MLP':
                # Ensure both positions and features are not used, only view
                self.pos_posenc = -1
                self.feature_posenc = -1

            # Set the module
            self.render_module = RenderModule(self.appearance_dims, self.view_posenc, self.pos_posenc, self.feature_posenc, self.numb_features).to(device)

        # Use the spherical harmonics rendering function
        elif self.shading_mode == 'SH':
            self.render_module = SHRender

        # Use the simple RGB feature pass rendering function
        elif self.shading_mode == 'RGB':
            self.render_module = RGBRender

        # Uh oh!
        else:
            print(f'Invalid shading module: {self.shading_mode}')
            exit()

        print(f'Positional encodings > position : {self.pos_posenc}, view : {self.view_posenc}, features : {self.feature_posenc}')
        print(f'Render module is: {self.shading_mode}')

        pass

    def update_step_size(self, grid_size):
        print(f'Axis-aligned bounding box: {self.bb.view(-1)}')
        print(f'Grid size: {grid_size}')
        self.bb_size = self.bb[1] - self.bb[0]
        self.inv_bb_size = 2.0 / self.bb_size
        self.grid_size = torch.LongTensor(grid_size).to(self.device)
        self.units = self.bb_size / (self.grid_size - 1)
        self.step_size = torch.mean(self.units) * self.step_ratio
        self.bb_diaganol = torch.sqrt(torch.sum(torch.square(self.bb_size)))
        self.number_samples = int((self.bb_diaganol / self.step_size).item()) + 1
        print(f'Sampling step size: {self.step_size}')
        print(f'Sampling number: {self.number_samples}')

    # Can be implemented by child class.
    # Initializes the singular value decomposition.
    def init_svd_volume(self, resolution, device):
        pass

    def compute_features(self, positions):
        pass

    def compute_density_features(self, positions):
        pass

    def compute_appearance_features(self, positions):
        pass

    def normalize_coords(self, positions):
        return (positions - self.bb[0]) * self.inv_bb_size - 1
    
    def get_optional_parameter_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'bounding_box': self.bb,
            'grid_size': self.grid_size,

            'numb_density_comps': self.numb_density_comps,
            'numb_appearance_comps': self.numb_appearance_comps,
            'appearance_dims': self.appearance_dims,
            'shading_mode': self.shading_mode,

            'alpha_mask_threshold': self.alpha_mask_threshold,
            'step_ratio': self.step_ratio,
            'near_far_ray_dists': self.near_far_ray_dists,
            'feature_density_act_function': self.feature_density_act_funct,
            
            'pos_posenc': self.pos_posenc,
            'view_posenc': self.view_posenc,
            'feature_posenc': self.feature_posenc,
            'numb_features': self.numb_features,

            'ray_march_weight_threshold': self.ray_march_weight_threshold,
            'density_shift': self.density_shift,
            'distance_scale': self.distance_scale
        }

    def save(self, path):
        kwargs = self.get_kwargs() # Get kwargs
        checkpoint = {'kwargs': kwargs, 'state_dict': self.state_dict()} # use kwargs and torch module state dict
        if self.alpha_mask is not None: # If the alpha mask exists, add it too
            alpha_volume = self.alpha_mask.alpha_volume.bool().cpu().numpy() # get volume
            # add volume features
            checkpoint.update({
                'alpha_mask.shape': alpha_volume.shape, 
                'alpha_mask.mask': np.packbits(alpha_volume.reshape(-1)), 
                'alpha_mask.bb': self.alpha_mask.bb.cpu()
                }) 
        torch.save(checkpoint, path)

    def load(self, checkpoint):
        # If the alpha mask is saved
        if 'alpha_mask.bb' in checkpoint:
            # Get length of bit packings
            length = np.prod(checkpoint['alpha_mask.shape'])
            # Read and reshape the packed bits
            alpha_volume = torch.from_numpy(np.unpackbits(checkpoint['alpha_mask.mask'])[:length].reshape(checkpoint['alpha_mask.shape']))
            # Set the alpha mask
            self.alpha_mask = AlphaGridMask(self.device, checkpoint['alpha_mask.bb'].to(self.device), alpha_volume.float().to(self.device))
        # Load in the state dict for the nn.Module class
        self.load_state_dict(checkpoint['state_dict'])

    def sample_ray(self, ray_origins, ray_directions, training=True, numb_samples=-1):
        numb_samples = numb_samples if numb_samples > 0 else self.number_samples # get numb samples
        step_size = self.step_size # Get step size
        near, far = self.near_far_ray_dists # Get near, far distances for rays
        # Avoid dividing by 0 by making 0s equal to very small numbers
        vec = torch.where(ray_directions == 0, torch.full_like(ray_directions, 1e-6), ray_directions)
        # Calculate "rates" for bounding boxes - how far along bounding boxes the ray is
        rate_a, rate_b = (self.bb[1] - ray_origins) / vec, (self.bb[0] - ray_origins) / vec
        # Calcualte the minimum distance to the bounding box
        min_d = torch.minimum(rate_a, rate_b).amax(-1).clamp(near=near, far=far)

        sample_range = torch.arange(numb_samples)[None].float() # Create values according to number of samples
        if training: # If training, add range for each ray and create noise
            sample_range = sample_range.repeat(ray_directions.shape[-2], 1) # Repeat the range for each ray in batch
            sample_range += torch.rand_like(sample_range[:, [0]]) # Add random noise to the range, augmenting data
        step = step_size * range.to(ray_origins.device) # Calculate step sizes
        interpolated_distances = (min_d[..., None] + step) # Calculated interpolated distances along rays
        # Compute sampled points along the rays
        ray_samples = ray_origins[..., None, :] + ray_directions[..., None, :] * interpolated_distances[..., None]
        # Get a filter of all points sampled outside of the bounding box
        external_mask = ((ray_samples < self.bb[0]) | (ray_samples > self.bb[1])).any(dim=-1)
        # Return positions, interpolations, and the opposte of the mask containing rays outside the bounding box
        return ray_samples, interpolated_distances, ~external_mask

    def feature_to_density_compute(self, density_features):
        if self.feature_density_act_funct == 'softplus':
            return F.softplus(density_features + self.density_shift)
        elif self.feature_density_act_funct == 'relu':
            return F.relu(density_features)
        else:
            return F.relu(density_features)
    
    # Forward pass for rendering
    def forward(self, ray_chunks, white_bg=True, training=False, numb_samples=-1):
        # Get view directions and input positions from ray chunks
        view_directions = ray_chunks[:, 3:6] # 3, 4, 5
        in_positions = ray_chunks[:, :3] # 0, 1, 2
        # Calculate sampled positions from ray marching
        positions, interpolated_distances, valid_ray_positions = self.sample_ray(in_positions, view_directions, training=training, numb_samples=numb_samples)
        # Calculate distances between adjacent sampled positions
        distances = torch.cat((interpolated_distances[:, 1:] - interpolated_distances[:, :-1], torch.zeros_like(interpolated_distances[:, :1])), dim=-1)
        # Match view directions to the shape of the sampled positions
        view_directions = view_directions.view(-1, 1, 3).expand(positions.shape)

        # Apply alpha mask if necessary
        if self.alpha_mask is not None:
            # Sample the alpha using the alpha mask module
            alphas = self.alpha_mask.sample_alpha(positions[valid_ray_positions])
            alpha_mask = alphas > 0 # Filter to alphas > 0
            invalid_ray_positions = ~valid_ray_positions # Get opposite
            # filter invalid ray positions so that anywhere outside the alpha mask is invalid
            invalid_ray_positions[valid_ray_positions] |= ~alpha_mask 
            # Take opposite again for final result
            valid_ray_positions = ~invalid_ray_positions

        # Initialize tensors for sigma and rgb values
        sigma = torch.zeros(positions.shape[:-1 ], device=positions.device)
        rgb = torch.zeros((*positions.shape[:2], 3), device=positions.device)

        # Compute sigma for valid rays
        if valid_ray_positions.any():
            positions = self.normalize_coords(positions)
            sigma_features = self.compute_density_features(positions[valid_ray_positions]) # use vectorization
            valid_sigmas = self.feature_to_density_compute(sigma_features)
            sigma[valid_ray_positions] = valid_sigmas

        # Get alpha computation based on sigma and distances computed
        alpha, weights, bg_weights = raw_to_alpha_compute(sigma, distances * self.distance_scale)

        # Create appearance mask according to the ray march weight threshold
        appearance_mask = weights > self.ray_march_weight_threshold

        # Calculate appearance features and render!
        if appearance_mask.any():
            # Calculate appearance features
            appearance_features = self.compute_appearance_features(positions[appearance_mask])
            # Get valid rgb positions based on the masked positions, view directions, and features
            valid_rgb_points = self.render_module(positions[appearance_mask], view_directions[appearance_mask], appearance_features)
            rgb[appearance_mask] = valid_rgb_points

        # Aggregate weights and RGB values for the maps
        accumulated_map = torch.sum(weights, -1)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        # If using a white background or
        # If training and 50% chance
        if white_bg or (training and torch.rand((1,)) < 0.5):
            # add the accumulated map to the rgb map so that the
            # background is moved towards white (1) by taking 1 - weights
            # and applying that to the rgb
            rgb_map = rgb_map + (1.0 - accumulated_map[..., None])
        
        # clamp RGB values from 0 to 1
        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weights * interpolated_distances, -1)
            depth_map = depth_map + (1.0 - accumulated_map) * ray_chunks[..., -1]

        return rgb_map, depth_map

        
