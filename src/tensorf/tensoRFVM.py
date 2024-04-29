from .tensoRF import TensoRFBase

import torch
import torch.nn.functional as F

# Represents the class for the TensoRF-VM module
class TensoRFVM(TensoRFBase):
    """
    Adapted from Reference: https://github.com/apchenstu/TensoRF/blob/main/models/tensoRF.py 
    """
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoRFVM, self).__init__(aabb, gridSize, device, **kargs)
        
        self.init_svd_volume(gridSize[0], device)

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.numb_appearance_comps + self.numb_density_comps, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.numb_appearance_comps + self.numb_density_comps, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.numb_appearance_comps * 3, self.appearance_dims, bias=False, device=device)

    
    def get_optional_parameter_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.render_module, torch.nn.Module):
            grad_vars += [{'params':self.render_module.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.mat_operations_order[0]], xyz_sampled[..., self.mat_operations_order[1]], xyz_sampled[..., self.mat_operations_order[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vec_operations_order[0]], xyz_sampled[..., self.vec_operations_order[1]], xyz_sampled[..., self.vec_operations_order[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.numb_density_comps:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.numb_density_comps:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.numb_appearance_comps], coordinate_plane, align_corners=True).view(3 * self.numb_appearance_comps, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.numb_appearance_comps], coordinate_line, align_corners=True).view(3 * self.numb_appearance_comps, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_density_features(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.mat_operations_order[0]], xyz_sampled[..., self.mat_operations_order[1]], xyz_sampled[..., self.mat_operations_order[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vec_operations_order[0]], xyz_sampled[..., self.vec_operations_order[1]], xyz_sampled[..., self.vec_operations_order[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.numb_density_comps:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.numb_density_comps:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appearance_features(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.mat_operations_order[0]], xyz_sampled[..., self.mat_operations_order[1]], xyz_sampled[..., self.mat_operations_order[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vec_operations_order[0]], xyz_sampled[..., self.vec_operations_order[1]], xyz_sampled[..., self.vec_operations_order[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.numb_appearance_comps], coordinate_plane, align_corners=True).view(3 * self.numb_appearance_comps, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.numb_appearance_comps], coordinate_line, align_corners=True).view(3 * self.numb_appearance_comps, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_component_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.numb_density_comps:]) + self.vectorDiffs(self.line_coef[:,:self.numb_appearance_comps])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i]
            mat_id_0, mat_id_1 = self.mat_operations_order[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.update_step_size(res_target)
        print(f'upsampling to {res_target}')


class TensoRFVMSplit(TensoRFBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoRFVMSplit, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.numb_density_comps, self.grid_size, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.numb_appearance_comps, self.grid_size, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.numb_appearance_comps), self.appearance_dims, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vec_operations_order)):
            vec_id = self.vec_operations_order[i]
            mat_id_0, mat_id_1 = self.mat_operations_order[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.render_module, torch.nn.Module):
            grad_vars += [{'params':self.render_module.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)



    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))