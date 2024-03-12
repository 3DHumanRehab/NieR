#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH,NORM2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.norm_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self._features_dc_norm = torch.empty(0)
        self._features_rest_norm = torch.empty(0)
        self._features_dc_inlight = torch.empty(0)
        self._features_rest_inlight = torch.empty(0)
        
        self.norm_mlp1_n_input=9
        self.norm_mlp1_n_output=3
        self.norm_mlp2_n_input=9
        self.norm_mlp2_n_output=3
        
        self.opacity_mlp1_n_input=12
        self.opacity_mlp1_n_output=3
        self.opacity_mlp2_n_input=12
        self.opacity_mlp2_n_output=3
        
        self.norm_mlp1_weight = torch.empty(0)
        self.norm_mlp2_weight = torch.empty(0)
        self.norm_mlp1_bias = torch.empty(0)
        self.norm_mlp2_bias = torch.empty(0)
        self.opacity_mlp1_weight = torch.empty(0)
        self.opacity_mlp2_weight = torch.empty(0)
        self.opacity_mlp1_bias = torch.empty(0)
        self.opacity_mlp2_bias = torch.empty(0)
        self.opacity_weight=torch.tensor([0.1,0.2,0.7],device='cuda',requires_grad=False).reshape(3,1)
        self.specular_coef=torch.empty(0)
        
        self.setup_functions()


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.norm_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,

            self._features_dc_norm ,
            self._features_rest_norm,
            self._features_dc_inlight ,
            self._features_rest_inlight,
            self.norm_mlp1_weight,
            self.norm_mlp2_weight,
            self.norm_mlp1_bias,
            self.norm_mlp1_bias,
            self.opacity_mlp1_weight,
            self.opacity_mlp2_weight,
            self.opacity_mlp1_bias,
            self.opacity_mlp2_bias,
            self.specular_coef,
            
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        norm_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,

        self._features_dc_norm ,
        self._features_rest_norm,
        self._features_dc_inlight ,
        self._features_rest_inlight,
        self.norm_mlp1_weight, #weight=(3,9),linear为[9,3]
        self.norm_mlp2_weight,
        self.norm_mlp1_bias, # 3
        self.norm_mlp2_bias,
        self.opacity_mlp1_weight,
        self.opacity_mlp2_weight,
        self.opacity_mlp1_bias,
        self.opacity_mlp2_bias,
        self.specular_coef,
        
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.norm_gradient_accum = norm_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    @property
    def get_specular_coef(self):
        return self.specular_coef
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features_norm(self):
        features_dc_norm = self._features_dc_norm
        features_rest_norm = self._features_rest_norm
        return torch.cat((features_dc_norm, features_rest_norm), dim=1)
    
    @property
    def get_features_inlight(self):
        features_dc_inlight = self._features_dc_inlight
        features_rest_inlight = self._features_rest_inlight
        return torch.cat((features_dc_inlight, features_rest_inlight), dim=1)
    
    
    
    def norm_mlp1(self,x:torch.Tensor):#b,9
        w=torch.transpose(self.norm_mlp1_weight.reshape(-1,self.norm_mlp1_n_output,self.norm_mlp1_n_input), 1,2) # b,9,3
        return torch.bmm(x.reshape(-1,1,self.norm_mlp1_n_input),w).reshape([-1,self.norm_mlp1_n_output])+self.norm_mlp1_bias
        
    def norm_mlp2(self,x:torch.Tensor):#b,9
        w=torch.transpose(self.norm_mlp2_weight.reshape(-1,self.norm_mlp2_n_output,self.norm_mlp2_n_input), 1,2) # b,9,3
        return torch.bmm(x.reshape(-1,1,self.norm_mlp2_n_input),w).reshape([-1,self.norm_mlp2_n_output])+self.norm_mlp2_bias

    def opacity_mlp1(self,x:torch.Tensor):#b,9
        w=torch.transpose(self.opacity_mlp1_weight.reshape(-1,self.opacity_mlp1_n_output,self.opacity_mlp1_n_input), 1,2) # b,9,3
        return torch.bmm(x.reshape(-1,1,self.opacity_mlp1_n_input),w).reshape([-1,self.opacity_mlp1_n_output])+self.opacity_mlp1_bias
    def opacity_mlp2(self,x:torch.Tensor):
        w=torch.transpose(self.opacity_mlp2_weight.reshape(-1,self.opacity_mlp2_n_output,self.opacity_mlp2_n_input), 1,2) # b,9,3
        return torch.bmm(x.reshape(-1,1,self.opacity_mlp2_n_input),w).reshape([-1,self.opacity_mlp2_n_output])+self.opacity_mlp2_bias
  

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        fused_norm = NORM2SH(torch.tensor(np.asarray(pcd.normals)).float().cuda())
        features_norm = torch.zeros((fused_norm.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_norm[:, :3, 0 ] = fused_norm
        features_norm[:, 3:, 1:] = 0.0
        
        fused_inlight = NORM2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_inlight = torch.zeros((fused_norm.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_inlight[:, :3, 0 ] = fused_inlight
        features_inlight[:, 3:, 1:] = 0.0
        

        norm_mlp1_weight=torch.rand([fused_point_cloud.shape[0],self.norm_mlp1_n_input*self.norm_mlp1_n_output]).float().cuda()
        norm_mlp2_weight=torch.rand([fused_point_cloud.shape[0],self.norm_mlp2_n_input*self.norm_mlp2_n_output]).float().cuda()
        norm_mlp1_bias=torch.rand([fused_point_cloud.shape[0],self.norm_mlp1_n_output]).float().cuda()
        norm_mlp2_bias=torch.rand([fused_point_cloud.shape[0],self.norm_mlp2_n_output]).float().cuda()

        opacity_mlp1_weight=torch.rand([fused_point_cloud.shape[0],self.opacity_mlp1_n_input*self.opacity_mlp1_n_output]).float().cuda()
        opacity_mlp2_weight=torch.rand([fused_point_cloud.shape[0],self.opacity_mlp2_n_input*self.opacity_mlp2_n_output]).float().cuda()
        opacity_mlp1_bias=torch.rand([fused_point_cloud.shape[0],self.opacity_mlp1_n_output]).float().cuda()
        opacity_mlp2_bias=torch.rand([fused_point_cloud.shape[0],self.opacity_mlp2_n_output]).float().cuda()

        specular_coef=torch.rand([fused_point_cloud.shape[0],1]).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self._features_dc_norm = nn.Parameter(features_norm[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_norm = nn.Parameter(features_norm[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self._features_dc_inlight = nn.Parameter(features_inlight[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_inlight = nn.Parameter(features_inlight[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.norm_mlp1_weight = nn.Parameter(norm_mlp1_weight.requires_grad_(True))
        self.norm_mlp2_weight = nn.Parameter(norm_mlp2_weight.requires_grad_(True))       
        self.norm_mlp1_bias = nn.Parameter(norm_mlp1_bias.requires_grad_(True))
        self.norm_mlp2_bias = nn.Parameter(norm_mlp2_bias.requires_grad_(True))       

        self.opacity_mlp1_weight= nn.Parameter(opacity_mlp1_weight.requires_grad_(True)) 
        self.opacity_mlp2_weight= nn.Parameter(opacity_mlp2_weight.requires_grad_(True)) 
        self.opacity_mlp1_bias= nn.Parameter(opacity_mlp1_bias.requires_grad_(True)) 
        self.opacity_mlp2_bias= nn.Parameter(opacity_mlp2_bias.requires_grad_(True)) 

        self.specular_coef = nn.Parameter(specular_coef.requires_grad_(True))
        


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.norm_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._features_dc_norm], 'lr': training_args.feature_norm_lr, "name": "nf_dc_norm"},
            {'params': [self._features_rest_norm], 'lr': training_args.feature_norm_lr / 20.0, "name": "nf_rest_norm"},
            {'params': [self._features_dc_inlight], 'lr': training_args.feature_inlight_lr, "name": "nf_dc_inlight"},
            {'params': [self._features_rest_inlight], 'lr': training_args.feature_inlight_lr / 20.0, "name": "nf_rest_inlight"},
            {'params': [self.norm_mlp1_weight], 'lr': training_args.norm_mlp1_lr, "name": "norm_mlp1_weight"},
            {'params': [self.norm_mlp2_weight], 'lr': training_args.norm_mlp2_lr, "name": "norm_mlp2_weight"},
            {'params': [self.norm_mlp1_bias], 'lr': training_args.norm_mlp1_lr, "name": "norm_mlp1_bias"},
            {'params': [self.norm_mlp2_bias], 'lr': training_args.norm_mlp2_lr, "name": "norm_mlp2_bias"},
            {'params': [self.opacity_mlp1_weight], 'lr': training_args.opacity_mlp1_lr, "name": "opacity_mlp1_weight"},
            {'params': [self.opacity_mlp2_weight], 'lr': training_args.opacity_mlp2_lr, "name": "opacity_mlp2_weight"},
            {'params': [self.opacity_mlp1_bias], 'lr': training_args.opacity_mlp1_lr, "name": "opacity_mlp1_bias"},
            {'params': [self.opacity_mlp2_bias], 'lr': training_args.opacity_mlp2_lr, "name": "opacity_mlp2_bias"},
            {'params': [self.specular_coef], 'lr': training_args.specular_coef, "name": "specular_coef"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        

        for i in range(self._features_dc_norm.shape[1]*self._features_dc_norm.shape[2]):
            l.append('nf_dc_norm_{}'.format(i))
        for i in range(self._features_rest_norm.shape[1]*self._features_rest_norm.shape[2]):
            l.append('nf_rest_norm_{}'.format(i)) 

        for i in range(self._features_dc_inlight.shape[1]*self._features_dc_inlight.shape[2]):
            l.append('nf_dc_inlight_{}'.format(i))
        for i in range(self._features_rest_inlight.shape[1]*self._features_rest_inlight.shape[2]):
            l.append('nf_rest_inlight_{}'.format(i)) 

        for i in range(self.norm_mlp1_weight.shape[1]):
            l.append('norm_mlp1_weight_{}'.format(i))
        for i in range(self.norm_mlp2_weight.shape[1]):
            l.append('norm_mlp2_weight_{}'.format(i))
        for i in range(self.norm_mlp1_bias.shape[1]):
            l.append('norm_mlp1_bias_{}'.format(i))
        for i in range(self.norm_mlp2_bias.shape[1]):
            l.append('norm_mlp2_bias_{}'.format(i))

        for i in range(self.opacity_mlp1_weight.shape[1]):
            l.append('opacity_mlp1_weight_{}'.format(i))
        for i in range(self.opacity_mlp2_weight.shape[1]):
            l.append('opacity_mlp2_weight_{}'.format(i))
        for i in range(self.opacity_mlp1_bias.shape[1]):
            l.append('opacity_mlp1_bias_{}'.format(i))
        for i in range(self.opacity_mlp2_bias.shape[1]):
            l.append('opacity_mlp2_bias_{}'.format(i))
        l.append('specular_coef')

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        nf_dc_norm = self._features_dc_norm.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        nf_rest_norm = self._features_rest_norm.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        nf_dc_inlight = self._features_dc_inlight.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        nf_rest_inlight = self._features_rest_inlight.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        norm_mlp1_weight = self.norm_mlp1_weight.detach().cpu().numpy()
        norm_mlp2_weight = self.norm_mlp2_weight.detach().cpu().numpy()
        norm_mlp1_bias = self.norm_mlp1_bias.detach().cpu().numpy()
        norm_mlp2_bias = self.norm_mlp2_bias.detach().cpu().numpy()
        opacity_mlp1_weight = self.opacity_mlp1_weight.detach().cpu().numpy()
        opacity_mlp2_weight = self.opacity_mlp2_weight.detach().cpu().numpy()
        opacity_mlp1_bias = self.opacity_mlp1_bias.detach().cpu().numpy()
        opacity_mlp2_bias = self.opacity_mlp2_bias.detach().cpu().numpy()
        specular_coef=self.specular_coef.detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest,
                                     opacities, scale, rotation,
                                     nf_dc_norm,nf_rest_norm,nf_dc_inlight,nf_rest_inlight,norm_mlp1_weight,norm_mlp2_weight,norm_mlp1_bias,norm_mlp2_bias,
                                     opacity_mlp1_weight,opacity_mlp1_bias,opacity_mlp2_weight,opacity_mlp2_bias,specular_coef
                                     ), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        # print(plydata.elements[0])
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        specular_coef = np.asarray(plydata.elements[0]["specular_coef"])[..., np.newaxis]
        

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        features_dc_norm = np.zeros((xyz.shape[0], 3, 1))
        features_dc_norm[:, 0, 0] = np.asarray(plydata.elements[0]["nf_dc_norm_0"])
        features_dc_norm[:, 1, 0] = np.asarray(plydata.elements[0]["nf_dc_norm_1"])
        features_dc_norm[:, 2, 0] = np.asarray(plydata.elements[0]["nf_dc_norm_2"])
    

        extra_f_names_norm = [p.name for p in plydata.elements[0].properties if p.name.startswith("nf_rest_norm_")]
        extra_f_names_norm = sorted(extra_f_names_norm, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_norm)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_norm = np.zeros((xyz.shape[0], len(extra_f_names_norm)))
        for idx, attr_name in enumerate(extra_f_names_norm):
            features_extra_norm[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_norm = features_extra_norm.reshape((features_extra_norm.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
    
        features_dc_inlight = np.zeros((xyz.shape[0], 3, 1))
        features_dc_inlight[:, 0, 0] = np.asarray(plydata.elements[0]["nf_dc_inlight_0"])
        features_dc_inlight[:, 1, 0] = np.asarray(plydata.elements[0]["nf_dc_inlight_1"])
        features_dc_inlight[:, 2, 0] = np.asarray(plydata.elements[0]["nf_dc_inlight_2"])
        

        extra_f_names_inlight = [p.name for p in plydata.elements[0].properties if p.name.startswith("nf_rest_inlight_")]
        extra_f_names_inlight = sorted(extra_f_names_inlight, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_inlight)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_inlight = np.zeros((xyz.shape[0], len(extra_f_names_inlight)))
        for idx, attr_name in enumerate(extra_f_names_inlight):
            features_extra_inlight[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_inlight = features_extra_inlight.reshape((features_extra_inlight.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
    


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        norm_mlp1_weight_names=[p.name for p in plydata.elements[0].properties if p.name.startswith("norm_mlp1_weight_")]
        norm_mlp1_weight_names = sorted(norm_mlp1_weight_names, key = lambda x: int(x.split('_')[-1]))
        norm_mlp1_weight = np.zeros((xyz.shape[0], len(norm_mlp1_weight_names)))
        for idx, attr_name in enumerate(norm_mlp1_weight_names):
            norm_mlp1_weight[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        norm_mlp2_weight_names=[p.name for p in plydata.elements[0].properties if p.name.startswith("norm_mlp2_weight_")]
        norm_mlp2_weight_names = sorted(norm_mlp2_weight_names, key = lambda x: int(x.split('_')[-1]))
        norm_mlp2_weight = np.zeros((xyz.shape[0], len(norm_mlp2_weight_names)))
        for idx, attr_name in enumerate(norm_mlp2_weight_names):
            norm_mlp2_weight[:, idx] = np.asarray(plydata.elements[0][attr_name])

        norm_mlp1_bias_names=[p.name for p in plydata.elements[0].properties if p.name.startswith("norm_mlp1_bias_")]
        norm_mlp1_bias_names = sorted(norm_mlp1_bias_names, key = lambda x: int(x.split('_')[-1]))
        norm_mlp1_bias = np.zeros((xyz.shape[0], len(norm_mlp1_bias_names)))
        for idx, attr_name in enumerate(norm_mlp1_bias_names):
            norm_mlp1_bias[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        norm_mlp2_bias_names=[p.name for p in plydata.elements[0].properties if p.name.startswith("norm_mlp2_bias_")]
        norm_mlp2_bias_names = sorted(norm_mlp2_bias_names, key = lambda x: int(x.split('_')[-1]))
        norm_mlp2_bias = np.zeros((xyz.shape[0], len(norm_mlp2_bias_names)))
        for idx, attr_name in enumerate(norm_mlp2_bias_names):
            norm_mlp2_bias[:, idx] = np.asarray(plydata.elements[0][attr_name])
       

        opacity_mlp1_weight_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_mlp1_weight_")]
        opacity_mlp1_weight_names = sorted(opacity_mlp1_weight_names, key=lambda x: int(x.split('_')[-1]))
        opacity_mlp1_weight = np.zeros((xyz.shape[0], len(opacity_mlp1_weight_names)))
        for idx, attr_name in enumerate(opacity_mlp1_weight_names):
            opacity_mlp1_weight[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacity_mlp2_weight_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_mlp2_weight_")]
        opacity_mlp2_weight_names = sorted(opacity_mlp2_weight_names, key=lambda x: int(x.split('_')[-1]))
        opacity_mlp2_weight = np.zeros((xyz.shape[0], len(opacity_mlp2_weight_names)))
        for idx, attr_name in enumerate(opacity_mlp2_weight_names):
            opacity_mlp2_weight[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacity_mlp1_bias_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_mlp1_bias_")]
        opacity_mlp1_bias_names = sorted(opacity_mlp1_bias_names, key=lambda x: int(x.split('_')[-1]))
        opacity_mlp1_bias = np.zeros((xyz.shape[0], len(opacity_mlp1_bias_names)))
        for idx, attr_name in enumerate(opacity_mlp1_bias_names):
            opacity_mlp1_bias[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacity_mlp2_bias_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_mlp2_bias_")]
        opacity_mlp2_bias_names = sorted(opacity_mlp2_bias_names, key=lambda x: int(x.split('_')[-1]))
        opacity_mlp2_bias = np.zeros((xyz.shape[0], len(opacity_mlp2_bias_names)))
        for idx, attr_name in enumerate(opacity_mlp2_bias_names):
            opacity_mlp2_bias[:, idx] = np.asarray(plydata.elements[0][attr_name])



        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc_norm = nn.Parameter(torch.tensor(features_dc_norm, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_norm = nn.Parameter(torch.tensor(features_extra_norm, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._features_dc_inlight = nn.Parameter(torch.tensor(features_dc_inlight, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_inlight = nn.Parameter(torch.tensor(features_extra_inlight, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self.norm_mlp1_weight = nn.Parameter(torch.tensor(norm_mlp1_weight, dtype=torch.float, device="cuda").requires_grad_(True))
        self.norm_mlp2_weight = nn.Parameter(torch.tensor(norm_mlp2_weight, dtype=torch.float, device="cuda").requires_grad_(True))
        self.norm_mlp1_bias = nn.Parameter(torch.tensor(norm_mlp1_bias, dtype=torch.float, device="cuda").requires_grad_(True))
        self.norm_mlp2_bias = nn.Parameter(torch.tensor(norm_mlp2_bias, dtype=torch.float, device="cuda").requires_grad_(True))

        self.opacity_mlp1_weight = nn.Parameter(torch.tensor(opacity_mlp1_weight, dtype=torch.float, device="cuda").requires_grad_(True))
        self.opacity_mlp2_weight = nn.Parameter(torch.tensor(opacity_mlp2_weight, dtype=torch.float, device="cuda").requires_grad_(True))
        self.opacity_mlp1_bias = nn.Parameter(torch.tensor(opacity_mlp1_bias, dtype=torch.float, device="cuda").requires_grad_(True))
        self.opacity_mlp2_bias = nn.Parameter(torch.tensor(opacity_mlp2_bias, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.specular_coef = nn.Parameter(torch.tensor(specular_coef, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._features_dc_norm = optimizable_tensors["nf_dc_norm"]
        self._features_rest_norm = optimizable_tensors["nf_rest_norm"]
        self._features_dc_inlight = optimizable_tensors["nf_dc_inlight"]
        self._features_rest_inlight = optimizable_tensors["nf_rest_inlight"]
        self.norm_mlp1_weight=optimizable_tensors["norm_mlp1_weight"]
        self.norm_mlp2_weight=optimizable_tensors["norm_mlp2_weight"]
        self.norm_mlp1_bias=optimizable_tensors["norm_mlp1_bias"]
        self.norm_mlp2_bias=optimizable_tensors["norm_mlp2_bias"]

        self.opacity_mlp1_weight=optimizable_tensors["opacity_mlp1_weight"]
        self.opacity_mlp2_weight=optimizable_tensors["opacity_mlp2_weight"]
        self.opacity_mlp1_bias=optimizable_tensors["opacity_mlp1_bias"]
        self.opacity_mlp2_bias=optimizable_tensors["opacity_mlp2_bias"]

        self.specular_coef=optimizable_tensors["specular_coef"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.norm_gradient_accum = self.norm_gradient_accum[valid_points_mask]
        

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation,
                              new_features_dc_norm, new_features_rest_norm,new_features_dc_inlight, new_features_rest_inlight,
                              norm_mlp1_weight,norm_mlp2_weight,norm_mlp1_bias,norm_mlp2_bias,
                              opacity_mlp1_weight,opacity_mlp2_weight,opacity_mlp1_bias,opacity_mlp2_bias,specular_coef
                              ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "nf_dc_norm": new_features_dc_norm,
        "nf_rest_norm": new_features_rest_norm,
        "nf_dc_inlight": new_features_dc_inlight,
        "nf_rest_inlight": new_features_rest_inlight,
        "norm_mlp1_weight":norm_mlp1_weight,
        "norm_mlp2_weight":norm_mlp2_weight,
        "norm_mlp1_bias":norm_mlp1_bias,
        "norm_mlp2_bias":norm_mlp2_bias,
        "opacity_mlp1_weight":opacity_mlp1_weight,
        "opacity_mlp2_weight":opacity_mlp2_weight,
        "opacity_mlp1_bias":opacity_mlp1_bias,
        "opacity_mlp2_bias":opacity_mlp2_bias,
        "specular_coef":specular_coef,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._features_dc_norm = optimizable_tensors["nf_dc_norm"]
        self._features_rest_norm = optimizable_tensors["nf_rest_norm"]
        self._features_dc_inlight = optimizable_tensors["nf_dc_inlight"]
        self._features_rest_inlight = optimizable_tensors["nf_rest_inlight"]

        self.norm_mlp1_weight=optimizable_tensors["norm_mlp1_weight"]
        self.norm_mlp2_weight=optimizable_tensors["norm_mlp2_weight"]
        self.norm_mlp1_bias=optimizable_tensors["norm_mlp1_bias"]
        self.norm_mlp2_bias=optimizable_tensors["norm_mlp2_bias"]
        
        self.opacity_mlp1_weight=optimizable_tensors["opacity_mlp1_weight"]
        self.opacity_mlp2_weight=optimizable_tensors["opacity_mlp2_weight"]
        self.opacity_mlp1_bias=optimizable_tensors["opacity_mlp1_bias"]     
        self.opacity_mlp2_bias=optimizable_tensors["opacity_mlp2_bias"]   
        
        self.specular_coef=optimizable_tensors["specular_coef"]   
          

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.norm_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_specular_coef = self.specular_coef[selected_pts_mask].repeat(N,1)
        

        new_features_dc_norm = self._features_dc_norm[selected_pts_mask].repeat(N,1,1)
        new_features_rest_norm = self._features_rest_norm[selected_pts_mask].repeat(N,1,1)
        new_features_dc_inlight = self._features_dc_inlight[selected_pts_mask].repeat(N,1,1)
        new_features_rest_inlight = self._features_rest_inlight[selected_pts_mask].repeat(N,1,1)

        new_norm_mlp1_weight = self.norm_mlp1_weight[selected_pts_mask].repeat(N,1)
        new_norm_mlp2_weight = self.norm_mlp2_weight[selected_pts_mask].repeat(N,1)
        new_norm_mlp1_bias = self.norm_mlp1_bias[selected_pts_mask].repeat(N,1)
        new_norm_mlp2_bias = self.norm_mlp2_bias[selected_pts_mask].repeat(N,1)
        new_opacity_mlp1_weight = self.opacity_mlp1_weight[selected_pts_mask].repeat(N,1)
        new_opacity_mlp2_weight = self.opacity_mlp2_weight[selected_pts_mask].repeat(N,1)
        new_opacity_mlp1_bias = self.opacity_mlp1_bias[selected_pts_mask].repeat(N,1)
        new_opacity_mlp2_bias = self.opacity_mlp2_bias[selected_pts_mask].repeat(N,1)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_features_dc_norm,new_features_rest_norm,new_features_dc_inlight,new_features_rest_inlight,
                                   new_norm_mlp1_weight,new_norm_mlp2_weight,new_norm_mlp1_bias,new_norm_mlp2_bias,
                                   new_opacity_mlp1_weight,new_opacity_mlp2_weight,new_opacity_mlp1_bias,new_opacity_mlp2_bias,new_specular_coef)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_features_dc_norm = self._features_dc_norm[selected_pts_mask]
        new_features_rest_norm = self._features_rest_norm[selected_pts_mask]
        new_features_dc_inlight = self._features_dc_inlight[selected_pts_mask]
        new_features_rest_inlight = self._features_rest_inlight[selected_pts_mask]

        new_norm_mlp1_weight = self.norm_mlp1_weight[selected_pts_mask]
        new_norm_mlp2_weight = self.norm_mlp2_weight[selected_pts_mask]
        new_norm_mlp1_bias = self.norm_mlp1_bias[selected_pts_mask]
        new_norm_mlp2_bias = self.norm_mlp2_bias[selected_pts_mask]
        new_opacity_mlp1_weight = self.opacity_mlp1_weight[selected_pts_mask]
        new_opacity_mlp2_weight = self.opacity_mlp2_weight[selected_pts_mask]
        new_opacity_mlp1_bias = self.opacity_mlp1_bias[selected_pts_mask]
        new_opacity_mlp2_bias = self.opacity_mlp2_bias[selected_pts_mask]
        new_specular_coef = self.specular_coef[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_features_dc_norm,new_features_rest_norm,new_features_dc_inlight,new_features_rest_inlight,
                                   new_norm_mlp1_weight,new_norm_mlp2_weight,new_norm_mlp1_bias,new_norm_mlp2_bias,
                                   new_opacity_mlp1_weight,new_opacity_mlp2_weight,new_opacity_mlp1_bias,new_opacity_mlp2_bias,new_specular_coef)



    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,N=4,use_norm_grads=0,norm_grad_weight=0.1):
        grads = self.xyz_gradient_accum / self.denom
        
        if use_norm_grads:
            grads=(1-norm_grad_weight)*grads+norm_grad_weight*self.norm_gradient_accum / self.denom
            
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent,N=N)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor,norm_tensor, update_filter,use_norm_grads=0):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        if use_norm_grads:
            assert norm_tensor is not None
            self.norm_gradient_accum[update_filter] += torch.norm(norm_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
