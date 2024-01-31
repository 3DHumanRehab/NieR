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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh





def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    colors_precomp = None

    pipe.convert_SHs_python=True
    if override_color is None:
        if pipe.convert_SHs_python:
            
            
            
            xyz_norm=pc.get_xyz.norm(dim=1, keepdim=True)
            xyz_normlized=pc.get_xyz/xyz_norm
            
            #  rgb precomputed
            
            # shs_view/shs_view_norm:[b,3,16]
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            shs_view_norm = pc.get_features_norm.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            
            # dir_pp/dir_pp_normalized :[b,3],dir_pp_norm:[b,1]
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_norm=dir_pp.norm(dim=1, keepdim=True)
            dir_pp_normalized = dir_pp/dir_pp_norm

            # sh2rgb,sh2norm:[b,3],norm_norm/dot_product/cosine:[b,1]
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            sh2norm = eval_sh(pc.active_sh_degree, shs_view_norm, dir_pp_normalized)
            sh2rgb_norm=torch.norm(sh2rgb, dim=-1, keepdim=True)
            sh2norm_norm = torch.norm(sh2norm, dim=-1, keepdim=True)
            sh2rgb_normalized=sh2rgb/sh2rgb_norm
            sh2norm_normalized=sh2norm/sh2norm_norm
            cosine = -torch.sum(dir_pp_normalized*sh2norm_normalized, dim=-1, keepdim=True).view(-1, 1)
            
            
            
            
            
            
            # # wr:[b,3]
            wr=pc.norm_mlp1(torch.cat([xyz_normlized,dir_pp_normalized,sh2norm_normalized],dim=1))
            # # rgb:[b,3],in [0,1]
            rgb=torch.sigmoid(sh2norm_norm*pc.norm_mlp2(torch.cat([wr,sh2rgb_normalized,torch.sin(cosine),torch.cos(cosine),torch.tanh(cosine)],dim=1)))
            # # exp(2*(rgb-0.5)) in [1/e,e]
            sh2rgb=torch.mul(sh2rgb,torch.exp(2*(rgb-0.5)))
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
            
            
            
            # weight sampling by directing attr.opacity  
            out=pc.opacity_mlp1(torch.cat([xyz_normlized,dir_pp_normalized,sh2norm_normalized,torch.sin(cosine),torch.cos(cosine),torch.tanh(cosine)],dim=1))
            multi_opacities=torch.sigmoid(sh2norm_norm*out)
            weight=torch.matmul(multi_opacities,pc.opacity_weight)
            opacity=torch.mul(opacity,torch.exp(2*(weight-0.5)))
            
            
            

        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
