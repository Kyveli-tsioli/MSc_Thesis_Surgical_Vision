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
import numpy as np #added that 
import os #added that
from PIL import Image  # added this
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
#from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer #added 0107 for surface normals
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time




#render a scene from the perspective of a given viewpoint camera using the gaussian model
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None, viewpoint_idx=None, iteration=None): #added iteration argument and viewpoint_idx 11/06
    #add back iteratio to revert back to 11/06
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
    
    means3D = pc.get_xyz #extract 3d means from the gaussian model
    if cam_type != "PanopticSports":
        #compute the field of view
        #field of view is the extent of observable world seen at any given moment through a camera 
        #typically expressed in degrees 
        #in graphics and 3D rendering the FoV determines how much of the scene 
        #is visible to the camera
        #a wider FoV allows more of the scene to be captured but can cause distortion, 
        #while a narrower FoV captures less of the scene but maintains a mroe realistic perspective

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height), #camera transformations
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else: #if camera is "PanopticSports" use the camera settings directly from viewpoint
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    #extract properties from the gaussian model
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    #deformation handling:
    if "coarse" in stage: #use the original values
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage: #apply deformation using the deformation model to get final properties 
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    #APPLY activation functions to the final properties
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None: #if override color is not provided, compute the colors using SH
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color
     
    #added 3006 but give me cuda error 
    #scales_final = pc.scaling_activation(scales_final)
    #rotations_final = pc.rotation_activation(rotations_final)
    #opacity_final = pc.opacity_activation(opacity_final)
    
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    #unormalised depth: measure the error betwenen this depth and the ground truth depth
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.


    #ADDED 11/05: save depth map so that I can compare with the GT depth
    #if iteration == final_iteration:
   ##2806 commented out starts here####
    ##depth_map = depth #REMOVE DETACH 2806
    ##depth_map_folder = "/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_2806_endo/depth_maps"
    ##os.makedirs(depth_map_folder, exist_ok=True)

    # Ensure unique filenames if multiple depth maps are saved
    #depth_map_filename = f"depth_map_{iteration}.npy" was this


    ##depth_map_filename = f"depth_map_{stage}_{iteration}_{viewpoint_idx}.npy" #was this 11/06 and was generating th depth map of a particular viewpoint over the course of the training
    #depth_map_png_filename = f"depth_map_idx_{viewpoint_camera.image_index}.png"  # Ensure this is defined
    #depth_map_png_filename = f"depth_map_{viewpoint_idx}.png"
    ####depth_map_filename=f"depth_map_{viewpoint_idx}.npy" # ADDED 12/06
    
    #depth_map_path = os.path.join(depth_map_folder, depth_map_png_filename) ####was depth_map_filename
    ##depth_map_path = os.path.join(depth_map_folder, depth_map_filename)
    

 # Temporarily disable gradient tracking for saving
    ##with torch.no_grad():
        ##depth_map_cpu = depth_map.cpu().numpy()
        ##np.save(depth_map_path, depth_map_cpu)

        ##depth_map_png_filename = f"depth_map_{stage}_{iteration}_{viewpoint_idx}.png"
        ##depth_map_png_path = os.path.join(depth_map_folder, depth_map_png_filename)
        ##depth_normalized = (depth_map_cpu - depth_map_cpu.min()) / (depth_map_cpu.max() - depth_map_cpu.min())
        ##depth_image = (depth_normalized * 255).astype(np.uint8)
        ##depth_image = Image.fromarray(depth_image.squeeze())
        ##depth_image.save(depth_map_png_path)
        ##print(f"Saved depth map .png at {depth_map_png_path}")
####commented out ends here 2806#####

    ####np.save(depth_map_path, depth_map)
    #print(f"saved depth map .png at {depth_map_path}")


    # Convert depth map to PNG and save it- added 11/06
    #depth_map_png_filename = f"depth_map_{iteration}.png" was this 
    #depth_map_png_filename = f"depth_map_{iteration}_{viewpoint_idx}.png" #was this 11/06 #uncommented just now
    #####depth_map_png_filename= f"depth_map_{stage}_{iteration}_{viewpoint_idx}.png" # ADDED 12/06

    ####depth_map_png_path = os.path.join(depth_map_folder, depth_map_png_filename) # 11/06
    #depth_map_png_path = os.path.join(depth_map_folder, depth_map_png_filename) #ADDED
    
    ####depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize depth values to [0, 1]
    ####depth_image = (depth_normalized * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    ####depth_image = Image.fromarray(depth_image.squeeze())  # Remove single-dimensional entries and convert to Image
    ####depth_image.save(depth_map_png_path)  # Save as PNG
    ####print(f"Saved depth map .png at {depth_map_png_path}")
    
    # Print depth map values
    #print("Depth map values:", depth_map)
    
    #added this for smoothness loss on 24/06
    ##pc.update_depths(torch.tensor(depth_normalized, dtype=torch.float, device="cuda")) 

    #pc.update_depths(depth) commented out 3006
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points, #2d points on the screen space
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth} #depth values for the points 

#added a separate function for saving 2806
#def save_depth_map(depth_map,stage, iteration, viewpoint_idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0207_torchl1_norm_smooth_diagnostics_minmaxscaling_in_smooth_0.3_tensorb_beta3.5/depth_maps"):
def save_depth_map(depth_map,stage, iteration, viewpoint_idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/FINAL/2508_office_0_best/depth_maps"):
#def save_depth_map(depth_map, stage, iteration, viewpoint_idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0107_endo_norm/depth_maps"): 
    depth_map_folder = os.path.join(output_dir, "depth_maps")
    os.makedirs(depth_map_folder, exist_ok=True)

    depth_map_filename = f"depth_map_{stage}_{iteration}_{viewpoint_idx}.npy"
    depth_map_path = os.path.join(depth_map_folder, depth_map_filename)

    # Convert depth map to numpy array without detaching
    depth_map_cpu = depth_map.detach().cpu().numpy()
    
    # Save the depth map as a numpy array
    with torch.no_grad():
        np.save(depth_map_path, depth_map_cpu)

    depth_map_png_filename = f"depth_map_{stage}_{iteration}_{viewpoint_idx}.png"
    depth_map_png_path = os.path.join(depth_map_folder, depth_map_png_filename)
    
    # Convert depth map to PNG and save it
    depth_normalized = (depth_map_cpu - depth_map_cpu.min()) / (depth_map_cpu.max() - depth_map_cpu.min())
    depth_image = (depth_normalized * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_image.squeeze())
    
    with torch.no_grad():
        depth_image.save(depth_map_png_path)

    print(f"Saved depth map .png at {depth_map_png_path}")
