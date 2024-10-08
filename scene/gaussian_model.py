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
import matplotlib.pyplot as plt #added by me June 3rd 
from pathlib import Path #added by me June 3rd 

import torch

import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
class GaussianModel():

    def setup_functions(self):
        #COMPUTES THE COVARIANCE MATRIX OF EACH GAUSSIAN SPLAT FROM THE SCALING AND ROTATION MATRICES 
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            #scaling represents the scale of the gaussian splat along each principal axis
            #rotation: defines the rotation of the gaussian splat
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) #reflects the paper: Σ= RSS.T R.T= RS(RS).T and  L = R @ L (this ensures that cov matrix is positive semi-definite and symmetric by design)
            symm = strip_symmetric(actual_covariance) #ensure cov matrix is symmetric (to account for numerical errors)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid #sigmoid to constrain the opacity values to (0,1]
        self.inverse_opacity_activation = inverse_sigmoid #to transform values from the (0,1) to their original domain

        self.rotation_activation = torch.nn.functional.normalize #ensures that rotation vectors are normalised because rotations are represented with quaternions


    def __init__(self, sh_degree : int, args):
       
        #to INITIALISE the GaussianModel instance with default values and set up necessary attributes
        #this method is called when a new instance of the model is created 
        self.active_sh_degree = 0 #determines the level of detail used in the current model which can be adjusted during training
        #to improve the representation and detail as needed
        self.max_sh_degree = sh_degree  
        #each 3D Gaussian is characterized by the following attributes: 
        #POSITION, COLOUR defined by spherical harmonic coefficients, OPACITY, ROTATION FACTOR, SCALING FACTOR 
        self._xyz = torch.empty(0)#holds the positions of Gaussians
        print("self._xyz shape initialisation", self._xyz.shape)
        # self._deformation =  torch.empty(0)
        self._deformation = deform_network(args) #network for modeling deformation over time (initialised with args)
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0) #current features (color)
        self._features_rest = torch.empty(0) #residual features ??
        self._scaling = torch.empty(0) #scaling factors for each gaussian
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)#keep track of the max radius of 2D gaussian splat (determines the spatial extend of a gaussian splat) to decide which ones can be pruned
        self.xyz_gradient_accum = torch.empty(0) #accumulated gradients for positions
        
        

        self.denom = torch.empty(0)#denominator for normalisation
        

        self.optimizer = None
        self.percent_dense = 0 
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)

        #added this for smoothness loss on 24/06
        #self.depths= torch.empty(0) commented out 3006


        self.setup_functions()

    def capture(self): #to SAVE the current state of the model (enable checkpointing during training)
        return (
            self.active_sh_degree, #captures the current spherical harmonics degree 
            self._xyz, #stores the position of all gaussians
            self._deformation.state_dict(), #captures the state of the deformation network
            self._deformation_table,
            # self.grid,
            self._features_dc, #stores the color info
            self._features_rest,
            self._scaling, #stores scaling factors 
            self._rotation, #rotation matrices (stores orientation of each gaussian)
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    #restore method to LOAD (checkpoints) a saved state of the GaussianModel, it reconstructs the model from previously saved parameters allowing the training to resume from a specific point
    #crucial method for resuming interrupted training sessions or for evaluating a saved model
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args) #sets up training configurations using training_args
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    #concatenates features_dc and features_rest to provide a combined feature set for each gaussian
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) #provides a comprehensive set of features for each gaussian essential for accurate 
    #rendering and representation of the scene
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) #returns the transformed opacity values for each gaussian
    
    #####added the following three properties for smoothness loss term on 24/06
    @property
    def get_depths(self):
        return self.depths
    
    @property
    def get_positions(self):
        return self._xyz
    
    def update_depths(self, new_depths):
        self.depths = new_depths.requires_grad_(True) #added the requires_grad 2806
    #########
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self): #increment the SH degree to allow the model to capture more detailed angular variations in light and color
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            #starts with a lower complexity to speed up initial training and then gradually increases the detail to capture finer details in the scene as the model improces

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        #initialises the GaussianModel from a point cloud
        self.spatial_lr_scale = spatial_lr_scale #set the learning rate scale for spatial parameters to control how quickly spatial parameters are updated
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() #converts point cloud coordinates from the 'pcd' object to a pytorch tensor and transfers it to the GPU
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) #converts the RGB colors of the point cloud to their SH representation and transfers them to the GPU
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() #stores the spherical harmonics coefficients that represent the color info for each point in the point cloud
        features[:, :3, 0 ] = fused_color #assign color to features
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) #computes the squared distances between points in the point cloud
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        #inverse sigmoid to bring them to R instead of constraining them into [0,1] so that optimiser can operate effectively and avoid vanishing gradients (when values are very close to 0 or 1)
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        #converts tensors to model parameters with gradients enabled (requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

  
    def training_setup(self, training_args):
        #initialises various training parameters and the optimizer for the GaussianModel
        self.percent_dense = training_args.percent_dense #sets the percentage of gaussians that will be actively used in training
        #initialises tensors for accumulating gradients and denominators for normalisation during training
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        
        #preparing parameter groups for optimiser, each group corresponds to a different aspect of the model (positions, deformation, grid, features, opacity, scaling, rotation)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]
        #sets up exponential learning rate schedulers for the position, deormation and grid parameters 
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self): #generates a comprehensive list of attribute names for each gaussian
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz'] #basic positional and normal coordinates
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
        return l
    def compute_deformation(self,time):
        #deformation of gaussians over time and updates their position
        deform = self._deformation[:,:,:time].sum(dim=-1) #sums the deformation vectors up to the specified time step
        xyz = self._xyz + deform #adds the deformation to the original positions (self._xyz) to get the updated positions
        return xyz
    # def save_ply_dynamic(path):
    #     for time in range(self._deformation.shape(-1)):
    #         xyz = self.compute_deformation(time)
    def load_model(self, path):
        print("loading model from exists{}".format(path)) #the path from which the model is being loaded
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda") #loads the deformation network's weights from 'deformation.pth' and transfers them to the GPU
        #deformation table and accumulator:
        #load them from their respective files ensures that the model can resume its defomration behavior accurately based on previously saved states
        #the deformation table keeps track of which gaussians are actively undergoing deformation
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")): #loading logic: if the file 'deformation_table.pth' exists, it loads this table from the file to resume the exact deformation states from the previous training session
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda") #torch.load to load the saved tensors ensuring they are moved to the GPU 
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
    def save_ply(self, path):
        #saves the current state of the model in PLY format (standard format for storing 3D data)
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy() #positions of the points 
        normals = np.zeros_like(xyz) #store normal vectors 
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full) #comvines all attributes into a structured array
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        #opacity values might drift due to transformations and updates so resetting ensures the opacity value are within a stable range 
        #first rransform values back from the normalised range to their original scale
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        #capping at 0.01 provides control over the max opacity maintaining desired transparency levels
        #in rendering we prefer low opacity values to ensure that gaussians remain semi-transparent which helps in blending
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity") #updates the optimiser to use the new opacity values 
        self._opacity = optimizable_tensors["opacity"] #updates the model's internal 'opacity' attribute to use the new tensor

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis] #extract opacities

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

        #convert data to tensors
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
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
        #handles the removal (pruning) of certain Gaussian components and updates the optimizer's state after pruning
        #the function ensures that after pruning, the optimizer's state (including the moving averages used for adaptive learning rates)
        #is consistent with the updated model parameters to ensure that adam optimizer continues functioning correctly with the pruned model
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            #'state' of the optimizer is internal variables and data structures maintained by the optimizer
            #to perform updates to the model's parameters. specifically for the adam optimizer, states include:
            #exponential moving average of gradients, exponential moving average of squared gradients, and a state ict where it stores exp_avg and exp_avg_sq for EACH parameter it is optimizing
            if stored_state is not None: #mask= boolean tensor that indicates which gaussian components should be kept (True) anf which should be removed (False) 
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors #returns a dictionary containing the updated parameter tensors 

    def prune_points(self, mask):
        #removes certain gaussian components based on a mask, which indicates which components should be pruned
        #mask is a boolean tensor where 'True' indicates a gaussian component to be pruned
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask) #adjusts optimizer's state to discard the pruned gaussian components and returns
        #takes the valid_points_mask and removes the parameters of the pruned gaussian components from the optimizer's state
        #a dictionary of the remaining(unpruned) tensors that can be optimized
        #optimizable_tensors is a dictionary containing the update parameters for the components that are kept
        #update model parameters to reflect pruning using the pruned tensors returned by prune_optimizer
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        #the opposite of the prune_optimizer which removes specific parameters from the optimizer's state
        #this function here adds new parameters to the optimizer's state
        optimizable_tensors = {} #dict will store the new parameters that are to be added to the optimizer
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table):
        #updates the model’s parameters to include the new gaussians
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        #aims to increase the resolution and detail of the gaussian representation by adding more gaussians
        #this function refines the gaussian representation by splitting high-gradient gaussians into multiple new gaussians
        

        #steps: gradient condition check, scaling condition check. generation of new gaussians (replication of scaling values, generation of new sample positions, application of rotation for correct orientation of gaussians and scaling down of the new gaussians), updates the model through calling ensification_postfix, prunes the original gaussians
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        #selects gaussians where the gradient magnitude is greater than or equal to the gradient threshold. this ientifies gaussians that are contributing significantly to the error and need refinement
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        #identifies the critical gaussians that need to be refined by splitting based on the magnitude of the gradient (areas that meet the gradient threshold indicate areas needing refinement)
        # breakpoint()
        #this line REFINES THE SELECTION (2nd condition) by ensuring that the selected gaussians also have scaling factors larger than a specifid threshold, the specified threshold
        #indicates that these gaussians are too broad and need to be split into smaller gaussians to capture finer details
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any(): #if no points meet the gradient threshold exit the function early as there is nothing to densify
            return
        #if such gaussians exist, it prepares to split each selected gaussian into multiple new gaussians. this involves:
        #replicating the scaling values of the selected gaussians
        #generating new sample positions for the new gaussians based on these scaling values
        #adjust the new positions by applying the rotations to ensure the gaussians are correctly oriented
        #it scales down the new gaussians appropriately to maintain the correct level of detail
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
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter) #prune original gaussians

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        #creates a mask that selects gaussians where the norm of their gradients is greater than or equal to a specified threshold
        #steps: gradient condition check, density condition check, cloning of gaussians, update of the model through densiication_postfix
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 主动增加稀疏点云
        # if not hasattr(self,"voxel_size"):
        #     self.voxel_size = 8  
        # if not hasattr(self,"density_threshold"):
        #     self.density_threshold = density_threshold
        # if not hasattr(self,"displacement_scale"):
        #     self.displacement_scale = displacement_scale
        # point_cloud = self.get_xyz.detach().cpu()
        # sparse_point_mask = self.downsample_point(point_cloud)
        # _, low_density_points, new_points, low_density_index = addpoint(point_cloud[sparse_point_mask],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        # sparse_point_mask = sparse_point_mask.to(grads_accum_mask)
        # low_density_index = low_density_index.to(grads_accum_mask)
        # if new_points.shape[0] < 100 :
        #     self.density_threshold /= 2
        #     self.displacement_scale /= 2
        #     print("reduce diplacement_scale to: ",self.displacement_scale)
        # global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(grads_accum_mask)
        # global_mask[sparse_point_mask] = low_density_index
        # selected_pts_mask_grow = torch.logical_and(global_mask, grads_accum_mask)
        # print("降采样点云:",sparse_point_mask.sum(),"选中的稀疏点云：",global_mask.sum(),"梯度累计点云：",grads_accum_mask.sum(),"选中增长点云：",selected_pts_mask_grow.sum())
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        ##the exact opposite condition than in the densify_and_split function
        #clones gaussians in regiions where the scaling factor is small so that we can increase the density in areas that need more detail 
        # breakpoint()        
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        # if opt.add_point:
        # selected_xyz, grow_xyz = self.add_point_by_mask(selected_pts_mask_grow.to(self.get_xyz.device), self.displacement_scale)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        # print("被动增加点云：",selected_xyz.shape[0])
        # print("主动增加点云：",selected_pts_mask.sum())
        # if model_path is not None and iteration is not None:
        #     point = combine_pointcloud(self.get_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy(), selected_xyz.detach().cpu().numpy())
        #     write_path = os.path.join(model_path,"add_point_cloud")
        #     os.makedirs(write_path,exist_ok=True)
        #     o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        #     print("write output.")
    @property
    def get_aabb(self): #axis-aligned bounding box is a bounding box that is aligned with the coordinate axes
        #represents the spatial extent of an object or a set of objects
        #spatial extent: represents the spatial limits of the deformation applied to the points in the scene
        return self._deformation.get_aabb
    

    def get_displayment(self,selected_point, point, perturb):#generates new positions for selected point by adding random displacements ensuring they stay within the axis-aligned bounding boxes
        xyz_max, xyz_min = self.get_aabb #retrieves the max and min bounds of the aabb for the object providing constraints within which new gaussians can be placed
        #generate random displacements using a norma; distribution
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb #generate random displacemetns for each selected point
        #adds displacements to selected points to create new positions for the gaussians
        #randomness helps in exploring different spatial configurations 
        final_point = selected_point + displacements #adds the random disp;acements to the selected points to get the new positions 

        #ensure the new positions stay within the AABB
        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
        #in gaussian splatting, adding random points is for densification: increase the number of points to improve the representation and detail of the scene
    
        # while (mask_d.sum()/final_point.shape[0])<0.5:
        #     perturb/=2
        #     displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        #     final_point = selected_point + displacements
        #     mask_a = final_point<xyz_max 
        #     mask_b = final_point>xyz_min
        #     mask_c = mask_a & mask_b
        #     mask_d = mask_c.all(dim=1)
        #     final_point = final_point[mask_d]
        return final_point, mask_d    
    

    def add_point_by_mask(self, selected_pts_mask, perturb=0): #to add new points to the Gaussian model based on a selection mask, ensuring that these 
        #new points are valid and enhacing the model's density and detai;
        selected_xyz = self._xyz[selected_pts_mask] #selects point from the current set based on the provided mas
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)
        # displacements = torch.randn(selected_xyz.shape[0], 3).to(self._xyz) * perturb

        # new_xyz = selected_xyz + displacements
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        return selected_xyz, new_xyz
    def downsample_point(self, point_cloud):
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        point_downsample = point_cloud
        flag = False 
        while point_downsample.shape[0]>1000:
            if flag:
                self.voxel_size+=8
            point_downsample = downsample_point_cloud_open3d(point_cloud,voxel_size=self.voxel_size)
            flag = True
        print("point size:",point_downsample.shape[0])
        # downsampled_point_mask = torch.eq(point_downsample.view(1,-1,3), point_cloud.view(-1,1,3)).all(dim=1)
        downsampled_point_index = find_indices_in_A(point_cloud, point_downsample)
        downsampled_point_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(point_downsample.device)
        downsampled_point_mask[downsampled_point_index]=True
        return downsampled_point_mask
    
    def grow(self, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        #function that aims to increase the density of the Gaussian model by adding new points, particularly in low-density areas
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        if not hasattr(self,"density_threshold"):
            self.density_threshold = density_threshold
        if not hasattr(self,"displacement_scale"):
            self.displacement_scale = displacement_scale
        flag = False

        #retrieve the current positions of the gaussians
        point_cloud = self.get_xyz.detach().cpu()
        point_downsample = point_cloud.detach() 
        downsampled_point_index = self.downsample_point(point_downsample)
        #identify points in low-density areas, newly generated points to increase density and indices of low-density points in the original point cloud

        _, low_density_points, new_points, low_density_index = addpoint(point_cloud[downsampled_point_index],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        if new_points.shape[0] < 100 :
            self.density_threshold /= 2
            self.displacement_scale /= 2
            print("reduce diplacement_scale to: ",self.displacement_scale)

        elif new_points.shape[0] == 0:
            print("no point added")
            return
        global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool)

        global_mask[downsampled_point_index] = low_density_index
        global_mask
        selected_xyz, new_xyz = self.add_point_by_mask(global_mask.to(self.get_xyz.device), self.displacement_scale)
        print("point growing,add point num:",global_mask.sum())
        if model_path is not None and iteration is not None:
            point = combine_pointcloud(point_cloud, selected_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy())
            write_path = os.path.join(model_path,"add_point_cloud")
            os.makedirs(write_path,exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        return
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        #opacity-based pruning: creates a mask identifying gaussians with opacity values below a specified threshold
        #these gaussians contribute little to the overall image and can be removed
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size: #if max_screen_size is provided, the function identifies gaussians that are too large to be efficiently rendered
            big_points_vs = self.max_radii2D > max_screen_size #checks if the 2D projected size of the gaussian excees max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent #checks if the scaling factor (i.e. size of the gaussian in 3D space) exceeds a 10% fraction of the scene extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) #gaussians that meet either of these conditions are marked for pruning
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        #calculates the view-space positional gradients and normalises them
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        #calls the densify_and_clone or ensify_and_split 
        #the scaling condition to check if the gaussians are too large (i.e. large scaling factor) is implemented in the densify_and_split
        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        #updates the gradient accumulation and denominator for normalisation during densification
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()


#added by me 3 June
def plot_opacity_custom(model, output_dir, index=0):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    opacity_values= model.get_opacity.detach().cpu().numpy()
    print("opacity values:", opacity_values)

    plt.figure(figsize=(10,10))
    plt.imshow(opacity_values, cmap='jet', aspect='auto')
    plt.colorbar(label='opacity')
    plt.title('Gaussian Opacity')
    plt.xlabel("Gaussians")
    plt.ylabel('Opacity Value')

    #save the plot
    plt.savefig(Path(output_dir)/ f"{index}_opacity.png")
    plt.close()


