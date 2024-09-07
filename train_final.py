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

####HIGH LEVEL OVERVIEW OF THE TRAIN.PY####
#Uses FineSampler (custom sampler to create permutations of frames and maintain temporal coherence) for sampling camera viewpoints if specified, otherwise uses a standard DataLoader
#uses the render function (takes as input camera viewpoints and the gaussian model) to generate images from current camera viewpoints
#computes L1 loss and PSNR between rendered images and ground truth images
#computes gradients and performs an optimiser step
#updates model and saves checkpoints at specified intervals


##initialise the gaussian model, load checkpoints if available, set up background color and timing events, handle data loading using custom sampler,
#perform the training loop: select random camera viewpoints, render images from current viewpoints, compute losses and perform optimisation, save checkpoints and evaluate the model at specified intervals
#adjust point cloud density (densification and pruning)


import matplotlib.pyplot as plt
from PIL import Image #0708
import matplotlib.cm as cm #0708

import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, l1_loss_depth, l1_loss_mask
from gaussian_renderer import render, network_gui, save_depth_map
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid #lib that generates unique identifiers, often used for uniquely identifying models or sessions
from tqdm import tqdm #creates progress bars to track progress of long-running tasks
from utils.image_utils import psnr #Peak Signal-to-Noise-ratio, a metric used to measure the quality of reconstructed images compared to the original ones 
from argparse import ArgumentParser, Namespace #a class for parsing command-line arguments 

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams #classes or unctions for managing various configuration parameters related to the model, pipeline, optimisation, hidden settings 

from torch.utils.data import DataLoader #pytorch utility that provides an efficient way to iterate over datasets supporting batching, shuffling, parallel data loading

from utils.timer import Timer 
from utils.loader_utils import FineSampler, get_stamp_list
import lpips #deep-learning based library for calculating the Learned Perceptual Image Patch Similarity (LPIPS) which measures perceptual similarity between images
from utils.scene_utils import render_training_image 
from time import time
import copy
from scene.gaussian_model import plot_opacity_custom
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import cv2 

import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


import torchvision.models as models
import torch.nn.functional as F
from heatmap import save_depth_discrepancy_heatmap


    
torch.cuda.empty_cache()


def free_up_memory(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8) #clip the pixel intensities in (0,1)
#8-bit-unsigned integer data type
#move tensor to CPU and then convert to NumPy: necessary step because most python libraries 
#for data processing (like numpy) and visualisation (like matplotlib) operate on CPU memory and work with numpy arrays
#converting back to [0,255] for visualisation: the to8b function is specifically for preparing
#the data for visualisation or saving as image files, NOT FOR training 



# Flag to enable or disable smooth torch entirely
enable_smooth_torch = True # Set to True to allow smooth torch, False for baseline 3DGS 
use_smooth_torch = enable_smooth_torch and (random.random() < 0.8)



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
print(f"TENSORBOARD_FOUND: {TENSORBOARD_FOUND}")

def log_memory_usage(stage, iteration):
    print(f"[{stage}][Iteration {iteration}] Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{stage}][Iteration {iteration}] Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def normalize_tensor(tensor):
    return tensor / (tensor.max() + 1e-6)


def update_ema(new_value, ema, alpha=0.4):
    return alpha * new_value + (1 - alpha) * ema



def error_intensity_overlay(predicted, ground_truth, output_dir, image_index):

    predicted_np = predicted.cpu().detach().numpy()
    ground_truth_np = ground_truth.cpu().detach().numpy()
    difference = np.abs(predicted_np - ground_truth_np)
   

    # Normalize the predicted depth map
    norm_predicted = (predicted_np - predicted_np.min()) / (predicted_np.max() - predicted_np.min())

    # Normalize the error for mapping to intensity
    error_intensity = difference / difference.max()

    # Create an intensity map where high errors are darker
    intensity_map = 1 - error_intensity

    # Apply the intensity map to the normalized predicted map
    overlay = norm_predicted * intensity_map
    overlay = np.squeeze(overlay)  # This removes any singleton dimensions


    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(overlay, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Depth Map with Error Intensity Overlay for Image {image_index}")
    plt.savefig(f"{output_dir}/error_overlay_{image_index}.png")
    plt.close()

    # Calculate and return the mean error
    mean_diff = np.mean(difference)
    print(f"Image {image_index}: Mean depth difference = {mean_diff}")

    return mean_diff

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths."""
    # Transfer to CPU, detach from the computation graph, and convert to numpy
    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt((np.log(gt + 1e-6) - np.log(pred + 1e-6)) ** 2).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    


    # Open a file for logging depth discrepancies
    log_file_path = os.path.join(scene.model_path, f"depth_discrepancies_brain_2908_further_exp_fulll1rendered_0.7depth_{stage}.log")
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Logging started for {stage} stage\n")
        log_file.flush()  # Ensure the initial message is written to the disk


    #INITIALISE THE GAUSSIAN MODEL, LOAD CHECKPOINT IF AVAILABLE.
        weight_factor=0.5
        first_iter = 0
        print(vars(opt)) #added for debugging

        gaussians.training_setup(opt) #initialise the Gaussian model with the options
        if checkpoint:
            # breakpoint()
            if stage == "coarse" and stage not in checkpoint:
                print("start from fine stage, skip coarse stage.")
                # process is in the coarse stage, but start from fine stage
                return
            if stage in checkpoint: 
                (model_params, first_iter) = torch.load(checkpoint) #loads the checkpoint file, retrieving the saved model parameters and the iteration to resume from
                gaussians.restore(model_params, opt) #restores the gaussian model parameters to the state saved in the checkpoint


        #SET UP BACKGROUND COLOR, TIMING EVENTS, EXPONENTIAL MOVING AVERAGE FOR LOSS AND PSNR
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #data type set to "torch.float32" to match the precision requirements for rendering operations
        #while pixel intensities are in [0,255], float data type ensures percision during rendering operations such as interpolation, blending 

        #measure iteration time for performance monitoring 
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = None #will later hold a collection of camera viewpoints for rendering 
        ema_loss_for_log = 0.0 #Exponential Moving Average version of loss
        ema_psnr_for_log = 0.0

        final_iter = train_iter
    
        progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
        first_iter += 1
        # lpips_model = lpips.LPIPS(net="alex").cuda()

        #retrieve sets of camera viewpoints from the 'scene' object  
        video_cams = scene.getVideoCameras()
        
        test_cams = scene.getTestCameras()
        number_of_view_test = len(test_cams)  # This assumes 'test_cameras' is an iterable (like a list)
        print("Number of views in the test camera:", number_of_view_test)

        
        train_cams = scene.getTrainCameras()
        number_of_views_train = len(train_cams)  # This assumes 'test_cameras' is an iterable (like a list)
        print("Number of views in the TRAIN camera:", number_of_views_train)


        if not viewpoint_stack and not opt.dataloader: #viewpoint stack contains the camera viewpoints that are used for rendering during training
            # dnerf's branch
            #Initialise viewpoint_stack with camera viewpoints to ensure that each training iter can render images from different perspectives
            viewpoint_stack = [i for i in train_cams] #creates a list that holds camera viewpoints during training and assigns it to 'viewpoint_stack'
            #each camera viewpoint in this list provides the necessary parameters (position, orientation, field of view) to render images of the scene from different perspectives
            #the rendered images are then used to train the neural rendering model
            temp_list = copy.deepcopy(viewpoint_stack) #deep copy is an independent copy of 'viewpoint_stack' so any modifications to one will NOT affect the other 
    
        batch_size = opt.batch_size
        print("data loading done")
        if opt.dataloader: #uses FineSampler to load data batches according to the custom data sampling pattern
            viewpoint_stack = scene.getTrainCameras() #stack of camera viewpoints used during the training process 
            if opt.custom_sampler is not None: #creates a FineSampler instance (our custom sampler) to load batches according to a custom sampling pattern
                sampler = FineSampler(viewpoint_stack) #load batches of viewpoints 
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
                random_loader = False
            else: #otherwise use a standard DataLoader
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
                random_loader = True
            loader = iter(viewpoint_stack_loader)
    
    
        # dynerf, zerostamp_init
        # breakpoint()
        #
        if stage == "coarse" and opt.zerostamp_init:
            load_in_memory = True
            # batch_size = 4
            #ensure training on frames that correspond to the same timestamp across different poses (i.e. camera positions, i.e. viewpoints)
            temp_list = get_stamp_list(viewpoint_stack,0)
            viewpoint_stack = temp_list.copy()
        else:
            load_in_memory = False 
                             
        count = 0
        smooth_torch = torch.tensor(0.0, device='cuda')  # Default value for smooth_torch
        ema_smooth_torch_loss_for_log = 0.0    #added 0407  

        for iteration in range(first_iter, final_iter+1): 
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    #receives data from the 'network_gui': camera settings, flag incidaing whether to continue training, parameters for SH, covariance computation 
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                
                    #iteration-based (i.e. count) selection of viewpoints during training to ensure that the model is exposed uniformly to multiple viewpoints, both forwards and backwards (temporal coherence)
                    if custom_cam != None:
                        count +=1
                        viewpoint_index = (count ) % len(video_cams)
                    
                        if (count //(len(video_cams))) % 2 == 0: #forward viewing 
                            viewpoint_index = viewpoint_index
                        else: #backward viewing
                            viewpoint_index = len(video_cams) - viewpoint_index - 1
                        # print(viewpoint_index)
                        viewpoint = video_cams[viewpoint_index]
                        custom_cam.time = viewpoint.time #adjusts the custom camera’s timestamp to match the current viewpoint’s timestamp
                        # print(custom_cam.time, viewpoint_index, count)
                        ##renders the scene from the custom camera's perspective
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                        #use the render function to generate the rendered images from the Gaussian model given the current camera viewpoints 
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path) #send the rendered image back to the newtork_gui for real-time moniroting and further processing
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                        break
                except Exception as e:
                    print(e)
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration) #update the learning rate based on the current iteration

        # Every 1000 we increase the levels of SH up to a maximum degree
        ##increasing the degree of SH incrementally during training is a strategy to progressively refine the model's ability to represent
        #detailed angular variations in light and colour
        #start with low degree of SH for faster convergence in the early stages of learning
        #then gradually refine as the complexity of the scenes and the nuances of light and color variations become more apparent
        #coarse-to-fine strategy in the early stages of training (using high order SH might lead to overfitting the noise initially)
        #high order SH capture high-frequency details (sharp edges, details in lighting and colour variations)
            if iteration % 1000 == 0: #we start by optimising only the 0-th order component and then introduce one band of the SH after every 1000 iterations until all 4 bands of SH are represented
                gaussians.oneupSHdegree()

            # Pick a random Camera
            # dynerf's branch
            if opt.dataloader and not load_in_memory:
                try: #attempts to load the next batch of camera viewpoints unless all batches have been exhausted
                    viewpoint_cams = next(loader) #load the next batch of camera viewpoints 
                except StopIteration: #otherwise sample camera viewpoints from 'viewpoint_stack'
                    print("reset dataloader into random dataloader.")
                    if not random_loader: #creates new dataloader with shuffling to ensure random sampling of viewpoints in the next iteration
                        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                        random_loader = True
                    loader = iter(viewpoint_stack_loader)

            else: #manually select camera_viewpoints if not using the dataloader
                idx = 0
                viewpoint_cams = [] #holds the selected camera viewpoints 
                #manually select randomly viewpoints from the viewpoint_stack until the batch size is reached
                while idx < batch_size : #viewpoint sampling: randomly select camera viewpoints from the viewpoint_stack for rendering
                    #loading all viewpoints into memory is not efficient hence we load in batches 
                    
                    viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1)) #randomly select a camera viewpoint from the stack and remove it from the stack
                    if not viewpoint_stack : #if stack is empty after popping a viewpoint then refill
                        viewpoint_stack =  temp_list.copy()
                    viewpoint_cams.append(viewpoint_cam)
                    idx +=1
                if len(viewpoint_cams) == 0:
                    continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            images = []
            gt_images = []
            radii_list = [] #defines the spatial extent of each gaussian splat in the rendering
            #the radii are used to filter which points are visible from a given camera viewpoint
            #radii is critical during densification and pruning, as they determine which points should be kept, duplicated or removed
        
            gt_depths = []
            depths =[] 
            masks= []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
       
            for idx,viewpoint_cam in enumerate(viewpoint_cams): #for each selected camera viewpoint, generate the rendered image
        
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, iteration= iteration, viewpoint_idx= idx) # Pass iteration and idx- 11/06
                print(f"Rendering at iteration {iteration} in stage {stage}")
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                #image is the rendered image
                depth = render_pkg["depth"].to('cuda')  #added to test the l1 depth loss (ground truth depth with generated depth comparison)
                depth.requires_grad_(True) 
                save_depth_map(depth, stage, iteration, idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2608/brain_0.6l1rendered/depth_maps") 
                gt_depth = torch.tensor(viewpoint_cam.depth, device='cuda')
                mask = viewpoint_cam.mask
                print(f"Loading GT depth for camera {viewpoint_cam.depth}")
                mask = (gt_depth > 0).float()  
           
                if scene.dataset_type!="PanopticSports":
                    gt_image = viewpoint_cam.original_image.cuda() #retrieves the ground truth image for the current viewpoint
                    print("gt_image is", gt_image)
                    print("gt_image_shape", gt_image.shape)
                else:
                    gt_image  = viewpoint_cam['image'].cuda()
             
                if mask is not None:
                    mask = mask.cuda()
                    masks.append(mask.unsqueeze(0))
    
                images.append(image.unsqueeze(0))
                dep_mask= torch.logical_and(gt_depth>0, depth>0) 
                gt_depth = gt_depth * dep_mask
                depth = depth * dep_mask
                depths.append(depth.unsqueeze(0))
                gt_depths.append(gt_depth.unsqueeze(0))
                gt_images.append(gt_image.unsqueeze(0)) #ground truth images corresponding to the viewpoints are retrieved for comparison
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)

            
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0) #.any(dim=0) checks if any of the visibility filters are 'True' for each gaussian splat.
            #criticial for ensuring that the model focuses on visible splats for computational efficiency
        

            if len(masks) != 0:
                mask_tensor = torch.cat(masks, 0)
            else:
                mask_tensor = None
        
            image_tensor = torch.cat(images,0) * mask_tensor
            depth_tensor = torch.cat(depths, 0) * mask_tensor
            gt_image_tensor = torch.cat(gt_images,0) * mask_tensor
            gt_depth_tensor = torch.cat(gt_depths, 0) * mask_tensor

    
            Ll1 = l1_loss_mask(image_tensor, gt_image_tensor, mask_tensor.unsqueeze(0))
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #edw 1907
        

            #BELOW UNCOMMENT IF YOU'RE RUNNING THE BASELINE#
            #normalized_depth = depth / (depth.max() + 1e-6) #check baseline 3007
            #normalized_gt_depth = gt_depth / (gt_depth.max() + 1e-6) #check baseline 3007
            #depth_discrepancy = l1_loss_depth(normalized_depth, normalized_gt_depth) #check baseline 3007
            #print('Depth Discrepancy for baseline', depth_discrepancy)
            #depth_discrepancy = l1_loss_depth(normalized_depth, normalized_gt_depth)
            #print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)

            # Print and log the discrepancy
            #discrepancy_message = f"Iteration {iteration}, Image Index {idx}: Depth Discrepancy = {depth_discrepancy}\n"
            #print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)
            #log_file.write(discrepancy_message)
            #log_file.flush()

        
            if use_smooth_torch: 
                weight_factor=0.5
            
            # Calculate the masked L1 loss for depth
                normalized_depth = depth / (depth.max() + 1e-6)
                normalized_gt_depth = gt_depth / (gt_depth.max() + 1e-6)
                masked_depth_loss = torch.abs(normalized_depth - normalized_gt_depth) * mask
                masked_depth_loss = masked_depth_loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0).to(depth.device)
                depth_loss = l1_loss_mask(depth_tensor/(depth_tensor.max()+1e-6), gt_depth_tensor/(gt_depth_tensor.max()+1e-6), \
                    mask=mask_tensor.unsqueeze(0))* weight_factor 
                depth_discrepancy = l1_loss_depth(normalized_depth, normalized_gt_depth)
                print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)

                # Print and log the discrepancy
                discrepancy_message = f"Iteration {iteration}, Image Index {idx}: Depth Discrepancy = {depth_discrepancy}\n"
                print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)
                log_file.write(discrepancy_message)
                

                # Call the function to save the heatmap
                output_dir_heatmap = '/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2908_brain_fulll1rendered_0.7depth' 
                save_depth_discrepancy_heatmap(normalized_depth, normalized_gt_depth, output_dir_heatmap, iteration, idx)        
                
                loss =  0.8*Ll1 + depth_loss  #+ opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                
            else:
                loss = Ll1 
            
            loss.backward() #if this is placed here it means im training only for coarse stage
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double() #measures image quality
       
        
            if iteration % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {iteration} - Initial L1 Loss: {Ll1.item():.4f}, PSNR: {psnr_.item():.2f}")
       
            if iteration == final_iter:
                print(f"Performing evaluation at iteration {iteration}")
                test_cams = scene.getTestCameras()  # Load test dataset cameras
                output_dir_heatmap = '/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2608/heatmaps/2908_brain_fulll1rendered_0.7depth'
                os.makedirs(output_dir_heatmap, exist_ok=True)  


                # Open a file for logging depth discrepancies during evaluation
                log_file_path = os.path.join(output_dir_heatmap, "evaluation_depth_discrepancies_further_exp_2908_brain_fulll1rendered_0.7depth_METRICS.log")
                with open(log_file_path, 'a') as log_file:
                    log_file.write("Logging started for evaluation\n")
                    cumulative_mean_diff = 0.0  # Initialize the cumulative mean difference
                    metrics_sum = np.zeros(7)  # To hold sum of metrics for averaging


                    for idx, cam in enumerate(test_cams):
                        # Rendering the predicted depth from the model
                        render_pkg_eval = render(cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                        normalized_predicted_depth = render_pkg_eval["depth"] / (render_pkg_eval["depth"].max() + 1e-6)
            
                        # Normalizing the ground truth depth
                        normalized_gt_depth = torch.tensor(cam.depth, device='cuda') / (torch.tensor(cam.depth, device='cuda').max() + 1e-6)

                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(normalized_gt_depth, normalized_predicted_depth)
                        metrics_sum += np.array([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                        mean_diff= error_intensity_overlay(normalized_predicted_depth, normalized_gt_depth, output_dir_heatmap, idx)
                        cumulative_mean_diff += mean_diff
                        # Log the mean difference for each image
                        discrepancy_message_eval = f"Image {idx}: Mean depth difference = {mean_diff}\n"
                        # Log the metrics for each image
                        log_file.write(discrepancy_message_eval)
                        log_file.write(f"Image {idx}: Abs Rel: {abs_rel}, Sq Rel: {sq_rel}, RMSE: {rmse}, RMSE Log: {rmse_log}, a1: {a1}, a2: {a2}, a3: {a3}\n")
                    final_mean_depth_error = cumulative_mean_diff / len(test_cams)
                    print("length of test_cams", len(test_cams))
                    log_file.write(f"Final Mean Depth Error: {final_mean_depth_error}\n")  # Log the final mean depth error
                    print(f"Final Mean Depth Error: {final_mean_depth_error}")  # Optionally print the final mean depth error
                    # Average the metrics over all images
                    metrics_avg = metrics_sum / len(test_cams)
                    log_file.write(f"Average Metrics: Abs Rel: {metrics_avg[0]}, Sq Rel: {metrics_avg[1]}, RMSE: {metrics_avg[2]}, RMSE Log: {metrics_avg[3]}, a1: {metrics_avg[4]}, a2: {metrics_avg[5]}, a3: {metrics_avg[6]}\n")
                    log_file.write("Logging for evaluation completed\n")

    
            if stage == "fine" and hyper.time_smoothness_weight != 0:
                if iteration % 100 == 0:  # Print every 100 iterations
                    print(f"Iteration {iteration} - TV Loss: {tv_loss.item():.4f}")

                tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
                loss += tv_loss
               
                print(f"Iteration {iteration} - Total Loss after all components: {loss.item():.4f}")
        
                if iteration % 100 == 0:
                    tb_writer.add_scalar('Loss/TV Loss', tv_loss.item(), iteration)
                    tb_writer.add_scalar('Loss/Smooth Loss', smooth_torch.item(), iteration)
                    tb_writer.add_scalar('Loss/Total Loss', loss.item(), iteration)
                positions= gaussians.get_positions
           
                # Print SSIM loss values
                if iteration % 100 == 0:  # Print every 100 iterations
                    print(f"Iteration {iteration} - SSIM Loss: {(opt.lambda_dssim * (1.0 - ssim)).item():.4f}")

              
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        # Print total loss values
            if iteration % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {iteration} - Total Loss: {loss.item():.4f}")

                loss_dict = {"Loss": f"{Ll1.item():.{4}f}",
                        "psnr": f"{psnr_:.{2}f}"}
                if stage == "fine" and hyper.time_smoothness_weight != 0:
                    loss_dict['tv_loss'] = f"{tv_loss:.{4}f}" #formatted to 4 decimal places
            
                os.execv(sys.executable, [sys.executable] + sys.argv) #restarts the program to recover from the NaN error state and continue training from the last saved checkpoint
       
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            #aggregate the gradients of the viewspace point tensors across all viewpoint sin the CURRENT batch
            #the aggregation ensures that the updates to the gaussian parameters take into account the contributions from all viewpoints
            #for maintaining a consistent ad accurate representation of the scene from multiple perspectives 
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            #viewspace_point_tensor: positions of gaussian splats in the view space (coordinate space defined by the viewpoint camera)
            #the gradients of these tensors are crucial for adjusdting the positions of the splats to minimise the loss function
            #aggregation ensures that the updates to the gaussian splats are influenced by the views from all perspectives - maintain a consistent representation of the scene across all viewpoints
            iter_end.record()
       

            with torch.no_grad(): #only updating the metrics so we dont need to track gradients
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
               
                ema_smooth_torch_loss_for_log = update_ema(smooth_torch.item(), ema_smooth_torch_loss_for_log)
                total_point = gaussians._xyz.shape[0]
           
                if iteration % 10 == 0:
                    string_dict = {"Loss": f"{Ll1.item():.{4}f}",
                                        "psnr": f"{psnr_:.{2}f}"} 
                    if stage == 'fine':
                        string_dict['tv'] = f"{tv_loss:.{4}f}"
                progress_bar.update(10)
                
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                timer.pause()
                

                training_report(tb_writer, iteration, Ll1, loss, l1_loss,smooth_torch, ema_smooth_torch_loss_for_log,iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, weight_factor, use_smooth_torch, ssim, lambda_dssim=0.2)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, stage)
               
                if (iteration in saving_iterations):
                    print(f"\n[ITER {iteration}] Saving Gaussians at {args.model_path}")
                    scene.save(iteration, stage)


                #periodically render and save images from both the training and test cameras to monitor the visual quality of the model outputs
                if dataset.render_process:
                #for the first 1000 iterations, images are rendered every 10 iterations
                # for iterations between 1000 and 3000, images are rendered every 50 iterations
                # for iterations between 3000 and 600000, images are rendered every 100 iterations
                    if (iteration < 1000 and iteration % 10 == 9) \
                        or (iteration < 3000 and iteration % 50 == 49) \
                            or (iteration < 60000 and iteration %  100 == 99) :
                    
                            #render from both train camera viewpoints and from test camera viewpoints (the latter for evaluation to check how well the model generalises to unseen viewpoints for which we have the ground truth images in the dataset)
                            #here the two folders coarsetrain_render and coarsetest_render and their pointclouds subfolders are being created
                            #this is where we check how well the model performs in generating accurate and high-quality images from new angles.
                            #Robustness: Its ability to handle different lighting, occlusions, and other scene-specific challenges that might not have been explicitly covered during training.
                            #testing phase: if we want to generate data for augmentation we can use the render_training)image function with test camera viewpoint to create images
                            #from "unseen" viewpoints 
                            render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            print("length of test cameras at the render_training_image function", len(test_cams))
                            render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                timer.start()
                # Densification
                ##adjust the point cloud density during training (update max radii for pruning, add densification stats, perform densification and pruning based on thresholds and intervals specified in the options)
                if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                #this step updates the max radii in image-space for the gaussians that are visible in the current iteration. this is essential for determining which gaussians might need to be densified or pruned
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    print("max radii for the radius in smoothness", gaussians.max_radii2D)
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)#increment the denominator for the selected points and add the norm of the view-space point tensor to the gradient accumulator
                
                    #densification thresholds: depending on whether the current stage is "coarse" or "fine", diferent thresholds for opacity and gradient-based densification are set
                    if stage == "coarse":
                        opacity_threshold = opt.opacity_threshold_coarse
                        densify_threshold = opt.densify_grad_threshold_coarse
                    else:     #calculate them based on the iteration number and predefined values 
                        opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                        densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                    #size_threshold = 20 if iteration > opt.opacity_reset_interval else None #addd because of the bug 24/06

                    #check if the current iteration is appropriate for densification and if the number of splats is below a certain threshold
                    if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000: #checks if the iteration is eligible for densification and if the number of gaussians is below a certain threshold
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        #densification is performed periodically 
                        #add gaussians where there are insufficient to increase the density and improve scene coverage
                        #densify aims to fill gaps in the scene and it is threshold-based, i.e. typically occurs when opacity and gradient related thresholds are met
                        #focus is on adding splats based on the gradient of the loss function
                        gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000: #pruning if the iteration count is appropriate and the number of gaussians exceeds a certain threshold
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None 
                        #opacity and gradient thresholds to decide which splats to prune

                        gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    #regular addition: adds new Gaussian splats to the point cloud at regular intervals and is less focused on specific
                    #criteria like gradient-based densification 
                    #critetion: grow is triggered at regular intervals independently of gradients
                        gaussians.grow(5,5,scene.model_path,iteration,stage)
                        # torch.cuda.empty_cache()
                    if iteration % opt.opacity_reset_interval == 0:
                        print("reset opacity")
                        gaussians.reset_opacity()

                # Optimizer step
                #updates the parameters of the gaussian splats using the gradients calculated during backprop
                #optimizer adjusts the positions, colours, opacities, sizes and cov matrices
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True) #zero the gradients

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth") #the parameters are saved and restored from checkpoints to allow for training to resume
       

    final_checkpoint_path = os.path.join(scene.model_path, f"final_model_checkpoint_{stage}.pth")
    torch.save({
    'sh_degree': dataset.sh_degree,
    'iteration': iteration,
    'args': hyper  # Make sure 'hyper' contains the data you expect
    }, final_checkpoint_path)
    print(f"Final model checkpoint saved at: {final_checkpoint_path}")

            
#main training function
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    print("Entering training function")  # Debugging print
    tb_writer = prepare_output_and_logger(expname)
    print("Tensorboard writer prepared")  # Debugging print
    #initialise the gaussian model with the specified spherical harmonics degree and hyperparameters
    gaussians = GaussianModel(dataset.sh_degree, hyper) #parameters of the Gaussian model: position, color, opacity, size, covariance of each Gaussian splat
    dataset.model_path = args.model_path
    timer = Timer()
    #initialises the scene with the dataset and gaussian model
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start() #measure the duration of each training stage

    #calls the scene reconstruction twice: first for the "coarse" stage up to "opt.coarse_iterations"
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer) 
    #call the function again for the "fine" stage up to "opt.iterations"
    #scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         #checkpoint_iterations, checkpoint, debug_from,
                         #gaussians, scene, "fine", tb_writer, opt.iterations,timer)

#creates an output folder for the experiment using the specified experiment name
def prepare_output_and_logger(expname):    
    #output_path_tensorb = "/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0207_torchl1_norm_smooth_diagnostics_minmaxscaling_in_smooth_0.3_tensorb_beta3.5/logs"
    print("Entering prepare_output_and_logger") #added 0407
    model_path = os.path.join("./output_kyveli/", expname) #added 0407
    print(f"Model path set to:{model_path}")
    print("Output folder:{}".format(model_path)) #added 0407
    os.makedirs(model_path, exist_ok=True) #added 0407

    with open(os.path.join(model_path, "cfg_args"),'w') as cfg_log_f: #added 0407
         cfg_log_f.write("Dummy config for testing")

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
        print("TensorBoard SummaryWriter initialized")  # Debugging print
    else:
        print("Tensorboard not available: not logging progress")

    print("TensorBoard logs will be written to: ", model_path)
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss,smooth_torch,ema_smooth_torch_loss_for_log, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, weight_factor, use_smooth_torch, ssim, lambda_dssim=0.2):
    #training and validation losses, PSNR and other metrics on tensorBoard
    print("Entering training_report function")
    print("Tensorboard writer:", tb_writer)
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patches/total_loss', loss.item(), iteration)
        if use_smooth_torch:
            tb_writer.add_scalar(f'{stage}/train_loss_patches/smooth_loss', smooth_torch.item(), iteration) 
            tb_writer.add_scalar(f'{stage}/train_loss_patches/ema_depth_loss', ema_smooth_torch_loss_for_log, iteration)
        
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    #log validation losses
    if iteration in testing_iterations:
        print(f"Evaluating validation metrics at iteration {iteration}")  # Debugging print

        torch.cuda.empty_cache() #default from 4DGS
        
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                smooth_torch_test=0.0 
                overall_loss_test= 0.0 

                for idx, viewpoint in enumerate(config['cameras']):
                    #image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0) #commented out 0407 to add validation metrics
                    #added the following 3 lines for validation 0407
                    render_output_test= renderFunc(viewpoint, scene.gaussians, stage= stage, cam_type=dataset_type, *renderArgs)
                    image_test = torch.clamp(render_output_test["render"], 0.0, 1.0)
                    depth_test = render_output_test["depth"]
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_depth_test= torch.tensor(viewpoint.depth, device='cuda') #added 0407 for validation
                    print(f"Iteration {iteration} - Validation Viewpoint {idx}: GT Image Mean: {gt_image.mean().item()}, Std: {gt_image.std().item()}") #added 0507 to ensure not the same gt depth image is selected because the loss is flat and constant

                    try:
                        #logs rendered images and the corresponding ground truth iages for visualisation durng the validation/testing phase 
                        if tb_writer and (idx < 5):  #only the first 5 samples in the current batch are logged to tensorboard
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image_test[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image_test, gt_image).mean().double()
                    psnr_test +=psnr(image_test,gt_image, mask=None).mean().double()
                    if use_smooth_torch:
                        smooth_torch_test = l1_loss(depth_test/(depth_test.max() +1e-6), gt_depth_test/(gt_depth_test.max() +1e-6)) *weight_factor #no mask is applied during evaluation
                        
                
                    
                #after the loop divide the accumulated losses by the number of cameras to get average losses    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                if use_smooth_torch:
                    smooth_torch_test /= len(config['cameras'])
                    overall_loss_test =0.4* l1_test + smooth_torch_test #+ lambda_dssim * (1.0 - ssim(image_test, gt_image)) #ADJUST THIS!!! 1507
                else:
                    overall_loss_test = l1_test       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                #log the averaged losses to tensorboard using the add_scalar method
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if use_smooth_torch:
                        tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - smooth_loss', smooth_torch_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - overall_loss', overall_loss_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)

        torch.cuda.empty_cache() #default from 4DGS 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache() #free up unused memory in the GPU, default from 4DGS 
    parser = ArgumentParser(description="Training script parameters") #create an 'ArgumentParser' object that will be used to handle command-line arguments for the script
    setup_seed(6666) #reproducibility, consistent results across runs
    
    
    #adding arguments to the parser
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")  #'--ip' is the name of the command-line argument
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "") #path to a configuration file for additional parameters
    

    args = parser.parse_args(sys.argv[1:]) #after defining arguments, parser.parse_args() is called to parse the input and
    #return an object with the argument values as attributes 
    args.save_iterations.append(args.iterations)
    print("Testing iterations:", args.test_iterations)
    
    
    #check if a configuration file is provided via the '--configs' argument. If it is,
    #the script imports additional libraries and merges configurations from the file with 
    #the command-line arguments 
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs) #loads the configuration file
        args = merge_hparams(args, config) #merges the loaded configuration with the existing command-line arguments,
        #ensuring that any configurations specified in the file are applied
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) #for monitoring and interacting with the training process
    #using the IP address and port specified in the arguments 
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
  
    
