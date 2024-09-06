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
#densification: adjusts point cloud density based on specified thresholds and intervals



##initialise the gaussian model, load checkpoints if available, set up bacground color and timing events, handle data loading using custom sampler,
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
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, l1_loss_depth, masked_l1_loss, l1_loss_mask
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
import cv2 #2806 experiment with loss 

import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


#output_path_tensorb= "/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0207_torchl1_norm_smooth_diagnostics_minmaxscaling_in_smooth_0.3_tensorb_beta3.5/logs"
#Create the TensorBoard SummaryWriter
#tb_writer = SummaryWriter(log_dir=output_path_tensorb)
import torchvision.models as models
import torch.nn.functional as F
from heatmap import save_depth_discrepancy_heatmap


    

smooth_l1_loss = nn.SmoothL1Loss(beta=3) #default is beta=1
l1_loss_torch= nn.L1Loss()

torch.cuda.empty_cache()


def free_up_memory(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()

#def setup_cuda():
    #if not torch.cuda.is_available():
       # print("CUDA is not available. Exiting.")
       # sys.exit(1)
    #torch.cuda.empty_cache()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(device)
    #return device

#device = setup_cuda()
##end of additions for cude-related error

#added this for debugging- 24/06 afternoon
#def check_for_nan(tensor, name):
    #if torch.isnan(tensor).any():
        #print(f"{name} contains NaNs")
        #return True
    #return False
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


#use_smooth_torch= True
# Flag to enable or disable smooth torch entirely
enable_smooth_torch = True # Set to True to allow smooth torch, False for baseline

# Use smooth_torch only if enabled and random chance is met
use_smooth_torch = enable_smooth_torch and (random.random() < 0.8)
#use_smooth_torch = random.random() < 0.3


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
print(f"TENSORBOARD_FOUND: {TENSORBOARD_FOUND}")

def log_memory_usage(stage, iteration):
    print(f"[{stage}][Iteration {iteration}] Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{stage}][Iteration {iteration}] Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")



#added 0107 to troubleshoot why loss is oscillating instead of steadily decreasing
def normalize_tensor(tensor):
    return tensor / (tensor.max() + 1e-6)

#added 0407
def update_ema(new_value, ema, alpha=0.4):
    return alpha * new_value + (1 - alpha) * ema

def add_noise_to_depth(depth_tensor, noise_level):
    noise = torch.randn(depth_tensor.size()) * noise_level
    noisy_depth = depth_tensor + noise.to(depth_tensor.device)
    return noisy_depth
 


def calculate_mean_depth_difference(predicted_depth, gt_depth):
    """
    Calculate the mean absolute difference between predicted and ground truth depth maps.
    """
    if predicted_depth is None or gt_depth is None:
        print("Depth data is missing.")
        return None

    # Ensure both tensors are on the same device and have the same data type
    predicted_depth = predicted_depth.to(gt_depth.device)
    
    # Calculate element-wise absolute difference and its mean
    difference = torch.abs(predicted_depth - gt_depth)
    mean_difference = torch.mean(difference).item()  # Converts to Python float

    return mean_difference


def save_heatmap(predicted, ground_truth, output_dir, image_index):
    difference = np.abs(predicted.cpu().detach().numpy() - ground_truth.cpu().detach().numpy())


    difference = np.squeeze(difference)
    #normalized_discrepancy = np.squeeze(discrepancy)

    # Ensure the normalized_discrepancy is not empty after squeezing
    if difference.size == 0:
        raise ValueError("No valid image data to display after squeezing dimensions.")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(difference, cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Depth Discrepancy Heatmap for Viewpoint {image_index}")
    plt.savefig(os.path.join(output_dir, f"heatmap_viewpoint{image_index}.png"))
    #plt.title(f"Depth Discrepancy Heatmap for Image {image_index}")
    #plt.savefig(os.path.join(output_dir, f"heatmap_{image_index}.png"))
    plt.close()
    mean_diff = np.mean(difference)
    print(f"Image {image_index}: Mean depth difference = {mean_diff}")
    return mean_diff

def save_predicted_heatmap(predicted, ground_truth, output_dir, image_index):

    
    
    # Compute the absolute difference
    difference = np.abs(predicted.cpu().detach().numpy() - ground_truth.cpu().detach().numpy())
    difference = np.squeeze(difference)
    
    if difference.size == 0:
        raise ValueError("No valid image data to display after squeezing dimensions.")
    
    # Normalize the predicted depth map for visualization
    #predicted_depth_map = predicted_depth_map.cpu().numpy()
    #norm_predicted_depth = predicted_depth_map / (predicted_depth_map.max() + 1e-6)
    predicted_np = predicted.cpu().detach().numpy().squeeze()
    norm_predicted = (predicted_np - predicted_np.min()) / (predicted_np.max() - predicted_np.min() + 1e-6)
    norm_difference = (difference - difference.min()) / (difference.max() - difference.min() + 1e-6)

    #norm_predicted = (predicted.cpu().detach().numpy().squeeze() - predicted.min()) / (predicted.max() - predicted.min() + 1e-6) #comment this 


    # Create a colormap for the depth map
    cmap = plt.get_cmap('coolwarm')  # You can choose any colormap that suits your preference
    rgba_image = cmap(norm_predicted) #comment this 
    ###rgba_image = cmap(predicted.cpu().detach().numpy().squeeze())  # Convert the normalized depth map to RGBA colors based on the colormap

    #rgba_image = cmap(norm_predicted_depth)  # This converts the normalized depth map to RGBA colors based on the colormap

    # Overlay the difference as a heatmap
#####new additions 0708 ##
    # Apply a transparency mask where difference is less to show the predicted depth map more clearly
    #alpha_mask = 1 - difference / difference.max()
    alpha_mask = 1 - np.sqrt(norm_difference)
    #####alpha_mask = np.interp(difference, (difference.min(), difference.max()), (0.4, 1))  # Scale between 40% to 100% opacity

    rgba_image[..., 3] = alpha_mask  # Adjust the alpha channel to make lower discrepancies more transparent
#### new additions 0708###

    ###rgba_image[..., 0:3] *= (1 - np.expand_dims(difference, -1))  # Modulate the RGB channels based on the difference

    #plt.figure(figsize=(10, 5))
    plt.clf()  # Clear figure
    plt.cla()  # Clear axis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(rgba_image, interpolation='nearest')
    plt.colorbar(ax.imshow(rgba_image, interpolation='nearest'))
    ax.set_title(f"Depth Discrepancy Heatmap for Viewpoint {image_index}")
    fig.savefig(os.path.join(output_dir, f"heatmap_viewpoint_{image_index}.png"))
    plt.close(fig)  # Close the figure to free memory

    #plt.imshow(rgba_image, interpolation='nearest')
    #plt.colorbar()
    #plt.title(f"Depth Discrepancy Heatmap for Viewpoint {image_index}")
    #plt.savefig(os.path.join(output_dir, f"heatmap_viewpoint_{image_index}.png"))
    #plt.close()

    mean_diff = np.mean(difference)
    print(f"Image {image_index}: Mean depth difference = {mean_diff}")
    return mean_diff

def visualize_depth_and_error(predicted, ground_truth, output_dir, image_index):
    plt.clf()  # Clear any existing plot data
    plt.cla()

    # Normalize the predicted depth map
    predicted_np = predicted.cpu().detach().numpy().squeeze()
    norm_predicted = (predicted_np - predicted_np.min()) / (predicted_np.max() - predicted_np.min())

    # Compute and normalize the difference
    ground_truth_np = ground_truth.cpu().detach().numpy().squeeze()
    difference = np.abs(predicted_np - ground_truth_np)
    norm_difference = (difference - difference.min()) / (difference.max() - difference.min())

    # Set up subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display predicted depth map
    ax1 = axes[0]
    im1 = ax1.imshow(norm_predicted, cmap='gray')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title(f"Predicted Depth Map for Viewpoint {image_index}")

    # Display discrepancy map
    ax2 = axes[1]
    im2 = ax2.imshow(norm_difference, cmap='coolwarm')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title(f"Discrepancy Map for Viewpoint {image_index}")

    # Overlay depth and discrepancy for combined visualization
    ax3 = axes[2]
    ax3.imshow(norm_predicted, cmap='gray', alpha=0.6)  # Reduced alpha to see overlay
    im3 = ax3.imshow(norm_difference, cmap='coolwarm', alpha=0.4)  # Transparent error overlay
    fig.colorbar(im3, ax=ax3)
    ax3.set_title("Combined Visualization")

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"combined_viewpoint_{image_index}.png"))
    plt.close(fig)

    mean_diff = np.mean(difference)
    print(f"Image {image_index}: Mean depth difference = {mean_diff}")
    return mean_diff

def visualize_predicted_errors(predicted, ground_truth, output_dir, image_index):
    # Compute the absolute difference

    predicted_np = predicted.cpu().detach().numpy()
    ground_truth_np = ground_truth.cpu().detach().numpy()
    difference = np.abs(predicted_np - ground_truth_np)
   
    
    # Normalize the predicted depth map for visualization
    norm_predicted = (predicted_np - predicted_np.min()) / (predicted_np.max() - predicted_np.min())

    
    # Create a colormap for the depth map
    cmap_base = plt.get_cmap('gray')  # For the background
    cmap_error = plt.get_cmap('viridis')  # For the error heatmap

    # Create the base image
    base_image = cmap_base(norm_predicted)

    # Create the error heatmap
    error_image = cmap_error(difference / difference.max())

    # Adjust transparency based on the error magnitude
    error_image[..., 3] = difference / difference.max()

    # Overlay the heatmap onto the base image
    combined_image = base_image.copy()
    combined_image[..., :3] += error_image[..., :3]  # Add color
    combined_image[..., 3] = 1  # No transparency in the final image
    combined_image = np.squeeze(combined_image)  # Ensure no unnecessary dimensions are present


    # Calculate mean difference
    mean_diff = np.mean(difference)
    print(f"Image {image_index}: Mean depth difference = {mean_diff}")

    # Plot and save the result
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_image)
    plt.colorbar(label='Error magnitude', orientation='vertical')
    plt.title(f"Predicted Depth Map with Error Heatmap Overlay for Image {image_index}")
    plt.savefig(f"{output_dir}/heatmap_viewpoint_{image_index}.png")
    plt.close()

    return mean_diff

def error_intensity_overlay(predicted, ground_truth, output_dir, image_index):
    # Calculate the absolute difference
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
    # Increase max_split_size_mb before entering fine stage

    ###ADDED 3007###
    # Open a file for logging depth discrepancies
    log_file_path = os.path.join(scene.model_path, f"depth_discrepancies_brain_2908_further_exp_fulll1rendered_0.7depth_{stage}.log")
    with open(log_file_path, 'a') as log_file:
        # Continue with existing setup...
        log_file.write(f"Logging started for {stage} stage\n")
        log_file.flush()  # Ensure the initial message is written to the disk
    #### END OF ADDITION 3007###

    #INITIALISE THE GAUSSIAN MODEL, LOAD CHECKPOINT IF AVAILABLE.
        weight_factor=0.5
        first_iter = 0
        print(vars(opt)) #added for debugging
        #training_setup called upon model instantiation
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
            #### COMMENTED OUT 1107###
            #if stage == "fine":
            # Clear memory before entering the fine stage
            #del viewpoint_stack, temp_list
            #torch.cuda.empty_cache()
            
            #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        #elif stage == "coarse":
            #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
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
        
            gt_depths = [] #added 2806
            depths =[] #added 2806

            masks= []

            visibility_filter_list = []
            viewspace_point_tensor_list = []
        #11/06: added idx and did enumerate 
        #for viewpoint_cam in viewpoint_cams: this is the original in 4dgs
        #def print_grad(grad):
            #print("depth tensor gradient:", grad)
            for idx,viewpoint_cam in enumerate(viewpoint_cams): #for each selected camera viewpoint, generate the rendered image
        
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, iteration= iteration, viewpoint_idx= idx) # Pass iteration and idx- 11/06
                print(f"Rendering at iteration {iteration} in stage {stage}")
            
            
                ##print("generated depth 2806 is", generated_depth.shape)
                ##print("generated depth 2806", type(generated_depth))
            
                #original is: render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
                #image_index= viewpoint_cam.image_index #ADDED 12/06
                #render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type, iteration=iteration, viewpoint_idx=image_index)####, final_iteration= final_iter)  # Pass the image index ADDED 12/06 

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                #image is the rendered image
            
                depth = render_pkg["depth"].to('cuda')  #added 2806 to test the l1 depth loss (ground truth depth with generated depth comparison)
                depth.requires_grad_(True) #added 2806
                save_depth_map(depth, stage, iteration, idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2608/brain_0.6l1rendered/depth_maps") #0407
 
                
                print("depth tensor requires_grad", depth.requires_grad)
                depth.register_hook(lambda grad: print("Depth tensor gradient:", grad)) #added 3006
                ############
                #depth = add_noise_to_depth(depth, noise_level=100)
                #########
                gt_depth = torch.tensor(viewpoint_cam.depth, device='cuda') #added 2806
                mask = viewpoint_cam.mask
                print(f"Loading GT depth for camera {viewpoint_cam.depth}")
                print(f"GT depth stats before processing: min={np.min(viewpoint_cam.depth)}, max={np.max(viewpoint_cam.depth)}")
                mask = (gt_depth > 0).float()  #2807 change
           
                print("gt_depth tensor values", gt_depth)
           
            
                if scene.dataset_type!="PanopticSports":
                    gt_image = viewpoint_cam.original_image.cuda() #retrieves the ground truth image for the current viewpoint
                    print("gt_image is", gt_image)
                    print("gt_image_shape", gt_image.shape)
                else:
                    gt_image  = viewpoint_cam['image'].cuda()
                    #added for debugging 25/06 morning
                    print("gt_image is", gt_image)
                    print("gt_image_shape", gt_image.shape)


                if mask is not None:
                    mask = mask.cuda()
                    masks.append(mask.unsqueeze(0))
            # Create a mask where ground truth depth is greater than zero
            #mask = (gt_depth > 0).float() #2707
            



            # #added 2806
            ###ADDED 2707###
                print("des edw ta values")
                print(gt_depth.min(), gt_depth.max(), gt_depth.mean())
            #dep_mask = torch.logical_and(gt_depth > 0, depth > 0)
            #dep_mask = (gt_depth > 0).float() #2707 #PROSOXI prepei na einai gt_depth>0 kanonika 
            #print("EDW MASK INSPECTION", dep_mask)
            #print("mask shape", dep_mask.shape)
            #dep_mask = torch.logical_and(gt_depth >0) #, depth > 0)
            #gt_depth = gt_depth * dep_mask
            #depth = depth * dep_mask
            ##############

                images.append(image.unsqueeze(0))
                dep_mask= torch.logical_and(gt_depth>0, depth>0) #ORIGINAL ENDO-4DGS
                #dep_mask=(gt_depth>0).float() Exoun thema ta dimensions
                gt_depth = gt_depth * dep_mask
                depth = depth * dep_mask
                depths.append(depth.unsqueeze(0))
                gt_depths.append(gt_depth.unsqueeze(0)) #added 2806
                gt_images.append(gt_image.unsqueeze(0)) #ground truth images corresponding to the viewpoints are retrieved for comparison
            
                radii_list.append(radii.unsqueeze(0))
            #visibility_filter_list contains boolean tensors indicating whether each Gaussian splat is visible from the corresponding viewpoint
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)

            # Save the depth map, commented out 3006
            #save_depth_map(depth, stage, iteration, idx, args.model_path, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0107_endo_norm/depth_maps") #2806 added, commented out 0107
            #save_depth_map(depth, stage, iteration, idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/multipleview/brain_2907_mask_tensor_gpu35_weight0.3_prob0.8_l1full/depth_maps") #0407
            #save_depth_map(depth, stage, iteration, idx, output_dir="/vol/bitbucket/kt1923/4DGaussians/output/multipleview/office_0_0207_torchl1_norm_smooth_diagnostics_minmaxscaling_in_smooth_0.3_tensorb_beta3.5/depth_maps") #2806 added
        #contains the radii for each gaussian splat from ALL the viewpoints in the CURRENT batch.
        #concatenation of the tensors along the first dimension resulting in a single tensor containing all the radii
        #max(dim=0) computes the max radius for EACH gaussian splat across ALL viewpoints.
        #this ensures that we consider the largest extent of each splat- crucial for determining their visibility and spatial influence 
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

        #mean_depth_difference = calculate_mean_depth_difference(depth_tensor, gt_depth_tensor) #added 2907
        #print(f"Mean Depth Difference at iteration {iteration}: {mean_depth_difference}") #added 2907

        
        
        #image_tensor = torch.cat(images,0) #list of tensors representing the rendered image for each viewpoint in the current batch
        #gt_image_tensor = torch.cat(gt_images,0) #shape is [1,3,480,640]

        
        #depth_tensor= torch.cat(depths,0)
        #gt_depth_tensor= torch.cat(gt_depths,0)
            print("shape of image tensor", image_tensor.shape) #[1,3,480,640]

            log_memory_usage(stage, iteration)
        #the above lists of tensors are essential for batching operations during loss calculation and backprop
        # Loss
        # breakpoint()
        #first Ll1 is the original one

        ####HAOZHENG 11/07###
      
        # Loss
        #Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])


        #if iteration <= 12000:
            #use_smooth_torch = True
       # else:
            #use_smooth_torch = False
        ######Ll1 = l1_loss(image, gt_image)#, mask=None) #adjusted 2707
        
            Ll1 = l1_loss_mask(image_tensor, gt_image_tensor, mask_tensor.unsqueeze(0))
        #loss= Ll1
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        
        #loss = (1.0 - opt.lambda_dssim) * Ll1 
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #edw 1907
        # norm

            #BELOW UNCOMMENT IF YOU'RE RUNNING THE BASELINE- 3007 POST-MEETING##
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

        

        
            if use_smooth_torch: #now use_smooth_torch is based on the prob:use_smooth_torch = random.random() < 0.3
                print("CHECKING FOR THE L1 LOSS VALUES-RANGE", image_tensor.max())
                print("checking for l1 loss value", Ll1)
                weight_factor=0.5
                
                
            
            #smooth_torch = l1_loss(depth/(depth.max()+1e-6), gt_depth/(gt_depth.max()+1e-6))*weight_factor #ORIGINAL


            # Calculate the masked L1 loss for depth
            #smooth_torch = (torch.abs(depth - gt_depth) * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0).to(depth.device)
                normalized_depth = depth / (depth.max() + 1e-6)
                normalized_gt_depth = gt_depth / (gt_depth.max() + 1e-6)


                masked_depth_loss = torch.abs(normalized_depth - normalized_gt_depth) * mask
                masked_depth_loss = masked_depth_loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0).to(depth.device)

            # Apply mask and calculate L1 loss for depth
            #masked_depth_loss = (torch.abs(normalized_depth - normalized_gt_depth) * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0).to(depth.device)
            #masked_depth_loss= l1_loss_mask(normalized_depth, normalized_gt_depth, mask=dep_mask)

            #masked_depth_loss= l1_loss(normalized_depth, normalized_gt_depth)




            #smooth_torch = l1_loss(depth/(depth.max()+1e-6), gt_depth)*weight_factor #trial 2207

            #smooth_torch = l1_loss(depth - depth.min()) / (depth.max() - depth.min() + 1e-6), (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-6) * weight_factor
            #smooth_torch= smooth_l1_loss(depth - depth.min()) / (depth.max() - depth.min() + 1e-6), (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-6)* weight_factor
            
            #normalized_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            #normalized_gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-6)   
            #smooth_torch = smooth_l1_loss(normalized_depth, normalized_gt_depth) * weight_factor

            #if normalized_depth.size(1) == 1:
                #normalized_depth = normalized_depth.repeat(1, 3, 1, 1)
                #normalized_gt_depth = normalized_gt_depth.repeat(1, 3, 1, 1)     
          

            #perceptual_depth_loss = perceptual_loss_fn(normalized_depth, normalized_gt_depth) * 0.6


            #smooth_torch = l1_loss(depth, gt_depth)*weight_factor
            #loss += smooth_torch
            #loss = 0.5* Ll1 + (smooth_torch *dep_mask).sum()/dep_mask.sum() + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #ORIGINAL
                depth_loss = l1_loss_mask(depth_tensor/(depth_tensor.max()+1e-6), gt_depth_tensor/(gt_depth_tensor.max()+1e-6), \
                    mask=mask_tensor.unsqueeze(0))* weight_factor 
                depth_discrepancy = l1_loss_depth(normalized_depth, normalized_gt_depth)
                print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)

                # Print and log the discrepancy
                discrepancy_message = f"Iteration {iteration}, Image Index {idx}: Depth Discrepancy = {depth_discrepancy}\n"
                print("Depth Discrepancy between Ground Truth and Predicted Depth Map", depth_discrepancy)
                log_file.write(discrepancy_message)
                # glog_file.flush()

                # Call the function to save the heatmap
                output_dir_heatmap = '/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2908_brain_fulll1rendered_0.7depth'  # Ensure this directory exists or is created
                save_depth_discrepancy_heatmap(normalized_depth, normalized_gt_depth, output_dir_heatmap, iteration, idx)        
                
                loss =  0.8*Ll1 + depth_loss  #+ opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                
            ###loss =  Ll1 + masked_depth_loss * weight_factor + opt.lambda_dssim * (1.0 - ssim(image, gt_image))*weight_factor  #2707
            else:
            #loss = loss
                loss = Ll1 
            
            loss.backward() #if this is here it means im training only for coarse stage




        ####MY IMPLEMENTATION AS OF 11/07 BEFORE HAOZHENG####
        #Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:]) #optimise the rendered images to match the ground truth
        #print('L1 loss range', Ll1)

        
       
        
        
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double() #measures image quality
       
        #if random.random()<0.5:
            #1107 try the l1 loss from utils.py
            #smooth_torch = l1_loss(
                #(depth - depth.min()) / (depth.max() - depth.min() + 1e-6), 
                #(gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-6)
                #) * weight_factor
           
       # else:
            #smooth_torch= torch.tensor(0.0, device='cuda')
        #print("EDW SMOOTH TORCH RANGE VALUES", smooth_torch)
        #print("Ll1 loss range values", Ll1)
        # norm
        

        #loss = Ll1+smooth_torch #measures absolute differences between predicted and actual pixel values optimising the rendered images to match the ground truth
        #the total loss is INITIALLY set to the L1 loss
        # Print initial loss values
            if iteration % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {iteration} - Initial L1 Loss: {Ll1.item():.4f}, PSNR: {psnr_.item():.2f}")
        #add this for debugging- 25/06 morning
        #if iteration == 2999:
            #print("iteration 2999 gaussians -what it is", gaussians)
            #print("iteration 2999", gt_image_tensor.shape)
            #edge_aware_smoothness_loss(gaussians, gt_image_tensor)



        #ADDITION 0608###
            if iteration == final_iter:
                # This is the final iteration; perform evaluation here
                print(f"Performing evaluation at iteration {iteration}")
                test_cams = scene.getTestCameras()  # Load test dataset cameras
                # Assume create_heatmaps is a function you define to handle heatmap creation
                output_dir_heatmap = '/vol/bitbucket/kt1923/4DGaussians/output/further_exp_2608/heatmaps/2908_brain_fulll1rendered_0.7depth'
                os.makedirs(output_dir_heatmap, exist_ok=True)  # Ensure output directory exists


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

                        # Save depth discrepancy heatmap
                        #mean_diff= save_heatmap(normalized_predicted_depth, normalized_gt_depth, output_dir_heatmap, idx) #THIS WORKS, 0608 implementation
                        #mean_diff=save_predicted_heatmap(normalized_predicted_depth, normalized_gt_depth, output_dir_heatmap, idx) #0708 
                        #mean_diff= visualize_depth_and_error(normalized_predicted_depth, normalized_gt_depth, output_dir_heatmap, idx)
                        #mean_diff= visualize_predicted_errors(normalized_predicted_depth, normalized_gt_depth, output_dir_heatmap, idx)

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

        
         
        #added this for debugging- 24/06 afternoon
        #if check_for_nan(Ll1, "Ll1"):
            #print("Exiting due to NaN in Ll1")
            #exit(1)
            if stage == "fine" and hyper.time_smoothness_weight != 0:
                #added 2606 to troubleshoot why its not saving the depth maps at the fine stage
                print(f"Entering fine stage at iteration {iteration}") #added 2606

            #Render call for fine stage to ensure depth maps are saved
            ### 2706 commented out to bring it back to the original implementation
            ###for idx, viewpoint_cam in enumerate(viewpoint_cams): #added 2606
                ###render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type, iteration=iteration, viewpoint_idx=idx) #added 2606
                ###print(f"Rendered at iteration {iteration}, viewpoint_idx {idx} in fine stage") #addd 2606
            #more granular loss function if the stage is 'fine' to capture subtle temporal changes 
            #total variational loss (TV loss) is added to encourage temporal coherence between consecutive timeframes (maintain consistent motion and apearace of objects across frames)
            # tv_loss = 0
            # Print TV loss values
                if iteration % 100 == 0:  # Print every 100 iterations
                    print(f"Iteration {iteration} - TV Loss: {tv_loss.item():.4f}")

                tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
                print("tv loss range at fine stage", tv_loss)
                loss += tv_loss
                print("depth tensor before scaling in smooth loss", depth)
                print("gt depth tensor before scaling in smooth loss", gt_depth)
                print('max depth tensor value', depth.max())
                print('max gt_depth tensor value', gt_depth.max())
            
            #smooth_torch= smooth_l1_loss(depth/depth.max(), gt_depth/gt_depth.max())
            #Min-Max scaling for depth tensors normalisation. transforms the depth values to a [0,1] range
            #directly based on the observed min and max values.
            #cons: sensitive to outliers, 
   
            #smooth_torch= smooth_l1_loss((depth- depth.min())/(depth.max()- depth.min() +1e-6), (gt_depth- gt_depth.min())/(gt_depth.max()- gt_depth.min() +1e-6)) *weight_factor
                print("smooth torch value when applied min max scaling", smooth_torch)
            
                print('type of smooth torch object', type(smooth_torch))
                #loss += smooth_torch
                print(f"Iteration {iteration} - Total Loss after all components: {loss.item():.4f}")
        
                if iteration % 100 == 0:
                    tb_writer.add_scalar('Loss/TV Loss', tv_loss.item(), iteration)
                    tb_writer.add_scalar('Loss/Smooth Loss', smooth_torch.item(), iteration)
                    tb_writer.add_scalar('Loss/Total Loss', loss.item(), iteration)


            #add this on 2706 to test the new attempt for loss
                positions= gaussians.get_positions
            #depths = gaussians.get_depths #for custom smoothing loss only commented out 2806
            ####print("depths in optimisation", depths) commenteed out on 2806
            ####print("shape of depths in optimisation", depths.shape) #[1,480,640] commented out 2806
            #print("type of depths in optimisation", type(depths)) #tensor
            #print(f"Iteration {iteration} positions:\n{positions}") #debug 2706
            #print(f"Iteration {iteration} depths:\n{depths}") #debug 2706
            #smoothness_loss= find_neighbors_and_smooth_splats(positions, depths, k=10)
            #loss += opt.lambda_smoothness * smoothness_loss


            ##2806 experiment with endo-4dgs depth smooth loss
            
        
            #add the smoothness loss based on neighboring splats- 24/06 and 25/06
            ###smoothness_loss= depth_smoothness_splats_vector(gaussians, radius=1.0) 
            ###loss += opt.lambda_smoothness * smoothness_loss

            ##added edge-aware experiment 24/06
            #smoothness_loss = edge_aware_smoothness_loss(gaussians, gt_image_tensor) #disable edge_aware to try the neighborhood loss first
            #added this for debugging- 24/06 afternoon
            #if check_for_nan(smoothness_loss, "smoothness_loss"):
                #print("Exiting due to NaN in smoothness_loss")
                #exit(1)
            #loss += opt.lambda_smoothness * smoothness_loss
        ##if opt.lambda_dssim != 0: #lambda_dssim is a weighing factor that determines the contribution of the SSIM loss to the total loss
            #SSIM is incorporated to focus on preserving structural info and perceptual qualtity rather than just pixel-wise accuracy
            ##ssim_loss = ssim(image_tensor,gt_image_tensor) #ssim loss optionally added to the total loss based on lambda_dssim hyperparameter
            #if check_for_nan(ssim_loss, "ssim_loss"):
                #print("Exiting due to NaN in ssim_loss")
                #exit(1)
            ##loss += opt.lambda_dssim * (1.0-ssim_loss)
            # Print SSIM loss values
                if iteration % 100 == 0:  # Print every 100 iterations
                    print(f"Iteration {iteration} - SSIM Loss: {(opt.lambda_dssim * (1.0 - ssim)).item():.4f}")

              
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        # Print total loss values
            if iteration % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {iteration} - Total Loss: {loss.item():.4f}")

        #loss.backward() #backprop to compute gradients and update the Gaussian parameters based on the gradients computed from the loss function 
        #free_up_memory(image_tensor, gt_image_tensor, depth_tensor, gt_depth_tensor, viewspace_point_tensor_list, radii_list, visibility_filter_list) #commented out 1107

            if depth_tensor.grad is None:
                print("gradient for depth_tensor are not being computed")
            else:
                print("gradients for depth_tensor are computed")
            if depth.grad is None:
                print("Gradient for depth tensor is not being computed")
            else:
                print("gradient for depth tensor is being computed". depth.grad)


            if gaussians._xyz.grad is None:
                print("Gradients for positions (_xyz) are not being computed.")
            else:
                print("Gradients for positions (_xyz) are being computed.")        


        #added 0407
            if torch.all(depth_tensor == 0):
                print(f"Depth tensor becomes 0 at iteration {iteration}, training terminates.")
                break #added 0507    
            if torch.isnan(loss).any() or torch.isnan(depth).any():
                print("NaN detected in loss or depth, skipping iteration.")
                continue
            if torch.isnan(loss).any():
                print("loss is nan iteration {iteration}, end training, reexecv program now.")
                #added 3006
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
        ####COMMENTED OUT CLEAR CACHE 1107###
        ##del image_tensor, gt_image_tensor, depth_tensor, gt_depth_tensor, render_pkg, viewspace_point_tensor_list, depths, gt_depths, images, gt_images #added 0907

        ##torch.cuda.empty_cache()  # added that 2606
        

            with torch.no_grad(): #only updating the metrics so we dont need to track gradients
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
                #added the following line 0407
                ema_smooth_torch_loss_for_log = update_ema(smooth_torch.item(), ema_smooth_torch_loss_for_log)
                total_point = gaussians._xyz.shape[0]
            #smooth_torch_loss = torch.tensor(0.0, device='cuda')  # Add this line 3006
                if iteration % 10 == 0:
                    string_dict = {"Loss": f"{Ll1.item():.{4}f}",
                                        "psnr": f"{psnr_:.{2}f}"} #added 3006
                    if stage == 'fine':
                        string_dict['tv'] = f"{tv_loss:.{4}f}"
                
                progress_bar.update(10)
                #added 3006
                
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save
                timer.pause()
                    ##training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss,smooth_torch, ema_smooth_torch_loss_for_log,iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, weight_factor, use_smooth_torch, ssim, lambda_dssim=0.2)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, stage)
               # plot_opacity_custom(gaussians, output_dir= args.model_path, index=iteration) #added by me June 3rd
            #added this 18.27
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
                    # breakpoint()
                        #render from both train camera viewpoints and from test camera viewpoints (the latter for evaluation to check how well the model generalises to unseen viewpoints for which we have the ground truth images in the dataset)
                        #here the two folders coarsetrain_render and coarsetest_render and their pointclouds subfolders are being created
                        #this is where we check how well the model performs in generating accurate and high-quality images from new angles.
                        #Robustness: Its ability to handle different lighting, occlusions, and other scene-specific challenges that might not have been explicitly covered during training.
                        #testing phase: if we want to generate data for augmentation we can use the render_training)image function with test camera viewpoint to create images
                        #from "unseen" viewpoints 
                            render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            print("length of test cameras at the render_training_image function", len(test_cams))
                            render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
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
        #tb_writer.close()  
       # log_file.write("Logging completed\n")  #commented out 0608


    #ADDED 3007
    #final_checkpoint_path = os.path.join(scene.model_path, f"final_model_checkpoint_{stage}.pth")
    #torch.save({
        #'model_state_dict': gaussians.state_dict(),
        #'optimizer_state_dict': gaussians.optimizer.state_dict(),
        #'iteration': iteration
    #}, final_checkpoint_path)
    #print(f"Final model checkpoint saved at: {final_checkpoint_path}")
    #print("edw to final checkpoint path", final_checkpoint_path)
    ### end of addition 3007



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

### commented out 0407###
    ##if not args.model_path:
        ##unique_str = expname #added by me 
        ##args.model_path = os.path.join("./output_kyveli/", unique_str) #added by me
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        ##unique_str = expname

        ##args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    ##print("Output folder: {}".format(args.model_path))
    ##os.makedirs(args.model_path, exist_ok = True)
    #save the configuration arguments to a file in the output folder ("cfg_args")
    ##with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        ##cfg_log_f.write(str(Namespace(**vars(args))))
#commented out 0307
    # Create Tensorboard writer
    ##tb_writer = None
    ##if TENSORBOARD_FOUND:
        ##tb_writer = SummaryWriter(args.model_path)
        #tb_writer = SummaryWriter(output_path_tensorb)
    ##else:
        #print("Tensorboard not available: not logging progress")
    
    #print("TensorBoard logs will be written to: ", args.model_path)     
    #return tb_writer

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
                smooth_torch_test=0.0 #added 0407 for validation
                overall_loss_test= 0.0 #added 0407 for validation

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
                        #smooth_torch_test += l1_loss_torch((depth_test- depth_test.min())/(depth_test.max()- depth_test.min() +1e-6), (gt_depth_test- gt_depth_test.min())/(gt_depth_test.max()- gt_depth_test.min() +1e-6)) *weight_factor
                        smooth_torch_test = l1_loss(depth_test/(depth_test.max() +1e-6), gt_depth_test/(gt_depth_test.max() +1e-6)) *weight_factor #ORIGINAL
                        #smooth_torch_test = l1_loss(depth_test/(depth_test.max() +1e-6), gt_depth_test) *weight_factor #2207

                   
                    # mask=viewpoint.mask
                    
                #after the loop dive the accumulated losses by the number of cameras to get average losses    
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

 #the construct if__name__=="__main__" is used to ensure that certain code blocks
#are only executed when the script is run directly, and not when it is imported 
#as a module in another script    
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
    #parser.add_argument('--lambda_smoothness', type=float, default=0.5)  # Added for smoothness loss- 24/06
   

  

    
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
  
    
