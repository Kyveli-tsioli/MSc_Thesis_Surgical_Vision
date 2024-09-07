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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
from sklearn.neighbors import KDTree #added this 25/06
from scipy.spatial import cKDTree
import numpy as np


def lpips_loss(img1, img2, lpips_model): #uses a pre-trained net to compute perceptual similarity (focus on high-level features rathen than pixel-wise differences)
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean() #mean absolute error: optimise the rendered images to match ground truth images 

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean() #mean squared error between the networ output and the ground truth (for penalising large deviations in pixel values)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True): #measures similarity between two images, focusing on strcutural info (perceptual similarity)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def l1_loss_depth(network_output, gt):
    #nan_mask = ~torch.isnan(network_output)  # Create a mask to ignore NaN values
    #return torch.abs(network_output[nan_mask] - gt[nan_mask]).mean()
    return torch.abs(network_output - gt).mean()


def l1_loss_mask(network_output, gt, mask=None):
    nan_mask = ~torch.isnan(network_output)
    if mask is not None:
        if mask.shape[1] != network_output.shape[1]:
            mask = mask.expand(-1, network_output.shape[1], -1, -1)
    if mask is not None:
        return (torch.abs((network_output[nan_mask] - gt[nan_mask]))*mask[nan_mask]).mean()
    else:
        return (torch.abs((network_output[nan_mask] - gt[nan_mask]))).mean()



    

def find_neighbors_and_smooth_splats(positions, depths,k=10):

    #extract 3D positions and depths
    #remove detach
    positions_np= positions.detach().cpu().numpy() #tensor containing the 3D coordinates of each gaussian splat get detached from the comp graph, then moved to cpu then converted to numpy array
    print("positions", positions_np)
    print("positions shape", positions_np.shape) #(120517,3) i.e. (n,3) where n=120517 is the number of gaussian splats. is that normal? paper says at optimised state 1-5 million splats for all scenes tested
    #each row in positions_np represents the 3D coordinates of a single gaussian splat
    #positions shape: (84031,3)
    print("positions type", type(positions_np))


    # Flatten the depths to match the splat positions
    depths_flat = depths.view(-1).detach().cpu().numpy() #flatten the depth tensor into a 1D tensor
    print("depths shape", depths_flat.shape) #(3072000,)
    
    #create KD-tree for finding nearest neighbors 
    tree= cKDTree(positions_np) #root node of the KDTree selected based on the median value of the points along the x coordinate (taken from the 3D gaussians positions)
    #tree is constructed by recursively partitioning the 3D space using the median values along each dimension
    
    #initialise smoothness loss
    smoothness_loss=0.0

    #find k-nearest neighbors for EVERY splat (efficient because it uses the partitioning of the KD Tree)
    distances, indices= tree.query(positions_np, k=k) #finds the k nearest neighbors for each point in positions_np by traversing the constructed KDTree without a for loop!
    #returns the distances to the k nearest neighbors and the indices of these neighbors
    #distances is (n,k) and indices also (n,k)
    for i in range(len(positions_np)): #iterate over each splat 
        #get the neigbors' depths and distances
        neighbor_indices= indices[i] #retrieve the indices of the k nearest neighbors
        neighbor_depths= depths_flat[neighbor_indices] #fetch the depths of these neighbors
        neighbor_distances= distances[i] #retrieve the distances of the k nearest neighbors
        #calculate weights based on inverse distance
        weights= 1/ (neighbor_distances +1e-6) 
        weights /= np.sum(weights) #normalise weights to sum to 1
        #compute weighted average depth
        weighted_avg_depth= np.sum(weights* neighbor_depths)


        # Calculate smoothness loss as L2 norm
        #smoothness_loss += (depths_flat[i] - weighted_avg_depth) ** 2 l2
        smoothness_loss += np.abs(depths_flat[i] - weighted_avg_depth) #l1

    smoothness_loss = torch.tensor(smoothness_loss, device=positions.device)
    return  smoothness_loss / len(positions_np)




def depth_smoothness_splats(gaussians, radius=1):
    #gaussians: gaussian object (initialised in the scene>init.py)
    #radius: radius within which to enforce depth smoothness
    
    #get positions and depths of Gaussians
    positions = gaussians.get_xyz
    depths = gaussians.get_depths.squeeze(0)  #removes batch dimension, now depths shape is [480, 640]
    print("positions shape in the loss function", positions.shape) # [83052,3]: where probably 83052 is the number of gaussian splats and 3 represents the 3D coordinates for each splat
    print("depths shape in the loss function", depths.shape)
    print("depths in the loss", depths)
    print("positions in the loss", positions)
    smoothness_loss = 0.0
    count = 0

    # Iterate over each Gaussian splat
    for i in range(len(positions)):
        pos_i = positions[i]
        depth_i = depths[i // depths.shape[1], i% depths.shape[1]] #get the depth of the current gaussian splat?
        print("let's check the pos_i",pos_i)
        print("let's check the depth_i", depth_i)
        counter_iter=0
        for j in range(len(positions)):
            if i != j:
                pos_j = positions[j]
                distance = torch.norm(pos_i - pos_j).item()

                if distance < radius:
                    depth_j = depths[j // depths.shape[1], j% depths.shape[1]]  # Get the depth of the neighboring Gaussian splat
                    smoothness_loss += torch.abs(depth_i - depth_j)
                    count += 1
                    counter_iter +=1
                    print("counter_iter", counter_iter)

    return smoothness_loss / count if count > 0 else smoothness_loss

    #comment on normalisation step:
    #normalisation: averaging the smoothness loss by the count of neighbor pairs
    #ensures that the loss is not dependent on the number of neighbors. this makes the loss value consistent regardless of the density of the splats 
    #also avoid the scenario where the smoothness loss grows large if there are many neighbors, making the overall loss (the rest of the losses) unstable

def depth_smoothness_splats_vector(gaussians, radius=1):
    # Get positions and depths of Gaussians
    positions = gaussians.get_xyz
    depths = gaussians.get_depths.squeeze(0)  # Removes batch dimension
    smoothness_loss = 0.0

    # Compute pairwise distances between splats
    distances = torch.cdist(positions, positions)

    # Create a mask for neighbors within the radius
    neighbor_mask = (distances < radius) & (distances > 0)  # Exclude self-distances (i.e., distance > 0)

    # Get indices of neighbors
    neighbor_indices = neighbor_mask.nonzero(as_tuple=True)

    # Compute depth differences for neighbors
    depth_diffs = torch.abs(depths.view(-1)[neighbor_indices[0]] - depths.view(-1)[neighbor_indices[1]])

    # Compute smoothness loss as the mean of depth differences
    if len(depth_diffs) > 0:
        smoothness_loss = depth_diffs.mean()

    return smoothness_loss



def depth_smoothness_splats_KD(gaussians, radius=1.0):
    # Get positions and depths of Gaussians
    positions = gaussians.get_xyz
    depths = gaussians.get_depths.squeeze(0)  # removes batch dimension, now depths shape is [480, 640]

    print("positions shape in the loss function", positions.shape)  # [83052, 3]: where probably 83052 is the number of Gaussian splats and 3 represents the 3D coordinates for each splat
    print("depths shape in the loss function", depths.shape)
    print("depths in the loss", depths)
    print("positions in the loss", positions)

    # Flatten the depths for easier access
    depths_flat = depths.view(-1)
    print("depths flat is", depths_flat)
    print("depths flat shape", depths_flat.shape)


    # Use KD-Tree for efficient neighbor search
    kdtree = cKDTree(positions.detach().cpu().numpy())
    smoothness_loss = 0.0
    count = 0

    # Iterate over each Gaussian splat
    for i in range(len(positions)):
        pos_i = positions[i].detach().cpu().numpy()
        depth_i = depths_flat[i]

        # Find neighbors within the specified radius
        neighbors = kdtree.query_ball_point(pos_i, r=radius)
        for j in neighbors:
            if i != j:
                depth_j = depths_flat[j]
                smoothness_loss += torch.abs(depth_i - depth_j)
                count += 1

    return smoothness_loss / count if count > 0 else smoothness_loss



def edge_aware_smoothness_loss(gaussians, rgb):
    depth= gaussians.get_depths
    # Add a dummy dimension to depth to match the shape of rgb for debugging on nan- 24/06 afternoon
    depth = depth.unsqueeze(1)
    print("depth unasqueeze(1) shape at inter 2999", depth.shape)

    #im comparing the rendered image (gaussians) and the training image (rgb)
    #so the shapes should be rendered image: [batch, H, W, 3]
    #rgb: [batch, H, W, 3]
   
    grad_depth_x = torch.abs(depth[..., :, :-1] - depth[..., :, 1:])
    grad_depth_y = torch.abs(depth[..., :-1, :] - depth[..., 1:, :])        

    grad_img_x= torch.mean(torch.abs(rgb[..., :, :-1, :]- rgb[..., :, 1:, :]), dim=1, keepdim=True)
    grad_img_y= torch.mean(torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), dim=1, keepdim=True)


    # Adjust the shapes to make sure they match
    if grad_depth_x.shape[-1] != grad_img_x.shape[-1]:
        #added for debugging 25/06 morning
        print("MPIKE STO IF STATEMENT")
        min_dim = min(grad_depth_x.shape[-1], grad_img_x.shape[-1])
        grad_depth_x = grad_depth_x[..., :min_dim]
        grad_img_x = grad_img_x[..., :min_dim]

    if grad_depth_y.shape[-2] != grad_img_y.shape[-2]:
        min_dim = min(grad_depth_y.shape[-2], grad_img_y.shape[-2])
        grad_depth_y = grad_depth_y[..., :min_dim, :]
        grad_img_y = grad_img_y[..., :min_dim, :]

    grad_depth_x *= torch.exp(-grad_img_x.squeeze(1))
    grad_depth_y *= torch.exp(-grad_img_y.squeeze(1))

    if torch.isnan(grad_depth_x).any() or torch.isnan(grad_depth_y).any():
        print("grad_depth_x contains NaNs")
        print("grad_depth_y contains NaNs")

    return grad_depth_x.mean() + grad_depth_y.mean()
