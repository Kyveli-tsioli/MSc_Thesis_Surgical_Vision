import torch
import os
import numpy as np

def load_checkpoint(checkpoint_path):
    """
    Load the checkpoint from the specified path.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint

def normalize_tensor(tensor):
    """
    Normalize the tensor by its maximum element.
    """
    return tensor / tensor.max()

def calculate_mean_depth_difference(predicted_depth, gt_depth):
    """
    Calculate the mean absolute difference between predicted and ground truth depth maps.
    """
    if predicted_depth is None or gt_depth is None:
        print("Depth data is missing.")
        return None

    # Ensure both tensors are on the same device and have the same data type
    predicted_depth = predicted_depth.to(gt_depth.device)
    
    # Normalize both depth tensors
    normalized_predicted_depth = normalize_tensor(predicted_depth)
    normalized_gt_depth = normalize_tensor(gt_depth)
    
    # Calculate element-wise absolute difference and its mean
    difference = torch.abs(normalized_predicted_depth - normalized_gt_depth)
    mean_difference = torch.mean(difference).item()  # Converts to Python float

    return mean_difference

def main(checkpoint_path, predicted_depth_map_path, gt_depth_map_path):
    """
    Main function to load the model, perform depth map comparisons, and compute the mean discrepancy.
    """
    # Load the model checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint['model_state_dict']  # You might need to adjust how you load or use the model depending on your architecture

    # Load depth maps
    predicted_depth = torch.load(predicted_depth_map_path)
    gt_depth = torch.load(gt_depth_map_path)

    # Perform the comparison
    mean_difference = calculate_mean_depth_difference(predicted_depth, gt_depth)
    
    print(f"Mean depth discrepancy: {mean_difference}")

if __name__ == "__main__":
    checkpoint_path = "/vol/bitbucket/kt1923/4DGaussians/final_model_checkpoint_coarse.pth"  # Adjust this path
    predicted_depth_map_path = "path_to_predicted_depth_map.pth"  # Adjust this path
    gt_depth_map_path = "path_to_gt_depth_map.pth"  # Adjust this path

    main(checkpoint_path, predicted_depth_map_path, gt_depth_map_path)
