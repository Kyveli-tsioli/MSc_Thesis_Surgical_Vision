import cv2
import numpy as np
import os

# Path to the directory of the ground truth images
depth_images_dir = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/ns_images2/colmap/gt/depth'
# Path to the directory where normalized depth images will be saved
output_dir = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/ns_images2/colmap/gt/normalized_depth'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def apply_clahe(image):
    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    return cl1

def process_depth_image(image_path):
    # Load depth image
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image was loaded correctly
    if depth_image is None:
        print(f"Error: Could not load image {image_path}.")
        return
    
    # Check the range of depth values
    min_depth = np.min(depth_image)
    max_depth = np.max(depth_image)
    print(f"Depth range for {os.path.basename(image_path)}: {min_depth} to {max_depth}")
    
    # Normalize depth values to range [0, 255]
    normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth = normalized_depth.astype(np.uint8)  # Convert the normalized values to 8-bit
    
    # Apply CLAHE
    clahe_depth = apply_clahe(normalized_depth)
    
    # Save the CLAHE enhanced depth image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, clahe_depth)
    print(f"Saved CLAHE enhanced depth image to {output_path}")

# Iterate over all images in the directory
for filename in os.listdir(depth_images_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(depth_images_dir, filename)
        process_depth_image(image_path)

print("All depth images processed and saved.")
