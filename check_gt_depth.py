import cv2
import numpy as np
import os 

import os
import cv2
import numpy as np

# Path to the directory of the ground truth images from office_0 scene
depth_images_dir = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/gt/depth'
# Path to the directory where normalized depth images WILL BE saved
output_dir = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/gt/normalized_depth'

# Create the output directory (if it doesn't exist)
os.makedirs(output_dir, exist_ok=True)

# Function to process and save depth images
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
    normalized_depth = normalized_depth.astype(np.uint8) #convert the normalised values to 8-bit for compatibility with OpenCV library

    # Save the normalized depth image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, normalized_depth)
    print(f"Saved normalized depth image to {output_path}")

# Iterate over all images in the directory 
for filename in os.listdir(depth_images_dir): #os.listdir lists all files in the directory
    if filename.endswith('.png'): #iterate over each file and if it ends with '.png' construct the full image path
        image_path = os.path.join(depth_images_dir, filename)
        process_depth_image(image_path)

print("All depth images processed and saved.")