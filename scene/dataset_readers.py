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


###Construct the CameraInfo object which stores info about each camera (rotation matrix of the camera (i.e. orientation), translation vector (i.e. position of the camera),
#field of view in x and y directions, image data associated with the camera, name of the image file, width of the image, height of the image, time based on the order of the camera, mask

###Construct SceneInfo object that includes training and testing camera info and normalisation params
#this means: point cloud data from the scene,list of training 'CameraInfo' objects, list of testing 'CameraInfo' objects
#list of cameras used for video generation (same as training cameras in this context)
#normalisation arams for the camera coordinates and path to the PLY file containing the point cloud data


import os
import sys
from PIL import Image
from scene.cameras import Camera
import cv2 #added 2807

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm

class CameraInfo(NamedTuple): #holds info about the camera (rotation,translation, field of view, image data)
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    #COMMENTED OUT 0207
    depth: np.array  #= None #added 2806
    pc= np.array #added 2806

   
class SceneInfo(NamedTuple): #holds info about the entire scene including point clouds, train/test/video cameras, normalisation params and path to the ply file
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        #centering the camera positions so that center of all cameras is at the origin of the coordinate system
        #and scaling to ensure that the entire scene is visible and that it fits the camera's field of view
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depth_folder): #images_folder: the folder path where the images corresponding to the cameras are stored
    #reads camera extrinsic and intrinsics from COLMAP files and constructs 'CameraInfo' objects which contain info about each camera (parameters and the corresponding image)
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        #compute the field of view based on the camera model
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        #construct the file path to the image associated with the current camera
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        print(f"Attempting to open image at: {image_path}")  # Debugging output added that


        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}") #error debugging



        #added 2806
        # Load depth map
        ##depth_path= os.path.join(depth_folder, f"depth_{idx}.npy")
        ##depth_path_png = os.path.join(depth_folder, f"depth_{idx}.png") #added 2806
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
         # Load depth map
        #depth_path_npy = os.path.join(depth_folder, f"{idx}.npy") commented out 3007
        #depth_path_png = os.path.join(depth_folder, f"{idx}.png") #added 2806   commented out 3007
        #print("DEPTH PATH PNG".capitalize, depth_path_png) commented out 3007
        #####adjusted 2807###
         # Initialize depth to zeros in case no valid depth file is found
        #depth = None  # Start with None to explicitly catch unassigned cases
   
        #depth_folder = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/gt/depth' #for office_0
        #depth_folder = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/ns_images2/colmap/gt/normalized_depth'
        #depth_folder = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/gt/normalized_depth'
        depth_folder = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/ns_images2/colmap/gt/depth'
        print("Listing files in directory:", depth_folder)
        #print(os.listdir(depth_folder))

        #depth_path_png = os.path.join(depth_folder, f"depth_{idx}.png") #for office_0
        depth_path_png = os.path.join(depth_folder, f"{idx}.png") #for brain phantom
        print("Depth path PNG:", depth_path_png)
        if os.path.exists(depth_path_png):
            #depth_raw = cv2.imread(depth_path_png, cv2.IMREAD_UNCHANGED)  # Read the image unchanged
            depth_raw = cv2.imread(depth_path_png, cv2.IMREAD_ANYDEPTH) 
            print("mpike sto depth_raw")
            if depth_raw is not None:
                depth = depth_raw.astype(np.float32)  # Convert type after confirming it's loaded
                print(f"Loaded depth image with shape {depth.shape} and dtype {depth.dtype}")
                print(f"Resolution (total pixels): {depth.shape[0] * depth.shape[1]}")

                print(f"Depth Statistics - Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}")
            else:
               print(f"Failed to load depth image from {depth_path_png}")
            
        else:
            print(f"Depth file does not exist: {depth_path_png}")

            #if depth_raw is None:
                #depth = np.zeros((height, width), dtype=np.float32)  # Fallback to zeros if none loaded
                #print("Defaulting to zero-filled depth array due to loading issues.")

        ### end of adjustment 2807###

        ######BRING IT BACK TO THE ORIGINAL BELOW### 2807
        # SOS EDW EINAI TO THEMA, TA KANEI 0##
        #if os.path.exists(depth_path_npy):
            #depth = np.load(depth_path_npy)
       # elif os.path.exists(depth_path_png):
            #depth = np.array(Image.open(depth_path_png))
        #else:
            #print("mpike na ta kanei 0s")
            #depth = np.zeros((height, width))  

    
        # Load depth map
        ##depth_path = os.path.join(images_folder, f"depth_{idx}.png") #added 2806
        ##depth = np.load(depth_path) #added 2806
        #each camera is associated with its corresponding image and intrinsic parameters
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path): #reads point cloud data from a PLY file
    try:
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    

        ###added 1607 for debugging
         # Check for normals
        if 'nx' in vertices:
            
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            print("nx in vertices") #enters successfully here
        else:
           
            normals = np.zeros_like(positions)  # handle missing normals
        ### end of addition
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T #this shows that there is an expectation for the PLY
        #file to contain normal data ('nx', 'ny', 'nz') for the normal vectors at each point and this was not present in the original initial .PLY file.
        #this is why we added the if 'nx' in vertices statement in the fetchPLY and this is why we passed the initial PLY
        #throgh the compute_normals.py

        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    except Exception as e:
        print(f"Failed to load PLY file: {e}")

def storePly(path, xyz, rgb): #writes point cloud data to a PLY file
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

#ADDED 2807
def read_depth_maps(depth_folder):
    depth_maps = []
    for filename in os.listdir(depth_folder):
        if filename.endswith('.png'):  # Assuming the depth maps are stored as PNG files
            filepath = os.path.join(depth_folder, filename)
            depth_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # Read the depth image with full bit depth
            if depth_image is not None:
                depth_maps.append(depth_image)
            else:
                print(f"Failed to load depth image from {filepath}")
    return depth_maps
###
def readColmapSceneInfo(path, images, eval, llffhold=8):
    #reads COLMAP scene info, including camera parameters and point cloud data
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    #added now
    reading_dir = os.path.join(path, "images") if images is None else images
    print("READING DIR IS", reading_dir)
    #reading_dir_gt_depth = os.path.join(path, "gt/normalized_depth") #was this up until 2707
    #reading_dir_gt_depth = os.path.join(path, "gt/visualized_depth")
    reading_dir_gt_depth = os.path.join(path, "gt/depth") #ADJUSTMENT 2707
    print("READING DIR GT DEPTH", reading_dir_gt_depth)
    #reading_dir = "images" if images == None else images
    depth_maps = read_depth_maps(reading_dir_gt_depth) #ADDED 2807
    depth_maps = read_depth_maps(reading_dir_gt_depth) #added 2807
    for depth_map in depth_maps: #added 2807
        print(f"Depth map stats: Min={np.min(depth_map)}, Max={np.max(depth_map)}") #added 2807
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), depth_folder= reading_dir_gt_depth)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) #ordering helps avoid mismatches between images and their corresponding camera parameters
    # breakpoint()

    #depth_maps = read_depth_maps(reading_dir_gt_depth ) #ADDED 2807
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) #transform the camera coordinates to a normalized step to ensure the input to the model is in a consistent range for stabilising and improving convegence (check NeRF paper)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    #print("EDW KOITAS 1607 KYVE", ply_path)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    #print("BIN PATH KYVE", bin_path)
    txt_path = os.path.join(path, "sparse/0/points3D.txt")   
    #below is the original#   
    #if not os.path.exists(ply_path):

        #print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        #try:
            #xyz, rgb, _ = read_points3D_binary(bin_path)
        #except:
            #xyz, rgb, _ = read_points3D_text(txt_path)
        #storePly(ply_path, xyz, rgb)
    
    #try:
        #pcd = fetchPly(ply_path)  
        
    #except:
        #pcd = None
    
    #HERE IS MY TRIAL 1607#
    print("Checking PLY path:", ply_path, os.path.exists(ply_path))
    print("Checking BIN path:", bin_path, os.path.exists(bin_path))
    print("Checking TXT path:", txt_path, os.path.exists(txt_path))

    if not os.path.exists(ply_path):
        print("No .ply file found. Checking for .bin and .txt.")
        if os.path.exists(bin_path):
            print("Found .bin file. Converting to .ply.")
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        elif os.path.exists(txt_path):
            print("Found .txt file. Converting to .ply.")
            xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        else:
            print("No suitable point cloud file found. Please check your data.")
    
    try:
        pcd = fetchPly(ply_path)
        if pcd.is_empty():
            raise ValueError("Loaded PLY file is empty.")
    except Exception as e:
        print(f"Failed to load PLY file: {e}")
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius): #generates a camera to world transformation matrix for a camera positioned on a spherical coordinate system with give theta, phi angles and radius
        #this places the camera at a specific point in the world space and orients it towards the scene
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    #render_poses: a sequence of camera poses is generated by varying the theta angle. this simulates a camera orbiting around the scene
    #this is pose generation: render_poses creates a set of novel camera poses (i.e. viewpoints) on a spherical path around the scene
    #these render poses are computed in spherical coordinates using the rotation (angles) and translation (radius) to define the camera's position and orientation in the world coordinate system
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0) 
    #associates each pose with a specific time value for temporal simulation
    render_times = torch.linspace(0,maxtime,render_poses.shape[0]) #generates a sequence of time values linearly spaced between 0 and maxtime (temporal aspects of the scene)
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file) #read JSON file to extract camera intrinsic (i.e. field of view)
        try:
            fovx = template_json["camera_angle_x"] 
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break

    # format information
    #now to render the scene from these viewpoints we compute the inverse of each poses's transformation matrix
    #this inversion transforms the poses from the world coordinate system to the camera coordinate system
    #crucial step for understanding how the scene should be viewed from each camera's perspective
    #transformation computation: computes the transformation matrices to convert form world coordinates to camera coordinates
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        #tranformations: computes the inverse of the pose matrix to get the transformation from the world coordinate to the camera coordinate system
        matrix = np.linalg.inv(np.array(poses)) 
        R = -np.transpose(matrix[:3,:3])
        #extract rotation and translation from the transformation matrix
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]

        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    #reads camera data from a transforms file
    #the focus here is in converting existing camera and image data applying normalisation and preparing the data for rendering or training
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            #calculate the field of view based on image dimensions and focal lengths
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos
def read_timeline(path): #reads and maps timestamps from transform JSON files
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval): #loads and processes camera and point cloud data from the specified dataset directory
    #load the training and test camera info from the specified directory 'datadir'
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points) #convert point cloud data to a numpy array

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max): #adds random points to the point cloud data for data augmentation
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float() #world to camera coordinate system transformation matrix converted to a pytorch tensor and moved to the GPU
    cam_center = torch.inverse(w2c)[:3, 3] # the camera's position in the world coordinate system (i.e. location of the camera in the world space)
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    #in computer graphics it is common to transform objects from the world coordinate system to the camera's coordinate system for rendering
    #this transformation allows the renderer to understand how objects in the scene should appear from the camera's perspective
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2) # is this the projection matrix from the 3D camera coordinate sysetm to the camera 2d image plane?
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz): #plots camera orientations for visualisation 
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        # breakpoint()
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            
            # breakpoint()
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            # breakpoint()
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            # breakpoint()
            # print(w2c,focal,image_name)
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # breakpoint()
    return cam_infos, max_time, scene_radius 

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readMultipleViewinfos(datadir,llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="test")

    train_cam_infos_ = format_infos(train_cam_infos,"train")
    nerf_normalization = getNerfppNorm(train_cam_infos_)

    ##ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    ply_path = os.path.join(datadir, "points3D.ply")
    #bin_path = os.path.join(datadir, "points3D_multipleview.bin") #original
    bin_path= os.path.join(datadir,"points3D.bin") #my attempt
    print("EDW TO BIN_PATH", bin_path)
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos.video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
 
#my addition starts here
def readCustomDatasetInfo(path, images, eval, llffhold=8):
    print("PATHHHH", path)
    try:
        #read camera extrinsics (position, orientation) and intrinsics (focal length, principal point etc.) 
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinic_file)
    #reading_dir = "colmap/images" if images is None else images old
    #reading_dir = os.path.join("colmap","images") if images is None else images
    reading_dir = os.path.join("colmap") if images is None else images
    image_folder_path = os.path.join(path, reading_dir) #added 16:13
    #depth_folder_path = os.path.join(path, "colmap/gt/normalized_depth")  # added 2806
    depth_folder_path = os.path.join(path, "colmap/gt/depth") #2707 ADJUSTMENT alla den kalw auti tin sinartisi outws i allws

    print("reading_dir", reading_dir)
    # Debugging: Print the image folder path
    print(f"Image folder path: {image_folder_path}") #added 16:13
    cam_infos_unsorted = readColmapCameras(cam_extrinsics= cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder =image_folder_path) #constructs CameraInfo objects (each image is associated with the corresponding camera and its parameters- intrinsics and extrinsics)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name) #sorts CameraInfo objects by image name to ensure that the images and their corresponding camera parameters are correctly aligned
    #split cameras for training and testing 
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    else: #all cameras are used for training
        train_cam_infos = cam_infos
        test_cam_infos =[]

    nerf_normalization = getNerfppNorm(train_cam_infos) #normalise the camera coordinates to ensure consistent input to the model (improves training convergence)
    ply_path = os.path.join(path, "colmap/sparse/0/points3D.ply")
    bin_path = os.path.join(path, "colmap/sparse/0/points3D.bin")
    #txt_path = os.path.join(path, "colmap/sparse/0/points3D.txt")
    if not os.path.exists(ply_path): #checks if the point cloud data ile exists 
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        #try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
        #except:
            #xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    #construct SceneInfo object containing the point cloud, camera info and normalisation parameters
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path) #SceneInfo object contains the point cloud, camera infom normalisation parameters
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos,
    "CustomDataset": readCustomDatasetInfo
}
