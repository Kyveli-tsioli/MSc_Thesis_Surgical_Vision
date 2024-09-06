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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset #scene is the directory and 'dataset' is the module (i.e. dataset.py) and FourDGSdatase is a class or a function defined within the 'dataset' module
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
import open3d as o3d
import numpy as np

#provides the necessary data for rendering images from different viewpoints 

class Scene:

    gaussians : GaussianModel
    #init method loads the appropriate dataset based on the source path provided.
    #class is initialised with parameters 'args', a 'GaussianModel' instance and optional parameters for loading iterations, shuffling, resolution scales and coarse loading

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        #when the init method is called, the parameters are used to initialise the attributes of the class
        #'gaussians': instance of the 'GaussianModel' class 
        #'load_iteration': specific iteration to load 
        #'shuffle', 'resolution_scales', 'load_coarse': additional
        #  parameters for loading and processing the scene
        """b
        :param path: Path to colmap scene main folder.
        """
        #ATTRIBUTES: variables that belong to an instance of a class. attributes are accessible as long as the object (i.e. instance of the class) exists. they have a broader scope than parameters of the function
       
        self.model_path = args.model_path #where the data is stored

      
        print("self model path", self.model_path, flush=True)
        self.loaded_iter = None
        self.gaussians = gaussians #the gaussian model instance used for rendering and training
        

        if load_iteration: #for resuming training from a particular point or loading a previously trained model
            if load_iteration == -1: #the latest iteration should be loaded
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud")) #constructs the path to the diretory where the point cloud data is stored 
            else: #if load_iteration is provided, it determines which iteration to load
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter)) #the highest iteration number found in the 'point_cloud' directory 
     #####added 1607##
        # Attempt to load the dataset
        #scene_info = self.load_dataset(args)
        #if not scene_info or not hasattr(scene_info, 'point_cloud'):
            #raise ValueError("Failed to load dataset or point cloud is not present.")

        # Validate and process the point cloud
        #self.process_point_cloud(scene_info.point_cloud)
    #########



        #initialise camera dictionaries (one for training, one for test and video cameras)
        #dictionaries to store camera data

  ###### Set up cameras based on evaluation mode- 0608 ADDITION####
        #if args.eval:
            #print("Loading Test Cameras for Evaluation")
            #self.test_cameras = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        #else:
            #print("Loading Training and Video Cameras for Training")
            #self.train_cameras = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
            #self.video_cameras = FourDGSdataset(scene_info.video_cameras, args, dataset_type)
####END OF 0608 ADDITION####


        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        #sceneLoadTypeCallbacks for different dataset types encapsulates the logif
        #for parsing and loading these datasets, allowing for flexibility in the data formats 
        #and robustness in loading different types of scenes 


        #identifies the dataset type based on the presence of specific files in the source_path and calls the appropriate loader function from 'sceneLoadTypeCallbacks' 
        #to load the dataset
        print("edw koitas", args.source_path)
        #load dataset based on the source path
        if os.path.exists(os.path.join(args.source_path, "sparse")): #checks the source path for specific files that indicate the type of the dataset
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
            #if a directory named 'sparse' exists, the code assumes it is a colmap dataset
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            #scenLoadTypeCallbacks is called to load the scene info 
            dataset_type="blender"
            #if a file named 'transforms_train.json' exists in the source path, it is assumed to be a Blneder dataset
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"

        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        #added this condition
        elif os.path.exists(os.path.join(args.source_path,"colmap/sparse/0/points3D.ply")):
            print("source path", args.source_path)
            scene_info = sceneLoadTypeCallbacks["CustomDataset"](args.source_path, args.images, args.eval,args.llffhold)
            dataset_type ="customDataset"
            print("custom dataset entered")
        else:
            assert False, "Could not recognize scene type!"

        #ATTRIBUTES
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        #camera data (i.e. scene_info.train_cameras, scene_info.test_cameras and scene_info,video_cameras) are wrapped in FourdDGSdataset instances to standardise access to camera data using the 'getitem' and 'len' methods
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)



        #finds bounding box for point cloud by computing the min and max x,y,z coordinates of the points in the point cloud (bounding box for the deformation network)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.") 
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min)) #updating scene info: the updated point cloud is assigned back to "scne_info" using the "_replace" method
            #which creates a new instance with updated fields 
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min) #configures the deformation network by setting the axis-aligned bounding box using xyz_max and xyz_min
        if self.loaded_iter: #if a specific iteration is set, then we need to load an existing Gaussian model from a previous iteration
            #loads point cloud data from a PLY file named 'point_cloud.ply'
            #the point cloud data is then used by the Gaussian model for further processing
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   )) #take the gaussian model's state allowing the training to continue from this point
        else: #if self.loaded_iter is not set, a NEW gaussian model is created using the point cloud data from 'scene_info'
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime) #max time value for time-based modelling in the 4D space
            #this happens if no iteration is set, then a new gaussian 

    def save(self, iteration, stage): #saves the scenemodel is created from the point cloud data
        if stage == "coarse": 
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0): 
        return self.train_camera #returns the training camera dataset wrapped in the 'FourDGSdataset'
    def getTestCameras(self, scale=1.0):
        return self.test_camera 
    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    

    #####added 1607
    # def load_dataset(self, args):
    #     paths = {
    #         "sparse": "Colmap",
    #         "transforms_train.json": "Blender",
    #         "poses_bounds.npy": "dynerf",
    #         "dataset.json": "nerfies",
    #         "train_meta.json": "PanopticSports",
    #         "points3D_multipleview.ply": "MultipleView",
    #         "colmap/sparse/0/points3D.ply": "CustomDataset"
    #     }
    #     for path, loader in paths.items():
    #         full_path = os.path.join(args.source_path, path)
    #         print("full path", full_path)
    #         if os.path.exists(full_path):
    #             print(f"Attempting to load dataset using {loader} from {full_path}")
    #             try:
    #                 loaded_data = sceneLoadTypeCallbacks[loader](args.source_path, args.images, args.eval, args.llffhold)
    #                 if hasattr(loaded_data, 'point_cloud') and isinstance(loaded_data.point_cloud, o3d.geometry.PointCloud):
    #                     if not loaded_data.point_cloud.is_empty():
    #                         print(f"Valid point cloud loaded with {len(loaded_data.point_cloud.points)} points.")
    #                         return loaded_data
    #                     else:
    #                         print("Loaded point cloud is empty.")
    #                 else:
    #                     print("Loaded data does not contain a valid point cloud.")
    #             except Exception as e:
    #                 print(f"Failed to load or validate point cloud due to: {e}")
    #         else:
    #             print(f"No dataset found at {full_path}")
    # raise ValueError("Dataset could not be identified or point cloud is invalid.")

   