from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov

#'FourDGSdataset' c;ass is designed to handle the dataset of camera info making it compatible with pytorch's dataloader.
#The 'Scene' object utilises this class to load camera data for training, testing and video rendering.


class FourDGSdataset(Dataset): #FourDGSdataset class inherits from torch.utils.data.Dataset making it compatible with Pytorch's DataLoader
    #this class wraps the dataset of camera info and provides methods to access individual data points '__getitem__' and the length of the dataset '__len__'
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset # a collection of camera info
        self.args = args #command-line arguments 
        self.dataset_type=dataset_type #colmap/blender/dynerf/nerfies/panopticSpors
    
    def __getitem__(self, index): #this method retrieves and processes the camera properties to create a 'Camera' object, which is used for rendering the scene from the specific viewpoint defined by the camera
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index] #extracts 'image', 'world to camera tansformation' and 'time'
                R,T = w2c #computes rotation of the camera in the world, translation of the camera in the world
                # the world to camera transformation helps in understanding how the scene should be transformed
                #from the world coordinates to the camera's coordinate system (essential for ray tracing where rays are cast from the camera into the scene)
                FovX = focal2fov(self.dataset.focal[0], image.shape[2]) #field of view in X direction
                FovY = focal2fov(self.dataset.focal[0], image.shape[1]) #field of view in Y direction
                #the field of view determines how the 3d points are projected onto the 2d image plane. 
                #it affects the scaling and perspective of the rendered image.
                mask=None
                #the world to camera transormation is crucial for the 3D rendering. it defined how points in the world
                #coordinate system are mapped to the camera coordinate system. translation: the position of the camera in the WORLD system
                #rotation: the orientation of the camera in the WORLD system
                #together these transformations allow the system to understand where the camera is located and how it is oriented
                #relative to the scene. this is essential for rendering the scene from the correct viewpoint.
                #field of view defines how wide the camera's view is and affects the perspective and scaling of the scene
                #it defines how much of the scene is visible through the camera and it determines how the 3d points are projected onto the 2d image plane
            except: #if the first method fails, it falls back to extracting these properies from 'caminfo'
                caminfo = self.dataset[index] #all the info for rendering the scene from a specific viewpoint
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask #caminfo is a different way of storing the same info
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask) #the camera object contains all necessary info to render the scene from this specfic viepwoint
        #because to render the scene accurately you need to know the camera's position and orientation (translation and rotation)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset) #length of Camera object, (image, rotation(= orientation), translation (=position in the scene), field of view, time the image was captured)
