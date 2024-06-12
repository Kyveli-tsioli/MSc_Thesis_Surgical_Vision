
import os
import cv2
import random
import numpy as np
from PIL import Image
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
import random
def get_stamp_list(dataset, timestamp):
    frame_length = int(len(dataset)/len(dataset.dataset.poses))
    # print(frame_length)
    if timestamp > frame_length:
        raise IndexError("input timestamp bigger than total timestamp.")
    print("select index:",[i*frame_length+timestamp for i in range(len(dataset.dataset.poses))])
    return [dataset[i*frame_length+timestamp] for i in range(len(dataset.dataset.poses))]
class FineSampler(Sampler):
    #FineSampler class for efficiently sampling viewpoints (camera positions) during the training process
    def __init__(self, dataset):
        self.len_dataset = len(dataset) # number of frames or viewpoints in the dataset (how many captures=rames there are per unique camera pose)
        self.len_pose = len(dataset.dataset.poses) #number of DISTINCT poses (camera positions) captured in the dataset
        self.frame_length = int(self.len_dataset/ self.len_pose) #average number of frames (= individual captures of the scene at different time instances or slight variations) per pose (=unique camera viewpoints or angles from which the scene is observed)

        sample_list = [] #stores the sampled indices
        for i in range(self.frame_length): #iterates over the number of frames per pose
            for j in range(4):
                idx = torch.randperm(self.len_pose) *self.frame_length + i #permutation of the viewpoints indices to ensure that each pose is sampled in a random order, such that each training epoch sees a different mix of frames (generalisation)
                # print(idx)
                # breakpoint()
                now_list = []
                cnt = 0
                for item in idx.tolist():
                    now_list.append(item)
                    cnt+=1
                    if cnt % 2 == 0 and len(sample_list)>2: #reinserting prevous samples to preerve some temporal coherence?  
                        select_element = [x for x in random.sample(sample_list,2)]
                        now_list += select_element
            
            sample_list += now_list
            
        self.sample_list = sample_list
        # print(self.sample_list)
        # breakpoint()
        print("one epoch containing:",len(self.sample_list))
    def __iter__(self): #iterator for sample_list 

        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)
