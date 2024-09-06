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
#higher PSNR values indicate better reconstruction quality
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        #the mask is used to selectively compute the PSNR over specific regions of the images
        #rather than the entire image.


        mask = mask.flatten(1).repeat(3,1) #flatten the mask to match the flatten image and repeats the mask across the colour channels
        mask = torch.where(mask!=0,True,False) #convert the mas to boolean tensor. 'True' indicates the pixels to be considered in the PSNR calculation
        img1 = img1[mask]
        img2 = img2[mask]
        

        #computes the squared difference between corresponding pixels of img1 and img2
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        #reshapes the result to a 2D tensor where each row corresponds to an image in the batch

    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr
