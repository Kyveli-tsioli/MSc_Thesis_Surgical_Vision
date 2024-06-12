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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder): #utility function
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    #splits each filename 'fname' into parts based on the underscore '_' character
    #[-1]: takes the last part of the split filename, which is expected to be the iteration number
    #saved_iters is a list of integers representing the iteration numbers from the filenames
    return max(saved_iters)
