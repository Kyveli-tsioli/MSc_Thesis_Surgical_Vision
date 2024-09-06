#!/bin/bash

#SBATCH --job-name=my_job_name  # Adjust the job name as necessary
#SBATCH --output=output.log     # Output log file
#SBATCH --error=error.log       # Error log file
#SBATCH --time=72:00:00         # Adjust the time limit as necessary

#SBATCH --gres=gpu:teslaa40:1   # Request a 48GB Nvidia A40 GPU
#SBATCH --mail-type=ALL         # Send email notifications
#SBATCH --mail-user=kt1923@ic.ac.uk  # Replace <your_username> with your email address

# Load the CUDA module
source /vol/cuda/12.0.0/setup.sh

# Set up Conda environment
source /vol/bitbucket/kt1923/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

# Set environment variables from .bashrc
export CUDA_HOME=/vol/cuda/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/vol/bitbucket/kt1923/4DGaussians:$PYTHONPATH

# Display GPU and system information
/usr/bin/nvidia-smi
uptime

# Run your training script
python train_torchl1_0907.py -s data/multipleview/office_0/colmap --port 6054 --expname "multipleview/office_0_1007_factor0.5_prob0.5_60000_job2" --configs arguments/multipleview/brain.py

