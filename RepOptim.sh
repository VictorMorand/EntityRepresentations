#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=RepOptim                 # Job name
#SBATCH --output=../logs/output_%j.err      # Output file (%j will be replaced with job ID)
#SBATCH --error=../logs/output_%j.err       # Error file (%j will be replaced with job ID)
#SBATCH --ntasks=1                          # Number of tasks (usually 1 for a Python script)
#SBATCH --account=cdt@v100
#SBATCH --constraint=v100-32g
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --gpus-per-node=1                   # Request 1 GPU (can increase if needed)
#SBATCH --time=06:00:00                     # Time limit (hh:mm:ss), in Jean ZAY < 20h

# Activate Python environment (optional, if using virtualenv or conda)
source $WORK/envs/env/bin/activate

# Run your Python script
python RepresentationOptimization.py