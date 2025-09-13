#!/bin/bash

# Environment Setup / Requirement / Installation
python3 -m pip install ".[train]"

# Data / Checkpoint / Weight Download (URL)
# Download Navier-Stokes dataset
bash scripts/download_navier_stokes.sh

# Download spring mesh dataset
bash scripts/download_spring_mesh.sh

# Download SST data (manual download required)
# wget https://zenodo.org/record/7259555/files/oisstv2.zip
# unzip oisstv2.zip -d $HOME/data/oisstv2

# Training
# First stage: Train interpolator network
python run.py experiment=spring_mesh_interpolation

# Second stage: Train forecaster network (replace <WANDB_RUN_ID> with actual ID from first stage)
python run.py experiment=spring_mesh_dyffusion diffusion.interpolator_run_id=i73blbh0

# Train Dropout baseline on spring mesh dataset
python run.py experiment=spring_mesh_time_conditioned

# Debug training with fewer trajectories
python run.py experiment=spring_mesh_interpolation datamodule.num_trajectories=1

# Debug SST training with fewer boxes
python run.py experiment=spring_mesh_interpolation 'datamodule.boxes=[88]'

# Enable mixed precision training
python run.py experiment=spring_mesh_interpolation trainer.precision=16

# Inference / Demonstration
# Test trained model using wandb run ID
python run.py mode=test logger.wandb.id=i73blbh0

# Testing / Evaluation
# Test model from local checkpoint
python run.py mode=test logger.wandb.id=i73blbh0 ckpt_path=path/to/local/checkpoint.ckpt
