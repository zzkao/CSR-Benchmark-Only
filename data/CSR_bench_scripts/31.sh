#!/bin/bash

# Environment Setup / Requirement / Installation
# Create and activate conda environment
conda create --name DAD-3DHeads python=3.8
conda activate DAD-3DHeads

# Clone repository and install dependencies
git clone https://github.com/PinataFarms/DAD-3DHeads.git
cd DAD-3DHeads
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Create dataset directory structure
mkdir -p dataset/DAD-3DHeadsDataset/{train,val,test}/{images,annotations}

# Note: Dataset needs to be downloaded manually after filling form at:
# https://docs.google.com/forms/d/e/1FAIpQLSdo8RPxtFR1xHBJ7gkNHbEse0eYOsHR739b9zZ4BtGWQv49LQ/viewform

# Training
# Run training with default configuration
python train.py

# Inference / Demonstration
# Create output directory
mkdir -p outputs

# Run demo with different visualization options
# 2D landmarks visualization
python demo.py images/demo_heads/1.jpeg outputs 68_landmarks
python demo.py images/demo_heads/1.jpeg outputs 191_landmarks
python demo.py images/demo_heads/1.jpeg outputs 445_landmarks

# Mesh visualization
python demo.py images/demo_heads/1.jpeg outputs face_mesh
python demo.py images/demo_heads/1.jpeg outputs head_mesh
python demo.py images/demo_heads/1.jpeg outputs pose

# Export formats
python demo.py images/demo_heads/1.jpeg outputs 3d_mesh
python demo.py images/demo_heads/1.jpeg outputs flame_params

# Testing / Evaluation
# Visualize ground truth labels
python visualize.py train <item_id>  # For training set
python visualize.py val <item_id>    # For validation set
python visualize.py test <item_id>   # For test set