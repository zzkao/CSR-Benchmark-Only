#!/bin/bash

# Environment Setup / Requirement / Installation
# Option 1: Using conda environment file
conda env create -f env.yml
conda activate dpm-pc-gen

# Option 2: Manual installation for CUDA 11+ (install required packages)
conda install pytorch>=1.7.0 cudatoolkit=11.0 -c pytorch
conda install h5py tqdm tensorboard=2.5.0 numpy=1.20.2 scipy=1.6.2 scikit-learn=0.24.2

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Preprocess data (convert to .xyz or .npy format)
python data_preprocess.py

# Training
# Train auto-encoder model
python train_ae.py \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --categories airplane \
    --test_size 400

# Train generator model
python train_gen.py \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --categories airplane \
    --test_size 400

# Inference / Demonstration
# No explicit inference commands - inference is part of testing

# Testing / Evaluation
# Test auto-encoder model
python test_ae.py \
    --ckpt ./pretrained/AE_all.pt \
    --categories all

# Test generator model
python test_gen.py \
    --ckpt ./pretrained/GEN_airplane.pt \
    --categories airplane