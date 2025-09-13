#!/bin/bash

# Environment Setup / Requirement / Installation
# Install dependencies
conda install -c pytorch pytorch torchvision
conda install cython scipy
conda install scikit-learn numpy pyyaml

# Compile Cython sampler
python graphsaint/setup.py build_ext --inplace

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Download datasets from Google Drive or BaiduYun (manual download required)
# Google Drive: https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz
# BaiduYun: https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg (code: f1ao)

# Convert dataset format (optional)
python convert.py ppi  # Example for PPI dataset

# Training
# Train on CPU
python -m graphsaint.tensorflow_version.train \
    --data_prefix ./data/ppi \
    --train_config train_config/table2/ppi.yml \
    --gpu -1

# PyTorch version
python -m graphsaint.pytorch_version.train \
    --data_prefix ./data/ppi \
    --train_config train_config/table2/ppi.yml \
    --gpu 0

# Inference / Demonstration
# No explicit inference commands in README
# Inference is part of training/evaluation

# Testing / Evaluation
# Evaluation is included in training commands with validation/test set metrics