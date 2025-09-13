#!/bin/bash

# Environment Setup / Requirement / Installation
# Install required packages (versions from README)
pip install torch>=1.8.0 torchvision>=0.9.0 tensorboard>=1.15.0
pip install scikit-learn>=0.23.2 numpy>=1.19.2 pandas>=1.1.3 matplotlib>=3.3.2

# Data / Checkpoint / Weight Download (URL)
# Create dataset directory if not exists
mkdir -p dataset

# Training
# Train basic RRL model on tic-tac-toe dataset
python3 experiment.py \
    -d tic-tac-toe \
    -bs 32 \
    -s 1@16 \
    -e 401 \
    -lrde 200 \
    -lr 0.002 \
    -ki 0 \
    -i 0 \
    -wd 0.0001 \
    --print_rule

# Train RRL with Novel Logical Activation Functions (NLAF)
python3 experiment.py \
    -d tic-tac-toe \
    -bs 32 \
    -s 1@64 \
    -e 401 \
    -lrde 200 \
    -lr 0.002 \
    -ki 0 \
    -i 0 \
    -wd 0.001 \
    --nlaf \
    --alpha 0.9 \
    --beta 3 \
    --gamma 3 \
    --temp 0.01 \
    --print_rule

# Train with recommended settings for better performance
python3 experiment.py \
    -d tic-tac-toe \
    -bs 64 \
    -s 1@1024 \
    -e 401 \
    -lrde 200 \
    -lr 0.002 \
    -ki 0 \
    -i 0 \
    -wd 0.0001 \
    --temp 0.1 \
    --save_best \
    --print_rule

# Inference / Demonstration
# Model will be saved as model.pth and rules in rrl.txt
# No separate inference command needed as evaluation is done during training

# Testing / Evaluation
# Monitor training progress with tensorboard
tensorboard --logdir=log_folder

# View help for all available options
python3 experiment.py --help
