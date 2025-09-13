#!/bin/bash

# Environment Setup / Requirement / Installation
# Install dependencies
conda install -c pytorch pytorch torchvision
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Data / Checkpoint / Weight Download (URL)
# Create directories for datasets
mkdir -p path/to/imagenet
mkdir -p path/to/coco/{annotations,train2017,val2017}

# Download COCO dataset (manual download required)
# https://cocodataset.org/#download
# Download ImageNet dataset (manual download required)

# Training
# Pre-training on ImageNet (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --lr_drop 40 \
    --epochs 60 \
    --pre_norm \
    --num_patches 10 \
    --batch_size 32 \
    --feature_recon \
    --fre_cnn \
    --imagenet_path path/to/imagenet \
    --output_dir path/to/save_model

# Fine-tuning on COCO (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --use_env detr_main.py \
    --lr_drop 200 \
    --epochs 300 \
    --lr_backbone 5e-5 \
    --pre_norm \
    --coco_path path/to/coco \
    --pretrain path/to/save_model/checkpoint.pth

# Inference / Demonstration
# No explicit inference commands in README
# See visualization notebook: https://colab.research.google.com/github/dddzg/up-detr/blob/master/visualization.ipynb

# Testing / Evaluation
# Evaluate on COCO val5k
python detr_main.py \
    --batch_size 2 \
    --eval \
    --no_aux_loss \
    --pre_norm \
    --coco_path path/to/coco \
    --resume path/to/save_model/checkpoint.pth