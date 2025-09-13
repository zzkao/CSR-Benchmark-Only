#!/bin/bash

# Environment Setup / Requirement / Installation
# Install required packages
pip install torch==1.7.0 torchvision==0.8.1
pip install advertorch==0.2.2 pretrainedmodels==0.7.4

# Data / Checkpoint / Weight Download (URL)
# Create data directory and download dataset
mkdir -p SubImageNet224
# Note: Download dataset manually from Google Drive or Baidu Drive and extract to SubImageNet224/

# Create directory for adversarial images
mkdir -p adv_images

# Training
# Generate adversarial examples using ResNet-152 as source model
python attack_sgm.py \
    --gamma 0.2 \
    --output_dir adv_images \
    --arch densenet201 \
    --batch-size 40

# Generate adversarial examples using DenseNet-201 as source model
python attack_sgm.py \
    --gamma 0.5 \
    --output_dir adv_images \
    --arch resnet152 \
    --batch-size 40

# Inference / Demonstration
# No explicit inference commands - adversarial example generation is the main task

# Testing / Evaluation
# Evaluate transferability using VGG19 with batch normalization as target model
python evaluate.py --input_dir adv_images --arch vgg19_bn

# Evaluate transferability using other architectures (examples)
python evaluate.py --input_dir adv_images --arch resnet152
python evaluate.py --input_dir adv_images --arch densenet201
