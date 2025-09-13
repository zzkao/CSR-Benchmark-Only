#!/bin/bash

# Environment Setup / Requirement / Installation
# Create and activate Python environment
conda create -n class_balanced_loss python=3.6
conda activate class_balanced_loss

# Install dependencies
pip install tensorflow==1.14

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Download CIFAR data in tfrecords format
wget https://drive.google.com/file/d/1NY3lWYRfsTWfsjFPxJUlPumy-WFeD7zK/ -O data/cifar_tfrecords.zip
unzip data/cifar_tfrecords.zip -d data/

# For TPU training, download raw data
mkdir -p tpu/raw_data
wget https://drive.google.com/file/d/1ZHhMFJxsgXItJYKiM_VJ0lznj8XClgWF/ -O tpu/raw_data.zip
unzip tpu/raw_data.zip -d tpu/

# Training
# Train on original CIFAR dataset
sh cifar_trainval.sh

# Train on long-tailed CIFAR dataset
sh cifar_im_trainval.sh

# Train with class-balanced loss
sh cifar_im_trainval_cb.sh

# For TPU training (after setting up Google Cloud)
# Convert datasets to tfrecords and upload to GCS
cd tpu/tools/datasets
python dataset_to_gcs.py \
    --project=your_project_name \
    --gcs_output_path=gs://your_bucket/data \
    --local_scratch_dir=/tmp/tfrecords \
    --raw_data_dir=../../raw_data

# Train on different datasets with TPU
cd ../../
sh run_ILSVRC2012.sh  # For ImageNet
sh run_inat2017.sh    # For iNaturalist 2017
sh run_inat2018.sh    # For iNaturalist 2018

# Inference / Demonstration
# No explicit inference commands in README
# Inference is part of evaluation in training scripts

# Testing / Evaluation
# View training progress and results
tensorboard --logdir=./results --port=6006