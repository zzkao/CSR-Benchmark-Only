#!/bin/bash

# Environment Setup / Requirement / Installation
# Install required packages
pip install mtcnn
conda install -c conda-forge moviepy
conda install -c cogsci pygame
conda install -c conda-forge requests
conda install -c conda-forge pytables

# Create data directory
mkdir -p data

# Data / Checkpoint / Weight Download (URL)
# Download and preprocess datasets
cd data
# Download IMDB-WIKI dataset (manual download required)
# Download from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
# Download Morph2 dataset (requires application)
# https://www.faceaginggroup.com/?page_id=1414

# Process datasets
python TYY_IMDBWIKI_create_db.py --db imdb --output imdb.npz
python TYY_IMDBWIKI_create_db.py --db wiki --output wiki.npz
python TYY_MORPH_create_db.py --output morph_db_align.npz
cd ..

# Training
# Train MobileNet and DenseNet models
cd training_and_testing
sh run_all.sh

# Train SSR-Net model
sh run_ssrnet.sh

# Train gender model
sh run_ssrnet_gender.sh

# Plot training results (for IMDB dataset example)
cp plot.sh ssrnet_plot.sh plot_reg.py ./imdb_models/
cd imdb_models
sh plot.sh
sh ssrnet_plot.sh
cd ..

# Inference / Demonstration
# Video demo with MTCNN (CPU only)
cd ../demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py TGOP.mp4

# Video demo with frame skip option (skip 3 frames)
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py TGOP.mp4 '3'

# Real-time webcam demo with LBP face detector
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_ssrnet_lbp_webcam.py

# Testing / Evaluation
# Testing is included in training scripts
cd ../training_and_testing