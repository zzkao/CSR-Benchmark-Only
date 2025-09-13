#!/bin/bash

# Environment Setup / Requirement / Installation
# Install PyTorch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
# Install requirements
pip install -r requirements.txt
# Install PointNet++ CUDA modules
cd lib/pointnet2 && python setup.py install && cd ../..

# Data / Checkpoint / Weight Download (URL)
# Download ScanRefer dataset (after getting access)
wget <download_link>
# Download GLoVE embeddings
wget http://kaldir.vc.in.tum.de/glove.p -P data/
# Download ENet weights for multiview features
wget http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth -P data/
# Process ScanNet data
cd data/scannet && python batch_load_scannet_data.py && cd ../..
# Extract and project multiview features (optional)
python script/compute_multiview_features.py
python script/project_multiview_features.py --maxpool
# Download benchmark data
wget http://kaldir.vc.in.tum.de/scanrefer_benchmark_data.zip
unzip scanrefer_benchmark_data.zip -d data/

# Training
# Basic training with XYZ coordinates only
python scripts/train.py --no_lang_cls
# Training with RGB values
python scripts/train.py --use_color
# Training with RGB and normals
python scripts/train.py --use_color --use_normal
# Training with multiview features
python scripts/train.py --use_multiview
# Training with multiview and normals
python scripts/train.py --use_multiview --use_normal

# Inference / Demonstration
# Visualize processed scene data
python data/scannet/visualize.py --scene_id scene0000_00
# Visualize feature projections
python script/project_multiview_labels.py --scene_id scene0000_00 --maxpool
# Generate predictions for benchmark
python scripts/predict.py --folder <folder_name> --use_color

# Testing / Evaluation
# Evaluate trained model
python scripts/eval.py --folder <folder_name> --reference --use_color --no_nms --force --repeat 5
# Visualize predictions
python scripts/visualize.py --folder <folder_name> --scene_id <scene_id> --use_color