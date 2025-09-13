#!/bin/bash

# Environment Setup / Requirement / Installation
# Install Python dependencies
pip install -r requirements.txt

# Compile tf_cpn library
cd mvpose/backend/tf_cpn/lib/
make
cd ./lib_kernel/lib_nms
bash compile.sh

# Compile light_head_rcnn library
cd mvpose/backend/light_head_rcnn/lib/
bash make.sh

# Compile pictorial function
cd mvpose/src/m_lib/
python setup.py build_ext --inplace

# Data / Checkpoint / Weight Download (URL)
# Create directories for models and datasets
mkdir -p backend/light_head_rcnn/output/model_dump
mkdir -p backend/tf_cpn/log/model_dump
mkdir -p backend/CamStyle/logs
mkdir -p datasets

# Download and extract datasets
wget http://campar.cs.tum.edu/files/belagian/multihuman/CampusSeq1.tar.bz2 -P datasets/
wget http://campar.cs.tum.edu/files/belagian/multihuman/Shelf.tar.bz2 -P datasets/
tar -xjf datasets/CampusSeq1.tar.bz2 -C datasets/
tar -xjf datasets/Shelf.tar.bz2 -C datasets/

# Generate camera parameters
python ./src/tools/mat2pickle.py /parameter/dir ./datasets/CampusSeq1

# Training
# No explicit training commands in README

# Inference / Demonstration
# Run demo on Campus dataset
python ./src/m_utils/demo.py -d Campus

# Run demo on Shelf dataset
python ./src/m_utils/demo.py -d Shelf

# Testing / Evaluation
# Preprocess datasets for faster evaluation
python src/tools/preprocess.py -d Campus -dump_dir ./datasets/Campus_processed
python src/tools/preprocess.py -d Shelf -dump_dir ./datasets/Shelf_processed

# Evaluate on Campus dataset
python ./src/m_utils/evaluate.py -d Campus
# Evaluate with preprocessed data
python ./src/m_utils/evaluate.py -d Campus -dumped ./datasets/Campus_processed

# Evaluate on Shelf dataset
python ./src/m_utils/evaluate.py -d Shelf
# Evaluate with preprocessed data
python ./src/m_utils/evaluate.py -d Shelf -dumped ./datasets/Shelf_processed