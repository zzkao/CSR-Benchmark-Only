#!/bin/bash

# Environment Setup / Requirement / Installation
# Create and setup virtual environment
git clone https://github.com/SimonKohl/probabilistic_unet.git .
cd prob_unet/
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -e .

# Install batch-generators and dependencies
cd ..
git clone https://github.com/MIC-DKFZ/batchgenerators
cd batchgenerators
pip3 install nilearn scikit-image nibabel
pip3 install -e .
cd prob_unet

# Data / Checkpoint / Weight Download (URL)
# Create data directories
mkdir -p data/cityscapes
cd data/cityscapes

# Download Cityscapes dataset (requires login)
# Manual download required from:
# https://www.cityscapes-dataset.com/file-handling/?packageID=1 (gtFine_trainvaltest.zip)
# https://www.cityscapes-dataset.com/file-handling/?packageID=3 (leftImg8bit_trainvaltest.zip)

# Preprocess Cityscapes dataset
cd cityscapes
python3 preprocessing.py
cd ..

# Download pretrained weights
wget https://zenodo.org/record/1419051/files/pretrained_weights.tar.gz
tar -xvzf pretrained_weights.tar.gz -C /model

# Training
# Train probabilistic U-Net
cd training
python3 train_prob_unet.py --config prob_unet_config.py
cd ..

# Inference / Demonstration
# Generate samples from validation set
cd evaluation
python3 eval_cityscapes.py --write_samples

# Testing / Evaluation
# Evaluate generated samples
python3 eval_cityscapes.py --eval_samples

# View results in Jupyter notebook
jupyter notebook evaluation_plots.ipynb

# Run tests
cd ../tests/evaluation
python3 -m pytest eval_tests.py