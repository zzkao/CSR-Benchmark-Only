#!/bin/bash
# This is an example file. Name it with the id(#line).sh
# Environment Setup
conda create -n myenv python=3.9 -y
conda activate myenv
pip install -r requirements.txt

# Data Download
wget https://example.com/dataset.zip -O dataset.zip
unzip dataset.zip -d data/

# Training
python train.py \
  --data_dir data/ \
  --epochs 1 \
  --batch_size 32

# Evaluation
python eval.py \
  --model checkpoints/model.pt \
  --data_dir data/test/

# Inference / Demo
python demo.py --input "Hello world"
