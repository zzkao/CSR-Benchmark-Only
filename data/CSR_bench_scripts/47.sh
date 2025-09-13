#!/bin/bash

# Environment Setup / Requirement / Installation
# For LRV-V1 (MiniGPT4-based)
git clone https://github.com/FuxiaoLiu/LRV-Instruction.git
cd LRV-Instruction
conda env create -f environment.yml --name LRV
conda activate LRV

# For LRV-V2 (mplug-owl-based)
# Install mplug-owl dependencies according to their repo

# Data / Checkpoint / Weight Download (URL)
# Create necessary directories
mkdir -p MiniGPt-4/cc_sbu_align/image

# Download required files (paths need to be set according to download.txt)
# - Vicuna weights
# - LRV-V1 checkpoint
# - LRV-V2 checkpoint and lora weights
# - Visual Genome images
# - Dataset annotations

# Training
# No explicit training commands provided in README

# Inference / Demonstration
# For LRV-V1 (MiniGPT4-based)
cd ./MiniGPT-4
# Run local demo
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
# Run inference
python inference.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0

# For LRV-V2 (mplug-owl-based)
# Run local demo
python -m serve.web_server --base-model /path/to/mplug-owl-checkpoint --bf16
# Run inference
python -m serve.inference --base-model /path/to/mplug-owl-checkpoint --bf16

# Testing / Evaluation
# Run GAVIE evaluation
# First download VG annotations and generate evaluation prompts
python Evaluation/evaluate.py