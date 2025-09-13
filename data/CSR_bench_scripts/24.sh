#!/bin/bash

# Environment Setup / Requirement / Installation
# Set platform and install dependencies
platform=cpu  # Options: cpu, cu118, cu121, rocm5.7
make torch-${platform}
pip install -r requirements/core.${platform}.txt -e .[train,test]

# Update dependency pins (optional)
pip install pip-tools
make clean-reqs reqs

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Download LMDB datasets (manual download required)
# 1. https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE
# 2. https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC

# Training
# Train with default configuration
./train.py

# Train with specific experiment/model variant
./train.py +experiment=parseq-tiny

# Train with pretrained weights
./train.py +experiment=parseq-tiny pretrained=parseq-tiny

# Train with specific character set
./train.py charset=94_full  # Options: 36_lowercase, 62_mixed-case

# Train with specific dataset
./train.py dataset=real  # Options: real, synth

# Train with custom parameters
./train.py model.img_size=[32,128] model.max_label_length=25 model.batch_size=384
./train.py data.root_dir=data data.num_workers=2 data.augment=true
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2

# Resume training from checkpoint
./train.py +experiment=parseq-tiny ckpt_path=outputs/parseq/timestamp/checkpoints/checkpoint.ckpt

# Inference / Demonstration
# Benchmark model compute requirements
./bench.py model=parseq model.decode_ar=false model.refine_iters=3

# Benchmark latency vs output length
./bench.py model=parseq model.decode_ar=false model.refine_iters=3 +range=true

# Testing / Evaluation
# Test with default settings (lowercase alphanumeric)
./test.py outputs/model/timestamp/checkpoints/last.ckpt

# Test with pretrained model
./test.py pretrained=parseq

# Test with different character sets
./test.py outputs/model/timestamp/checkpoints/last.ckpt --cased  # mixed-case alphanumeric
./test.py outputs/model/timestamp/checkpoints/last.ckpt --cased --punctuation  # full character set

# Test on challenging datasets
./test.py outputs/model/timestamp/checkpoints/last.ckpt --new

# Test with different rotations
./test.py outputs/model/timestamp/checkpoints/last.ckpt --cased --punctuation --rotation 90
./test.py outputs/model/timestamp/checkpoints/last.ckpt --cased --punctuation --rotation 180
./test.py outputs/model/timestamp/checkpoints/last.ckpt --cased --punctuation --rotation 270