#!/bin/bash

# Environment Setup / Requirement / Installation
conda create --name tsdiff --yes python=3.8 && conda activate tsdiff
pip install --editable "."

# Data / Checkpoint / Weight Download (URL)
# Note: The repository doesn't explicitly provide data download commands
# Using sample datasets mentioned in the paper (Solar, KDDCup, M4, Uber)

# Training
# Basic TSDiff models
python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml
python bin/train_model.py -c configs/train_tsdiff/train_m4.yaml

# TSDiff-Cond models
python bin/train_cond_model.py -c configs/train_tsdiff-cond/uber_tlc_hourly.yaml
python bin/train_cond_model.py -c configs/train_tsdiff-cond/m4_hourly.yaml

# Training with missing values support
python bin/train_model.py -c configs/train_tsdiff/train_missing_uber_tlc.yaml
python bin/train_model.py -c configs/train_tsdiff/train_missing_kdd_cup.yaml
python bin/train_cond_model.py -c configs/train_tsdiff-cond/missing_RM_uber_tlc_hourly.yaml
python bin/train_cond_model.py -c configs/train_tsdiff-cond/missing_BM-B_kdd_cup_2018_without_missing.yaml
python bin/train_cond_model.py -c configs/train_tsdiff-cond/missing_BM-E_kdd_cup_2018_without_missing.yaml

# Inference / Demonstration
# Observation Self-Guidance examples
# For checkpoint paths, I used a default `./checkpoints/model.ckpt` since actual paths weren't provided
python bin/guidance_experiment.py -c configs/guidance/guidance_solar.yaml --ckpt ./checkpoints/model.ckpt
python bin/guidance_experiment.py -c configs/guidance/guidance_kdd_cup.yaml --ckpt ./checkpoints/model.ckpt

# Testing / Evaluation
# Refinement evaluation
python bin/refinement_experiment.py -c configs/refinement/solar_nips-linear.yaml --ckpt ./checkpoints/model.ckpt
python bin/refinement_experiment.py -c configs/refinement/m4_hourly-deepar.yaml --ckpt ./checkpoints/model.ckpt

# Train-on-Synthetic-Test-on-Real (TSTR) evaluation
python bin/tstr_experiment.py -c configs/tstr/solar_nips.yaml --ckpt ./path/to/ckpt
python bin/tstr_experiment.py -c configs/tstr/kdd_cup_2018_without_missing.yaml --ckpt ./checkpoints/model.ckpt
