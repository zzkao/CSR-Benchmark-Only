#!/bin/bash

# Environment Setup / Requirement / Installation
# Clone repository and setup conda environment
git clone https://github.com/YyzHarry/SubpopBench.git
cd SubpopBench/
conda env create -f environment.yml
conda activate subpop_bench

# Data / Checkpoint / Weight Download (URL)
# Download and prepare datasets (replace <data_path> with actual path)
DATA_PATH="./data"
python -m subpopbench.scripts.download --data_path ${DATA_PATH} --download

# Training
# Train single model with unknown attributes (basic example)
python -m subpopbench.train \
    --algorithm ERM \
    --dataset ColoredMNIST \
    --train_attr no \
    --data_dir ${DATA_PATH} \
    --output_dir ./outputs \
    --output_folder_name stage1

# Train two-stage model (DFR) with known attributes
python -m subpopbench.train \
    --algorithm DFR \
    --dataset ColoredMNIST \
    --train_attr yes \
    --data_dir ${DATA_PATH} \
    --output_dir ./outputs \
    --output_folder_name dfr_run \
    --stage1_folder ./outputs/stage1 \
    --stage1_algo ERM

# Launch hyperparameter sweep with unknown attributes
python -m subpopbench.sweep launch \
    --algorithms ERM GroupDRO \
    --dataset ColoredMNIST \
    --train_attr no \
    --n_hparams 5 \
    --n_trials 1

# Launch sweep with fixed hyperparameters and multiple seeds
python -m subpopbench.sweep launch \
    --algorithms ERM GroupDRO \
    --dataset ColoredMNIST \
    --train_attr no \
    --best_hp \
    --input_folder ./outputs/sweep_results \
    --n_trials 3

# Inference / Demonstration
# No explicit inference commands - evaluation is done during training

# Testing / Evaluation
# Collect and analyze sweep results
python -m subpopbench.scripts.collect_results --input_dir ./outputs/sweep_results
