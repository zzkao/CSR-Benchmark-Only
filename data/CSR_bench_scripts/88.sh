#!/bin/bash

# Environment Setup / Requirement / Installation
# Install Torch per Pytorch
pip install -r requirements.txt
mkdir result

# Data / Checkpoint / Weight Download (URL)
# Download and prepare datasets
cd data && bash download_dataset.sh && cd ..
python tools/generate_k_shot_data.py

# Generate SBERT embeddings for demonstration filtering
bash tools/get_sbert_embedding.sh roberta-large

# Training
# Basic prompt-based fine-tuning with demonstrations (single run)
python run.py \
    --task_name SST-2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-demo \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed 42 \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16

# Generate automatic templates using T5
python tools/generate_template.py \
    --output_dir my_auto_template \
    --task_name SST-2 \
    --seed 13 21 42 87 100 \
    --t5_model t5-base \
    --beam 100

# Generate automatic label mappings
bash tools/run_generate_labels.sh

# Inference / Demonstration
# Zero-shot inference
TAG=zero-shot TYPE=prompt TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--no_train"

# GPT3-style in-context learning
TAG=gpt3-in-context TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--no_train --num_sample 1 --gpt3_in_context_head --gpt3_in_context_num 32 --truncate_head --use_full_length"

# Testing / Evaluation
# Evaluate with demonstration filtering
TAG=exp TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--demo_filter --demo_filter_model sbert-roberta-large"

# Evaluate with automatic template
TAG=exp TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--template_path auto_template/SST-2/16-42.sort.txt --template_id 0"

# Evaluate with automatic label mapping
TAG=exp TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--mapping_path auto_label_mapping/SST-2/16-42.sort.txt --mapping_id 0"

# Gather results
python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-2', 'few_shot_type': 'prompt-demo'}"
