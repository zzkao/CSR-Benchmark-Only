#!/bin/bash

# Environment Setup / Requirement / Installation
# Install PyTorch and dependencies (tested versions)
pip install torch==2.1.0.dev20230514+cu118 -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html

pip install transformers==4.28.1
pip install accelerate==0.17.1

# Data / Checkpoint / Weight Download (URL)
# Download and prepare datasets for medium models
cd data && bash download_dataset.sh && cd ..

# Generate k-shot data splits
for K in 16 512; do
    python data/download_dataset.sh --mode k-shot-1k-test --k $K
done

# Training
# Medium Models (RoBERTa-large)
# Standard fine-tuning
cd medium_models
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh

# Fine-tuning with prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-2 MODEL=roberta-large EXTRA_TAG=prefix bash finetune.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# Fine-tuning with LoRA
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16

# MeZO training
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-6 EPS=1e-3 MODEL=roberta-large bash mezo.sh

# MeZO with prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-2 EPS=1e-1 MODEL=roberta-large EXTRA_TAG=prefix bash mezo.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# MeZO with LoRA
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16

# Large Models (OPT)
# Zero-shot evaluation
cd large_models
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0

# In-context learning
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh

# Full-parameter fine-tuning (single GPU)
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh

# Full-parameter fine-tuning (multi-GPU)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=4 bash finetune_fsdp.sh

# MeZO training for large models
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh

# Inference / Demonstration
# Run zero-shot inference
python run.py --trainer none --model_name facebook/opt-13b --task_name SST2 --num_train 0

# Run in-context learning inference
python run.py --trainer none --model_name facebook/opt-13b --task_name SST2 --num_train 32

# Testing / Evaluation
# Gather results for medium models
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-ft', 'task_name': 'sst-2'}"

# Run non-differentiable objective evaluation (SQuAD F1)
MODEL=facebook/opt-13b TASK=SQuAD MODE=prefix LR=1e-2 EPS=1e-1 bash mezo.sh --non_diff --evaluation_strategy no --save_strategy no --save_model
