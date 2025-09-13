#!/bin/bash
# Environment Setup / Requirement / Installation
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Note: Data and weights are automatically downloaded from Hugging Face Hub when running the scripts

# Training
python finetune.py --base_model 'decapoda-research/llama-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca'
python finetune.py --base_model 'decapoda-research/llama-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca' --batch_size 128 --micro_batch_size 4 --num_epochs 1 --learning_rate 1e-4 --cutoff_len 512 --val_set_size 2000 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules '[q_proj,v_proj]' --train_on_inputs --group_by_length
python finetune.py --base_model='decapoda-research/llama-7b-hf' --num_epochs=1 --cutoff_len=512 --group_by_length --output_dir='./lora-alpaca' --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --lora_r=16 --micro_batch_size=8

# Inference / Demonstration
python generate.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' --lora_weights 'tloen/alpaca-lora-7b'

# Testing / Evaluation
# Note: No specific testing/evaluation commands found in README

# Docker Setup (Alternative)
docker build -t alpaca-lora .
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' --lora_weights 'tloen/alpaca-lora-7b'

# Docker Compose Setup (Alternative)
docker-compose up -d --build
docker-compose logs -f
docker-compose down --volumes --rmi all
