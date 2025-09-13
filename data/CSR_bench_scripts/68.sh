#!/bin/bash
# Environment Setup / Requirement / Installation
cd $HOME && git clone https://github.com/mistralai/mistral-finetune.git
cd mistral-finetune
pip install -r requirements.txt
pip install pandas pyarrow
pip install wandb
pip install mistral_inference

# Data / Checkpoint / Weight Download (URL)
mkdir -p ${HOME}/mistral_models
cd ${HOME} && wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar
tar -xf mistral-7B-v0.3.tar -C mistral_models
cd $HOME && mkdir -p data && cd $HOME/data

# Training
cd $HOME/mistral-finetune
torchrun --nproc-per-node 1 --master_port 29500 -m train example/7B.yaml
python -m utils.extend_model_vocab --original_model_ckpt ${HOME}/mistral_models/7B --extended_model_ckpt ${HOME}/mistral_models/7B_extended

# Inference / Demonstration
mistral-chat ${HOME}/mistral_models/7B/ --max_tokens 256 --temperature 1.0 --instruct --lora_path ${HOME}/ultra_chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors

# Testing / Evaluation
python -m utils.validate_data --train_yaml example/7B.yaml
python -m utils.validate_data --train_yaml example/7B.yaml --create_corrected
python -m utils.reformat_data ${HOME}/data/ultrachat_chunk_train.jsonl
python -m utils.reformat_data ${HOME}/data/ultrachat_chunk_eval.jsonl
python -m utils.reformat_data_glaive ${HOME}/data/glaive_train.jsonl
python -m utils.reformat_data_glaive ${HOME}/data/glaive_eval.jsonl