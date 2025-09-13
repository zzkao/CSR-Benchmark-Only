#!/bin/bash

# Environment Setup / Requirement / Installation
# Install stable release
pip install alpaca-farm

# Install development version
pip install git+https://github.com/tatsu-lab/alpaca_farm.git

# Set OpenAI API key for evaluations
export OPENAI_API_KEY="your_api_key"

# Data / Checkpoint / Weight Download (URL)
# Download pre-trained models (replace paths with actual values)
LLAMA_PATH="/path/to/llama-7b-hf"
SAVE_DIR="models"

# Download all AlpacaFarm models
python -m pretrained_models.recover_model_weights \
    --llama-7b-hf-dir ${LLAMA_PATH} \
    --alpaca-farm-model-name all \
    --models-save-dir ${SAVE_DIR}

# Training
# Supervised Fine-tuning (SFT)
bash examples/scripts/sft.sh \
    "${SAVE_DIR}/sft10k" \
    "sft_run" \
    ${LLAMA_PATH}

# Reward Modeling with simulated preferences
bash examples/scripts/reward_modeling.sh \
    "${SAVE_DIR}/reward_model_sim" \
    "rm_sim_run" \
    "${SAVE_DIR}/sft10k" \
    "alpaca_noisy_multi_preference"

# RLHF with PPO using simulated preferences
bash examples/scripts/rlhf_ppo.sh \
    "${SAVE_DIR}/ppo_sim" \
    "ppo_sim_run" \
    "${SAVE_DIR}/reward_model_sim" \
    "${SAVE_DIR}/sft10k" \
    0.0067

# Expert Iteration
bash examples/scripts/expiter.sh \
    "${SAVE_DIR}/expiter" \
    "expiter_run" \
    "${SAVE_DIR}/sft10k" \
    "${SAVE_DIR}/expiter_data"

# DPO training
bash examples/scripts/dpo.sh \
    "${SAVE_DIR}/dpo" \
    "dpo_run" \
    "${SAVE_DIR}/sft10k"

# Quark training
bash examples/scripts/rlhf_quark.sh \
    "${SAVE_DIR}/quark" \
    "quark_run" \
    "${SAVE_DIR}/reward_model_sim" \
    "${SAVE_DIR}/sft10k" \
    0.0067

# Inference / Demonstration
# Best-of-n decoding for evaluation
python examples/best_of_n.py \
    --task "run_best_of_n" \
    --decoder_name_or_path "${SAVE_DIR}/sft10k" \
    --scorer_name_or_path "${SAVE_DIR}/reward_model_sim" \
    --num_return_sequences 16 \
    --per_device_batch_size 4 \
    --split "eval" \
    --mixed_precision "bf16" \
    --tf32 True \
    --flash_attn True \
    --output_path "${SAVE_DIR}/best_of_n_outputs.json"

# Run OpenAI baseline models
python examples/oai_baselines.py \
    --model_name "text-davinci-003" \
    --save_path "${SAVE_DIR}/oai_outputs.json"

# Testing / Evaluation
# Run automatic evaluation using AlpacaEval
python -c "
from alpaca_farm.auto_annotations import alpaca_leaderboard
alpaca_leaderboard('${SAVE_DIR}/best_of_n_outputs.json', name='best_of_n_model')
"
