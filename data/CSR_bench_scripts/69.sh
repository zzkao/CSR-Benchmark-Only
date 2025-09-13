#!/bin/bash
# Environment Setup / Requirement / Installation
pip install torch fairscale fire sentencepiece
pip install -e .
chmod +x download.sh

# Data / Checkpoint / Weight Download (URL)
./download.sh

# Training
# Note: No specific training commands found in this repository (inference-focused)

# Inference / Demonstration
torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4

# Testing / Evaluation
# Note: No specific testing/evaluation commands found in README