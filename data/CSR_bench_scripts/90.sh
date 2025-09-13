#!/bin/bash

# Environment Setup / Requirement / Installation
# Install stable release
pip install alpaca-eval

# Install optional dependencies
pip install alpaca_eval[all]

# Install development version
pip install git+https://github.com/tatsu-lab/alpaca_eval

# Data / Checkpoint / Weight Download (URL)
# No explicit data download commands in README

# Training
# This is an evaluation tool, no training commands

# Inference / Demonstration
# Basic evaluation using model outputs
export OPENAI_API_KEY=your_api_key
alpaca_eval --model_outputs 'example/outputs.json'

# Evaluate from model directly
alpaca_eval evaluate_from_model \
    --model_name "gpt-3.5-turbo" \
    --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
    --reference_model "gpt4_turbo"

# Testing / Evaluation
# Make a leaderboard
alpaca_eval make_leaderboard \
    --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
    --reference_model "gpt4_turbo"

# Analyze evaluators
alpaca_eval analyze_evaluators \
    --evaluator "weighted_alpaca_eval_gpt4_turbo"

# Run evaluation with specific parameters
alpaca_eval evaluate \
    --model_outputs 'example/outputs.json' \
    --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
    --reference_outputs "gpt4_turbo" \
    --output_path "results/"
