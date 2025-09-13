#!/bin/bash
# Environment Setup / Requirement / Installation
conda create -n LTSF_Linear python=3.6.9
conda activate LTSF_Linear
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
mkdir dataset
# Note: Download datasets from Google Drive: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
# Put all datasets in the ./dataset directory

# Training
mkdir -p logs/LongForecasting
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv --model_id Exchange_336_96 --model DLinear --data custom --features M --seq_len 336 --pred_len 96 --enc_in 8 --des 'Exp' --itr 1 --batch_size 8 --learning_rate 0.0005 > logs/LongForecasting/DLinear_Exchange_336_96.log
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv --model_id Exchange_336_192 --model DLinear --data custom --features M --seq_len 336 --pred_len 192 --enc_in 8 --des 'Exp' --itr 1 --batch_size 8 --learning_rate 0.0005 > logs/LongForecasting/DLinear_Exchange_336_192.log
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv --model_id Exchange_336_336 --model DLinear --data custom --features M --seq_len 336 --pred_len 336 --enc_in 8 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.0005 > logs/LongForecasting/DLinear_Exchange_336_336.log
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv --model_id Exchange_336_720 --model DLinear --data custom --features M --seq_len 336 --pred_len 720 --enc_in 8 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.0005 > logs/LongForecasting/DLinear_Exchange_336_720.log
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh

# Inference / Demonstration
mkdir -p weights_plot
python weight_plot.py

# Testing / Evaluation
# Note: Evaluation is integrated into the training process above
# Results are logged in logs/LongForecasting/ directory
# Model checkpoints are saved in checkpoints/ directory for further analysis
