#!/bin/bash
# Environment Setup / Requirement / Installation
pip install torch>=1.4
pip install pandas
pip install tensorboardX
python -m mdlt.scripts.download --data_dir ./data

# Data / Checkpoint / Weight Download (URL)
# CSV files for train/val/test splits are provided in mdlt/dataset/split/
# Pre-trained models available at Google Drive links in README
# Place CSV files in corresponding dataset folders under data_dir

# Training
python -m mdlt.train --algorithm ERM --dataset VLCS --output_folder_name vlcs_erm_run --data_dir ./data --output_dir ./output
python -m mdlt.train --algorithm BoDA --dataset VLCS --output_folder_name vlcs_boda_run --data_dir ./data --output_dir ./output
python -m mdlt.train --algorithm CRT --dataset VLCS --output_folder_name vlcs_crt_run --data_dir ./data --output_dir ./output --stage1_folder vlcs_erm_run --stage1_algo ERM


python -m mdlt.train --algorithm BoDA --dataset PACS --output_folder_name pacs_boda_run --data_dir ./data --output_dir ./output --stage1_folder pacs_erm_run --stage1_algo ERM
python -m mdlt.train --algorithm ERM --dataset ImbalancedDigits --imb_type eee --imb_factor 0.01 --selected_envs 1 2 --output_folder_name digits_erm_run --data_dir ./data --output_dir ./output
python -m mdlt.train --algorithm BoDA --dataset ImbalancedDigits --imb_type eee --imb_factor 0.01 --selected_envs 1 2 --output_folder_name digits_boda_run --data_dir ./data --output_dir ./output

# Inference / Demonstration
python -u -m mdlt.evaluate.eval_best_hparam --algorithm BoDA --dataset VLCS --data_dir ./data --output_dir ./output --folder_name vlcs_boda_run
python -u -m mdlt.evaluate.eval_best_hparam --algorithm ERM --dataset PACS --data_dir ./data --output_dir ./output --folder_name pacs_erm_run
python -u -m mdlt.evaluate.eval_checkpoint --algorithm BoDA --dataset VLCS --data_dir ./data --checkpoint ./output/vlcs_boda_run/model.pkl

# Testing / Evaluation
python -m mdlt.sweep launch --algorithms ERM BoDA --dataset VLCS --n_hparams 5 --n_trials 1 --data_dir ./data --output_dir ./output
python -m mdlt.sweep launch --algorithms ERM BoDA CRT --dataset PACS --n_hparams 3 --n_trials 1 --data_dir ./data --output_dir ./output
python -m mdlt.sweep launch --algorithms ERM BoDA --dataset VLCS --best_hp --input_folder ./output/sweep_results --n_trials 3 --data_dir ./data --output_dir ./output
python -m mdlt.scripts.collect_results --input_dir ./output
