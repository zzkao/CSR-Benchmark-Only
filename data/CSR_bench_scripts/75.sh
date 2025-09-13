#!/bin/bash
# Environment Setup / Requirement / Installation
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install ipykernel==6.23.1 numpy==1.24.3 tokenizers==0.13.3 torch==2.0.1 tqdm==4.65.0 transformers==4.29.2 datasets==2.12.0 beautifulsoup4==4.12.2 wandb==0.15.3 hydra-core==1.3.2 tensor-parallel==1.2.4
ulimit -n 64000

# Data / Checkpoint / Weight Download (URL)
# Note: Datasets are downloaded automatically during training

# Training
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false model.archive=/path/to/checkpoint/from/sft/LATEST/policy.pt
python -u train.py model=pythia69 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false
python -u train.py model=blank_model model.name_or_path=gpt2 model.block_name=GPT2Block datasets=[hh,shp] loss=sft exp_name=anthropic_shp_sft_gpt2 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 loss.label_smoothing=0.1 exp_name=conservative_dpo_pythia28 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false model.archive=/path/to/checkpoint/from/sft/LATEST/policy.pt
python -u train.py model=pythia28 datasets=[hh] loss=ipo loss.beta=0.1 exp_name=ipo_pythia28 gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=false n_epochs=1 wandb.enabled=false model.archive=/path/to/checkpoint/from/sft/LATEST/policy.pt

# Inference / Demonstration
# Note: Inference/evaluation is built into the training process with sample_during_eval=true

# Testing / Evaluation
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28_eval gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 trainer=BasicTrainer sample_during_eval=true n_epochs=1 wandb.enabled=false do_first_eval=true n_eval_examples=64