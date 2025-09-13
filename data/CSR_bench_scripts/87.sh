#!/bin/bash

# Environment Setup / Requirement / Installation
# Install Torch per PyTorch selector
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html || pip install torch==1.7.1

# Install other requirements
pip install -r requirements.txt

# Install SimCSE package
pip install simcse || python setup.py install

# Data / Checkpoint / Weight Download (URL)
# Download evaluation datasets
cd SentEval/data/downstream/ && bash download_dataset.sh && cd -

# Download training data
cd data && bash download_wiki.sh && bash download_nli.sh && cd -

# Training
# Unsupervised SimCSE training (single GPU/CPU)
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir results/unsup-simcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --mlp_only_train \
    --temp 0.05 \
    --do_mlm \
    --mlm_weight 0.1 \
    --mlm_probability 0.15

# Supervised SimCSE training (multi-GPU)
python -m torch.distributed.launch --nproc_per_node 8 train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir results/sup-simcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --temp 0.05

# Inference / Demonstration
# Convert checkpoint to HuggingFace format
python simcse_to_huggingface.py --path results/unsup-simcse-bert-base-uncased

# Testing / Evaluation
# Evaluate on STS tasks (--mode test isn't explicitly in README)
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode test

# Evaluate on transfer tasks
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set transfer \
    --mode test

# Fast evaluation for development
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode dev
    
