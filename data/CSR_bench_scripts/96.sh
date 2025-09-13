#!/bin/bash

# Environment Setup / Requirement / Installation
# Install dependencies
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Create data directories
mkdir -p data/NER_data data/RE_data

# Download and prepare RE data
cd data
wget 120.27.214.45/Data/re/multimodal/data.tar.gz
tar -xzvf data.tar.gz
mv data RE_data
cd ..

# Data: add NER datasets (manual via Drive) 
# Download Twitter2015 and Twitter2017 from the README Drive links and place under:
#   data/NER_data/twitter2015
#   data/NER_data/twitter2017
# (Each containing train/valid/test.txt and the *_images / *_aux_images folders.)


# Training
# Train NER model on Twitter2015 dataset
bash run_twitter15.sh

# Train NER model on Twitter2017 dataset
bash run_twitter17.sh

# Train RE model on MRE dataset
bash run_re_task.sh

# Inference / Demonstration
# Run inference on NER model (Twitter15/17)
python -u run.py \
    --dataset_name=twitter15 \
    --bert_name=bert-base-uncased \
    --seed=1234 \
    --only_test \
    --max_seq=80 \
    --use_prompt \
    --prompt_len=4 \
    --sample_ratio=1.0 \
    --load_path=ckpt/ner/twitter15

# Run inference on RE model
python -u run.py \
    --dataset_name=MRE \
    --bert_name=bert-base-uncased \
    --seed=1234 \
    --only_test \
    --max_seq=80 \
    --use_prompt \
    --prompt_len=4 \
    --sample_ratio=1.0 \
    --load_path=ckpt/re

# Testing / Evaluation
# Testing is included in the inference commands above
# The model will output evaluation metrics during inference
