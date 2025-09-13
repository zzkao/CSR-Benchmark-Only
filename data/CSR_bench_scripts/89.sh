#!/bin/bash

# Environment Setup / Requirement / Installation
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Download and setup SciERC dataset
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
mkdir -p scierc_data && tar -xf sciERC_processed.tar.gz -C scierc_data && rm -f sciERC_processed.tar.gz
scierc_dataset=scierc_data/processed_data/json/

# Download pre-trained models
mkdir -p scierc_models && cd scierc_models

# Entity model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip
unzip ent-scib-ctx0.zip && rm -f ent-scib-ctx0.zip
scierc_ent_model=scierc_models/ent-scib-ctx0/

# Relation model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip
unzip rel-scib-ctx0.zip && rm -f rel-scib-ctx0.zip
scierc_rel_model=scierc_models/rel-scib-ctx0/

# Approximation relation model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx0.zip
unzip rel_approx-scib-ctx0.zip && rm -f rel_approx-scib-ctx0.zip
scierc_rel_model_approx=scierc_models/rel_approx-scib-ctx0/

cd ..

# Training
# Train entity model
python run_entity.py \
    --do_train --do_eval \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window 0 \
    --task scierc \
    --data_dir ${scierc_dataset} \
    --model allenai/scibert_scivocab_uncased \
    --output_dir ${scierc_ent_model} \
    --num_train_epochs 1

# Train relation model
python run_relation.py \
    --task scierc \
    --do_train --train_file ${scierc_dataset}/train.json \
    --do_eval \
    --model allenai/scibert_scivocab_uncased \
    --do_lower_case \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --context_window 0 \
    --max_seq_length 128 \
    --entity_output_dir ${scierc_ent_model} \
    --output_dir ${scierc_rel_model}

# Train approximation relation model
python run_relation_approx.py \
    --task scierc \
    --do_train --train_file ${scierc_dataset}/train.json \
    --do_eval \
    --model allenai/scibert_scivocab_uncased \
    --do_lower_case \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --context_window 0 \
    --max_seq_length 128 \
    --entity_output_dir ${scierc_ent_model} \
    --output_dir ${scierc_rel_model_approx}

# Inference / Demonstration
# Run entity model inference
python run_entity.py \
    --do_eval --eval_test \
    --context_window 0 \
    --task scierc \
    --data_dir ${scierc_dataset} \
    --model allenai/scibert_scivocab_uncased \
    --output_dir ${scierc_ent_model}

# Run relation model inference
python run_relation.py \
    --task scierc \
    --do_eval --eval_test \
    --model allenai/scibert_scivocab_uncased \
    --do_lower_case \
    --context_window 0 \
    --max_seq_length 128 \
    --entity_output_dir ${scierc_ent_model} \
    --output_dir ${scierc_rel_model}

# Run approximation relation model inference with batch computation
python run_relation_approx.py \
    --task scierc \
    --do_eval --eval_test \
    --model allenai/scibert_scivocab_uncased \
    --do_lower_case \
    --context_window 0 \
    --max_seq_length 250 \
    --entity_output_dir ${scierc_ent_model} \
    --output_dir ${scierc_rel_model_approx} \
    --batch_computation

# Testing / Evaluation
# Evaluate relation model predictions
python run_eval.py --prediction_file ${scierc_rel_model}/predictions.json

# Evaluate approximation relation model predictions
python run_eval.py --prediction_file ${scierc_rel_model_approx}/predictions.json
