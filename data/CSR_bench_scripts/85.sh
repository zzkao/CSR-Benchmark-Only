#!/bin/bash

# Environment Setup / Requirement / Installation
# Install Python 3.8 and required packages
conda create --name ontoprotein python=3.8 --yes && conda activate ontoprotein
pip install biopython==1.37 goatools
pip install torch==1.9.0 transformers==4.5.1 deepspeed==0.5.1 lmdb tape-proteins

# Data / Checkpoint / Weight Download (URL)
# Download pre-training data (ProteinKG25)
# (Manually) Download from Drive link https://drive.google.com/uc?export=download&id=1iTC2-zbvYZCDhWM_wxRufCvV6vvPk8HR 
# Download downstream task datasets
# (Manually) Download from Drive link https://drive.google.com/uc?export=download&id=12d5wzNcuPxPyW8KIzwmvGg2dOKo0K0ag

# Download pre-trained models
# (Manually) Download Models from HuggingFace
# ProtBERT link https://huggingface.co/Rostlab/prot_bert
mkdir -p data/model_data/ProtBERT 
# PubMedBERT link https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
mkdir -p data/model_data/PubMedBERT 

# Training
# Generate pre-training data
python gen_onto_protein_data.py

# Pre-train OntoProtein
bash script/run_pretrain.sh

# Inference / Demonstration
# No specific inference commands provided in README

# Testing / Evaluation
# Run downstream tasks (examples for different tasks) Had to switch paths because read me is wrong with --model model_data/ProtBertModel(PubMedBERT) cause its created differently above 
bash run_main.sh --model data/model_data/ProtBert --output_file ss3-ProtBert --task_name ss3 --do_train True --epoch 5 --optimizer AdamW --per_device_batch_size 2 --gradient_accumulation_steps 8 --eval_step 100 --eval_batchsize 4 --warmup_ratio 0.08 --frozen_bert False
bash run_main.sh --model data/model_data/PubMedBERT --output_file ss3-ProtBert --task_name ss3 --do_train True --epoch 5 --optimizer AdamW --per_device_batch_size 2 --gradient_accumulation_steps 8 --eval_step 100 --eval_batchsize 4 --warmup_ratio 0.08 --frozen_bert False

# Run other tasks by changing task_name (supported tasks: ss8, contact, remote_homology, fluorescence, stability)
bash run_main.sh --model data/model_data/ProtBert --output_file ss8-ProtBert --task_name ss8 --do_train True --epoch 5 --optimizer AdamW --per_device_batch_size 2 --gradient_accumulation_steps 8 --eval_step 100 --eval_batchsize 4 --warmup_ratio 0.08 --frozen_bert False
bash run_main.sh --model data/model_data/PubMedBERT --output_file ss8-ProtBert --task_name ss8 --do_train True --epoch 5 --optimizer AdamW --per_device_batch_size 2 --gradient_accumulation_steps 8 --eval_step 100 --eval_batchsize 4 --warmup_ratio 0.08 --frozen_bert False
