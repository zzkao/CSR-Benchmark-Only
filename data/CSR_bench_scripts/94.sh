#!/bin/bash

# Environment Setup / Requirement / Installation
# Create 
git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git 
cd hyena-dna 

# Create conda environment
conda create -n hyena-dna python=3.8
conda activate hyena-dna

# Install PyTorch with CUDA
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install Flash Attention
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e . --no-build-isolation

# Optional: Install fused layers for speed Note:path not set
cd csrc/layer_norm && pip install . --no-build-isolation && cd ../../..

# Data / Checkpoint / Weight Download (URL)
# Create data directory and download HG38 data
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed

# Training
# Quick start training on small genomics dataset
python -m train wandb=null experiment=hg38/genomic_benchmark_scratch

# Pretraining on Human Reference Genome
python -m train wandb=null experiment=hg38/hg38_hyena \
    model.d_model=128 \
    model.n_layer=2 \
    dataset.batch_size=256 \
    train.global_batch_size=256 \
    dataset.max_length=1024 \
    optimizer.lr=6e-4 \
    trainer.devices=1

# Train on GenomicBenchmarks dataset
python -m train wandb=null experiment=hg38/genomic_benchmark \
    dataset_name=human_enhancers_cohn \
    train.pretrained_model_path=/path/to/ckpt \
    dataset.max_length=500 \
    model.layer.l_max=1024

# Train on Nucleotide Transformer dataset
python -m train wandb=null experiment=hg38/nucleotide_transformer \
    dataset_name=enhancer \
    dataset.max_length=500 \
    model.layer.l_max=1026

# Species classification training
python -m train wandb=null experiment=hg38/species \
    dataset.species=[human,mouse,hippo,pig,lemur] \
    train.global_batch_size=256 \
    optimizer.lr=6e-5 \
    trainer.devices=1 \
    dataset.batch_size=1 \
    dataset.max_length=1024 \
    dataset.species_dir=/path/to/data/species/ \
    model.layer.l_max=1026 \
    model.d_model=128 \
    model.n_layer=2 \
    trainer.max_epochs=150 \
    decoder.mode=last \
    train.pretrained_model_path=null \
    train.pretrained_model_state_hook=null

# Inference / Demonstration
# Run soft prompting genomics evaluation
python -m evals.soft_prompting_genomics

# Run instruction tuned genomics evaluation
python -m evals.instruction_tuned_genomics

# Testing / Evaluation
# Load and evaluate finetuned model
python -m train wandb=null experiment=hg38/genomic_benchmark_load_finetuned_model

# Evaluate chromatin profile
python -m train wandb=null experiment=hg38/chromatin_profile \
    dataset.ref_genome_path=/path/to/fasta/hg38.ml.fa \
    dataset.data_path=/path/to/chromatin_profile \
    dataset.ref_genome_version=hg38
