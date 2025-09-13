#!/bin/bash

# Environment Setup / Requirement / Installation


# Create virtual environment (recommended)
python -m venv tape_env
source tape_env/bin/activate

# Install TAPE package
pip install tape_proteins

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Download all datasets using provided script
bash download_data.sh

# Alternative: Download individual datasets
# Download LMDB format datasets
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/pfam.tar.gz -P data/
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz -P data/
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/proteinnet.tar.gz -P data/
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz -P data/
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz -P data/
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz -P data/

# Extract datasets
cd data && for f in *.tar.gz; do tar xzf "$f"; done && cd ..

# Training
# Train language model using distributed training (recommended)
tape-train-distributed transformer masked_language_modeling \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --fp16 \
    --warmup_steps 10000 \
    --nproc_per_node 2 \
    --gradient_accumulation_steps 4

# Train downstream task (example: secondary structure prediction)
tape-train-distributed transformer secondary_structure \
    --from_pretrained results/pretrained_model \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --fp16 \
    --warmup_steps 10000 \
    --nproc_per_node 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --eval_freq 1 \
    --save_freq 10

# Inference / Demonstration
# Embed proteins from a FASTA file using UniRep model
tape-embed unirep input.fasta output_embeddings.npz babbler-1900 --tokenizer unirep

# Embed proteins with full sequence (instead of average)
tape-embed unirep input.fasta output_embeddings.npz babbler-1900 --tokenizer unirep --full_sequence_embed

# Testing / Evaluation
# Evaluate downstream model (example: secondary structure prediction)
tape-eval transformer secondary_structure results/trained_model --metrics accuracy

# Evaluate with multiple metrics
tape-eval transformer secondary_structure results/trained_model --metrics accuracy mse mae spearmanr
