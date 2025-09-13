#!/bin/bash

# Environment Setup / Requirement / Installation
# Install dependencies (versions from README)
pip install torch>=1.9.0
pip install dgl>=0.7.2
pip install pyyaml==5.4.1

# Data / Checkpoint / Weight Download (URL)
# Datasets will be downloaded automatically when running the code

# Training
# Node Classification - Transductive (e.g., Cora dataset)
python main_transductive.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0 --use_cfg

# Node Classification - Inductive (e.g., PPI dataset)
python main_inductive.py --dataset ppi --encoder gat --decoder gat --seed 0 --device 0 --use_cfg

# Graph Classification (e.g., IMDB-BINARY dataset)
python main_graph.py --dataset IMDB-BINARY --encoder gin --decoder gin --seed 0 --device 0 --use_cfg

# Alternative script-based training commands
# Transductive node classification
sh scripts/run_transductive.sh cora 0  # Options: cora/citeseer/pubmed/ogbn-arxiv

# Inductive node classification
sh scripts/run_inductive.sh ppi 0  # Options: ppi/reddit

# Graph classification
sh scripts/run_graph.sh IMDB-BINARY 0  # Options: IMDB-BINARY/IMDB-MULTI/PROTEINS/MUTAG/NCI1/REDDIT-BINERY/COLLAB

# Inference / Demonstration
# No explicit inference commands in README
# Models can be used for node/graph classification after training

# Testing / Evaluation
# Evaluation is included in training scripts
# Results will show metrics like Micro-F1 for node classification
# and Accuracy for graph classification
