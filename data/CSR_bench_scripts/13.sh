#!/bin/bash
# Environment Setup / Requirement / Installation
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh
source ~/.bashrc
conda install git
git clone https://github.com/graphdeeplearning/graphtransformer.git
cd graphtransformer
conda env create -f environment_cpu.yml
conda activate graph_transformer
mkdir -p data/molecules data/SBMs out/ZINC_sparse_LapPE_BN/results out/ZINC_sparse_LapPE_BN/checkpoints out/ZINC_sparse_LapPE_BN/logs

# Data / Checkpoint / Weight Download (URL)
cd data/
bash script_download_molecules.sh
bash script_download_SBMs.sh
cd molecules/
curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
cd ../SBMs/
curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl -o SBM_CLUSTER.pkl -J -L -k
curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_PATTERN.pkl -o SBM_PATTERN.pkl -J -L -k
cd ../..

# Training
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' --dataset ZINC --seed 41
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_LN.json' --dataset ZINC --seed 41
python main_SBMs_node_classification.py --config 'configs/SBMs_GraphTransformer_CLUSTER_500k_sparse_graph_LN.json' --dataset SBM_CLUSTER --seed 41
python main_SBMs_node_classification.py --config 'configs/SBMs_GraphTransformer_PATTERN_500k_sparse_graph_LN.json' --dataset SBM_PATTERN --seed 41

# Inference / Demonstration
tensorboard --logdir='out/ZINC_sparse_LapPE_BN/logs/' --port 6006
jupyter notebook scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC.ipynb
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' --dataset ZINC --seed 41 --epochs 10

# Testing / Evaluation
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k.sh
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k.sh
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' --dataset ZINC --seed 41 --epochs 100
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_BN.json' --dataset ZINC --seed 41 --epochs 100
python main_SBMs_node_classification.py --config 'configs/SBMs_GraphTransformer_LapPE_CLUSTER_500k_sparse_graph_BN.json' --dataset SBM_CLUSTER --seed 41 --epochs 100
python main_SBMs_node_classification.py --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --dataset SBM_PATTERN --seed 41 --epochs 100
