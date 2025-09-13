#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/MinkaiXu/GeoDiff.git
cd GeoDiff
conda env create -f env.yml
conda activate geodiff
conda install pytorch-geometric=1.7.2=py37_torch_1.8.0_cu102 -c rusty1s -c conda-forge
mkdir -p data/GEOM/QM9 data/GEOM/Drugs logs

# Data / Checkpoint / Weight Download (URL)
wget -O data/GEOM/QM9/train_data_40k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/QM9/val_data_5k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/QM9/test_data_1k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/QM9/qm9_property.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/Drugs/train_data_40k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/Drugs/val_data_5k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O data/GEOM/Drugs/test_data_1k.pkl "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
mkdir -p logs/qm9_default/checkpoints logs/drugs_default/checkpoints
wget -O logs/qm9_default/checkpoints/qm9_default.pt "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
wget -O logs/drugs_default/checkpoints/drugs_default.pt "https://drive.google.com/uc?export=download&id=1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh"
cp configs/qm9_default.yml logs/qm9_default/
cp configs/drugs_default.yml logs/drugs_default/

# Training
python train.py ./configs/qm9_default.yml
python train.py ./configs/drugs_default.yml
python train.py ./configs/drugs_1k_default.yml

# Inference / Demonstration
python test.py logs/qm9_default/checkpoints/qm9_default.pt --start_idx 0 --end_idx 100 --w_global 0.3
python test.py logs/drugs_default/checkpoints/drugs_default.pt --start_idx 0 --end_idx 100
python test.py logs/qm9_default/checkpoints/qm9_default.pt --start_idx 800 --end_idx 1000 --w_global 0.3
python test.py logs/drugs_default/checkpoints/drugs_default.pt --start_idx 800 --end_idx 1000
python test.py logs/qm9_default/checkpoints/qm9_default.pt --num_confs 50 --start_idx 0 --test_set data/GEOM/QM9/qm9_property.pkl --w_global 0.3

# Testing / Evaluation
python eval_covmat.py logs/qm9_default/sample/sample_all.pkl
python eval_covmat.py logs/drugs_default/sample/sample_all.pkl
python eval_prop.py --generated logs/qm9_default/sample/sample_all.pkl
python eval_prop.py --generated logs/drugs_default/sample/sample_all.pkl