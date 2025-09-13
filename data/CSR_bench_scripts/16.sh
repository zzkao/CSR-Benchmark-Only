#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/YyzHarry/imbalanced-semi-self.git
cd imbalanced-semi-self
pip install torch>=1.2 torchvision
pip install pyyaml scikit-learn tensorboardX
mkdir -p data log checkpoint
mkdir -p data/cifar-10-batches-py data/svhn data/imagenet data/inat2018

# Data / Checkpoint / Weight Download (URL)
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
tar -xzf data/cifar-10-python.tar.gz -C data/
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -P data/svhn/
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -P data/svhn/
wget https://drive.google.com/file/d/1SODQBUvv2qycDivBb4nhHaCk3TMzaVM4/view?usp=sharing -O data/ti_80M_selected.pickle
wget https://drive.google.com/file/d/1Z4rwaqzjNoNQ27sofx1aDl8OLH-etoyP/view?usp=sharing -O data/pseudo_labeled_cifar10_rho50.pickle
wget https://drive.google.com/file/d/19VeMQ07unVq3hIjLN5LiXWZNTI4CiN5F/view?usp=sharing -O data/pseudo_labeled_svhn_rho50.pickle

# Training
python pretrain_rot.py --dataset cifar10 --imb_factor 0.01 --arch resnet32 --epochs 50 --batch-size 64 --gpu 0
python pretrain_rot.py --dataset cifar100 --imb_factor 0.01 --arch resnet32 --epochs 50 --batch-size 64 --gpu 0
python pretrain_moco.py --dataset imagenet --data data/imagenet --arch resnet50 --epochs 50 --batch-size 64
python train.py --dataset cifar10 --imb_factor 0.01 --arch resnet32 --epochs 50 --batch-size 64 --pretrained_model checkpoint/pretrain_rot_cifar10_resnet32.pth --gpu 0
python train.py --dataset cifar100 --imb_factor 0.01 --arch resnet32 --epochs 50 --batch-size 64 --pretrained_model checkpoint/pretrain_rot_cifar100_resnet32.pth --gpu 0
python train_semi.py --dataset cifar10 --imb_factor 0.02 --imb_factor_unlabel 0.02 --arch resnet32 --epochs 50 --batch-size 128 --gpu 0
python train_semi.py --dataset svhn --imb_factor 0.02 --imb_factor_unlabel 0.02 --arch resnet32 --epochs 50 --batch-size 128 --gpu 0
python -m imagenet_inat.main --cfg imagenet_inat/config/imagenet_lt_resnext50.yaml --model_dir checkpoint/pretrain_moco_imagenet_resnet50.pth

# Inference / Demonstration
python gen_pseudolabels.py --resume checkpoint/cifar10_ce_resnet32.pth --data_dir data --output_dir data --output_filename pseudo_labeled_cifar10_demo.pickle --dataset cifar10 --arch resnet32
python train.py --dataset cifar10 --imb_factor 0.1 --arch resnet32 --epochs 10 --batch-size 32 --gpu 0
python train_semi.py --dataset cifar10 --imb_factor 0.1 --imb_factor_unlabel 0.1 --arch resnet32 --epochs 10 --batch-size 64 --gpu 0
python pretrain_rot.py --dataset cifar10 --imb_factor 0.1 --arch resnet32 --epochs 10 --batch-size 32 --gpu 0

# Testing / Evaluation
python train_semi.py --dataset cifar10 --resume checkpoint/semi_cifar10_resnet32_rho50.pth -e --arch resnet32 --gpu 0
python train_semi.py --dataset svhn --resume checkpoint/semi_svhn_resnet32_rho50.pth -e --arch resnet32 --gpu 0
python train.py --dataset cifar10 --resume checkpoint/ssp_cifar10_resnet32_rho100.pth -e --arch resnet32 --gpu 0
python train.py --dataset cifar100 --resume checkpoint/ssp_cifar100_resnet32_rho100.pth -e --arch resnet32 --gpu 0
python -m imagenet_inat.main --cfg imagenet_inat/config/imagenet_lt_resnext50.yaml --model_dir checkpoint/ssp_imagenet_resnet50.pth --test
python -m imagenet_inat.main --cfg imagenet_inat/config/inat_resnext50.yaml --model_dir checkpoint/ssp_inat2018_resnet50.pth --test
