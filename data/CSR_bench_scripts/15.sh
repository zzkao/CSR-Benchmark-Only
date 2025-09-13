#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/bnu-wangxun/Deep_Metric.git
cd Deep_Metric
pip install torch==1.0.0 torchvision==0.2.1
pip install numpy scipy matplotlib pillow tqdm
mkdir -p data ckps result
mkdir -p ckps/Weight ckps/Weight/cub ckps/Weight/car196
mkdir -p result/Weight result/Weight/cub result/Weight/car196

# Data / Checkpoint / Weight Download (URL)
cd data
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
tar -xzf car_devkit.tgz
mv devkit car196
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar -xzf images.tgz
wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
unzip Stanford_Online_Products.zip
cd ..

# Training
python train.py --net BN-Inception --data cub --data_root data --init random --lr 1e-5 --dim 512 --alpha 40 --num_instances 5 --batch_size 80 --epoch 100 --loss Weight --width 227 --save_dir ckps/Weight/cub/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80 --save_step 50 --ratio 0.16
python train.py --net BN-Inception --data car196 --data_root data --init random --lr 1e-5 --dim 512 --alpha 40 --num_instances 5 --batch_size 80 --epoch 100 --loss Weight --width 227 --save_dir ckps/Weight/car196/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80 --save_step 50 --ratio 0.16
python train.py --net BN-Inception --data cub --data_root data --init random --lr 1e-4 --dim 256 --alpha 30 --num_instances 4 --batch_size 64 --epoch 50 --loss Contrastive --width 227 --save_dir ckps/Contrastive/cub/BN-Inception-DIM-256 --save_step 25
python train.py --net BN-Inception --data cub --data_root data --init random --lr 1e-4 --dim 256 --alpha 30 --num_instances 4 --batch_size 64 --epoch 50 --loss Triplet --width 227 --save_dir ckps/Triplet/cub/BN-Inception-DIM-256 --save_step 25
python train.py --net BN-Inception --data cub --data_root data --init random --lr 1e-4 --dim 256 --alpha 30 --num_instances 4 --batch_size 64 --epoch 50 --loss Lifted --width 227 --save_dir ckps/Lifted/cub/BN-Inception-DIM-256 --save_step 25

# Inference / Demonstration
python test.py --net BN-Inception --data cub --data_root data --batch_size 100 --gallery_eq_query True --width 227 --resume ckps/Weight/cub/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep50.pth.tar --pool_feature False
python test.py --net BN-Inception --data car196 --data_root data --batch_size 100 --gallery_eq_query True --width 227 --resume ckps/Weight/car196/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep50.pth.tar --pool_feature False
python train.py --net BN-Inception --data cub --data_root data --init random --lr 1e-5 --dim 512 --alpha 40 --num_instances 5 --batch_size 32 --epoch 10 --loss Weight --width 227 --save_dir ckps/demo/cub --save_step 5 --ratio 0.16

# Testing / Evaluation
python test.py --net BN-Inception --data cub --data_root data --batch_size 100 --gallery_eq_query True --width 227 --resume ckps/Weight/cub/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep100.pth.tar --pool_feature False
python test.py --net BN-Inception --data car196 --data_root data --batch_size 100 --gallery_eq_query True --width 227 --resume ckps/Weight/car196/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep100.pth.tar --pool_feature False
python test.py --net BN-Inception --data cub --data_root data --batch_size 50 --gallery_eq_query False --width 227 --resume ckps/Contrastive/cub/BN-Inception-DIM-256/ckp_ep50.pth.tar --pool_feature False
python test.py --net BN-Inception --data cub --data_root data --batch_size 50 --gallery_eq_query False --width 227 --resume ckps/Triplet/cub/BN-Inception-DIM-256/ckp_ep50.pth.tar --pool_feature False
python test.py --net BN-Inception --data cub --data_root data --batch_size 50 --gallery_eq_query False --width 227 --resume ckps/Lifted/cub/BN-Inception-DIM-256/ckp_ep50.pth.tar --pool_feature False
