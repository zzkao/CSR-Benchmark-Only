#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/tjiiv-cprg/EPro-PnP.git
cd EPro-PnP
conda create -y -n epropnp_6dof python=3.6
conda activate epropnp_6dof
conda install -y pip
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
pip install opencv-python==4.5.1.48 pyro-ppl==1.4.0 PyYAML==5.4.1 matplotlib termcolor plyfile easydict scipy progress numba tensorboardx
conda create -y -n epropnp_det python=3.7
conda activate epropnp_det
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
conda install -y -c fvcore -c iopath -c conda-forge -c bottler fvcore iopath nvidiacub
git clone https://github.com/facebookresearch/pytorch3d && cd pytorch3d && git checkout v0.6.1 && pip install -v -e . && cd ..
cd EPro-PnP-Det && pip install -v -e . && cd ..

# Data / Checkpoint / Weight Download (URL)
mkdir -p EPro-PnP-6DoF/dataset/bg_images EPro-PnP-6DoF/dataset/lm EPro-PnP-6DoF/checkpoints
mkdir -p EPro-PnP-Det/data/nuscenes
wget -O EPro-PnP-6DoF/dataset/lm.zip "https://mega.nz/folder/w0sTxbYa#0w-huVv5gK953mO-eGpYVg"
curl -L -o EPro-PnP-6DoF/checkpoints/cdpn_stage_1.pth "https://drive.google.com/uc?export=download&id=1Jem2XsdHxr3ETRsZYqyTUmo5F3TmJGfO"
curl -L -o EPro-PnP-6DoF/checkpoints/epropnp_cdpn_init_long.pth "https://drive.google.com/uc?export=download&id=1Jem2XsdHxr3ETRsZYqyTUmo5F3TmJGfO"
curl -L -o EPro-PnP-Det/checkpoints/epropnp_det_v1b_220411.pth "https://drive.google.com/uc?export=download&id=1AWRg09fkt66I8rgrp33Lwb9l6-D6Gjrg"
cd EPro-PnP-Det && python tools/data_converter/nuscenes_converter.py data/nuscenes --version v1.0-trainval && cd ..

# Training
cd EPro-PnP-6DoF/tools && python main.py --cfg exps_cfg/epropnp_basic.yaml && cd ../..
cd EPro-PnP-6DoF/tools && python main.py --cfg exps_cfg/epropnp_cdpn_init.yaml && cd ../..
cd EPro-PnP-Det && python train.py configs/epropnp_det_basic.py --gpu-ids 0 && cd ..
cd EPro-PnP-Det && python train.py configs/epropnp_det_v1b_220411.py --gpu-ids 0 && cd ..

# Inference / Demonstration
cd EPro-PnP-Det && python demo/infer_imgs.py demo/ configs/epropnp_det_basic.py checkpoints/epropnp_det_v1b_220411.pth --intrinsic demo/nus_cam_front.csv --show-views 3d bev mc && cd ..
cd EPro-PnP-Det && python demo/infer_nuscenes_sequence.py --help && cd ..

# MANUAL
#jupyter notebook demo/fit_identity.ipynb
#tensorboard --logdir EPro-PnP-6DoF/exp
#tensorboard --logdir EPro-PnP-Det/work_dirs

# Testing / Evaluation
cd EPro-PnP-6DoF/tools && python main.py --cfg exps_cfg/epropnp_cdpn_init_long.yaml && cd ../..
cd EPro-PnP-Det && python test.py configs/epropnp_det_basic.py checkpoints/epropnp_det_v1b_220411.pth --val-set --eval nds --gpu-ids 0 && cd ..
cd EPro-PnP-Det && python test.py configs/epropnp_det_v1b_220411.py checkpoints/epropnp_det_v1b_220411.pth --val-set --eval nds --gpu-ids 0 && cd ..
cd EPro-PnP-Det && python test.py configs/epropnp_det_v1b_220411.py checkpoints/epropnp_det_v1b_220411.pth --format-only --eval-options jsonfile_prefix=results/test_results && cd ..
