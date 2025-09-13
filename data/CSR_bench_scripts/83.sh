#!/bin/bash
# Environment Setup / Requirement / Installation
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q '^raydf '; then
  conda create -n raydf python=3.8 -y
fi
conda activate raydf

if [ ! -d RayDF ]; then
  git clone https://github.com/vLAR-group/RayDF.git
fi
cd RayDF

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
sh datasets/download.sh blender
sh datasets/download.sh dmsr
sh datasets/download.sh scannet
mkdir -p datasets && cd datasets && gdown 1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM && unzip blender.zip && rm blender.zip && cd ..
mkdir -p datasets && cd datasets && gdown 14bxsM1a9QnP9b7GHBFuU6ln1nq03Qsy9 && unzip dmsr.zip && rm dmsr.zip && cd ..
mkdir -p datasets && cd datasets && gdown 1UzJzcgBkGo6KfhZMLFCikXoaXptbWPF- && unzip scannet.zip && rm scannet.zip && cd ..

# Training
python run_cls.py --config configs/blender_cls.txt --scene lego
python run_mv.py --config configs/blender.txt --scene lego
python run_mv.py --config configs/blender.txt --scene lego --rgb_layer 2
python run_cls.py --config configs/dmsr_cls.txt --scene bathroom
python run_mv.py --config configs/dmsr.txt --scene bathroom
python run_cls.py --config configs/scannet_cls.txt --scene scene0004_00
python run_mv.py --config configs/scannet.txt --scene scene0004_00
sh run.sh 0 blender lego
sh run.sh 0 dmsr bathroom

# Inference / Demonstration
python run_mv.py --config configs/blender.txt --scene lego --eval_only
python run_mv.py --config configs/blender.txt --scene lego --eval_only --denoise
python run_mv.py --config configs/blender.txt --scene lego --eval_only --grad_normal
python run_mv.py --config configs/dmsr.txt --scene bathroom --eval_only
python run_mv.py --config configs/scannet.txt --scene scene0004_00 --eval_only

# Testing / Evaluation
python run_cls.py --config configs/blender_cls.txt --scene lego --eval_only
python run_cls.py --config configs/dmsr_cls.txt --scene bathroom --eval_only
python run_cls.py --config configs/scannet_cls.txt --scene scene0004_00 --eval_only
python run_mv.py --config configs/blender.txt --scene lego --eval_only --denoise --grad_normal
