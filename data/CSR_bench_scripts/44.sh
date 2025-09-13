#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch.git
cd RGBD_Semantic_Segmentation_PyTorch
conda env create -f rgbd.yaml
conda activate rgbd
cd ./furnace/apex
python setup.py install --cpp_ext --cuda_ext
cd ../..
mkdir -p DATA/pytorch-weight DATA/NYUDepthv2/ColoredLabel DATA/NYUDepthv2/Depth DATA/NYUDepthv2/HHA DATA/NYUDepthv2/Label DATA/NYUDepthv2/RGB

# Data / Checkpoint / Weight Download (URL)
cd DATA/pytorch-weight
wget -O resnet101-5d3b4d8f.pth "https://drive.google.com/uc?export=download&id=1_1HpmoCsshNCMQdXhSNOq8Y-deIDcbKS"
cd ../NYUDepthv2
wget -O nyu_depth_v2_processed.tar.gz "https://drive.google.com/uc?export=download&id=1_1HpmoCsshNCMQdXhSNOq8Y-deIDcbKS"
tar -xzf nyu_depth_v2_processed.tar.gz
wget -O train.txt "https://raw.githubusercontent.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch/master/DATA/NYUDepthv2/train.txt"
wget -O test.txt "https://raw.githubusercontent.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch/master/DATA/NYUDepthv2/test.txt"
cd ../..

# Training
cd ./model/SA-Gate.nyu.432
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
cd ../SA-Gate.nyu
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
cd ../malleable2_5d.nyu.res101
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
cd ../..

# Inference / Demonstration
cd ./model/SA-Gate.nyu
python eval.py -e 300-400 -d 0-7 --save_path results
cd ../SA-Gate.nyu.432
python eval.py -e 300-400 -d 0-3 --save_path results
cd ../malleable2_5d.nyu.res101
python eval.py -e 300-400 -d 0-3 --save_path results
cd ../..

# Testing / Evaluation
cd ./model/SA-Gate.nyu
python eval.py -e 350 -d 0 --save_path single_scale_results
python eval.py -e 350 -d 0 --save_path multi_scale_results
cd ../SA-Gate.nyu.432
python eval.py -e 350 -d 0 --save_path single_scale_results
cd ../malleable2_5d.nyu.res101
python eval.py -e 350 -d 0 --save_path single_scale_results
cd ../..