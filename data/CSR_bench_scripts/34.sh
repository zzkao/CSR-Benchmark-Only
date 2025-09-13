#!/bin/bash
# Environment Setup / Requirement / Installation
pip install torch>=1.1.0 torchvision>=0.3.0
pip install cython
git clone https://github.com/jeffffffli/res-loglikelihood-regression
cd res-loglikelihood-regression
python setup.py develop
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install six terminaltables scipy==1.1.0 opencv-python matplotlib visdom tqdm tensorboardx easydict pyyaml munkres timm==0.1.20 natsort
mkdir data

# Data / Checkpoint / Weight Download (URL)
mkdir -p data/coco/annotations
mkdir -p data/coco/images/train2017
mkdir -p data/coco/images/val2017
mkdir -p data/mpii/annotations
mkdir -p data/mpii/images
mkdir -p data/h36m/annotations
mkdir -p data/h36m/images
wget -O data/coco/annotations/person_keypoints_train2017.json "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
wget -O data/coco/annotations/person_keypoints_val2017.json "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
wget -O data/coco/images/train2017.zip "http://images.cocodataset.org/zips/train2017.zip"
wget -O data/coco/images/val2017.zip "http://images.cocodataset.org/zips/val2017.zip"
wget -O data/mpii/annotations/annot_mpii.json "https://drive.google.com/file/d/1--EQZnCJI_XJIc9_bw-dzw3MrRFLMptw/view?usp=sharing"
wget -O coco-laplace-rle.pth "https://drive.google.com/file/d/1YBHqNKkxIVv8CqgDxkezC-4vyKpx-zXK/view?usp=sharing"
wget -O h36m-laplace-rle.pth "https://drive.google.com/file/d/1v2ZhembnFyJ_FXGHEOCzGaM-tAVFMy7A/view?usp=sharing"

# Training
./scripts/train.sh ./configs/256x192_res50_regress-flow.yaml train_rle_coco
./scripts/train.sh ./configs/256x192_res50_3d_h36mmpii-flow.yaml train_rle_h36m
python ./scripts/train.py --nThreads 8 --launcher pytorch --rank 0 --dist-url tcp://localhost:23456 --exp-id train_rle_coco --cfg ./configs/256x192_res50_regress-flow.yaml --seed 123123
python ./scripts/train.py --nThreads 8 --launcher pytorch --rank 0 --dist-url tcp://localhost:23457 --exp-id train_rle_h36m --cfg ./configs/256x192_res50_3d_h36mmpii-flow.yaml --seed 123123

# Inference / Demonstration
# No specific inference/demonstration commands found in README

# Testing / Evaluation
./scripts/validate.sh ./configs/256x192_res50_regress-flow.yaml ./coco-laplace-rle.pth
./scripts/validate.sh ./configs/256x192_res50_3d_h36mmpii-flow.yaml ./h36m-laplace-rle.pth
python ./scripts/validate.py --cfg ./configs/256x192_res50_regress-flow.yaml --valid-batch 32 --flip-test --checkpoint ./coco-laplace-rle.pth --launcher pytorch --rank 0 --dist-url tcp://localhost:23456
python ./scripts/validate.py --cfg ./configs/256x192_res50_3d_h36mmpii-flow.yaml --valid-batch 32 --flip-test --checkpoint ./h36m-laplace-rle.pth --launcher pytorch --rank 0 --dist-url tcp://localhost:23457