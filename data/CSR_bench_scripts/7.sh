#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/elliottwu/unsup3d.git
cd unsup3d
source ~/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate unsup3d
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
pip install neural_renderer_pytorch
pip install facenet-pytorch
conda install gxx_linux-64=7.3
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer && python setup.py install && cd ..

# Data / Checkpoint / Weight Download (URL)
cd data
curl -o synface.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/synface.zip" && unzip synface.zip
curl -o cat_combined.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/cat_combined.zip" && unzip cat_combined.zip
curl -o syncar.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/syncar.zip" && unzip syncar.zip
cd ../pretrained
curl -o pretrained_celeba.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/pretrained_celeba.zip" && unzip pretrained_celeba.zip
curl -o pretrained_cat.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/pretrained_cat.zip" && unzip pretrained_cat.zip
curl -o pretrained_synface.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/pretrained_synface.zip" && unzip pretrained_synface.zip
curl -o pretrained_syncar.zip "https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/pretrained_syncar.zip" && unzip pretrained_syncar.zip
cd ..

# Training
python run.py --config experiments/train_celeba.yml --gpu 0 --num_workers 2
python run.py --config experiments/train_cat.yml --gpu 0 --num_workers 2
python run.py --config experiments/train_synface.yml --gpu 0 --num_workers 2
python run.py --config experiments/train_syncar.yml --gpu 0 --num_workers 2

# Inference / Demonstration
mkdir -p demo/images/human_face demo/results/human_face
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth --gpu
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth --detect_human_face
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth --render_video --gpu

# Testing / Evaluation
python run.py --config experiments/test_celeba.yml --gpu 0 --num_workers 2
python run.py --config experiments/test_cat.yml --gpu 0 --num_workers 2
python run.py --config experiments/test_synface.yml --gpu 0 --num_workers 2
python run.py --config experiments/test_syncar.yml --gpu 0 --num_workers 2
