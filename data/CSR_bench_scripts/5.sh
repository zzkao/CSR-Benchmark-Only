#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/adobe/antialiased-cnns.git
cd antialiased-cnns
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh
mkdir -p /data/imagenet
./valprep.sh /data/imagenet/val

# Training
python main.py --data /data/imagenet -a resnet18_lpf4 --out-dir resnet18_lpf4 --lr 0.01 --epochs 1 --batch-size 32
python main.py --data /data/imagenet -a resnet50_lpf4 --out-dir resnet50_lpf4 --lr 0.01 --epochs 1 --batch-size 16
python main.py --data /data/imagenet -a alexnet_lpf4 --out-dir alexnet_lpf4 --gpu 0 --lr 0.01 --epochs 1 --batch-size 32
python main.py --data /data/imagenet -a vgg16_lpf4 --out-dir vgg16_lpf4 --lr 0.01 --epochs 1 --batch-size 16
python main.py --data /data/imagenet -a densenet121_lpf4 --out-dir densenet121_lpf4 --lr 0.01 --epochs 1 --batch-size 16
python main.py --data /data/imagenet -a resnet50_lpf4 --out-dir resnet50_lpf4_finetune --lr 0.01 --epochs 1 --finetune --batch-size 16

# Inference / Demonstration
python example_usage.py
python -c "import antialiased_cnns; model = antialiased_cnns.resnet50(pretrained=True); print('Model loaded successfully')"
python -c "import antialiased_cnns; import torch; C = 10; blurpool = antialiased_cnns.BlurPool(C, stride=2); ex_tens = torch.Tensor(1,C,128,128); print('BlurPool output shape:', blurpool(ex_tens).shape)"

# Testing / Evaluation
python main.py --data /data/imagenet -a resnet50_lpf4 --pretrained -e --batch-size 8
python main.py --data /data/imagenet -a resnet18_lpf4 --pretrained -e --batch-size 8
python main.py --data /data/imagenet -a alexnet_lpf4 --pretrained -e --gpu 0 --batch-size 8
python main.py --data /data/imagenet -a vgg16_lpf4 --pretrained -e --batch-size 8
python main.py --data /data/imagenet -a densenet121_lpf4 --pretrained -e --batch-size 8
python main.py --data /data/imagenet -a resnet50_lpf4 --pretrained -es -b 8
python main.py --data /data/imagenet -a resnet18_lpf4 --pretrained -es -b 8
python main.py -a resnet18_lpf4 --resume resnet18_lpf4/model_best.pth.tar --save_weights resnet18_lpf4/weights.pth.tar
