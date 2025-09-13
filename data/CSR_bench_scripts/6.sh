#!/bin/bash
# Environment Setup / Requirement / Installation
conda create -n spark python=3.8 -y
conda activate spark
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.5.4
pip install matplotlib numpy Pillow typed-argument-parser tensorboardx
git clone https://github.com/keyu-tian/SparK.git
cd SparK

# Data / Checkpoint / Weight Download (URL)
mkdir -p /data/imagenet/train /data/imagenet/val
wget -O resnet50_1kpretrained_timm_style.pth "https://drive.google.com/uc?export=download&id=1H8605HbxGvrsu4x4rIoNr-Wkd7JkxFPQ"
wget -O convnextS_1kpretrained_official_style.pth "https://drive.google.com/uc?export=download&id=1Ah6lgDY5YDNXoXHQHklKKMbEd08RYivN"
wget -O res50_withdecoder_1kpretrained_spark_style.pth "https://drive.google.com/uc?export=download&id=1STt3w3e5q9eCPZa8VzcJj1zG6p3jLeSF"

# Training
cd pretrain
python3 main.py --exp_name=debug --data_path=/data/imagenet --model=resnet50 --bs=32 --epochs=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 main.py --data_path=/data/imagenet --exp_name=resnet50_pretrain --exp_dir=./logs --model=resnet50 --bs=64 --epochs=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12346 main.py --data_path=/data/imagenet --exp_name=convnext_small_pretrain --exp_dir=./logs --model=convnext_small --bs=32 --epochs=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12347 main.py --data_path=/data/imagenet --exp_name=convnext_large_384 --exp_dir=./logs --model=convnext_large --input_size=384 --mask=0.75 --bs=32 --base_lr=4e-4 --epochs=1

# Inference / Demonstration
cd ..
python -c "import torch, timm; res50, state = timm.create_model('resnet50'), torch.load('resnet50_1kpretrained_timm_style.pth', 'cpu'); res50.load_state_dict(state.get('module', state), strict=False); print('ResNet50 loaded successfully')"
python -c "import torch, timm; model = timm.create_model('convnext_small'); state = torch.load('convnextS_1kpretrained_official_style.pth', 'cpu'); model.load_state_dict(state.get('module', state), strict=False); print('ConvNeXt-Small loaded successfully')"
jupyter nbconvert --to notebook --execute pretrain/viz_reconstruction.ipynb --output viz_reconstruction_executed.ipynb
jupyter nbconvert --to notebook --execute pretrain/viz_spconv.ipynb --output viz_spconv_executed.ipynb

# Testing / Evaluation
cd downstream_imagenet
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12348 main.py --data_path=/data/imagenet --exp_name=resnet50_finetune --exp_dir=./logs --model=resnet50 --resume_from=../resnet50_1kpretrained_timm_style.pth --epochs=1 --bs=32
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12349 main.py --data_path=/data/imagenet --exp_name=convnext_small_finetune --exp_dir=./logs --model=convnext_small --resume_from=../convnextS_1kpretrained_official_style.pth --epochs=1 --bs=16
tensorboard --logdir ./logs/tensorboard_log --port 6006 --host 0.0.0.0 &
