#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/wyhuai/DDNM.git
cd DDNM
pip install numpy torch blobfile tqdm pyYaml pillow
mkdir -p exp/logs/celeba exp/logs/imagenet exp/datasets/celeba exp/datasets/imagenet exp/datasets/celeba_hq exp/datasets/solvay exp/datasets/oldphoto exp/inp_masks hq_demo/data/pretrained hq_demo/data/datasets/gts/inet256

# Data / Checkpoint / Weight Download (URL)
wget -O exp/logs/celeba/celeba_hq.ckpt "https://drive.google.com/uc?export=download&id=1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21"
wget -O exp/logs/imagenet/256x256_diffusion_uncond.pt "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
wget -O hq_demo/data/pretrained/256x256_classifier.pt "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt"
wget -O hq_demo/data/pretrained/256x256_diffusion.pt "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt"

# Training
# DDNM is zero-shot and requires no training

# Inference / Demonstration
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0 -i demo
python main.py --ni --simplified --config celeba_hq.yml --path_y solvay --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.1 -i demo_realworld_sr
python main.py --ni --simplified --config oldphoto.yml --path_y oldphoto --eta 0.85 --deg "mask_color_sr" --deg_scale 2.0 --sigma_y 0.02 -i demo_oldphoto
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "colorization" --sigma_y 0 -i demo_colorization
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "inpainting" --sigma_y 0 -i demo_inpainting
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "deblur_gauss" --sigma_y 0 -i demo_deblur
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "denoising" --sigma_y 0.1 -i demo_denoising
cd hq_demo && python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/orange.png --class 950 --deg "sr_averagepooling" --scale 4 -i orange_hq && cd ..

# Testing / Evaluation
sh evaluation.sh
