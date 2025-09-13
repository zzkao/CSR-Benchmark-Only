#!/bin/bash
# Environment Setup / Requirement / Installation
pip install torch torchvision
pip install numpy scipy scikit-image opencv-python pillow
pip install tensorboardX pyyaml
pip install gdown
pip install --upgrade gdown
git clone https://github.com/layumi/Seg-Uncertainty
cd Seg-Uncertainty
mkdir -p data/Cityscapes/data/gtFine
mkdir -p data/Cityscapes/data/leftImg8bit
mkdir -p data/GTA5/images
mkdir -p data/GTA5/labels
mkdir -p data/synthia/RGB
mkdir -p data/synthia/GT
mkdir -p data/Oxford_Robot_ICCV19/train
mkdir -p snapshots

# Data / Checkpoint / Weight Download (URL)
gdown 1BMTTMCNkV98pjZh_rU0Pp47zeVqF3MEc
wget -O trained_model.zip "https://drive.google.com/uc?export=download&id=1smh1sbOutJwhrfK8dk-tNvonc0HLaSsw"
wget -O gta5_dataset.zip "https://download.visinf.tu-darmstadt.de/data/from_games/"
wget -O synthia_dataset.zip "http://synthia-dataset.net/download/808/"
wget -O cityscapes_dataset.zip "https://www.cityscapes-dataset.com/"
wget -O oxford_robot_dataset.zip "http://www.nec-labs.com/~mas/adapt-seg/adapt-seg.html"

# Training
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5 --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5 --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 --lambda-me-target 0 --lambda-kl-target 0.1 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0,1 --often-balance --use-se
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0,1,2 --often-balance --use-se --input-size 1280,640 --train_bn --autoaug False

# Inference / Demonstration
python generate_plabel_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth

# Testing / Evaluation
python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug/GTA5_25000.pth
python evaluate_cityscapes.py --restore-from ./trained_model.pth