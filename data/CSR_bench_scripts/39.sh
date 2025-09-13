#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/Owen-Liuyuxuan/visualDet3D
cd visualDet3D
pip3 install -r requirement.txt
pip3 install numpy torch torchvision pillow scikit-image fire matplotlib opencv-python numba easydict tensorflow cython tqdm pyquaternion
chmod +x make.sh
./make.sh
mkdir -p workdirs
mkdir -p workdirs/Mono3D
mkdir -p workdirs/Mono3D/log
mkdir -p workdirs/Mono3D/checkpoint
mkdir -p workdirs/Mono3D/output
mkdir -p workdirs/Mono3D/output/training
mkdir -p workdirs/Mono3D/output/validation
mkdir -p data/kitti_obj/training
mkdir -p data/kitti_obj/testing

# Data / Checkpoint / Weight Download (URL)
wget -O pretrained_mono3d.pth "https://github.com/Owen-Liuyuxuan/visualDet3D/releases/download/1.0/GroundAware_pretrained.pth"
wget -O pretrained_stereo3d.pth "https://github.com/Owen-Liuyuxuan/visualDet3D/releases/download/1.1/YOLOStereo3D_pretrained.pth"
wget -O kitti_training.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
wget -O kitti_testing.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip"
wget -O kitti_labels.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
wget -O kitti_calib.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"

# Training
cd config
cp Yolo3D_example mono3d_config.py
cp Stereo3D_example stereo3d_config.py
cp monodepth_config.py depth_config.py
cd ..
chmod +x launchers/det_precompute.sh
chmod +x launchers/train.sh
chmod +x launchers/eval.sh
chmod +x launchers/disparity_precompute.sh
./launchers/det_precompute.sh config/mono3d_config.py train
./launchers/det_precompute.sh config/mono3d_config.py test
./launchers/train.sh config/mono3d_config.py 0 mono3d_experiment
./launchers/det_precompute.sh config/stereo3d_config.py train
./launchers/det_precompute.sh config/stereo3d_config.py test
./launchers/disparity_precompute.sh config/stereo3d_config.py false
./launchers/train.sh config/stereo3d_config.py 0 stereo3d_experiment
./launchers/train.sh config/depth_config.py 0 depth_experiment

# Inference / Demonstration
python3 scripts/train.py --config=config/mono3d_config.py --experiment_name=demo_mono3d
python3 scripts/train.py --config=config/stereo3d_config.py --experiment_name=demo_stereo3d
python3 scripts/train.py --config=config/depth_config.py --experiment_name=demo_depth

# Testing / Evaluation
./launchers/eval.sh config/mono3d_config.py 0 workdirs/Mono3D/checkpoint/latest.pth validation
./launchers/eval.sh config/mono3d_config.py 0 workdirs/Mono3D/checkpoint/latest.pth test
./launchers/eval.sh config/stereo3d_config.py 0 workdirs/Stereo3D/checkpoint/latest.pth validation
./launchers/eval.sh config/stereo3d_config.py 0 workdirs/Stereo3D/checkpoint/latest.pth test
./launchers/eval.sh config/depth_config.py 0 workdirs/Depth/checkpoint/latest.pth validation
python3 scripts/eval.py --config=config/mono3d_config.py --gpu=0 --checkpoint_path=pretrained_mono3d.pth --split_to_test=validation
python3 scripts/eval.py --config=config/stereo3d_config.py --gpu=0 --checkpoint_path=pretrained_stereo3d.pth --split_to_test=validation
python3 scripts/imdb_precompute_3d.py --config=config/mono3d_config.py
python3 scripts/imdb_precompute_test.py --config=config/mono3d_config.py
python3 scripts/disparity_compute.py --config=config/stereo3d_config.py --use_point_cloud=false