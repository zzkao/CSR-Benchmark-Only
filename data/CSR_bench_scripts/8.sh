#!/bin/bash
# Environment Setup / Requirement / Installation.
conda create -n hybrik python=3.8 -y
conda activate hybrik
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
git clone https://github.com/Jeff-sjtu/HybrIK.git
cd HybrIK
pip install pycocotools
pip install numpy six terminaltables scipy cython opencv-python==4.1.2.30 matplotlib tqdm easydict chumpy pyyaml tb-nightly future ffmpeg-python joblib
python setup.py develop

# Data / Checkpoint / Weight Download (URL)
curl -L -o model_files.zip "https://drive.google.com/uc?export=download&id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV"
unzip model_files.zip
mkdir -p pretrained_models
curl -L -o pretrained_models/hybrik_hrnet.pth "https://drive.google.com/uc?export=download&id=1C-jRnay38mJG-0O4_um82o1t7unC1zeT"
curl -L -o pretrained_models/hybrik_resnet34.pth "https://drive.google.com/uc?export=download&id=19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx"
curl -L -o pretrained_models/hybrikx_rle_hrnet.pth "https://drive.google.com/uc?export=download&id=1R0WbySXs_vceygKg_oWeLMNAZCEoCadG"
mkdir -p data/h36m/annotations data/h36m/images data/pw3d/json data/pw3d/imageFiles data/3dhp data/coco/annotations data/coco/train2017 data/coco/val2017

# Training
python ./scripts/train_smpl_cam.py --nThreads 8 --launcher pytorch --rank 0 --dist-url tcp://127.0.0.1:23456 --exp-id test_3dpw --cfg configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml --seed 123123
python ./scripts/train_smpl_cam.py --nThreads 8 --launcher pytorch --rank 0 --dist-url tcp://127.0.0.1:23457 --exp-id test_hrnet --cfg configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml --seed 123123

# Inference / Demonstration
mkdir -p examples res_dance res
python scripts/demo_video.py --video-name examples/dance.mp4 --out-dir res_dance --save-pk --save-img --gpu 0
python scripts/demo_image.py --img-dir examples --out-dir res --gpu 0
python scripts/demo_video_x.py --video-name examples/dance.mp4 --out-dir res_dance_x --save-pk --save-img --gpu 0

# Testing / Evaluation
python ./scripts/validate_smpl_cam.py --batch 8 --gpus 0 --world-size 1 --flip-test --launcher pytorch --rank 0 --dist-url tcp://127.0.0.1:23458 --cfg configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml --checkpoint pretrained_models/hybrik_hrnet.pth
python ./scripts/validate_smpl_cam.py --batch 8 --gpus 0 --world-size 1 --flip-test --launcher pytorch --rank 0 --dist-url tcp://127.0.0.1:23459 --cfg configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml --checkpoint pretrained_models/hybrik_resnet34.pth
