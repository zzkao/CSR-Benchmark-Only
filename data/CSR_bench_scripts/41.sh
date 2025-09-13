#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/cure-lab/SmoothNet.git
cd SmoothNet
export CONDA_ENV_NAME=smoothnet-env
conda create -n $CONDA_ENV_NAME python=3.6 -y
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch -y
pip install tensorboard==2.8.0 pyyaml==6.0 yacs==0.1.8 progress==1.6 smplx==0.1.28 thop==0.0.31.post2005241907 scipy==1.5.4 chumpy==0.70 opencv-python==4.6.0.66 tqdm==4.64.0 matplotlib==3.3.4 trimesh==3.12.7 pyrender==0.1.45
mkdir -p data/checkpoints data/poses data/smpl results

# Data / Checkpoint / Weight Download (URL)
wget -O data/checkpoints/aist_vibe_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK"
wget -O data/checkpoints/h36m_fcn_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=1ZketGlY4qA3kFp044T1-PaykV2llNUjB"
wget -O data/checkpoints/pw3d_spin_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k"
mkdir -p data/checkpoints/h36m_fcn_3D data/checkpoints/pw3d_spin_3D data/checkpoints/aist_vibe_3D
mv data/checkpoints/h36m_fcn_3D_checkpoint.pth.tar data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar
mv data/checkpoints/pw3d_spin_3D_checkpoint.pth.tar data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar
mv data/checkpoints/aist_vibe_3D_checkpoint.pth.tar data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar

# Training
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name aist,pw3d --estimator vibe,spin --body_representation 3D,3D --slide_window_size 8
python train_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 16
python train_smoothnet.py --cfg configs/aist_vibe_3D.yaml --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 32

# Inference / Demonstration
python visualize_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 32 --visualize_video_id 2 --output_video_path ./visualize
python visualize_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8 --visualize_video_id 1 --output_video_path ./visualize
python visualize_smoothnet.py --cfg configs/aist_vibe_3D.yaml --checkpoint data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 16 --visualize_video_id 0 --output_video_path ./visualize

# Testing / Evaluation
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name mpiinf3dhp,mpiinf3dhp --estimator tcmr,vibe --body_representation 3D,3D --slide_window_size 8 --tradition oneeuro
python eval_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8 --tradition savgol
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 16 --tradition gaus1d
python eval_smoothnet.py --cfg configs/aist_vibe_3D.yaml --checkpoint data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 8
python eval_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator rle --body_representation 2D --slide_window_size 8