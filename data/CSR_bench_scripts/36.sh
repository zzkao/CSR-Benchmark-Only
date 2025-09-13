#!/bin/bash
# Environment Setup / Requirement / Installation
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install pyyaml scikit-image scikit-learn opencv
git clone https://github.com/inrainbws/transmomo.pytorch
cd transmomo.pytorch
pip install -r requirements.txt
pip install easydict imageio-ffmpeg matplotlib numpy Pillow protobuf PyYAML scikit-image scikit-learn scipy tensorboardX torch>=1.2.0 torchvision tqdm
mkdir -p data/mixamo/36_800_24/train
mkdir -p data/mixamo/36_800_24/test
mkdir -p data/solo_dance/train
mkdir -p out

# Data / Checkpoint / Weight Download (URL)
wget -O data/mixamo_data.zip "https://drive.google.com/uc?export=download&id=1lMa-4Bspn2_XV4wqo_s9Bfa35-19UAkB"
unzip data/mixamo_data.zip -d data/mixamo/
wget -O data/solo_dance_data.zip "https://drive.google.com/uc?export=download&id=1366FaH0W2VYVW26ZbQJUp1x5GgMyMXuo"
unzip data/solo_dance_data.zip -d data/solo_dance/
wget -O pretrained_models.zip "https://drive.google.com/drive/folders/1xZ2Pw7ObrDUIH89ipH1diyFZJxeXNDd8/view?usp=sharing"
mkdir -p transmomo_mixamo_36_800_24/checkpoints
sh scripts/preprocess.sh
python scripts/preprocess.py --data_dir data/mixamo/36_800_24/train -i 32 -s 64 -p 0
python scripts/preprocess.py --data_dir data/mixamo/36_800_24/test -i 60 -s 120 -p 0
python scripts/rotate_test_set.py --data_dir data/mixamo/36_800_24/test --out_dir data/mixamo/36_800_24/test_random_rotate
python scripts/preprocess_solo_dance.py --data_dir data/solo_dance/train -i 32 -s 64 -p 0

# Training
python train.py --config configs/transmomo.yaml
python train.py --config configs/transmomo_solo_dance.yaml

# Inference / Demonstration
python infer_pair.py --config configs/transmomo.yaml --checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt --source data/mixamo/36_800_24/test/sample_a.npy --target data/mixamo/36_800_24/test/sample_b.npy --source_width 1280 --source_height 720 --target_height 1920 --target_width 1080
python render_interpolate.py --config configs/transmomo.yaml --checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt

# Testing / Evaluation
python test.py --config configs/transmomo.yaml --checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt --out_dir transmomo_mixamo_36_800_24_results
python scripts/compute_mse.py --in_dir transmomo_mixamo_36_800_24_results
python test.py --config configs/transmomo_solo_dance.yaml --checkpoint transmomo_solo_dance/checkpoints/autoencoder_00200000.pt --out_dir transmomo_solo_dance_results
python scripts/compute_mse.py --in_dir transmomo_solo_dance_results