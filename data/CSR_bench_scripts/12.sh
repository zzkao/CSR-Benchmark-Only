#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/wpeebles/gangealing.git
cd gangealing
export PYTHONPATH="${PYTHONPATH}:${PWD}"
conda env create -f environment.yml
conda activate gg
pip install pillow scipy torchvision moviepy numpy opencv-python pandas plotly tensorboard tqdm wget lmdb ray ninja scikit-learn termcolor imageio==2.4.1
mkdir -p data assets/averages assets/masks assets/objects

# Data / Checkpoint / Weight Download (URL)
python prepare_data.py --input_is_lmdb --lsun_category cat --out data/lsun_cats --size 512 --max_images 1000
python prepare_data.py --spair_category cat --spair_split test --out data/spair_cats_test --size 256
python prepare_data.py --cub_acsm --out data/cub_val --size 256
./process_video.sh data/sample_video.mp4
python prepare_data.py --path data/video_frames --out data/my_video_dataset --pad center --size 1024

# Training
torchrun --nproc_per_node=1 train.py --ckpt cat --load_G_only --padding_mode border --vis_every 1000 --ckpt_every 10000 --iter 50000 --tv_weight 1000 --loss_fn vgg_ssl --exp-name lsun_cats_demo --batch 2
torchrun --nproc_per_node=1 train.py --ckpt celeba --load_G_only --padding_mode border --gen_size 128 --vis_every 1000 --ckpt_every 10000 --iter 50000 --tv_weight 2500 --ndirs 512 --inject 6 --loss_fn lpips --exp-name celeba_demo --batch 2
torchrun --nproc_per_node=1 train.py --ckpt bicycle --load_G_only --padding_mode border --vis_every 1000 --ckpt_every 10000 --iter 50000 --tv_weight 1000 --loss_fn vgg_ssl --exp-name bicycles_demo --batch 2
python train_cluster_classifier.py --ckpt car --real_data_path data/lsun_cars --real_size 512 --batch 8 --lr 0.001 --iter 10000

# Inference / Demonstration
python applications/vis_correspondence.py --ckpt cat --real_data_path data/lsun_cats --vis_in_stages --real_size 512 --output_resolution 512 --resolution 256 --label_path assets/masks/cat_mask.png --dset_indices 100 200 300 400
python applications/mixed_reality.py --ckpt cat --objects --label_path assets/objects/cat/cat_cartoon.png --sigma 0.3 --opacity 1 --real_size 1024 --resolution 4096 --real_data_path data/my_video_dataset --no_flip_inference --save_frames
python applications/propagate_to_images.py --ckpt cat --real_data_path data/lsun_cats --real_size 512 --dset_indices 100 200 300 400 500
python applications/propagate_to_images.py --ckpt cat --real_data_path data/lsun_cats --real_size 512 --label_path assets/objects/cat/cat_vr_headset.png --objects -s 0.3 -o 1 --resolution 4096 --dset_indices 100 200 300
python applications/propagate_to_images.py --ckpt cat --real_data_path data/lsun_cats --real_size 512 --n_mean 1000 --dset_indices 0

# Testing / Evaluation
torchrun --nproc_per_node=1 applications/pck.py --ckpt cat --real_data_path data/spair_cats_test --real_size 256
torchrun --nproc_per_node=1 applications/pck.py --ckpt cub --real_data_path data/cub_val --real_size 256 --num_pck_pairs 1000 --transfer_both_ways --vis_transfer
torchrun --nproc_per_node=1 applications/flow_scores.py --ckpt cat --real_data_path data/lsun_cats --real_size 512 --no_flip_inference
python prepare_data.py --path data/raw_images --out data/new_lmdb_data --pad none --size 0
torchrun --nproc_per_node=1 applications/congeal_dataset.py --ckpt cat --real_data_path data/new_lmdb_data --out data/my_new_aligned_dataset --real_size 0 --flow_scores data/lsun_cats/flow_scores.pt --fraction_retained 0.25 --output_resolution 512
