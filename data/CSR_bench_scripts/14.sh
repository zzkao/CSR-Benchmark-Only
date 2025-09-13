#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/YyzHarry/imbalanced-regression.git
cd imbalanced-regression
conda create -n sts python=3.6 -y
conda activate sts
pip install --upgrade pip
pip install torch torchvision tensorboard_logger numpy pandas scipy tqdm matplotlib pillow wget gdown tensorboardX scikit-learn ipdb
pip install --upgrade jupyter notebook
conda install pytorch=0.4.1 cuda92 -c pytorch -y
pip install nltk wget ipdb scikit-learn allennlp==0.5.0
pip install overrides==3.1.0
mkdir -p imdb-wiki-dir/data agedb-dir/data nyud2-dir/data sts-b-dir/glue_data/STS-B

# Data / Checkpoint / Weight Download (URL)
cd imdb-wiki-dir
python download_imdb_wiki.py
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
cd ../agedb-dir
python data/create_agedb.py
python data/preprocess_agedb.py
cd ../nyud2-dir
python download_nyud2.py
python preprocess_nyud2.py
cd ../sts-b-dir
python glove/download_glove.py
python glue_data/create_sts.py
cd ..

# Training
cd imdb-wiki-dir
python train.py --data_dir ./data --reweight none --epochs 10
python train.py --data_dir ./data --reweight sqrt_inv --epochs 10
python train.py --data_dir ./data --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --epochs 10
python train.py --data_dir ./data --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 10
python train.py --data_dir ./data --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 10
cd ../agedb-dir
python train.py --data_dir ./data --reweight none --epochs 10
python train.py --data_dir ./data --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --epochs 10
python train.py --data_dir ./data --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 10
cd ../nyud2-dir
python train.py --data_dir ./data --reweight none --epochs 10
python train.py --data_dir ./data --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --epochs 10
python train.py --data_dir ./data --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 10
cd ../sts-b-dir
python train.py --cuda 0 --reweight none --epochs 10
python train.py --cuda 0 --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --epochs 10
python train.py --cuda 0 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 10
cd ..

# Inference / Demonstration
jupyter notebook --port 8888 tutorial/tutorial.ipynb
cd imdb-wiki-dir
python train.py --data_dir ./data --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --epochs 1
cd ../agedb-dir
python train.py --data_dir ./data --loss focal_l1 --epochs 1
cd ../nyud2-dir
python train.py --data_dir ./data --reweight sqrt_inv --epochs 1
cd ../sts-b-dir
python train.py --cuda 0 --loss huber --huber_beta 0.3 --epochs 1
cd ..

# Testing / Evaluation
cd imdb-wiki-dir
python train.py --data_dir ./data --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --evaluate --resume ./checkpoints/model_best.pth
cd ../agedb-dir
python train.py --data_dir ./data --reweight inverse --evaluate --resume ./checkpoints/model_best.pth
cd ../nyud2-dir
python test.py --data_dir ./data --eval_model ./checkpoints/model_best.pth
cd ../sts-b-dir
python train.py --cuda 0 --reweight inverse --evaluate --eval_model ./checkpoints/model_best.pth
cd ..
