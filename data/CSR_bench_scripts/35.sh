#!/bin/bash
# Environment Setup / Requirement / Installation
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv
conda install matplotlib scipy scikit-learn
conda install pyyaml easydict
conda install termcolor
git clone https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation
cd Unsupervised-Semantic-Segmentation
pip install apex einops h5py imageio joblib pillow pycocotools pycparser pyquaternion pywavelets scikit-image six tifffile toolz
mkdir -p /path/to/root/dir
mkdir -p /path/to/datasets

# Data / Checkpoint / Weight Download (URL)
wget -O pretrained_supervised_saliency.pth "https://drive.google.com/uc?export=download&id=1UkzAZMBG1U8kTqO3yhO2nTtoRNtEvyRq"
wget -O linear_finetuned_supervised_saliency.pth "https://drive.google.com/uc?export=download&id=1C2iv8wFV8MNLYLKw2E0Do2aeO-eaWNw3"
wget -O pretrained_unsupervised_saliency.pth "https://drive.google.com/uc?export=download&id=1efL1vWVcrGAqeC6OLalX8pwec41c6NZj"
wget -O linear_finetuned_unsupervised_saliency.pth "https://drive.google.com/uc?export=download&id=1y-HZTHHTyAceiFDLAraLXooGOdyQqY2Z"
wget -O pascal_voc_dataset.zip "https://drive.google.com/uc?export=download&id=1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH"

# Training
cd pretrain
python main.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml
python main.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_unsupervised_saliency_model.yml
cd ../segmentation
python linear_finetune.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_supervised_saliency.yml
python linear_finetune.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_unsupervised_saliency.yml

# Inference / Demonstration
cd segmentation
python eval.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_supervised_saliency.yml --state-dict ../linear_finetuned_supervised_saliency.pth
python eval.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_supervised_saliency.yml --state-dict ../linear_finetuned_supervised_saliency.pth --crf-postprocess
python eval.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_unsupervised_saliency.yml --state-dict ../linear_finetuned_unsupervised_saliency.pth

# Testing / Evaluation
cd segmentation
python linear_finetune.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_supervised_saliency.yml
python kmeans.py --config_env configs/env.yml --config_exp configs/kmeans/kmeans_VOCSegmentation_supervised_saliency.yml
python kmeans.py --config_env configs/env.yml --config_exp configs/kmeans/kmeans_VOCSegmentation_unsupervised_saliency.yml
python retrieval.py --config_env configs/env.yml --config_exp configs/retrieval/retrieval_VOCSegmentation_supervised_saliency.yml
python retrieval.py --config_env configs/env.yml --config_exp configs/retrieval/retrieval_VOCSegmentation_unsupervised_saliency.yml