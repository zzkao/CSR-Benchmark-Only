#!/bin/bash
# Environment Setup / Requirement / Installation
conda env create -f environment.yml
conda activate multiperson
git clone https://github.com/JiangWenPL/multiperson
cd multiperson
cd neural_renderer/
python3 setup.py install
cd ../mmcv
python3 setup.py install
cd ../mmdetection
./compile.sh
python setup.py develop
cd ../sdf
python3 setup.py install
mkdir -p mmdetection/data
mkdir -p mmdetection/work_dirs

# Data / Checkpoint / Weight Download (URL)
wget -O mmdetection/data/model_data.zip "https://drive.google.com/uc?export=download&id=1y5aKzW9WL42wTfQnv-JJ0YSIgsdb_mJn"
unzip mmdetection/data/model_data.zip -d mmdetection/data/
wget -O mmdetection/data/neutral_smpl_mean_params.h5 "https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/neutral_smpl_mean_params.h5"
wget -O panoptic_test_sequences.zip "https://drive.google.com/uc?export=download&id=1k212OS4X9DXtt_adMK5Oq8QtFOmAGHkX"
wget -O panoptic_annotations.zip "https://drive.google.com/uc?export=download&id=1OyJTn_1SaYVasb4zQcbGFhjfJpnP_sml"
wget -O mupots_annotations.zip "https://drive.google.com/uc?export=download&id=1xXjzE-aN4Q6N1ODlF4EWu3VJfe75aduF"
wget -O coco_annotations.zip "https://drive.google.com/uc?export=download&id=1xd4TmldU_NdQ8VbGnPFYHu4W5U80cCPh"
wget -O posetrack_annotations.zip "https://drive.google.com/uc?export=download&id=1Im3-Tj9BgkPapx7VB-lHaZ9UEJ-htM94"
wget -O mpi_inf_3dhp_annotations.zip "https://drive.google.com/uc?export=download&id=1P2xAtTLkHJNUhoOVtVMHfioDVynfS6-I"
wget -O mpii_annotations.zip "https://drive.google.com/uc?export=download&id=1qvJEpyJ1kbitC_KFOsA6Yvm4sA4GTcXA"
mkdir -p mmdetection/data/Panoptic
mkdir -p mmdetection/data/mupots-3d/rcnn
mkdir -p mmdetection/data/h36m/rcnn
mkdir -p mmdetection/data/coco/annotations
mkdir -p mmdetection/data/posetrack2018/rcnn
mkdir -p mmdetection/data/mpi_inf_3dhp/rcnn
mkdir -p mmdetection/data/mpii/rcnn

# Training
cd mmdetection
python3 tools/train.py configs/smpl/pretrain.py --create_dummy
python3 tools/train.py configs/smpl/pretrain.py
python3 tools/train.py configs/smpl/baseline.py --load_pretrain ./work_dirs/pretrain/latest.pth
python3 tools/train.py configs/smpl/baseline.py
python3 tools/train.py configs/smpl/tune.py --load_pretrain ./work_dirs/baseline/latest.pth
python3 tools/train.py configs/smpl/tune.py

# Inference / Demonstration
cd mmdetection
python3 tools/demo.py --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
mkdir -p demo_images
mkdir -p results

# Testing / Evaluation
cd mmdetection
python3 tools/full_eval.py configs/smpl/tune.py full_h36m --ckpt ./work_dirs/tune/latest.pth
python3 tools/full_eval.py configs/smpl/tune.py haggling --ckpt ./work_dirs/tune/latest.pth
python3 tools/full_eval.py configs/smpl/tune.py mafia --ckpt ./work_dirs/tune/latest.pth
python3 tools/full_eval.py configs/smpl/tune.py ultimatum --ckpt ./work_dirs/tune/latest.pth
python3 tools/full_eval.py configs/smpl/tune.py mupots --ckpt ./work_dirs/tune/latest.pth
python3 tools/full_eval.py configs/smpl/tune.py posetrack --ckpt ./work_dirs/tune/latest.pth
cd ../misc/preprocess_datasets/full
python panoptic.py /path/to/Panoptic/ 160422_ultimatum1 /path/to/test_sequences/160422_ultimatum1.txt
python h36m.py /path/to/h36m/ ../../../mmdetection/data/h36m/rcnn --split=train
python coco.py ../../../mmdetection/data/coco ../../../mmdetection/data/coco/rcnn
python3 posetrack.py ../../../mmdetection/data/posetrack2018/annotations/train/ ../../../mmdetection/data/posetrack2018/rcnn/train.pkl
python3 mpi_inf_3dhp.py ../../../mmdetection/data/mpi_inf_3dhp/annotations/train/ ../../../mmdetection/data/mpi_inf_3dhp/rcnn/train.pkl
python3 mpii.py ../../../mmdetection/data/mpii/annotations/train/ ../../../mmdetection/data/mpii/rcnn/train.pkl