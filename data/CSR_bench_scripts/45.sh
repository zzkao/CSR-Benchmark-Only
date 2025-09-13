#!/bin/bash

# Environment Setup / Requirement / Installation
git clone https://github.com/xuchen-ethz/snarf.git
cd snarf
conda env create -f environment.yml
conda activate snarf
python setup.py install

# Additional dependencies
git clone https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.9.0
python setup.py develop
cd ..

# Data / Checkpoint / Weight Download (URL)
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_FEMALE.pkl
mv /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_MALE.pkl
sh ./download_data.sh
tar -xf DFaust67.tar.bz2 -C data
tar -xf MPILimits.tar.bz2 -C data

# Training
# For minimally clothed human
python train.py subject=50002
# For clothed human
python train.py datamodule=cape subject=3375 datamodule.clothing='blazerlong' +experiments=cape

# Inference / Demonstration
# Quick demo for clothed human
python demo.py expname=cape subject=3375 demo.motion_path=data/aist_demo/seqs +experiments=cape
# Demo for minimally clothed human
python demo.py expname='dfaust' subject=50002 demo.motion_path='data/aist_demo/seqs'

# Testing / Evaluation
# Preprocess datasets
python preprocess/sample_points.py --output_folder data/DFaust_processed
python preprocess/sample_points.py --output_folder data/MPI_processed --skip 10 --poseprior
# Evaluate within distribution (DFaust test split)
python test.py subject=50002
# Evaluate outside distribution (PosePrior)
python test.py subject=50002 datamodule=jointlim