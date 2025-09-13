#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/MIVRC/MSRN-PyTorch.git
cd MSRN-PyTorch/MSRN
pip install torch>=0.4.0 torchvision numpy scikit-image imageio matplotlib tqdm
mkdir -p Train/dataset Test/model Test/LR/LRBI Test/HR Test/SR Test/OriginalTestData

# Data / Checkpoint / Weight Download (URL)
cd Train/dataset
wget -O DIV2K.tar "https://cv.snu.ac.kr/research/EDSR/DIV2K.tar"
tar -xf DIV2K.tar
cd ../../Test
wget -O test_datasets.zip "https://www.jianguoyun.com/p/DcrVSz0Q19ySBxiTs4oB"
unzip test_datasets.zip
wget -O original_test_datasets.zip "https://www.jianguoyun.com/p/DaSU0L4Q19ySBxi_qJAB"
unzip original_test_datasets.zip -d OriginalTestData/
cd model
wget -O MSRN_x2.pt "https://www.jianguoyun.com/p/DQpSSlQQ19ySBxjH2IYB"
wget -O MSRN_x3.pt "https://www.jianguoyun.com/p/DQpSSlQQ19ySBxjH2IYB"
wget -O MSRN_x4.pt "https://www.jianguoyun.com/p/DQpSSlQQ19ySBxjH2IYB"
cd ..

# Training
cd Train/
python main.py --template MSRN --save MSRN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset
python main.py --template MSRN --save MSRN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset
python main.py --template MSRN --save MSRN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset
cd ..

# Inference / Demonstration
cd Test/code/
python main.py --data_test MyImage --scale 2 --model MSRN --pre_train ../model/MSRN_x2.pt --test_only --save_results --chop --save "MSRN" --testpath ../LR/LRBI --testset Set5
python main.py --data_test MyImage --scale 2 --model MSRN --pre_train ../model/MSRN_x2.pt --test_only --save_results --chop --self_ensemble --save "MSRN_plus" --testpath ../LR/LRBI --testset Set5
python main.py --data_test MyImage --scale 3 --model MSRN --pre_train ../model/MSRN_x3.pt --test_only --save_results --chop --save "MSRN" --testpath ../LR/LRBI --testset Set5
python main.py --data_test MyImage --scale 3 --model MSRN --pre_train ../model/MSRN_x3.pt --test_only --save_results --chop --self_ensemble --save "MSRN_plus" --testpath ../LR/LRBI --testset Set5
python main.py --data_test MyImage --scale 4 --model MSRN --pre_train ../model/MSRN_x4.pt --test_only --save_results --chop --save "MSRN" --testpath ../LR/LRBI --testset Set5
python main.py --data_test MyImage --scale 4 --model MSRN --pre_train ../model/MSRN_x4.pt --test_only --save_results --chop --self_ensemble --save "MSRN_plus" --testpath ../LR/LRBI --testset Set5
cd ..

# Testing / Evaluation
matlab -nodisplay -nosplash -nodesktop -r "run('Prepare_TestData_HR_LR.m'); exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('Evaluate_PSNR_SSIM.m'); exit;"