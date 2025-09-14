#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
bash prepare_data.sh

# Training
# No training commands found - 3DDFA_V2 is an inference-only system

# Inference / Demonstration
python demo.py -f examples/inputs/01.jpg
python demo.py -f examples/inputs/*.jpg
python demo.py -f examples/inputs/
python demo.py -f examples/inputs/video.mp4
python demo.py -f 0
python demo.py -f examples/inputs/01.jpg --onnx
python demo.py -f examples/inputs/01.jpg --pose
python demo.py -f examples/inputs/01.jpg --pncc
python demo.py -f examples/inputs/01.jpg --uv_tex
python demo.py -f examples/inputs/01.jpg --ply
python demo.py -f examples/inputs/01.jpg --obj
python demo.py -f examples/inputs/01.jpg --onnx --pose --ply --obj
python video_kit.py -f examples/inputs/video.mp4 --output_dir results/
python batch_process.py --input_dir examples/inputs/ --output_dir results/

# Testing / Evaluation
python latency.py
python latency.py --onnx
