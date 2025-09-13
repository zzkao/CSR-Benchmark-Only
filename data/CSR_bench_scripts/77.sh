#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/SJTU-IPADS/PowerInfer
cd PowerInfer
pip install -r requirements.txt
pip install numpy>=1.24.4 sentencepiece>=0.1.98 transformers>=4.33.2
cmake -S . -B build
cmake --build build --config Release
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake -S . -B build -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release

# Data / Checkpoint / Weight Download (URL)
huggingface-cli download --resume-download --local-dir ReluLLaMA-7B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
huggingface-cli download --resume-download --local-dir ReluLLaMA-13B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF
huggingface-cli download --resume-download --local-dir ReluFalcon-40B --local-dir-use-symlinks False PowerInfer/ReluFalcon-40B-PowerInfer-GGUF
huggingface-cli download --resume-download --local-dir Bamboo-base-7B --local-dir-use-symlinks False PowerInfer/Bamboo-base-v0.1-gguf
python convert.py --outfile ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf ./SparseLLM/ReluLLaMA-7B ./PowerInfer/ReluLLaMA-7B-Predictor
python convert-dense.py --outfile ./Bamboo-base-v0.1-gguf/bamboo-7b-base-v0.1.gguf --outtype f16 ./Bamboo-base-v0.1

# Training
# Note: PowerInfer is an inference engine, not a training framework

# Inference / Demonstration
./build/bin/main -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"
./build/bin/main -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time" --vram-budget 8
./build/bin/main -m ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.q4.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"
./build/bin/main -m ./Bamboo-base-v0.1-gguf/bamboo-7b-base-v0.1.gguf -n 128 -t 8 -p "Once upon a time" -ngl 12
./examples/server/server -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -c 2048 --vram-budget 8
curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json" --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'

# Testing / Evaluation
./build/bin/quantize ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.q4.powerinfer.gguf Q4_0
./build/bin/quantize ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.powerinfer.gguf ./ReluFalcon-40B-PowerInfer-GGUF/falcon-40b-relu.q4.powerinfer.gguf Q4_0
./examples/perplexity/perplexity -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -f ./examples/perplexity/wikitext-2-raw/wiki.test.raw
./build/bin/main -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time" --reset-gpu-index
./build/bin/main -m ./ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time" --disable-gpu-index