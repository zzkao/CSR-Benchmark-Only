#!/bin/bash
# Environment Setup / Requirement / Installation
pip install llama-cookbook
pip install llama-cookbook[tests,auditnlg,vllm,langchain]
git clone git@github.com:meta-llama/llama-cookbook.git
cd llama-cookbook
pip install -U pip setuptools
pip install -e .
pip install -e .[tests,auditnlg,vllm]
huggingface-cli login
pip freeze | grep transformers
git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
export CUDA_VISIBLE_DEVICES=0

# Data / Checkpoint / Weight Download (URL)
wget -P ../../src/llama_cookbook/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

# Training
FSDP_CPU_RAM_EFFICIENT_LOADING=1 python -m llama_cookbook.finetuning --use_peft --peft_method lora --quantization 8bit --model_name meta-llama/Llama-3.2-8B --output_dir ./peft_model --dataset samsum_dataset --num_epochs 1
torchrun --nnodes 1 --nproc_per_node 1 -m llama_cookbook.finetuning --enable_fsdp --model_name meta-llama/Llama-3.2-8B --use_peft --peft_method lora --output_dir ./peft_model --dataset samsum_dataset --num_epochs 1
python -m llama_cookbook.finetuning --use_peft --peft_method lora --quantization 8bit --dataset alpaca_dataset --model_name meta-llama/Llama-3.2-8B --output_dir ./peft_model --num_epochs 1
python -m llama_cookbook.finetuning --use_peft --peft_method lora --quantization 8bit --dataset grammar_dataset --model_name meta-llama/Llama-3.2-8B --output_dir ./peft_model --num_epochs 1

# Inference / Demonstration
python multi_modal_infer.py --image_path "path/to/image.jpg" --prompt_text "Describe this image" --model_name "meta-llama/Llama-3.2-11B-Vision-Instruct"
python multi_modal_infer.py --model_name "meta-llama/Llama-3.2-11B-Vision-Instruct" --gradio_ui
python inference.py --model_name meta-llama/Llama-3.2-8B --prompt_file samsum_prompt.txt --use_auditnlg
python chat_completion/chat_completion.py --model_name meta-llama/Llama-3.2-8B --prompt_file chat_completion/chats.json --quantization 8bit --use_auditnlg
python inference.py --model_name meta-llama/Llama-3.2-8B --prompt_file samsum_prompt.txt --use_auditnlg --use_fast_kernels

# Testing / Evaluation
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 8B --output_dir /output/path
python -m llama_cookbook.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path ./model_checkpoints --consolidated_model_path ./converted_model --HF_model_path_or_name meta-llama/Llama-3.2-8B