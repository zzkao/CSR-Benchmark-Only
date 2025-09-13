#!/bin/bash
# Environment Setup / Requirement / Installation
pip3 install "fschat[model_worker,webui]"
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip
pip3 install -e ".[model_worker,webui]"
pip3 install -e ".[train]"
pip3 install -e ".[model_worker,llm_judge]"
brew install rust cmake
source /opt/intel/oneapi/setvars.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export FASTCHAT_USE_MODELSCOPE=True

# Data / Checkpoint / Weight Download (URL)
python3 download_mt_bench_pregenerated.py

# Training
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py --model_name_or_path meta-llama/Llama-2-7b-hf --data_path data/dummy_conversation.json --bf16 True --output_dir output_vicuna --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 10 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True
torchrun --nproc_per_node=4 --master_port=9778 fastchat/train/train_flant5.py --model_name_or_path google/flan-t5-xl --data_path ./data/dummy_conversation.json --bf16 True --output_dir ./checkpoints_flant5_3b --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 --evaluation_strategy "no" --save_strategy "steps" --save_steps 300 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap T5Block --tf32 True --model_max_length 2048 --preprocessed_path ./preprocessed_data/processed.json --gradient_checkpointing True
deepspeed fastchat/train/train_lora.py --model_name_or_path meta-llama/Llama-2-7b-hf --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --data_path ./data/dummy_conversation.json --bf16 True --output_dir ./checkpoints --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 100 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --q_lora True --deepspeed playground/deepspeed_config_s2.json

# Inference / Demonstration
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device cpu
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --load-8bit
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device mps --load-8bit
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device xpu
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device npu
python3 -m fastchat.serve.cli --model-path lmsys/longchat-7b-32k-v1.5
python3 -m fastchat.serve.cli --model-path lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
python3 -m fastchat.serve.gradio_web_server
python3 -m fastchat.serve.gradio_web_server_multi
CPU_ISA=amx python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device cpu

# Testing / Evaluation
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
python gen_judgment.py --model-list vicuna-7b-v1.5 --parallel 2
python show_result.py --model-list vicuna-7b-v1.5
python show_result.py
python3 qa_browser.py --share
vllm serve lmsys/vicuna-7b-v1.5 --dtype auto
python gen_api_answer.py --model vicuna-7b-v1.5 --openai-api-base http://localhost:8000/v1 --parallel 50