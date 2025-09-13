#!/bin/bash
# Environment Setup / Requirement / Installation
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
pip install mamba-ssm[causal-conv1d]
pip install mamba-ssm[dev]
git clone https://github.com/state-spaces/mamba
cd mamba
pip install . --no-build-isolation
pip install lm-eval==0.4.2
sudo patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch

# Data / Checkpoint / Weight Download (URL)
# Note: Models are auto-downloaded by the generation scripts from Hugging Face

# Training
# Note: Mamba is primarily an inference repository, no training scripts provided

# Inference / Demonstration
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-370m" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --prompt "My cat wrote all this CUDA code for a new language model and" --minp 0.05 --topk 0 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --batch 8
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba2-130m" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-160m" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

# Testing / Evaluation
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-130m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 64
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-370m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 64
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba2-130m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --device cuda --batch_size 64
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-2.8b-slimpj --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 --device cuda --batch_size 64
lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-2.8b-slimpj --tasks mmlu --num_fewshot 5 --device cuda --batch_size 64
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 64