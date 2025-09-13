#!/bin/bash
# Environment Setup / Requirement / Installation
export OPENAI_API_KEY=YOUR_API_KEY
pip install tree-of-thoughts-llm
git clone https://github.com/princeton-nlp/tree-of-thought-llm
cd tree-of-thought-llm
pip install -r requirements.txt
pip install -e .
pip install aiohttp==3.8.4 aiosignal==1.3.1 async-timeout==4.0.2 attrs==23.1.0 backoff==2.2.1 certifi==2023.5.7 charset-normalizer==3.1.0 frozenlist==1.3.3 idna==3.4 mpmath==1.3.0 multidict==6.0.4 numpy==1.24.3 openai==0.27.7 requests==2.31.0 sympy==1.12 tqdm==4.65.0 urllib3==2.0.2 yarl==1.9.2 pandas==2.0.3

# Data / Checkpoint / Weight Download (URL)
# Note: This framework uses built-in tasks and data, no external downloads needed

# Training
# Note: Tree of Thought is an inference/reasoning framework, not a training system

# Inference / Demonstration
python run.py --task game24 --task_start_index 900 --task_end_index 910 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 910 --naive_run --prompt_sample standard --n_generate_sample 10 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 910 --naive_run --prompt_sample cot --n_generate_sample 10 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task text --task_start_index 0 --task_end_index 10 --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample 5 --n_evaluate_sample 5 --n_select_sample 1 --prompt_sample cot --temperature 1.0 --backend gpt-3.5-turbo
python run.py --task text --task_start_index 0 --task_end_index 10 --naive_run --prompt_sample standard --n_generate_sample 10 --temperature 1.0 --backend gpt-3.5-turbo
sh scripts/game24/standard_sampling.sh --backend gpt-3.5-turbo
sh scripts/game24/cot_sampling.sh --backend gpt-3.5-turbo
sh scripts/game24/bfs.sh --backend gpt-3.5-turbo
sh scripts/text/standard_sampling.sh --backend gpt-3.5-turbo
sh scripts/text/bfs.sh --backend gpt-3.5-turbo

# Testing / Evaluation
python run.py --task game24 --task_start_index 900 --task_end_index 1000 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 1000 --naive_run --prompt_sample standard --n_generate_sample 100 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task text --task_start_index 0 --task_end_index 100 --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample 5 --n_evaluate_sample 5 --n_select_sample 1 --prompt_sample cot --temperature 1.0 --backend gpt-3.5-turbo
python run.py --task crosswords --task_start_index 0 --task_end_index 10 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7