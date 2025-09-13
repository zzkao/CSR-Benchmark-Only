#!/bin/bash
# Environment Setup / Requirement / Installation
pip install mistral-inference
cd $HOME && git clone https://github.com/mistralai/mistral-inference
cd $HOME/mistral-inference && poetry install .
pip install packaging mamba-ssm causal-conv1d transformers
pip install --upgrade mistral-common
pip install xformers>=0.0.24 simple-parsing>=0.1.5 fire>=0.6.0 mistral_common>=1.5.4 safetensors>=0.4.0 pillow>=10.3.0

# Data / Checkpoint / Weight Download (URL)
export MISTRAL_MODEL=$HOME/mistral_models
mkdir -p $MISTRAL_MODEL
export 12B_DIR=$MISTRAL_MODEL/12B_Nemo
wget https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar
mkdir -p $12B_DIR
tar -xf mistral-nemo-instruct-2407.tar -C $12B_DIR
export M7B_DIR=$MISTRAL_MODEL/7B_Instruct
wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar
mkdir -p $M7B_DIR
tar -xf mistral-7B-Instruct-v0.3.tar -C $M7B_DIR
export M8x7B_DIR=$MISTRAL_MODEL/8x7b_instruct
wget https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar
mkdir -p $M8x7B_DIR
tar -xf Mixtral-8x7B-v0.1-Instruct.tar -C $M8x7B_DIR
export M22B_CODESTRAL=$MISTRAL_MODEL/Codestral-22B-v0.1
wget https://models.mistralcdn.com/codestral-22b-v0-1/codestral-22B-v0.1.tar
mkdir -p $M22B_CODESTRAL
tar -xf codestral-22B-v0.1.tar -C $M22B_CODESTRAL
export 7B_MATHSTRAL=$MISTRAL_MODEL/mathstral-7B-v0.1
wget https://models.mistralcdn.com/mathstral-7b-v0-1/mathstral-7B-v0.1.tar
mkdir -p $7B_MATHSTRAL
tar -xf mathstral-7B-v0.1.tar -C $7B_MATHSTRAL
export 7B_CODESTRAL_MAMBA=$MISTRAL_MODEL/mamba-codestral-7B-v0.1
wget https://models.mistralcdn.com/codestral-mamba-7b-v0-1/codestral-mamba-7B-v0.1.tar
mkdir -p $7B_CODESTRAL_MAMBA
tar -xf codestral-mamba-7B-v0.1.tar -C $7B_CODESTRAL_MAMBA

# Training
# Note: Mistral Inference is an inference-only repository, no training capabilities

# Inference / Demonstration
mistral-demo $M7B_DIR
mistral-demo $12B_DIR
mistral-chat $M7B_DIR --instruct --max_tokens 256 --temperature 0.35
mistral-chat $12B_DIR --instruct --max_tokens 1024 --temperature 0.35
mistral-chat $M22B_CODESTRAL --instruct --max_tokens 256
mistral-chat $7B_MATHSTRAL --instruct --max_tokens 256
mistral-chat $7B_CODESTRAL_MAMBA --instruct --max_tokens 256
torchrun --nproc-per-node 2 --no-python mistral-demo $M8x7B_DIR
torchrun --nproc-per-node 2 --no-python mistral-chat $M8x7B_DIR --instruct

# Testing / Evaluation
python -m pytest tests
docker build deploy --build-arg MAX_JOBS=8
python -c "from mistral_inference.transformer import Transformer; from mistral_inference.generate import generate; from mistral_common.tokens.tokenizers.mistral import MistralTokenizer; from mistral_common.protocol.instruct.messages import UserMessage; from mistral_common.protocol.instruct.request import ChatCompletionRequest; print('Mistral inference test completed')"