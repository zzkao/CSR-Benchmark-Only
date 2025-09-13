#!/bin/bash
# Environment Setup / Requirement / Installation
python --version
conda create -n metagpt python=3.9 && conda activate metagpt
pip install --upgrade metagpt
pip install --upgrade git+https://github.com/geekan/MetaGPT.git
git clone https://github.com/geekan/MetaGPT && cd MetaGPT && pip install --upgrade -e .
pip install 'metagpt[rag]'
pip install 'metagpt[ocr]'
pip install 'metagpt[search-ddg]'
pip install 'metagpt[selenium]'
npm install -g @mermaid-js/mermaid-cli
pip install pyppeteer
pip install playwright
playwright install --with-deps chromium
export PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
metagpt --init-config

# Data / Checkpoint / Weight Download (URL)
mkdir -p ~/.metagpt
mkdir -p /opt/metagpt/{config,workspace}
docker pull metagpt/metagpt:latest
docker run --rm metagpt/metagpt:latest cat /app/metagpt/config/config2.yaml > /opt/metagpt/config/config2.yaml

# Training
# Note: MetaGPT is a multi-agent framework, not a model training system

# Inference / Demonstration
metagpt "Create a 2048 game"
metagpt "Write a cli snake game"
metagpt "Create a simple calculator"
docker run --rm --privileged -v /opt/metagpt/config/config2.yaml:/app/metagpt/config/config2.yaml -v /opt/metagpt/workspace:/app/metagpt/workspace metagpt/metagpt:latest metagpt "Write a cli snake game"
docker run --name metagpt -d --privileged -v /opt/metagpt/config/config2.yaml:/app/metagpt/config/config2.yaml -v /opt/metagpt/workspace:/app/metagpt/workspace metagpt/metagpt:latest
docker exec -it metagpt /bin/bash

# Testing / Evaluation
git clone https://github.com/geekan/MetaGPT.git
cd MetaGPT && docker build -t metagpt:custom .
python3 --version