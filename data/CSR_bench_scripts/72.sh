#!/bin/bash
# Environment Setup / Requirement / Installation
pip install graphrag
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra dev
mkdir -p ./ragtest/input
sudo apt-get install llvm-9 llvm-9-dev
sudo apt-get install python3.10-dev
export LLVM_CONFIG=/usr/bin/llvm-config-9
export GRAPHRAG_LLM_THREAD_COUNT=1
export GRAPHRAG_EMBEDDING_THREAD_COUNT=1

# Data / Checkpoint / Weight Download (URL)
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt -o ./ragtest/input/book.txt
graphrag init --root ./ragtest

# Training
graphrag index --root ./ragtest

# Inference / Demonstration
graphrag query --root ./ragtest --method global --query "What are the top themes in this story?"
graphrag query --root ./ragtest --method local --query "Who is Scrooge and what are his main relationships?"
uv run poe query --root ./ragtest --method global --query "What are the main characters?"
uv run poe query --root ./ragtest --method local --query "What is the setting of the story?"

# Testing / Evaluation
uv run poe test_unit
uv run poe test_integration
uv run poe test_smoke
uv run poe check
uv run poe format
uv build
./scripts/start-azurite.sh
uv run semversioner add-change -t patch -d "Test changes for evaluation."