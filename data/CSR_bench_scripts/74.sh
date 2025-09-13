#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/SWE-agent/SWE-agent.git
cd SWE-agent
python -m pip install --upgrade pip && pip install --editable .
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs
which python
python --version
sweagent --help
python -m sweagent --help

# Data / Checkpoint / Weight Download (URL)
# Note: SWE-agent uses API models, no model weights to download

# Training
# Note: SWE-agent is an agent framework, not a model training system

# Inference / Demonstration
sweagent run --config config/default.yaml --agent.model.name gpt-4o --problem_statement "Fix the syntax error in the Python function" --repo https://github.com/SWE-agent/test-repo
sweagent run-batch --config config/default.yaml --agent.model.name gpt-4o --agent.model.per_instance_cost_limit 2.00 --instances.type swe_bench --instances.subset lite --instances.split dev --instances.slice :3 --instances.shuffle=True
sweagent run-batch --config config/default.yaml --agent.model.name gpt-4o --num_workers 3 --agent.model.per_instance_cost_limit 2.00 --instances.type swe_bench --instances.subset lite --instances.split dev --instances.slice :3 --instances.shuffle=True
sweagent run-batch --config config/default_mm_with_images.yaml --agent.model.name claude-sonnet-4-20250514 --agent.model.per_instance_cost_limit 2.00 --instances.type swe_bench --instances.subset multimodal --instances.split dev --instances.slice :3 --instances.shuffle=True
sweagent run-batch --config config/default.yaml --agent.model.name gpt-4o --instances.type file --instances.path instances.yaml --instances.slice :3 --instances.shuffle=True

# Testing / Evaluation
sweagent run-batch --config config/default.yaml --agent.model.name gpt-4o --agent.model.per_instance_cost_limit 2.00 --instances.type swe_bench --instances.subset lite --instances.split dev --instances.slice :3 --instances.shuffle=True --evaluate=True
sweagent merge-preds
git switch v0.7