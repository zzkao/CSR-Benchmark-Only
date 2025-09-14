#!/bin/bash
# Environment Setup / Requirement / Installation
pip install knowledge-storm
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# No explicit download commands found - datasets are available on HuggingFace but accessed via API

# Training
# No training commands found - STORM is an inference-only system

# Inference / Demonstration
python examples/storm_examples/run_storm_wiki_gpt.py --output-dir ./output --retriever bing --do-research --do-generate-outline --do-generate-article --do-polish-article
python examples/costorm_examples/run_costorm_gpt.py --output-dir ./output --retriever bing

# Testing / Evaluation
# No explicit testing commands found in the README.
