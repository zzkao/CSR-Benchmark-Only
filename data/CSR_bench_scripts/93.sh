#!/bin/bash

# Environment Setup / Requirement / Installation
# Install package from GitHub (requires python>=3.8)
pip install git+https://github.com/lxuechen/private-transformers.git
pip install pytest

# Run tests to verify installation (requires pytest and GPU)
pytest -s tests

# Data / Checkpoint / Weight Download (URL)
# No explicit data download commands in README
# Models are downloaded automatically through Hugging Face

# Training
# No direct training commands in README
# Training is done through Python API, example commands would be in examples/ directory
# Check examples/image_classification/main.py and other example folders for specific training scripts

# Inference / Demonstration
# No explicit inference commands in README
# Models can be used through Python API as shown in README examples

# Testing / Evaluation
# Main test command
pytest -s tests

# Run specific test files (examples)
pytest -s tests/test_privacy_engine.py
pytest -s tests/test_supported_models.py
