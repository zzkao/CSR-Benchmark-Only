#!/bin/bash
# Environment Setup / Requirement / Installation
python --version
python -m pip install --upgrade pip
pip install gptcache
git clone -b dev https://github.com/zilliztech/GPTCache.git
cd GPTCache
pip install -r requirements.txt
python setup.py install
pip install numpy cachetools requests
export OPENAI_API_KEY=YOUR_API_KEY
echo $OPENAI_API_KEY

# Data / Checkpoint / Weight Download (URL)
# Note: GPTCache automatically downloads required models and embeddings when needed

# Training
# Note: GPTCache is a caching library, not a model training framework

# Inference / Demonstration
python examples/adapter/api.py
gptcache_server
python -c "from gptcache import cache; from gptcache.adapter import openai; cache.init(); cache.set_openai_key(); print('GPTCache initialized successfully')"
python -c "from gptcache import cache; from gptcache.adapter.api import put, get; from gptcache.processor.pre import get_prompt; cache.init(pre_embedding_func=get_prompt); put('hello', 'foo'); print(get('hello'))"

# Testing / Evaluation
python examples/benchmark/benchmark_sqlite_faiss_onnx.py
python -c "import gptcache; print('GPTCache version:', gptcache.__version__)"
python -c "from gptcache import cache; from gptcache.embedding import Onnx; from gptcache.manager import CacheBase, VectorBase, get_data_manager; from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation; onnx = Onnx(); data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=onnx.dimension)); cache.init(embedding_func=onnx.to_embeddings, data_manager=data_manager, similarity_evaluation=SearchDistanceEvaluation()); print('Semantic cache initialized successfully')"