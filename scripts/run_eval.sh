#!/bin/bash
set -e

echo "=========================================="
echo "Link Search Agent Evaluation (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="link-search-agent-grpo"

# Default values
MODEL_PATH="${1:-outputs/grpo_linksearch_masked/final}"
NUM_QUERIES="${2:-100}"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Usage: $0 [model_path] [num_queries]"
    exit 1
fi

# Check if database exists
if [ ! -f "link_search_agent/data/profiles.db" ]; then
    echo "Error: Database not found at link_search_agent/data/profiles.db"
    echo "Please run: ./scripts/generate_database.sh"
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t $IMAGE_NAME .
fi

# Load environment variables
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
fi

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
fi

echo "Running evaluation..."
echo "  Model: $MODEL_PATH"
echo "  Queries: $NUM_QUERIES"
echo ""
echo "Volume mounts:"
echo "  - link_search_agent/data → /workspace/link_search_agent/data"
echo "  - outputs → /workspace/outputs"
echo "  - ~/.cache/huggingface → /root/.cache/huggingface"
echo ""

# Run evaluation in Docker
docker run --rm -it \
    $GPU_FLAGS \
    $ENV_FILE \
    -v $(pwd)/link_search_agent/data:/workspace/link_search_agent/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $HOME/.cache/pip:/root/.cache/pip \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.cache/modelscope:/root/.cache/modelscope \
    -e LINK_SEARCH_DB_PATH=/workspace/link_search_agent/data/profiles.db \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    $IMAGE_NAME \
    python eval_linksearch.py --model-path /workspace/$MODEL_PATH --num-queries $NUM_QUERIES

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
