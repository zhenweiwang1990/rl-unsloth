#!/bin/bash
set -e

echo "=========================================="
echo "Email Agent Evaluation (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="email-agent-grpo"

# Default values
MODEL_PATH="${1:-outputs/grpo/final}"
NUM_QUERIES="${2:-100}"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Usage: $0 [model_path] [num_queries]"
    exit 1
fi

# Check if database exists
if [ ! -f "data/enron_emails.db" ]; then
    echo "Error: Database not found at data/enron_emails.db"
    echo "Please run ./scripts/generate_database.sh first."
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

# Run evaluation in Docker
docker run --rm -it \
    $GPU_FLAGS \
    $ENV_FILE \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e EMAIL_DB_PATH=/workspace/data/enron_emails.db \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    $IMAGE_NAME \
    python eval.py --model-path /workspace/$MODEL_PATH --num-queries $NUM_QUERIES

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

