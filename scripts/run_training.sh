#!/bin/bash
set -e

echo "=========================================="
echo "Email Agent GRPO Training (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="email-agent-grpo"

# Check if .env file exists
if [ ! -f "env.example" ]; then
    echo "Warning: env.example not found. Using default configuration."
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

# Load environment variables from .env if it exists
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
    echo "Loading environment from .env file"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    GPU_FLAGS="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
else
    echo "Warning: No GPU detected. Training will be very slow."
    GPU_FLAGS=""
fi

# Run training in Docker
echo "Starting GRPO training in Docker..."
echo ""

docker run --rm -it \
    $GPU_FLAGS \
    $ENV_FILE \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e EMAIL_DB_PATH=/workspace/data/enron_emails.db \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME \
    python train_grpo.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $(pwd)/outputs"
