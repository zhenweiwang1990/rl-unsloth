#!/bin/bash
set -e

# Parse command line arguments
MODE="${1:-masked}"  # Default to masked mode

# Validate mode
if [[ ! "$MODE" =~ ^(simple|rollout|masked)$ ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Available modes:"
    echo "  simple   - Fast training with heuristic rewards (for testing)"
    echo "  rollout  - Training with real agent rollouts"
    echo "  masked   - Full implementation with token-level masking (RECOMMENDED, default)"
    echo ""
    exit 1
fi

echo "=========================================="
echo "Link Search Agent GRPO Training (Docker)"
echo "Mode: $MODE"
echo "=========================================="
echo ""

case $MODE in
    simple)
        echo "ℹ️  Simple mode: Fast training with heuristic rewards"
        echo "   Use this for quick testing and validation"
        ;;
    rollout)
        echo "ℹ️  Rollout mode: Training with real agent rollouts"
        echo "   More accurate but slower than simple mode"
        ;;
    masked)
        echo "ℹ️  Masked mode: Full token-level masking (RECOMMENDED)"
        echo "   Most accurate, only trains on model-generated tokens"
        ;;
esac
echo ""

# Docker image name
IMAGE_NAME="link-search-agent-grpo"

# Load environment variables
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
    echo "✓ Loading environment from .env file"
else
    echo "ℹ️  No .env file found, using default settings"
fi

# Check if profile database exists
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
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Training mode: $MODE"
echo ""
echo "Volume mounts:"
echo "  - link_search_agent/data → /workspace/link_search_agent/data"
echo "  - outputs → /workspace/outputs"
echo "  - ~/.cache → /root/.cache (HuggingFace, pip)"
echo ""
echo "💡 Tip: You can change modes with:"
echo "   ./scripts/run_training.sh simple|rollout|masked"
echo ""

docker run -d --restart always \
    --name link-search-training \
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
    -e HF_DATASETS_CACHE=/root/.cache/huggingface \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME \
    python train_grpo_linksearch.py --mode $MODE --enable-detailed-logging

echo ""
echo "✓ Training container started: link-search-training"
echo ""
echo "To view logs:     docker logs -f link-search-training"
echo "To stop:          docker stop link-search-training"
echo "To remove:        docker rm -f link-search-training"
