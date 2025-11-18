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
echo "Email Agent GRPO Training (Docker)"
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
IMAGE_NAME="email-agent-grpo"

# Load environment variables
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
    echo "✓ Loading environment from .env file"
else
    echo "ℹ️  No .env file found, using default settings"
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
echo "Cache directory: $HOME/.cache/huggingface"
ls -lah "$HOME/.cache/huggingface" 2>/dev/null || echo "⚠️  Cache directory not found"
echo ""
echo "💡 Tip: You can change modes with:"
echo "   ./scripts/run_training.sh simple|rollout|masked"
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
    -e HF_DATASETS_CACHE=/root/.cache/huggingface \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME \
    python train_grpo.py --mode $MODE

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Mode: $MODE"
echo "Checkpoints saved to: $(pwd)/outputs/grpo_$MODE"
echo ""
echo "To use a different mode, run:"
echo "  ./scripts/run_training.sh simple   # Fast testing"
echo "  ./scripts/run_training.sh rollout  # Real rollouts"
echo "  ./scripts/run_training.sh masked   # Token masking (default)"
echo ""