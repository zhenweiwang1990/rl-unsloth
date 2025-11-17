#!/bin/bash
set -e

echo "=========================================="
echo "Email Agent - Benchmark (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="email-agent-grpo"

# Default values
MODEL_PATH="${1:-}"
RUN_ID="${RUN_ID:-001}"
TEST_SET_SIZE="${TEST_SET_SIZE:-100}"
VERBOSE="${VERBOSE:-false}"
USE_CACHE_ONLY="${USE_CACHE_ONLY:-false}"

# Display usage information
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_benchmark.sh [MODEL_PATH]"
    echo ""
    echo "Environment variables:"
    echo "  RUN_ID          - Identifier for this run (default: 001)"
    echo "  TEST_SET_SIZE   - Number of queries to test (default: 100)"
    echo "  VERBOSE         - Enable detailed logs (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  ./scripts/run_benchmark.sh"
    echo ""
    echo "  # Run with verbose output (detailed logs)"
    echo "  VERBOSE=true ./scripts/run_benchmark.sh"
    echo ""
    echo "  # Run with 10 queries and verbose output"
    echo "  TEST_SET_SIZE=10 VERBOSE=true ./scripts/run_benchmark.sh"
    echo ""
    exit 0
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
    echo "Loading environment from .env file"
fi

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    GPU_FLAGS="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
else
    echo "Warning: No GPU detected. Benchmark will be slower."
fi

echo "Configuration:"
echo "  - Run ID: $RUN_ID"
echo "  - Test set size: $TEST_SET_SIZE"
echo "  - Verbose: $VERBOSE"
if [ -n "$MODEL_PATH" ]; then
    echo "  - Model: $MODEL_PATH"
else
    echo "  - Model: Base model (no fine-tuning)"
fi
echo ""

# Prepare Docker command
# Mount HuggingFace cache to reuse downloaded models
DOCKER_CMD="docker run --rm -it \
    $GPU_FLAGS \
    $ENV_FILE \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e EMAIL_DB_PATH=/workspace/data/enron_emails.db \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e RUN_ID=$RUN_ID \
    -e TEST_SET_SIZE=$TEST_SET_SIZE \
    -e VERBOSE=$VERBOSE \
    $IMAGE_NAME"

# Run benchmark
echo "Starting benchmark..."
echo ""

# Build benchmark command
BENCHMARK_CMD="python benchmark.py --limit $TEST_SET_SIZE"

# Add verbose flag if enabled
if [ "$VERBOSE" = "true" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --verbose"
fi

# Add model path if provided
if [ -n "$MODEL_PATH" ]; then
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        exit 1
    fi
    BENCHMARK_CMD="$BENCHMARK_CMD --model-path /workspace/$MODEL_PATH"
fi

# Run the benchmark
$DOCKER_CMD $BENCHMARK_CMD

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: benchmark_results_${RUN_ID}.csv"

