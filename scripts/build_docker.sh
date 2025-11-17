#!/bin/bash
set -e

echo "=========================================="
echo "Email Agent - Docker Build"
echo "=========================================="
echo ""

# Default values
IMAGE_NAME="${IMAGE_NAME:-email-agent-grpo}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NO_CACHE="${NO_CACHE:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --name NAME       Docker image name (default: email-agent-grpo)"
            echo "  --tag TAG         Docker image tag (default: latest)"
            echo "  --no-cache        Build without using cache"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  IMAGE_NAME        Docker image name (can be overridden by --name)"
            echo "  IMAGE_TAG         Docker image tag (can be overridden by --tag)"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --name my-agent --tag v1.0"
            echo "  $0 --no-cache"
            echo "  IMAGE_NAME=my-agent IMAGE_TAG=dev $0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Configuration:"
echo "  - Image name: $IMAGE_NAME"
echo "  - Image tag: $IMAGE_TAG"
echo "  - Full name: $FULL_IMAGE_NAME"
echo "  - No cache: $NO_CACHE"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Warning: requirements.txt not found"
    echo "The Docker build may fail"
fi

# Ensure cache directories exist on host
echo "Setting up cache directories..."
mkdir -p "$HOME/.cache/pip"
mkdir -p "$HOME/.cache/huggingface"
mkdir -p "$HOME/.cache/transformers"
echo "  ✓ Cache directories ready"
echo ""

# Check Docker version and BuildKit support
DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
echo "Docker version: $DOCKER_VERSION"
echo ""

# Build the Docker image
echo "Starting Docker build..."
echo "=================================================="

BUILD_CMD="DOCKER_BUILDKIT=1 docker build"

if [ "$NO_CACHE" = "true" ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

BUILD_CMD="$BUILD_CMD -t $FULL_IMAGE_NAME ."

echo "Build command: $BUILD_CMD"
echo ""

# Execute the build
eval $BUILD_CMD

BUILD_STATUS=$?

echo ""
echo "=================================================="

if [ $BUILD_STATUS -eq 0 ]; then
    echo "✓ Docker build completed successfully!"
    echo ""
    
    # Show image information
    echo "Image information:"
    docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -2
    echo ""
    
    echo "Cache directories (mounted at runtime):"
    echo "  - Host pip cache: ~/.cache/pip"
    echo "  - Host HF cache: ~/.cache/huggingface"
    echo "  - Container pip cache: /root/.cache/pip"
    echo "  - Container HF cache: /root/.cache/huggingface"
    echo ""
    
    echo "Next steps:"
    echo "  1. Generate database: ./scripts/generate_database.sh"
    echo "  2. Run training: ./scripts/run_training.sh"
    echo "  3. Run benchmark: ./scripts/run_benchmark.sh"
    echo ""
    echo "Or run container directly:"
    echo "  docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it \\"
    echo "    -v ~/.cache/pip:/root/.cache/pip \\"
    echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "    -v \$(pwd)/data:/workspace/data \\"
    echo "    -v \$(pwd)/outputs:/workspace/outputs \\"
    echo "    --env-file .env \\"
    echo "    $FULL_IMAGE_NAME bash"
else
    echo "✗ Docker build failed with status $BUILD_STATUS"
    exit $BUILD_STATUS
fi

echo ""
echo "=========================================="

