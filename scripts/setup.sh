#!/bin/bash
set -e

echo "=========================================="
echo "Link Search Agent GRPO Training - Setup (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="link-search-agent-grpo"

# Create necessary directories
echo "Creating directories..."
mkdir -p link_search_agent/data outputs .cache
touch outputs/.gitkeep

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from env.example..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✓ Created .env file. Please edit it and add your HF_TOKEN."
    else
        cat > .env << 'EOF'
# HuggingFace Token (required for private datasets)
HF_TOKEN=

# Model configuration
MODEL_NAME=unsloth/Qwen3-30B-A3B-128K

# Training configuration
TRAIN_DATASET_SIZE=1000
EVAL_DATASET_SIZE=100
MAX_STEPS=200
LEARNING_RATE=1e-5
PER_DEVICE_TRAIN_BATCH_SIZE=2
NUM_GENERATIONS=3

# Agent configuration
MAX_TURNS=15
MAX_TOKENS=4096
MAX_PROFILES=10

# WandB (optional)
WANDB_PROJECT=link-search-grpo
WANDB_MODE=online
EOF
        echo "✓ Created .env file. Please edit it and add your HF_TOKEN."
    fi
else
    echo ""
    echo ".env file already exists."
fi

# Check if profile database exists
echo ""
if [ -f "link_search_agent/data/profiles.db" ]; then
    echo "✓ Profile database found at link_search_agent/data/profiles.db"
    DB_SIZE=$(du -h link_search_agent/data/profiles.db | cut -f1)
    echo "  Size: $DB_SIZE"
else
    echo "⚠️  Profile database not found."
    echo ""
    echo "To export the database from PostgreSQL, run:"
    echo "  python scripts/export_to_sqlite.py"
    echo ""
    echo "Make sure to set the following environment variables:"
    echo "  PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your HF_TOKEN (for HuggingFace dataset access)"
echo "2. Export database: python scripts/export_to_sqlite.py (if not done)"
echo "3. Start training: ./scripts/run_training.sh"
echo ""
echo "All commands will run inside Docker containers."
echo ""
