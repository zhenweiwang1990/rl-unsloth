#!/bin/bash
set -e

echo "=========================================="
echo "Email Agent GRPO Training - Setup (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="email-agent-grpo"

# Create necessary directories
echo "Creating directories..."
mkdir -p data outputs .cache
touch data/.gitkeep outputs/.gitkeep

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from env.example..."
    cp env.example .env
    echo "✓ Created .env file. Please edit it and add your OpenAI API key."
else
    echo ""
    echo ".env file already exists."
fi

# Generate database
echo ""
echo "Do you want to generate the email database now? (y/n)"
read -p "> " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./scripts/generate_database.sh
else
    echo "Skipping database generation. Run ./scripts/generate_database.sh later."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Generate database: ./scripts/generate_database.sh (if not done)"
echo "3. Start training: ./scripts/run_training.sh"
echo ""
echo "All commands will run inside Docker containers."
echo ""
