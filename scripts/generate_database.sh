#!/bin/bash
set -e

echo "=========================================="
echo "Generating Email Database (Docker)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="email-agent-grpo"

# Check if Docker image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t $IMAGE_NAME .
fi

# Check if database already exists
if [ -f "data/enron_emails.db" ]; then
    echo "Database already exists at data/enron_emails.db"
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing database."
        exit 0
    fi
    echo "Removing existing database..."
    rm data/enron_emails.db
fi

echo "Generating database from Enron email dataset..."
echo "This may take several minutes..."
echo ""

# Run database generation in Docker
docker run --rm \
    -v $(pwd)/data:/workspace/data \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e EMAIL_DB_PATH=/workspace/data/enron_emails.db \
    $IMAGE_NAME \
    python -c "
from email_agent.data import generate_database
import os

os.environ['EMAIL_DB_PATH'] = '/workspace/data/enron_emails.db'
print('Starting database generation...')
generate_database(overwrite=True)
print('\n✓ Database generated successfully!')
print(f'Database location: /workspace/data/enron_emails.db')
"

echo ""
echo "=========================================="
echo "Database Generation Complete!"
echo "=========================================="
echo "Database saved to: $(pwd)/data/enron_emails.db"
