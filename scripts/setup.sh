#!/bin/bash
set -e

echo "=========================================="
echo "Link Search Agent GRPO Training - Setup"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="link-search-agent-grpo"

# Create necessary directories
echo "Creating directories..."
mkdir -p link_search_agent/data outputs
mkdir -p $HOME/.cache/pip $HOME/.cache/huggingface $HOME/.cache/modelscope
touch outputs/.gitkeep

echo "✓ Directories created"
echo "  - link_search_agent/data (for SQLite database)"
echo "  - outputs (for checkpoints and logs)"
echo "  - ~/.cache/pip (pip cache)"
echo "  - ~/.cache/huggingface (model cache)"
echo "  - ~/.cache/modelscope (ModelScope cache)"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo ""
echo "✓ Docker image built: $IMAGE_NAME"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from env.example..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✓ Created .env file"
    else
        echo "⚠️  env.example not found, creating minimal .env"
        cat > .env << 'EOF'
# HuggingFace Token (required for private datasets)
HF_TOKEN=

# Model configuration
MODEL_NAME=unsloth/Qwen3-30B-A3B-128K

# PostgreSQL Export Configuration
# PG_HOST=your-host.com
# PG_PORT=5432
# PG_USER=postgres
# PG_PASSWORD=your-password
# PG_DATABASE=your-database
EOF
    fi
    echo ""
    echo "⚠️  Please edit .env and configure:"
    echo "   1. HF_TOKEN (for HuggingFace dataset access)"
    echo "   2. PG_* variables (for database export)"
else
    echo ""
    echo "✓ .env file already exists"
fi

# Check if profile database exists
echo ""
if [ -f "link_search_agent/data/profiles.db" ]; then
    echo "✓ Profile database found"
    DB_SIZE=$(du -h link_search_agent/data/profiles.db | cut -f1)
    echo "  Size: $DB_SIZE"
    
    # Show row counts using Docker
    echo "  Contents:"
    docker run --rm \
        -v $(pwd)/link_search_agent/data:/workspace/link_search_agent/data \
        $IMAGE_NAME \
        sqlite3 /workspace/link_search_agent/data/profiles.db \
        "SELECT '    Profiles: ' || COUNT(*) FROM profiles; SELECT '    Experiences: ' || COUNT(*) FROM experiences; SELECT '    Educations: ' || COUNT(*) FROM educations;" 2>/dev/null || echo "    (unable to read)"
else
    echo "⚠️  Profile database not found"
    echo ""
    echo "To export the database from PostgreSQL:"
    echo "  1. Configure PG_* variables in .env"
    echo "  2. Run: ./scripts/generate_database.sh"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "All operations run inside Docker containers."
echo ""
echo "Next steps:"
echo "  1. Edit .env and configure HF_TOKEN and PG_* variables"
echo "  2. Export database: ./scripts/generate_database.sh"
echo "  3. Start training: ./scripts/run_training.sh"
echo ""
echo "Available commands:"
echo "  ./scripts/setup.sh           - This setup script"
echo "  ./scripts/build_docker.sh    - Rebuild Docker image"
echo "  ./scripts/generate_database.sh - Export PostgreSQL to SQLite"
echo "  ./scripts/run_training.sh    - Start GRPO training"
echo "  ./scripts/run_eval.sh        - Evaluate trained model"
echo "  ./scripts/run_benchmark.sh   - Run benchmark tests"
echo ""
