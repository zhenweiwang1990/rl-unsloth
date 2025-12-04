#!/bin/bash
set -e

echo "=========================================="
echo "Generating Profile Database (PostgreSQL → SQLite)"
echo "=========================================="
echo ""

# Docker image name
IMAGE_NAME="link-search-agent-grpo"

# Check if Docker image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t $IMAGE_NAME .
fi

# Check if database already exists
if [ -f "link_search_agent/data/profiles.db" ]; then
    echo "Database already exists at link_search_agent/data/profiles.db"
    DB_SIZE=$(du -h link_search_agent/data/profiles.db | cut -f1)
    echo "Current size: $DB_SIZE"
    echo ""
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing database."
        exit 0
    fi
    echo "Removing existing database..."
    rm link_search_agent/data/profiles.db
fi

# Load environment variables
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
    echo "✓ Loading environment from .env file"
else
    echo "⚠️  No .env file found"
fi

# Check for required environment variables
echo ""
echo "PostgreSQL connection details:"

# Try to read from .env if exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

if [ -z "$PG_HOST" ] || [ -z "$PG_USER" ] || [ -z "$PG_PASSWORD" ] || [ -z "$PG_DATABASE" ]; then
    echo "❌ PostgreSQL connection details required."
    echo ""
    echo "Please set the following in .env file:"
    echo "  PG_HOST=your-host.com"
    echo "  PG_PORT=5432"
    echo "  PG_USER=postgres"
    echo "  PG_PASSWORD=your-password"
    echo "  PG_DATABASE=your-database"
    exit 1
fi

echo "  Host: $PG_HOST"
echo "  Port: ${PG_PORT:-5432}"
echo "  User: $PG_USER"
echo "  Database: $PG_DATABASE"
echo ""

# Create data directory if it doesn't exist
mkdir -p link_search_agent/data

echo "Exporting database from PostgreSQL..."
echo "This may take several minutes..."
echo ""

# Run export script in Docker
docker run --rm \
    $ENV_FILE \
    -v $(pwd)/link_search_agent/data:/workspace/link_search_agent/data \
    -v $(pwd)/scripts:/workspace/scripts \
    -e PG_HOST=$PG_HOST \
    -e PG_PORT=${PG_PORT:-5432} \
    -e PG_USER=$PG_USER \
    -e PG_PASSWORD=$PG_PASSWORD \
    -e PG_DATABASE=$PG_DATABASE \
    $IMAGE_NAME \
    python scripts/export_to_sqlite.py

echo ""
echo "=========================================="
echo "Database Generation Complete!"
echo "=========================================="

if [ -f "link_search_agent/data/profiles.db" ]; then
    DB_SIZE=$(du -h link_search_agent/data/profiles.db | cut -f1)
    echo "Database saved to: link_search_agent/data/profiles.db"
    echo "Size: $DB_SIZE"
    echo ""
    
    # Show row counts using Docker
    echo "Contents:"
    docker run --rm \
        -v $(pwd)/link_search_agent/data:/workspace/link_search_agent/data \
        $IMAGE_NAME \
        sqlite3 /workspace/link_search_agent/data/profiles.db \
        "SELECT 'Profiles: ' || COUNT(*) FROM profiles; SELECT 'Experiences: ' || COUNT(*) FROM experiences; SELECT 'Educations: ' || COUNT(*) FROM educations;"
else
    echo "Error: Database file was not created"
    exit 1
fi
