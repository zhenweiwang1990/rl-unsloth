#!/bin/bash
set -e

echo "=========================================="
echo "Generating Profile Database (PostgreSQL → SQLite)"
echo "=========================================="
echo ""

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

# Check for required environment variables
if [ -z "$PG_HOST" ] || [ -z "$PG_USER" ] || [ -z "$PG_PASSWORD" ] || [ -z "$PG_DATABASE" ]; then
    echo "PostgreSQL connection details required."
    echo ""
    echo "Please set the following environment variables:"
    echo "  PG_HOST      - PostgreSQL host"
    echo "  PG_PORT      - PostgreSQL port (default: 5432)"
    echo "  PG_USER      - PostgreSQL user"
    echo "  PG_PASSWORD  - PostgreSQL password"
    echo "  PG_DATABASE  - PostgreSQL database name"
    echo ""
    echo "Example:"
    echo "  export PG_HOST=your-host.com"
    echo "  export PG_PORT=5432"
    echo "  export PG_USER=postgres"
    echo "  export PG_PASSWORD=your-password"
    echo "  export PG_DATABASE=your-database"
    echo "  ./scripts/generate_database.sh"
    echo ""
    
    # Try to load from .env
    if [ -f ".env" ]; then
        echo "Attempting to load from .env file..."
        set -a
        source .env
        set +a
        
        if [ -z "$PG_HOST" ] || [ -z "$PG_USER" ] || [ -z "$PG_PASSWORD" ] || [ -z "$PG_DATABASE" ]; then
            echo "PostgreSQL variables not found in .env"
            exit 1
        fi
        echo "✓ Loaded from .env"
    else
        exit 1
    fi
fi

echo "PostgreSQL Connection:"
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

# Run export script
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
    
    # Show row counts
    echo "Contents:"
    sqlite3 link_search_agent/data/profiles.db "SELECT 'Profiles: ' || COUNT(*) FROM profiles;"
    sqlite3 link_search_agent/data/profiles.db "SELECT 'Experiences: ' || COUNT(*) FROM experiences;"
    sqlite3 link_search_agent/data/profiles.db "SELECT 'Educations: ' || COUNT(*) FROM educations;"
else
    echo "Error: Database file was not created"
    exit 1
fi
