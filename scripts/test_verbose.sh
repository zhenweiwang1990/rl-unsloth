#!/bin/bash
# Quick test script to verify verbose logging works

set -e

echo "=================================================="
echo "Testing Verbose Logging Feature"
echo "=================================================="
echo ""
echo "This will run a benchmark with:"
echo "  - 2 queries only"
echo "  - Verbose mode enabled"
echo "  - Detailed logs for debugging"
echo ""

# Check if database exists
if [ ! -f "data/enron_emails.db" ]; then
    echo "Error: Database not found at data/enron_emails.db"
    echo "Please run ./scripts/generate_database.sh first."
    exit 1
fi

# Set environment variables for quick test
export VERBOSE=true
export TEST_SET_SIZE=2

# Run the benchmark
echo "Starting test..."
echo ""

./scripts/run_benchmark.sh

echo ""
echo "=================================================="
echo "Test Complete!"
echo "=================================================="
echo ""
echo "If you saw detailed logs for 2 queries, the verbose"
echo "logging feature is working correctly!"
echo ""
echo "To run full benchmark with verbose logging:"
echo "  VERBOSE=true TEST_SET_SIZE=10 ./scripts/run_benchmark.sh"
echo ""

