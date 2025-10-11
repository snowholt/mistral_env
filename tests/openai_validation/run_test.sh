#!/bin/bash
# Quick runner script for OpenAI validation test
# Usage: ./run_test.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting OpenAI API Validation Test${NC}"
echo ""

# Activate virtual environment
source /home/lumi/beautyai/backend/venv/bin/activate

# Run test
python test_openai_validation.py

echo ""
echo -e "${GREEN}✅ Test completed successfully!${NC}"
echo ""
echo "Results saved to: validation_results.json"
