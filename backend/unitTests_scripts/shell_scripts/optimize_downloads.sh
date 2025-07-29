#!/bin/bash

# HuggingFace Download Optimization Script
# This script sets up environment variables for faster downloads

echo "=== HuggingFace Download Optimization ==="

# Set environment variables for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1  # Use hf_transfer for faster downloads
export HF_HUB_DOWNLOAD_TIMEOUT=120  # Increase timeout for large files
export HF_DATASETS_CACHE="/home/lumi/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/lumi/.cache/huggingface/transformers"
export HF_HOME="/home/lumi/.cache/huggingface"

# Set parallel download settings
export HF_HUB_PARALLEL_DOWNLOADS=4  # Number of parallel downloads
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

# PyTorch memory optimization for faster loading
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if hf_transfer is installed
if ! python -c "import hf_transfer" 2>/dev/null; then
    echo "Installing hf_transfer for faster downloads..."
    pip install hf_transfer
else
    echo "✅ hf_transfer is already installed"
fi

# Display current settings
echo ""
echo "Current optimization settings:"
echo "  HF_HUB_ENABLE_HF_TRANSFER: $HF_HUB_ENABLE_HF_TRANSFER"
echo "  HF_HUB_PARALLEL_DOWNLOADS: $HF_HUB_PARALLEL_DOWNLOADS"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# Test download speed
echo "Testing download speed with optimizations..."
python3 -c "
from huggingface_hub import hf_hub_download
import time
import os

print('Environment variables:')
for key in ['HF_HUB_ENABLE_HF_TRANSFER', 'HF_HUB_PARALLEL_DOWNLOADS', 'HF_HOME']:
    print(f'  {key}: {os.getenv(key)}')

print('\\nTesting small file download speed...')
start_time = time.time()
try:
    hf_hub_download('gpt2', 'config.json', cache_dir=None)
    end_time = time.time()
    print(f'✅ Download test successful in {end_time - start_time:.2f} seconds')
except Exception as e:
    print(f'❌ Download test failed: {e}')
"

echo ""
echo "To use these optimizations, source this script before running your application:"
echo "  source scripts/optimize_downloads.sh"
echo "  # Then run your BeautyAI commands"
