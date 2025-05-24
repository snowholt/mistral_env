#!/bin/bash
# Setup script for vLLM support in the Mistral Inference Framework

set -e  # Exit on error

VENV_DIR="venv"

echo "=== Setting up vLLM Support for Mistral Inference Framework ==="

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install vLLM
echo "Installing vLLM (optimized inference for LLMs)..."
pip install vllm

# Install the package with vLLM extras
echo "Installing package with vLLM support..."
pip install -e ".[vllm]"

if [ $? -eq 0 ]; then
    echo ""
    echo "===== vLLM Setup completed successfully! ====="
    echo ""
    echo "To use the Mistral Inference Framework with vLLM:"
    echo "1. Activate the environment:   source $VENV_DIR/bin/activate"
    echo "2. Use the --engine vllm flag with any script, e.g.:"
    echo "./scripts/chat.py --engine vllm"
    echo ""
    echo "vLLM supports different models and quantization methods:"
    echo "- AWQ quantization:            --quantization awq"
    echo "- SqueezeLLM quantization:     --quantization squeezellm"
    echo "- No quantization:             --quantization none"
    echo ""
else
    echo "Error during vLLM installation. Please check the error messages above."
    exit 1
fi
