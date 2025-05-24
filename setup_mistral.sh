#!/bin/bash
# Consolidated setup script for Mistral Inference Framework

set -e  # Exit on error

VENV_DIR="venv"

echo "=== Mistral Inference Framework Setup ==="
echo "This script will create a virtual environment and install required packages."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are properly installed."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Ask user whether to install with vLLM support
read -p "Do you want to install with vLLM support? (recommended for performance) [y/N]: " install_vllm
install_vllm=${install_vllm:-n}

if [[ $install_vllm =~ ^[Yy]$ ]]; then
    echo "Installing package with vLLM support..."
    pip install -e ".[dev,vllm]"
    
    echo "Installing vLLM (optimized inference for LLMs)..."
    pip install vllm
else
    echo "Installing package without vLLM support..."
    pip install -e ".[dev]"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "===== Setup completed successfully! ====="
    echo ""
    echo "To use the Mistral Inference Framework:"
    echo "1. Activate the environment:   source $VENV_DIR/bin/activate"
    echo "2. Run the test script:        mistral-test"
    echo "3. Run the chat interface:     mistral-chat"
    echo "4. Run benchmarks:             mistral-benchmark"
    echo ""
    echo "For help, use the --help flag with any command, e.g.:"
    echo "mistral-chat --help"
    echo ""
    if [[ $install_vllm =~ ^[Yy]$ ]]; then
        echo "vLLM is installed. To use the vLLM engine, use the --engine vllm flag, e.g.:"
        echo "mistral-chat --engine vllm"
        echo ""
        echo "vLLM supports different quantization methods:"
        echo "- AWQ quantization:            --quantization awq"
        echo "- SqueezeLLM quantization:     --quantization squeezellm"
        echo "- No quantization:             --quantization none"
        echo ""
    else
        echo "If you want to add vLLM support later, reinstall with:"
        echo "pip install -e \".[vllm]\""
        echo ""
    fi
else
    echo "Error during package installation. Please check the error messages above."
    exit 1
fi
