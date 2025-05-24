#!/bin/bash
# Setup script for Mistral Inference Framework

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

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e ".[dev]"

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/test_model.py scripts/chat.py scripts/benchmark.py

if [ $? -eq 0 ]; then
    echo ""
    echo "===== Setup completed successfully! ====="
    echo ""
    echo "To use the Mistral Inference Framework:"
    echo "1. Activate the environment:   source $VENV_DIR/bin/activate"
    echo "2. Run the test script:        ./scripts/test_model.py"
    echo "3. Run the chat interface:     ./scripts/chat.py"
    echo "4. Run benchmarks:             ./scripts/benchmark.py"
    echo ""
    echo "For help, use the --help flag with any script, e.g.:"
    echo "./scripts/chat.py --help"
    echo ""
    echo "To install vLLM support, run:"
    echo "./scripts/setup_vllm.sh"
    echo ""
else
    echo "Error during package installation. Please check the error messages above."
    exit 1
fi
