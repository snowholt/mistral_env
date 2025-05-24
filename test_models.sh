#!/bin/bash

# Script to test different models with varying memory requirements
# This script helps users find a model that works with their hardware configuration

echo "===== Mistral Model Memory Test ====="
echo "This script will help you find a model that works with your GPU's memory."

# Check for Python and venv
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+ and try again."
    exit 1
fi

# Check for GPU
echo "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: 'nvidia-smi' not found. You may not have an NVIDIA GPU or drivers installed."
    echo "This package requires a CUDA-capable GPU for optimal performance."
    read -p "Continue anyway? (y/N): " continue_without_gpu
    if [[ ! "$continue_without_gpu" =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
else
    echo "GPU detected. Checking memory..."
    total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    echo "GPU Memory: $total_memory MB"
    
    # Set model recommendations based on available memory
    if [ "$total_memory" -ge 24000 ]; then
        echo "High memory GPU detected (≥24 GB). You can try the full Mistral Small model."
        default_model="mistral-small"
    elif [ "$total_memory" -ge 16000 ]; then
        echo "Medium memory GPU detected (≥16 GB). The Mixtral-8x7B model should work with quantization."
        default_model="mixtral-8x7b"
    elif [ "$total_memory" -ge 8000 ]; then
        echo "Low memory GPU detected (≥8 GB). Mistral-7B should work with 4-bit quantization."
        default_model="mistral-7b"
    else
        echo "Very low memory GPU detected (<8 GB). Only small models will work."
        default_model="tiny-llama"
    fi
fi

# Check for installed package
if ! pip show mistral-inference > /dev/null 2>&1; then
    echo "Error: mistral-inference package not installed."
    echo "Please run ./setup_mistral.sh first to install the package."
    exit 1
fi

# Create the model registry
echo -e "\n=== Setting up model registry ==="
cat << EOF > models.json
{
  "default_model": "${default_model}",
  "models": {
    "mistral-small": {
      "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
      "engine_type": "vllm",
      "quantization": "4bit",
      "dtype": "float16",
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95,
      "do_sample": true,
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "name": "mistral-small",
      "description": "Mistral Small 3.1 24B (4-bit quantized, requires 24GB+ VRAM)"
    },
    "mixtral-8x7b": {
      "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
      "engine_type": "vllm",
      "quantization": "awq",
      "dtype": "float16",
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95,
      "do_sample": true,
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "name": "mixtral-8x7b",
      "description": "Mixtral-8x7B (AWQ quantized, requires 16GB+ VRAM)"
    },
    "mistral-7b": {
      "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "float16",
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95,
      "do_sample": true,
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "name": "mistral-7b",
      "description": "Mistral-7B (4-bit quantized, requires 8GB+ VRAM)"
    },
    "tiny-llama": {
      "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "engine_type": "transformers",
      "quantization": "8bit",
      "dtype": "float16",
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95,
      "do_sample": true,
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "name": "tiny-llama",
      "description": "TinyLlama 1.1B (small model for low memory, requires 4GB+ VRAM)"
    }
  }
}
EOF

echo "Model registry created with recommended defaults for your hardware."
echo "Default model set to: $default_model"

# List models
echo -e "\n=== Available Models ==="
mistral-models list --models-file models.json

# Ask which model to test
echo -e "\n=== Testing a Model ==="
echo "Which model would you like to test? (default: $default_model)"
read -p "Enter model name (or press Enter for default): " test_model
test_model=${test_model:-$default_model}

# Test the model
echo -e "\nTesting model: $test_model"
echo "This will use the selected model to generate a short response."
echo "If successful, it means the model works with your hardware configuration."

# Simple benchmark
mistral-benchmark --model-name $test_model --models-file models.json --input-lengths 10 --output-length 20

# Final instructions
echo -e "\n=== Next Steps ==="
echo "To use these models in the future, use the --models-file option:"
echo
echo "  mistral-chat --model-name $test_model --models-file models.json"
echo "  mistral-benchmark --model-name $test_model --models-file models.json"
echo
echo "You can also add more models to your registry with:"
echo
echo "  mistral-models add --name \"custom-model\" --model-id \"path/to/model\" --models-file models.json"
echo
echo "Happy inferencing!"
