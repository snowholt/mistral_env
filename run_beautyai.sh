#!/bin/bash

# Configure memory for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

# Show environment info
echo "Running with the following environment:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')"

# Run the application with 4-bit quantization
python -m mistral_inference.cli.test_cli --engine transformers --quantization 4bit
