# Mistral Inference Framework

This framework provides a scalable, professional-grade interface for running inference with Mistral AI models. It supports both Hugging Face Transformers and vLLM inference backends, with various quantization options to optimize for different hardware requirements.

## Features

- **Multiple Inference Backends**: Use either Hugging Face Transformers (default) or vLLM for optimized inference
- **Flexible Quantization**: Support for 4-bit/8-bit quantization with Transformers and AWQ/SqueezeLLM with vLLM
- **Interactive Chat**: Real-time streaming chat interface with conversation history
- **Benchmarking Tools**: Measure loading time, inference speed, and memory usage
- **Modular Design**: Clean separation of components for easy maintenance and extension
- **Configuration Management**: Use JSON config files or command-line parameters

## Requirements

- NVIDIA GPU with sufficient VRAM (RTX 4090 with 24GB recommended)
- Python 3.10+
- CUDA drivers properly installed

## Installation

### Basic Setup (Transformers)

```bash
# Run the setup script
./scripts/setup.sh
```

### vLLM Setup (Optional, for better performance)

After running the basic setup, you can add vLLM support:

```bash
./scripts/setup_vllm.sh
```

### Authenticate with Hugging Face (required for the model):

```bash
huggingface-cli login
```

Follow the prompts and enter your Hugging Face token from https://huggingface.co/settings/tokens

## Usage

### Quick Start

After installation, you can use the framework with the included scripts:

```bash
# Activate the virtual environment
source venv/bin/activate

# Test the model with a simple prompt
./scripts/test_model.py

# Start an interactive chat session
./scripts/chat.py

# Run benchmarks
./scripts/benchmark.py
```

### Command-line Options

All scripts support command-line options for customization:

```bash
# Use a different model
./scripts/chat.py --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Use vLLM backend with AWQ quantization
./scripts/chat.py --engine vllm --quantization awq

# Run benchmarks with custom parameters
./scripts/benchmark.py --input-lengths 10,100,500 --output-length 100
```

Use `--help` with any script to see all available options:

```bash
./scripts/chat.py --help
```

### Configuration Files

You can also use JSON configuration files for more complex setups:

```bash
# Create a custom configuration file
cp mistral_inference/config/default_config.json my_config.json

# Edit the file to customize settings
nano my_config.json

# Use the configuration file with any script
./scripts/chat.py --config my_config.json
```

## Project Structure

```
mistral_inference/
├── __init__.py                  # Package initialization
├── cli/                         # Command-line interfaces
│   ├── __init__.py
│   ├── benchmark_cli.py         # CLI for benchmarking
│   ├── chat_cli.py              # CLI for interactive chat
│   └── test_cli.py              # CLI for testing
├── config/                      # Configuration management
│   ├── __init__.py
│   ├── config_manager.py        # Configuration dataclasses
│   └── default_config.json      # Default configuration
├── core/                        # Core components
│   ├── __init__.py
│   ├── model_factory.py         # Factory for creating models
│   └── model_interface.py       # Abstract base class for models
├── inference_engines/           # Engine implementations
│   ├── __init__.py
│   ├── transformers_engine.py   # Hugging Face backend
│   └── vllm_engine.py           # vLLM backend
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── memory_utils.py          # Memory tracking utilities
├── benchmarks/                  # Benchmark utilities
│   └── __init__.py
scripts/                         # Convenience scripts
├── benchmark.py                 # Entry point for benchmarking
├── chat.py                      # Entry point for chat
├── setup.sh                     # Setup script
├── setup_vllm.sh                # vLLM setup script
└── test_model.py                # Entry point for testing
setup.py                         # Package installation
README.md                        # Documentation
```

## Model Compatibility

This framework is primarily designed for Mistral AI models but should work with most Hugging Face-compatible models:

### Tested Models

- **mistralai/Mistral-Small-3.1-24B-Instruct-2503**: Works with 4-bit quantization on 24GB GPUs
- **mistralai/Mixtral-8x7B-Instruct-v0.1**: Works with vLLM AWQ quantization on ≥16GB GPUs
- **mistralai/Mistral-7B-Instruct-v0.2**: Works on ≥8GB GPUs

## Quantization Details

### Transformers

The framework supports 4-bit and 8-bit quantization using the BitsAndBytes library:

```
4-bit quantization:
- ~6GB VRAM for 7B models
- ~12GB VRAM for Mixtral 8x7B models
- ~12-16GB VRAM for 24B models

8-bit quantization:
- ~10GB VRAM for 7B models
- ~20GB VRAM for Mixtral 8x7B models
- ~24GB+ VRAM for 24B models
```

### vLLM

vLLM supports AWQ and SqueezeLLM quantization:

```
AWQ quantization:
- ~5GB VRAM for 7B models
- ~10GB VRAM for Mixtral 8x7B models
- Typically faster than Transformers with 4-bit

SqueezeLLM quantization:
- Similar memory footprint to AWQ
- Different performance characteristics
```

## Troubleshooting

If you encounter issues:

1. **Out-of-memory errors**:
   - Try a different quantization method
   - Reduce `max_new_tokens`
   - Use a smaller model

2. **Slow inference**:
   - Try the vLLM backend
   - Make sure you're using CUDA with correct drivers
   - Check if other processes are using the GPU

3. **Model not found**:
   - Make sure you're authenticated with Hugging Face
   - Check if the model requires access permissions

## License and Attribution

This implementation is licensed under the MIT license.

The Mistral AI models are subject to their respective licenses. Check the [Hugging Face model card](https://huggingface.co/mistralai) for the most up-to-date licensing information.

## References

- [Mistral AI](https://mistral.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [vLLM](https://github.com/vllm-project/vllm)
- [BitsAndBytes Library](https://github.com/TimDettmers/bitsandbytes)
