# BeautyAI Inference Framework

This framework provides a scalable, professional-grade CLI interface for running inference with various language models, specializing in Arabic AI models but supporting multilingual capabilities. It features a unified command structure with support for both Hugging Face Transformers and vLLM inference backends, with various quantization options to optimize for different hardware requirements.

## Features

- **Unified CLI Interface**: Single `beautyai` command with organized subcommands for all functionality
- **Multiple Inference Backends**: Use either Hugging Face Transformers (default) or vLLM for optimized inference
- **Flexible Quantization**: Support for 4-bit/8-bit quantization with Transformers and AWQ/SqueezeLLM with vLLM
- **Multiple Model Architectures**: Support for both causal language models (e.g., Qwen, Mistral, Llama) and sequence-to-sequence models (e.g., T5, Flan-T5)
- **Interactive Chat**: Real-time streaming chat interface with conversation history
- **Benchmarking Tools**: Measure loading time, inference speed, and memory usage
- **Model Registry**: Manage multiple model configurations and easily switch between them
- **Modular Design**: Clean separation of components for easy maintenance and extension
- **Configuration Management**: Use JSON config files or command-line parameters
- **Service-Oriented Architecture**: Ready for future API integration and web interfaces
- **Backward Compatibility**: Legacy command aliases still supported

## Requirements

- NVIDIA GPU with sufficient VRAM (RTX 4090 with 24GB recommended)
- Python 3.10+
- CUDA drivers properly installed

## Installation

### Setup

```bash
# Run the setup script
chmod +x setup_beautyai.sh
./setup_beautyai.sh
```

The setup script will guide you through the installation process and ask if you want to install with vLLM support (recommended for better performance).

### Authenticate with Hugging Face (required for the model):

```bash
huggingface-cli login
```

Follow the prompts and enter your Hugging Face token from https://huggingface.co/settings/tokens

## Usage

### Quick Start

After installation, you can use the framework with the provided command-line tools:

```bash
# Activate the virtual environment
### Quick Start

After installation, you can use the framework with the unified CLI interface:

```bash
# Activate the virtual environment
source venv/bin/activate

# Show all available commands
beautyai --help

# Test the model with a simple prompt
beautyai run test

# Start an interactive chat session
beautyai run chat

# Run benchmarks
beautyai run benchmark

# Manage model configurations
beautyai model list
```

### Unified CLI Commands

The BeautyAI framework uses a unified command structure organized into four main groups:

#### Model Registry Management
```bash
# List all available models
beautyai model list

# Add a new model configuration
beautyai model add --name "mistral-7b" --model-id "mistralai/Mistral-7B-Instruct-v0.2" --engine transformers --quantization 4bit

# Show details of a specific model
beautyai model show mistral-7b

# Set a model as default
beautyai model set-default mistral-7b

# Remove a model from registry
beautyai model remove mistral-7b
```

#### System Lifecycle Management
```bash
# Load a model into memory
beautyai system load mistral-7b

# Check system status and loaded models
beautyai system status

# Unload a specific model
beautyai system unload mistral-7b

# Unload all models
beautyai system unload-all

# Clear model cache
beautyai system clear-cache
```

#### Inference Operations
```bash
# Start interactive chat
beautyai run chat --model-name mistral-7b

# Run inference tests
beautyai run test --model Qwen/Qwen3-14B

# Run performance benchmarks
beautyai run benchmark --model-name mistral-7b --output-file results.json
```

#### Configuration Management
```bash
# Show current configuration
beautyai config show

# Update configuration settings
beautyai config set default_engine vllm

# Reset configuration to defaults
beautyai config reset
```

### Legacy Commands (Backward Compatibility)

For backward compatibility, the old command structure is still supported:

```bash
# These legacy commands still work:
beautyai-test         # -> beautyai run test
beautyai-chat         # -> beautyai run chat
beautyai-benchmark    # -> beautyai run benchmark
beautyai-models list  # -> beautyai model list
```

## Supported Model Architectures

The framework supports two main types of model architectures:

### Causal Language Models (CLMs)

Causal language models like Qwen, Mistral, Llama, and GPT-style models are best suited for conversational AI and text generation tasks. These models predict the next token in a sequence and are trained with a causal attention mask.

Example models:
- Qwen/Qwen3-14B (default)
- mistralai/Mistral-7B-Instruct-v0.2
- meta-llama/Llama-2-7b-chat-hf

### Sequence-to-Sequence Models (Seq2Seq)

Sequence-to-sequence models like T5, Flan-T5, and BART are designed for transforming an input sequence into an output sequence. They excel at tasks like translation, summarization, and question-answering.

Example models:
- google/flan-t5-base
- google/flan-t5-xl

**Note**: vLLM backend only supports causal language models. For sequence-to-sequence models, the system will automatically use the Transformers backend.

### Model Selection Tips

- For chat and general text generation: Use causal language models (the default)
- For specific transformation tasks: Use sequence-to-sequence models with targeted prompts
- For large language models: Enable quantization (4bit or 8bit) to reduce memory usage
- For fastest inference: Use vLLM backend with compatible models

### Command-line Options

All commands support command-line options for customization:

```bash
# Use a different model with unified CLI
beautyai run chat --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Use vLLM backend with AWQ quantization
beautyai run chat --engine vllm --quantization awq

# Run benchmarks with custom parameters
beautyai run benchmark --input-lengths 10,100,500 --output-length 100

# Load a model with specific quantization
beautyai system load --model-name my-model --quantization 4bit
```

Use `--help` with any command to see all available options:

```bash
beautyai --help                    # Show main help
beautyai model --help              # Show model management help
beautyai run chat --help           # Show chat-specific options
beautyai system status --help      # Show status command help
```

### Configuration Files

You can also use JSON configuration files for more complex setups:

```bash
# Create a custom configuration file
cp beautyai_inference/config/default_config.json my_config.json

# Edit the file to customize settings
nano my_config.json

# Use the configuration file with any script
./scripts/chat.py --config my_config.json
```

## Project Structure

```
beautyai_inference/
├── __init__.py                  # Package initialization
├── cli/                         # Command-line interfaces
│   ├── __init__.py
│   ├── benchmark_cli.py         # CLI for benchmarking
│   ├── chat_cli.py              # CLI for interactive chat
│   ├── model_manager_cli.py     # CLI for model configuration management
│   └── test_cli.py              # CLI for testing
├── config/                      # Configuration management
│   ├── __init__.py
│   ├── config_manager.py        # Configuration dataclasses
│   ├── default_config.json      # Default configuration
│   └── model_registry.json      # Model registry storage
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
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: Works on low-memory systems (≥4GB GPUs)

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
