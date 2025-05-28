# BeautyAI Inference Framework

A scalable, professional-grade inference framework for various language models, specializing in Arabic AI models but supporting multilingual capabilities. The framework features both unified CLI and REST API interfaces with support for multiple inference backends, quantization options, and comprehensive model lifecycle management.

## 🚀 Key Features

### Core Capabilities
- **Unified CLI Interface**: Single `beautyai` command with organized subcommands for all functionality
- **REST API**: FastAPI-based web service for programmatic access with comprehensive endpoints
- **Multiple Inference Backends**: Hugging Face Transformers (default) and vLLM for optimized inference
- **Flexible Quantization**: 4-bit/8-bit with Transformers, AWQ/SqueezeLLM with vLLM
- **Multi-Architecture Support**: Causal language models (Qwen, Mistral, Llama) and sequence-to-sequence models (T5, Flan-T5)
- **Interactive Chat**: Real-time streaming chat interface with session management
- **Performance Tools**: Comprehensive benchmarking and memory monitoring capabilities
- **Model Registry**: Centralized configuration management with validation and versioning

### Architecture Highlights
- **Service-Oriented Architecture**: 15+ specialized services for maximum modularity and testability
- **Factory Pattern**: Intelligent model engine selection with automatic fallback strategies
- **Singleton Pattern**: Centralized model lifecycle management with cross-process state tracking
- **Configuration Management**: JSON-based configuration with validation, backup, and migration support
- **Memory Optimization**: GPU memory monitoring with automatic cleanup and cache management
- **API-Ready Design**: Services built for both CLI and web interface deployment

## 📋 Requirements

- **Hardware**: NVIDIA GPU with sufficient VRAM (RTX 4090 with 24GB recommended)
- **Software**: Python 3.10+, CUDA drivers properly installed
- **Access**: Hugging Face account with valid token for model access

## 🔧 Installation

### Quick Setup

```bash
# Clone and navigate to the project directory
cd beautyai

# Run the automated setup script
chmod +x setup_beautyai.sh
./setup_beautyai.sh
```

The setup script will guide you through the installation process and ask if you want to install with vLLM support (recommended for better performance).

### Hugging Face Authentication

```bash
# Authenticate with Hugging Face (required for most models)
huggingface-cli login
```

Follow the prompts and enter your Hugging Face token from https://huggingface.co/settings/tokens

### Development Installation

```bash
# For development with all dependencies
pip install -e ".[dev,api,vllm]"

# Install pre-commit hooks
pre-commit install
```

## 🚀 Quick Start

### CLI Interface

After installation, activate the virtual environment and use the unified CLI:

```bash
# Activate the virtual environment
source venv/bin/activate

# Show all available commands and help
beautyai --help

# Quick model test with default configuration
beautyai run test

# Start an interactive chat session
beautyai run chat

# Run performance benchmarks
beautyai run benchmark

# List available models in registry
beautyai model list

# Check system status and memory usage
beautyai system status
```

### REST API

Start the FastAPI web server for programmatic access:

```bash
# Start the API server (development mode)
uvicorn beautyai_inference.api.app:app --reload --host 0.0.0.0 --port 8000

# Start with custom configuration
uvicorn beautyai_inference.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📚 Command Reference

### Model Registry Management
The model registry provides centralized configuration management for all your AI models:

```bash
# List all available models in the registry
beautyai model list

# Add a new model configuration
beautyai model add --name "mistral-7b" \
                   --model-id "mistralai/Mistral-7B-Instruct-v0.2" \
                   --engine transformers \
                   --quantization 4bit \
                   --description "Mistral 7B with 4-bit quantization"

# Show detailed information about a specific model
beautyai model show mistral-7b

# Update an existing model configuration
beautyai model update mistral-7b --quantization 8bit --engine vllm

# Set a model as the default for inference operations
beautyai model set-default mistral-7b

# Remove a model from the registry
beautyai model remove mistral-7b --clear-cache
```

### System Lifecycle Management
Manage model loading, memory usage, and system resources:

```bash
# Load a model into GPU memory
beautyai system load mistral-7b

# Check comprehensive system status
beautyai system status

# List all currently loaded models
beautyai system list-loaded

# Unload a specific model from memory
beautyai system unload mistral-7b

# Unload all models to free memory
beautyai system unload-all

# Clear model cache files
beautyai system clear-cache mistral-7b
```

### Inference Operations
Run various types of inference tasks:

```bash
# Start interactive chat with default model
beautyai run chat

# Chat with a specific model and custom settings
beautyai run chat --model-name mistral-7b \
                  --temperature 0.7 \
                  --max-tokens 512

# Run quick model tests with sample prompts
beautyai run test --model Qwen/Qwen3-14B

# Run comprehensive performance benchmarks
beautyai run benchmark --model-name mistral-7b \
                       --input-lengths 10,100,500 \
                       --output-length 100 \
                       --output-file results.json

# Save current chat session
beautyai run save-session --session-id "my-session" \
                          --output-file session.json

# Load a previously saved chat session
beautyai run load-session --input-file session.json
```

### Configuration Management
Manage framework settings and configurations:

```bash
# Show current configuration
beautyai config show

# Update specific configuration values
beautyai config set default_engine vllm
beautyai config set gpu_memory_utilization 0.85

# Validate configuration against schema
beautyai config validate

# Reset configuration to defaults
beautyai config reset --confirm

# Create configuration backup
beautyai config backup --backup-dir my_backups \
                       --label "pre-update-backup"

# Restore from backup
beautyai config restore backups/config_20250525_120000.json

# Migrate configuration to new format
beautyai config migrate --backup --verbose
```

## 🌐 REST API Reference

The BeautyAI framework provides a comprehensive REST API for programmatic access to all functionality:

### API Endpoints Overview

**Health & Status**
- `GET /health` - Service health check
- `GET /system/status` - System status and memory usage
- `GET /system/memory` - Detailed memory statistics

**Model Management**
- `GET /models` - List all models in registry
- `POST /models` - Add new model to registry
- `GET /models/{name}` - Get model details
- `PUT /models/{name}` - Update model configuration
- `DELETE /models/{name}` - Remove model from registry
- `POST /models/{name}/load` - Load model into memory
- `DELETE /models/{name}/unload` - Unload model from memory

**Inference Operations**
- `POST /inference/chat` - Start chat session
- `POST /inference/test` - Run model tests
- `POST /inference/benchmark` - Run performance benchmarks
- `POST /inference/sessions` - Save chat session
- `GET /inference/sessions/{id}` - Load chat session

**Configuration Management**
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `PUT /config` - Bulk update configuration
- `DELETE /config` - Reset configuration
- `POST /config/validate` - Validate configuration
- `POST /config/backup` - Create configuration backup
- `POST /config/restore` - Restore from backup

### Example API Usage

```python
import requests

# API base URL
base_url = "http://localhost:8000"

# List available models
response = requests.get(f"{base_url}/models")
models = response.json()

# Add a new model
model_data = {
    "name": "qwen3-model",
    "model_id": "Qwen/Qwen3-14B",
    "engine_type": "transformers",
    "quantization": "4bit",
    "description": "Qwen 14B with 4-bit quantization"
}
response = requests.post(f"{base_url}/models", json=model_data)

# Start a chat session
chat_data = {
    "model_name": "qwen3-model",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "generation_config": {
        "temperature": 0.7,
        "max_new_tokens": 512
    }
}
response = requests.post(f"{base_url}/inference/chat", json=chat_data)
```

### Authentication

The API supports JWT-based authentication for production deployments:

```python
# Login to get access token
auth_data = {"username": "admin", "password": "secure_password"}
response = requests.post(f"{base_url}/auth/login", json=auth_data)
token = response.json()["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{base_url}/models", headers=headers)
```

## 🔧 Supported Model Architectures

The framework supports a wide range of model architectures with intelligent backend selection:

### Causal Language Models (CLMs)
**Best for**: Chat, text generation, code completion, general-purpose tasks

**Supported Models**:
- **Qwen Series**: Qwen/Qwen3-14B (default), Qwen/Qwen3-7B, Qwen/Qwen3-72B
- **Mistral Family**: mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-Instruct-v0.1
- **Llama Models**: meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf
- **Code Models**: Qwen/CodeQwen1.5-7B-Chat, codellama/CodeLlama-7b-Instruct-hf

**Optimization**: Compatible with both Transformers and vLLM backends, full quantization support

### Sequence-to-Sequence Models (Seq2Seq)
**Best for**: Translation, summarization, question-answering, structured text tasks

**Supported Models**:
- **T5 Family**: google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl
- **BART Models**: facebook/bart-large-cnn, facebook/bart-large-xsum
- **mT5 Models**: google/mt5-base, google/mt5-large (multilingual support)

**Note**: vLLM backend only supports causal models. Seq2seq models automatically use Transformers backend with fallback.

### Arabic & Multilingual Specialization
**Arabic-Optimized Models**:
- **AceGPT**: FreedomIntelligence/AceGPT-7B-chat (Arabic-focused)
- **Jais**: core42/jais-13b-chat (Arabic large language model)
- **AraBART**: UBC-NLP/AraBART (Arabic text generation)

**Multilingual Support**:
- **mBERT derivatives**: bert-base-multilingual-cased
- **XLM-RoBERTa**: xlm-roberta-base, xlm-roberta-large
- **mT5**: google/mt5-base (100+ languages)

### Smart Model Selection
The framework automatically handles:
- **Backend Selection**: vLLM for causal models, Transformers for seq2seq
- **Architecture Detection**: Automatic model type identification
- **Quantization Compatibility**: Backend-appropriate quantization methods
- **Memory Optimization**: Automatic memory management based on available VRAM
- **Fallback Strategies**: Graceful degradation when preferred backend unavailable

## ⚙️ Advanced Configuration

### Quantization Strategies

**Transformers Backend**:
```bash
# 4-bit quantization (recommended for large models)
beautyai model add --name qwen-4bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 4bit

# 8-bit quantization (balance of speed and quality)
beautyai model add --name qwen-8bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 8bit
```

**vLLM Backend**:
```bash
# AWQ quantization (fastest inference)
beautyai model add --name qwen-awq --model-id Qwen/Qwen3-14B \
                   --engine vllm --quantization awq

# SqueezeLLM quantization (alternative method)
beautyai model add --name qwen-squeeze --model-id Qwen/Qwen3-14B \
                   --engine vllm --quantization squeezellm
```

### Memory Requirements

**GPU Memory Usage Guidelines**:
```
Model Size    | 4-bit | 8-bit | AWQ   | Full Precision
7B models     | ~6GB  | ~10GB | ~5GB  | ~28GB
13B models    | ~10GB | ~18GB | ~9GB  | ~52GB
Mixtral 8x7B  | ~12GB | ~20GB | ~10GB | ~90GB
70B models    | ~40GB | ~70GB | ~35GB | ~280GB
```

### Custom Configuration Files

Create specialized configuration files for different deployment scenarios:

**config/production.json**:
```json
{
  "default_engine": "vllm",
  "gpu_memory_utilization": 0.90,
  "tensor_parallel_size": 2,
  "generation_config": {
    "temperature": 0.1,
    "max_new_tokens": 1024,
    "do_sample": true
  },
  "api_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  }
}
```

**config/development.json**:
```json
{
  "default_engine": "transformers",
  "force_cpu": false,
  "quantization": "4bit",
  "generation_config": {
    "temperature": 0.7,
    "max_new_tokens": 512
  },
  "logging": {
    "level": "DEBUG",
    "console": true
  }
}
```

Use custom configs:
```bash
# Use production configuration
beautyai --config config/production.json run chat

# Use development configuration  
beautyai --config config/development.json model list
```

## 🏗️ Project Architecture

The BeautyAI framework follows a modern, layered architecture designed for scalability and maintainability:

### Service-Oriented Architecture

**15+ Specialized Services**:
```
📁 services/
├── 🏗️ base/              # Base service infrastructure and interfaces
├── 🔧 model/             # Model registry, lifecycle, and validation services  
├── 🚀 inference/         # Chat, testing, benchmarking, and session services
├── ⚙️ config/            # Configuration, validation, backup, and migration services
└── 💾 system/            # Memory monitoring, cache management, and status services
```

**API Layer**:
```
📁 api/
├── 🌐 endpoints/         # REST API route handlers (models, inference, config, system)
├── 🔌 adapters/          # Service integration adapters for API
├── 🛡️ middleware/        # Authentication, logging, and error handling
├── 📊 schemas/           # Request/response data models
└── 🔐 auth.py           # JWT-based authentication system
```

**Core Engine Layer**:
```
📁 core/
├── 🏭 model_factory.py   # Factory pattern for engine creation
├── 🔄 model_manager.py   # Singleton model lifecycle management
└── 📋 model_interface.py # Abstract base class for inference engines
```

**CLI Interface**:
```
📁 cli/
├── 🎯 unified_cli.py     # Main CLI entry point with command routing
├── ⚙️ argument_config.py # Standardized argument handling
└── 🔌 handlers/          # CLI adapters for service integration
```

### Key Design Patterns

**1. Factory Pattern**: Intelligent model engine selection
- Automatic backend selection (Transformers vs vLLM)
- Architecture-aware quantization selection
- Graceful fallback strategies

**2. Singleton Pattern**: Centralized resource management
- Cross-process model state tracking
- GPU memory optimization
- Configuration persistence

**3. Adapter Pattern**: Interface unification
- CLI-to-service bridge
- API-to-service integration
- Backward compatibility layer

**4. Service Layer**: Business logic isolation
- Single responsibility principle
- Dependency injection ready
- Test-friendly design

## 🧪 Testing & Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_unified_cli.py -v
python -m pytest tests/test_cli_integration.py -v

# Run with coverage reporting
python -m pytest tests/ --cov=beautyai_inference --cov-report=html
```

### Development Commands
```bash
# Install in development mode
pip install -e ".[dev]"

# Run code formatting
black beautyai_inference/
isort beautyai_inference/

# Run linting
flake8 beautyai_inference/
mypy beautyai_inference/

# Run security checks
bandit -r beautyai_inference/
```

### Adding New Models

1. **Add to Model Registry**:
```bash
beautyai model add --name "my-new-model" \
                   --model-id "organization/model-name" \
                   --engine transformers \
                   --quantization 4bit \
                   --description "Description of the model"
```

2. **Test the Model**:
```bash
# Load and test the model
beautyai system load my-new-model
beautyai run test --model-name my-new-model
```

3. **Validate Configuration**:
```bash
# Ensure configuration is valid
beautyai config validate
```

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use type hints, add comprehensive docstrings
2. **Architecture**: Use service-oriented patterns, maintain separation of concerns
3. **Testing**: Add unit tests for new features, integration tests for CLI commands
4. **Documentation**: Update README and inline documentation for new features
5. **Backward Compatibility**: Maintain compatibility with existing CLI interfaces

## 🔍 Troubleshooting

### Common Issues & Solutions

**1. Out-of-Memory Errors**
```bash
# Try different quantization methods
beautyai model update my-model --quantization 4bit

# Use CPU fallback for testing
beautyai run chat --force-cpu

# Check memory usage
beautyai system status

# Reduce token limits
beautyai run chat --max-tokens 256
```

**2. Model Loading Issues**
```bash
# Verify model exists in registry
beautyai model list

# Check Hugging Face authentication
huggingface-cli whoami

# Clear model cache and retry
beautyai system clear-cache my-model
beautyai system load my-model
```

**3. Slow Inference Performance**
```bash
# Try vLLM backend for compatible models
beautyai model update my-model --engine vllm

# Check GPU utilization
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Use optimized quantization
beautyai model update my-model --quantization awq
```

**4. API Connection Issues**
```bash
# Check API server status
curl http://localhost:8000/health

# Verify port availability
netstat -tulpn | grep :8000

# Check server logs
uvicorn beautyai_inference.api.app:app --log-level debug
```

**5. Configuration Problems**
```bash
# Validate current configuration
beautyai config validate

# Reset to defaults if corrupted
beautyai config reset --confirm

# Check configuration file syntax
python -m json.tool config/my_config.json
```

### Performance Optimization

**Memory Management**:
- Use appropriate quantization for your GPU VRAM
- Monitor memory usage with `beautyai system status`
- Unload unused models with `beautyai system unload-all`
- Clear cache regularly with `beautyai system clear-cache --all`

**Inference Speed**:
- Use vLLM backend for production workloads
- Enable AWQ quantization for fastest inference
- Use appropriate tensor parallelism for multi-GPU setups
- Optimize `max_new_tokens` for your use case

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
# Enable debug logging for all commands
beautyai --verbose system status
beautyai --log-level DEBUG run chat

# Check detailed error traces
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from beautyai_inference.cli.unified_cli import main
main()
"
```

### Getting Help

1. **Check Documentation**: Review this README and inline help (`--help`)
2. **Validate Setup**: Run `beautyai system status` to check system health
3. **Check Logs**: Enable verbose mode to see detailed error information
4. **Community Support**: Open an issue with system info and error logs
5. **Configuration Backup**: Always backup working configurations before changes

## 📝 License & Attribution

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

- **Hugging Face Transformers**: Apache 2.0 License
- **vLLM**: Apache 2.0 License  
- **FastAPI**: MIT License
- **PyTorch**: BSD 3-Clause License
- **BitsAndBytes**: MIT License

### Model Licenses

Individual AI models are subject to their respective licenses:
- **Qwen Models**: Tongyi Qianwen License
- **Mistral Models**: Apache 2.0 License
- **Llama Models**: Custom Meta License
- **T5/Flan-T5**: Apache 2.0 License

Check individual model cards on Hugging Face for specific licensing terms.

## 📚 References & Documentation

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Model Resources
- [Qwen Model Family](https://huggingface.co/Qwen)
- [Mistral AI Models](https://huggingface.co/mistralai)
- [Google T5 Models](https://huggingface.co/google)
- [Meta Llama Models](https://huggingface.co/meta-llama)

### Technical References
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)
- [Arabic NLP Resources](https://github.com/linuxscout/arabic-nlp)

---

**BeautyAI Inference Framework** - Empowering Arabic AI and multilingual language model deployment with professional-grade tools and scalable architecture.
