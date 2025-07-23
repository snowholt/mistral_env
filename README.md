# BeautyAI Inference Framework

A scalable, professional-grade inference framework for various language models, specializing in Arabic AI models but supporting multilingual capabilities. The framework features both unified CLI and REST API interfaces with advanced voice-to-voice conversation capabilities, multiple inference backends, quantization options, and comprehensive model lifecycle management.

## 🚀 Key Features

### Core Capabilities
- **Unified CLI Interface**: Single `beautyai` command with organized subcommands for all functionality
- **REST API**: FastAPI-based web service with comprehensive endpoints for programmatic access
- **Voice-to-Voice Conversations**: Real-time WebSocket and HTTP endpoints for voice interactions with automatic language detection
- **Multiple Inference Backends**: Hugging Face Transformers, LlamaCpp (GGUF), and optional vLLM support
- **Advanced Text-to-Speech**: Coqui TTS with multilingual support, Edge TTS integration
- **Speech Recognition**: Whisper models optimized for Arabic with multilingual capabilities
- **Flexible Quantization**: 4-bit/8-bit with Transformers, Q4_K_M/Q6_K GGUF quantization with LlamaCpp
- **Multi-Architecture Support**: Causal language models (Qwen, Mistral, DeepSeek) and sequence-to-sequence models
- **Interactive Chat**: Real-time streaming chat interface with advanced parameter control and thinking mode
- **Performance Tools**: Comprehensive benchmarking and memory monitoring capabilities
- **Model Registry**: Centralized configuration management with validation and versioning

### Advanced Voice Features
- **Real-time WebSocket Voice Chat**: Bidirectional voice conversation with streaming audio
- **Automatic Language Detection**: Smart detection of input language with confidence scoring
- **Voice-to-Voice Pipeline**: Complete STT → LLM → TTS pipeline with session management
- **Multi-TTS Engine Support**: Coqui TTS (primary), Edge TTS, with voice cloning capabilities
- **Arabic Voice Optimization**: Specialized Arabic TTS models with natural speech synthesis
- **Content Filtering**: Configurable content filtering with multiple strictness levels
- **Session Persistence**: Conversation history tracking across voice interactions

### Architecture Highlights
- **Service-Oriented Architecture**: 15+ specialized services for maximum modularity and testability
- **Factory Pattern**: Intelligent model engine selection with automatic fallback strategies
- **Singleton Pattern**: Centralized model lifecycle management with cross-process state tracking
- **Configuration Management**: JSON-based configuration with validation, backup, and migration support
- **Memory Optimization**: GPU memory monitoring with automatic cleanup and cache management
- **API-Ready Design**: Services built for both CLI and web interface deployment
- **Voice Service Architecture**: Modular voice services for transcription, synthesis, and conversation management

## 📋 Requirements

- **Hardware**: NVIDIA GPU with sufficient VRAM (RTX 4090 with 24GB recommended)
- **Software**: Python 3.11+, CUDA drivers properly installed
- **Access**: Hugging Face account with valid token for model access
- **Audio Dependencies**: For voice features - system audio libraries (ALSA/PulseAudio on Linux)

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

### Voice Features Quick Start

Test voice-to-voice conversation:

```bash
# Test voice endpoints
curl -X GET "http://localhost:8000/inference/voice-to-voice/status"

# Test WebSocket voice conversation
# Connect to: ws://localhost:8000/ws/voice-conversation
```

**WebSocket Voice Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/voice-conversation?preset=qwen_optimized');

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    if (response.type === 'voice_response' && response.audio_base64) {
        // Play received audio
        const audioBlob = base64ToBlob(response.audio_base64, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        new Audio(audioUrl).play();
    }
};

// Send audio (from MediaRecorder)
recorder.ondataavailable = (e) => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(e.data); // Send audio chunk
    }
};
```

### Service Management (Ubuntu/systemd)

For development convenience, BeautyAI includes systemd service management for running the API server as a system service:

```bash
# Install the systemd service (one-time setup)
./manage-api-service.sh install

# Start the API service
./manage-api-service.sh start

# Check service status and API health
./manage-api-service.sh status

# View service logs
./manage-api-service.sh logs

# Stop the service
./manage-api-service.sh stop

# Restart the service (useful after code changes)
./manage-api-service.sh restart

# Enable auto-start on boot (optional)
./manage-api-service.sh enable
```

**Standard systemctl commands** also work after installation:
```bash
sudo systemctl start beautyai-api     # Start service
sudo systemctl stop beautyai-api      # Stop service
sudo systemctl status beautyai-api    # Check status
sudo journalctl -u beautyai-api -f    # Follow logs
```

**Features:**
- Auto-restart on failure
- Development mode with `--reload` for automatic code reloading
- Proper security restrictions and resource limits
- Integration with Ubuntu system logging
- Easy start/stop for development workflows

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

### Enhanced Chat Interface Parameters

**Core Generation Parameters**:
```bash
# Optimized preset usage (recommended)
beautyai run chat --preset qwen_optimized

# Direct parameter control (25+ parameters available)
beautyai run chat --model-name qwen3-unsloth-q4ks \
                  --temperature 0.3 \
                  --top-p 0.95 \
                  --top-k 20 \
                  --repetition-penalty 1.1 \
                  --max-tokens 512

# Advanced sampling parameters
beautyai run chat --model-name qwen3-unsloth-q4ks \
                  --min-p 0.05 \
                  --typical-p 1.0 \
                  --diversity-penalty 0.1 \
                  --no-repeat-ngram-size 3

# Thinking mode and content filtering
beautyai run chat --model-name qwen3-unsloth-q4ks \
                  --thinking-mode \
                  --disable-content-filter \
                  --content-filter-strictness relaxed
```

**Available Presets** (based on actual optimization testing):
- `qwen_optimized`: Best performance from testing (temp=0.3, top_p=0.95, top_k=20)
- `high_quality`: Maximum quality (temp=0.1, top_p=1.0, rep_penalty=1.15)
- `creative_optimized`: Creative but efficient (temp=0.5, top_p=1.0, top_k=80)
- `speed_optimized`: Fastest response
- `balanced`: Good balance of quality and speed
- `conservative`: Safe, consistent responses
- `creative`: More creative and varied responses

### Voice Conversation Commands

```bash
# Check voice service status
curl -X GET "http://localhost:8000/inference/voice-to-voice/status"

# Test audio chat (STT → LLM, text response)
curl -X POST "http://localhost:8000/inference/audio-chat" \
  -F "audio_file=@input.wav" \
  -F "input_language=auto" \
  -F "preset=qwen_optimized"

# Voice-to-voice conversation (STT → LLM → TTS)
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "audio_file=@input.wav" \
  -F "input_language=auto" \
  -F "output_language=auto" \
  -F "preset=qwen_optimized" \
  -F "thinking_mode=false"
```

### Advanced Model Management

```bash
# List models with quantization info
beautyai model list --format detailed

# Add GGUF model with specific quantization
beautyai model add --name custom-qwen-q4ks \
                   --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp \
                   --quantization Q4_K_S \
                   --model-filename "Qwen3-14B-Q4_K_S.gguf"

# Load model with timer control
beautyai system load custom-qwen-q4ks
beautyai model set-timer custom-qwen-q4ks --minutes 60

# Monitor model performance
beautyai system status --format detailed
beautyai model show custom-qwen-q4ks --include-timers
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
- `GET /system/resources` - CPU, GPU, disk usage
- `GET /system/performance` - Performance metrics over time

**Model Management**
- `GET /models` - List all models in registry
- `POST /models` - Add new model to registry
- `GET /models/{name}` - Get model details
- `PUT /models/{name}` - Update model configuration
- `DELETE /models/{name}` - Remove model from registry
- `POST /models/{name}/load` - Load model into memory
- `DELETE /models/{name}/unload` - Unload model from memory
- `GET /models/loaded` - List currently loaded models
- `GET /models/{name}/status` - Get model status
- `GET /models/{name}/timer` - Get model timer info
- `POST /models/{name}/timer/reset` - Reset model timer
- `POST /models/default/{name}` - Set default model

**Enhanced Inference Operations**
- `POST /inference/chat` - Advanced chat with 25+ parameters, presets, thinking mode control
- `POST /inference/test` - Run model tests
- `POST /inference/benchmark` - Run performance benchmarks
- `POST /inference/audio-chat` - Audio to text chat (STT → LLM)
- `POST /inference/voice-to-voice` - Complete voice conversation (STT → LLM → TTS)
- `GET /inference/voice-to-voice/status` - Voice service status
- `POST /inference/sessions/save` - Save chat session
- `POST /inference/sessions/load` - Load chat session
- `GET /inference/sessions` - List available sessions
- `DELETE /inference/sessions/{session_name}` - Delete session

**Real-time Voice Communication**
- `WebSocket /ws/voice-conversation` - Real-time bidirectional voice chat with streaming

**Configuration Management**
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `PUT /config` - Bulk update configuration
- `DELETE /config` - Reset configuration
- `POST /config/validate` - Validate configuration
- `POST /config/backup` - Create configuration backup
- `POST /config/restore` - Restore from backup

**System Management**
- `GET /system/cache` - Get cache status
- `POST /system/cache/clear` - Clear model caches
- `POST /system/memory/clear` - Clear unused memory
- `GET /system/logs` - Get system logs (admin)
- `POST /system/restart` - Restart system services (admin)

### Enhanced Chat API Examples

The chat API now supports 25+ parameters with direct access (no nested JSON required):

```python
import requests

# API base URL
base_url = "http://localhost:8000"

# Simple chat with optimization-based preset
simple_chat = {
    "model_name": "qwen3-unsloth-q4ks",
    "message": "What is artificial intelligence?",
    "preset": "qwen_optimized"  # Best settings from actual testing
}
response = requests.post(f"{base_url}/inference/chat", json=simple_chat)

# Advanced parameter control (direct field access)
advanced_chat = {
    "model_name": "qwen3-unsloth-q4ks", 
    "message": "Explain quantum computing in detail",
    "temperature": 0.3,           # Core parameters
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
    "min_p": 0.05,               # Advanced parameters
    "typical_p": 1.0,
    "diversity_penalty": 0.1,
    "no_repeat_ngram_size": 3,
    "disable_content_filter": True,    # Content filtering control
    "thinking_mode": False,            # Thinking mode control
    "response_language": "en"          # Language control
}
response = requests.post(f"{base_url}/inference/chat", json=advanced_chat)

# Voice-to-Voice conversation
voice_files = {'audio_file': open('input.wav', 'rb')}
voice_data = {
    'input_language': 'auto',     # Automatic language detection
    'output_language': 'auto',    # Match input language
    'preset': 'qwen_optimized',
    'thinking_mode': False,
    'disable_content_filter': True
}
response = requests.post(f"{base_url}/inference/voice-to-voice", 
                        files=voice_files, data=voice_data)

# Audio chat (STT → LLM, text response)
audio_files = {'audio_file': open('question.wav', 'rb')}
audio_data = {
    'chat_model_name': 'qwen3-unsloth-q4ks',
    'stt_model_name': 'whisper-large-v3-turbo-arabic',
    'input_language': 'ar',
    'preset': 'high_quality'
}
response = requests.post(f"{base_url}/inference/audio-chat", 
                        files=audio_files, data=audio_data)
```

### Advanced Model Management

```python
# List available models
response = requests.get(f"{base_url}/models")
models = response.json()

# Add a new model with specific configuration
model_data = {
    "name": "custom-qwen-model",
    "model_id": "Qwen/Qwen3-14B",
    "engine_type": "transformers",
    "quantization": "4bit",
    "description": "Custom Qwen model configuration",
    "custom_generation_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }
}
response = requests.post(f"{base_url}/models", json=model_data)

# Load model with custom configuration
load_config = {"force_reload": False}
response = requests.post(f"{base_url}/models/custom-qwen-model/load", json=load_config)

# Get detailed model status including timer info
response = requests.get(f"{base_url}/models/custom-qwen-model/status")

# List currently loaded models with timer information
response = requests.get(f"{base_url}/models/loaded?include_timers=true")
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
**Best for**: Chat, text generation, code completion, reasoning tasks

**Currently Supported Models**:
- **Qwen3 Series**: 
  - `qwen3-model`: Qwen/Qwen3-14B (Transformers, 4-bit)
  - `qwen3-unsloth-q4ks`: Optimized GGUF (LlamaCpp, Q4_K_S, 8.0GB, fastest)
  - `qwen3-unsloth-q4km`: Balanced GGUF (LlamaCpp, Q4_K_M, 8.4GB, balanced)
  - `qwen3-unsloth-q6k`: High-quality GGUF (LlamaCpp, Q6_K, 12GB, best quality)
  - `qwen3-official-q4km`: Official GGUF (LlamaCpp, Q4_K_M, 8.4GB)
  - `qwen3-official-q6k`: Official GGUF (LlamaCpp, Q6_K, 12GB)
- **DeepSeek Series**:
  - `deepseek-r1-qwen-14b-multilingual`: DeepSeek R1 Distill with reasoning capabilities
- **Arabic Specialized Models**:
  - `bee1reason-arabic-q4ks`: Arabic reasoning model (8GB, fast)
  - `bee1reason-arabic-q4km-i1`: Arabic reasoning model (8.4GB, balanced)

### Speech and Audio Models
**Speech-to-Text (STT)**:
- `whisper-large-v3-turbo-arabic`: Optimized for Arabic transcription

**Text-to-Speech (TTS)**:
- `coqui-tts-arabic`: Native Arabic TTS with Coqui TTS engine
- `coqui-tts-multilingual`: XTTS v2 with voice cloning (16+ languages)
- `edge-tts`: Microsoft Edge TTS (cloud-based, 20+ languages)

### Backend Engine Mapping
**LlamaCpp Engine** (Primary for quantized models):
- All GGUF models (Q4_K_S, Q4_K_M, Q6_K quantization)
- Memory efficient with fast inference
- CUDA acceleration support

**Transformers Engine** (Alternative):
- Full precision and BitsAndBytes quantization (4-bit, 8-bit)
- Broader model compatibility
- Research and experimentation friendly

**Specialized Engines**:
- **Coqui TTS**: High-quality neural speech synthesis
- **Edge TTS**: Cloud-based multilingual TTS
- **Whisper/Transformers**: Speech recognition models

### Smart Model Selection
The framework automatically handles:
- **Quantization Selection**: GGUF for efficiency, Transformers for compatibility
- **Memory Management**: Auto-selection based on available VRAM
- **Language Optimization**: Arabic-specific models when appropriate
- **Performance Tuning**: Q4_K_S for speed, Q6_K for quality
- **Fallback Strategies**: Graceful degradation when preferred models unavailable

### Memory Requirements

**GPU Memory Usage Guidelines**:
```
Model Type          | Q4_K_S | Q4_K_M | Q6_K  | 4-bit Transform
Qwen3 14B          | ~8.0GB | ~8.4GB | ~12GB | ~10GB
DeepSeek R1 14B    | ~8.0GB | ~8.4GB | ~12GB | ~10GB  
Arabic Reasoning   | ~8.0GB | ~8.4GB | ~12GB | ~10GB
```

**TTS Models**:
```
Coqui TTS Arabic     | ~1.2GB VRAM
Coqui TTS Multilingual | ~2.1GB VRAM
Whisper Large V3     | ~3.1GB VRAM
```

## ⚙️ Advanced Configuration

### Quantization Strategies

**LlamaCpp Engine (Primary)**:
```bash
# Q4_K_S quantization (fastest, 8.0GB)
beautyai model add --name qwen-q4ks --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp --quantization Q4_K_S \
                   --model-filename "Qwen3-14B-Q4_K_S.gguf"

# Q4_K_M quantization (balanced, 8.4GB)
beautyai model add --name qwen-q4km --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp --quantization Q4_K_M \
                   --model-filename "Qwen3-14B-Q4_K_M.gguf"

# Q6_K quantization (high quality, 12GB)
beautyai model add --name qwen-q6k --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp --quantization Q6_K \
                   --model-filename "Qwen3-14B-Q6_K.gguf"
```

**Transformers Backend (Alternative)**:
```bash
# 4-bit quantization with BitsAndBytes
beautyai model add --name qwen-4bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 4bit

# 8-bit quantization for memory efficiency
beautyai model add --name qwen-8bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 8bit
```

**Voice Model Configuration**:
```bash
# Add Arabic TTS model
beautyai model add --name arabic-tts \
                   --model-id "tts_models/ar/tn_arabicspeech/vits" \
                   --engine coqui_tts

# Add multilingual TTS with voice cloning
beautyai model add --name multilingual-tts \
                   --model-id "tts_models/multilingual/multi-dataset/xtts_v2" \
                   --engine coqui_tts
```

### Custom Configuration Files

Create specialized configuration files for different deployment scenarios:

**config/production.json**:
```json
{
  "default_model": "qwen3-unsloth-q4ks",
  "default_engine": "llama.cpp",
  "gpu_memory_utilization": 0.90,
  "generation_config": {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 20,
    "max_new_tokens": 256,
    "do_sample": true
  },
  "api_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "voice_config": {
    "default_stt_model": "whisper-large-v3-turbo-arabic",
    "default_tts_model": "coqui-tts-arabic",
    "enable_voice_features": true
  }
}
```

**config/development.json**:
```json
{
  "default_model": "qwen3-unsloth-q4ks",
  "default_engine": "llama.cpp",
  "force_cpu": false,
  "quantization": "Q4_K_S",
  "generation_config": {
    "temperature": 0.7,
    "max_new_tokens": 256
  },
  "logging": {
    "level": "DEBUG",
    "console": true
  },
  "voice_config": {
    "enable_voice_features": true,
    "debug_audio": true
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
├── 🎤 voice/             # Voice conversation, transcription, and synthesis services
│   ├── conversation/     # Voice-to-voice conversation management
│   ├── transcription/    # Speech-to-text services (Whisper)
│   └── synthesis/        # Text-to-speech services (Coqui TTS, Edge TTS)
├── ⚙️ config/            # Configuration, validation, backup, and migration services
└── 💾 system/            # Memory monitoring, cache management, and status services
```

**API Layer**:
```
📁 api/
├── 🌐 endpoints/         # REST API route handlers
│   ├── models.py         # Model management endpoints
│   ├── inference.py      # Chat, voice, and inference endpoints
│   ├── system.py         # System monitoring and control
│   ├── websocket_voice.py # Real-time voice conversation WebSocket
│   ├── config.py         # Configuration management
│   └── health.py         # Health check endpoints
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

**Inference Engines**:
```
📁 inference_engines/
├── transformers_engine.py    # Hugging Face Transformers backend
├── llamacpp_engine.py        # LlamaCpp GGUF backend (primary)
├── vllm_engine.py           # vLLM backend (optional)
└── voice_engines/           # Voice-specific engines
    ├── whisper_engine.py    # Speech recognition
    ├── coqui_tts_engine.py  # Coqui TTS synthesis
    └── edge_tts_engine.py   # Microsoft Edge TTS
```

**CLI Interface**:
```
📁 cli/
├── 🎯 unified_cli.py     # Main CLI entry point with command routing
├── ⚙️ argument_config.py # Standardized argument handling
└── 🔌 handlers/          # CLI adapters for service integration
    └── unified_cli_adapter.py # Unified adapter for all CLI commands
```

### Key Design Patterns

**1. Factory Pattern**: Intelligent model engine selection
- Automatic backend selection (LlamaCpp vs Transformers vs specialized engines)
- Architecture-aware quantization selection (GGUF vs BitsAndBytes)
- Voice engine selection (Coqui TTS vs Edge TTS)
- Graceful fallback strategies

**2. Singleton Pattern**: Centralized resource management
- Cross-process model state tracking
- GPU memory optimization and cleanup
- Voice model lifecycle management
- Configuration persistence

**3. Adapter Pattern**: Interface unification
- CLI-to-service bridge with unified commands
- API-to-service integration
- Voice service integration
- Backward compatibility layer

**4. Service Layer**: Business logic isolation
- Single responsibility principle
- Voice conversation orchestration
- Dependency injection ready
- Test-friendly design

**5. WebSocket Pattern**: Real-time communication
- Bidirectional voice conversation streaming
- Connection lifecycle management
- Session state persistence
- Audio chunk processing

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

# Check GGUF model filename
beautyai model show my-model
```

**3. Slow Inference Performance**
```bash
# Use GGUF models for better performance
beautyai model update my-model --engine llama.cpp --quantization Q4_K_S

# Check GPU utilization
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Use optimized quantization
beautyai model update my-model --quantization Q4_K_M
```

**4. Voice Processing Issues**
```bash
# Check voice service status
curl http://localhost:8000/inference/voice-to-voice/status

# Test audio file formats (use WAV for best compatibility)
file audio_input.wav

# Check TTS library installation
python -c "from TTS import TTS; print('Coqui TTS available')"

# Test with Edge TTS as fallback
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "tts_model_name=edge-tts" \
  -F "audio_file=@test.wav"
```

**5. WebSocket Voice Connection Issues**
```bash
# Test WebSocket connection
wscat -c "ws://localhost:8000/ws/voice-conversation"

# Check WebSocket status endpoint
curl http://localhost:8000/ws/voice-conversation/status

# Verify audio format support (WebM recommended for browsers)
# Convert audio to supported format if needed
ffmpeg -i input.mp3 -f webm output.webm
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
- Use GGUF quantization for GPU VRAM efficiency (Q4_K_S for speed, Q6_K for quality)
- Monitor memory usage with `beautyai system status`
- Unload unused models with `beautyai system unload-all`
- Clear cache regularly with `beautyai system clear-cache --all`

**Inference Speed**:
- Use LlamaCpp backend with GGUF models for production workloads
- Enable Q4_K_S quantization for fastest inference
- Use optimization-based presets: `qwen_optimized`, `speed_optimized`
- Optimize `max_new_tokens` for your use case (256 for voice, 512 for text)

**Voice Performance**:
- Use Coqui TTS for high-quality local synthesis
- Edge TTS for cloud-based multilingual support
- Keep audio files small (< 10MB) for WebSocket streaming
- Use WAV format for best compatibility and quality

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
- **LlamaCpp Python**: MIT License
- **Coqui TTS**: Mozilla Public License 2.0
- **FastAPI**: MIT License
- **PyTorch**: BSD 3-Clause License
- **BitsAndBytes**: MIT License
- **Whisper**: MIT License
- **Edge-TTS**: MIT License

### Model Licenses

Individual AI models are subject to their respective licenses:
- **Qwen Models**: Tongyi Qianwen License
- **DeepSeek Models**: Apache 2.0 License
- **Whisper Models**: MIT License
- **TTS Models**: Various (check individual model licenses)

Check individual model cards on Hugging Face for specific licensing terms.

## 📚 References & Documentation

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [LlamaCpp Python](https://llama-cpp-python.readthedocs.io/)
- [Coqui TTS](https://docs.coqui.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Model Resources
- [Qwen Model Family](https://huggingface.co/Qwen)
- [DeepSeek Models](https://huggingface.co/lightblue)
- [Unsloth Optimized Models](https://huggingface.co/unsloth)
- [Arabic Speech Models](https://huggingface.co/mboushaba)
- [Coqui TTS Models](https://huggingface.co/coqui)

### Technical References
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [WebSocket Protocol](https://tools.ietf.org/html/rfc6455)
- [Arabic NLP Resources](https://github.com/linuxscout/arabic-nlp)

---

**BeautyAI Inference Framework** - Empowering Arabic AI and multilingual language model deployment with professional-grade tools, advanced voice capabilities, and scalable architecture.
