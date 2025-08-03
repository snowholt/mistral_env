# BeautyAI Inference Framework

A scalable, professional-grade inference framework for various language models, specializing in Arabic AI models but supporting multilingual capabilities. The framework features a dual-stack architecture with CLI interface, FastAPI backend, and beautiful web UI with advanced voice-to-voice conversation capabilities.

## 🚀 Key Features

- **🎯 Unified CLI Interface**: Single `beautyai` command with organized subcommands
- **🌐 FastAPI Backend**: High-performance REST API with WebSocket voice features  
- **🎨 Beautiful Web UI**: Flask interface with animated 3D backgrounds and voice controls
- **🎤 Ultra-Fast Voice Chat**: Real-time voice conversations with <2 second response times
- **🔧 Multiple Inference Backends**: Transformers (primary), LlamaCpp, optional vLLM
- **⚡ Smart Quantization**: 4-bit/8-bit quantization for memory efficiency
- **🌍 Multilingual Support**: Arabic specialization with broad language support
- **📊 Performance Tools**: Benchmarking, memory monitoring, and optimization
- **🔄 Model Management**: Centralized registry with lifecycle management
- **🎛️ Advanced Parameters**: 25+ chat parameters with optimized presets

## 🏗️ Architecture

```
BeautyAI/
├── 🔧 backend/          # FastAPI server (port 8000)
│   ├── CLI interface    # beautyai command
│   ├── REST API         # /docs, /models, /inference
│   ├── WebSocket        # /ws/voice-conversation
│   └── Services         # 15+ specialized services
└── 🎨 frontend/         # Flask web UI (port 5001)
    ├── Chat interface   # Animated 3D backgrounds
    ├── Voice controls   # One-click voice chat
    └── Model management # Real-time status
```

## 🚀 Quick Start

### 1. Backend Setup (Required)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start API server
python run_server.py
```
📚 **API Docs**: http://localhost:8000/docs

### 2. Frontend Setup (Optional - for Web UI)
```bash
cd frontend  
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start web interface
python src/app.py
```
🎨 **Web UI**: http://localhost:5001

### 3. CLI Usage
```bash
cd backend
source venv/bin/activate

# Show all commands
beautyai --help

# Start interactive chat
beautyai run chat

# List models and check status
beautyai model list
beautyai system status
```

## 🎤 Voice Features

### Ultra-Fast Voice Chat
- **Response Time**: <2 seconds end-to-end
- **Technology**: Edge TTS + WebSocket streaming
- **Languages**: Arabic and English with auto-detection
- **Access**: Web UI microphone button or WebSocket API

### WebSocket API Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/voice-conversation');

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    if (response.type === 'voice_response') {
        // Play audio response
        playAudio(response.audio_base64);
    }
};

// Send audio data
ws.send(audioBlob);
```

## ⚙️ Configuration

### Model Registry
Models are configured in `backend/src/model_registry.json`:
```json
{
  "default_model": "default",
  "models": {
    "default": {
      "model_id": "Qwen/Qwen3-14B",
      "engine_type": "transformers",
      "quantization": "4bit"
    }
  }
}
```

### Adding Models
```bash
beautyai model add --name "custom-model" \
                   --model-id "organization/model-name" \
                   --engine transformers \
                   --quantization 4bit
```

## 🚀 Production Deployment

### System Services
```bash
# Backend API service
cd backend/unitTests_scripts/shell_scripts
./manage-api-service.sh install
./manage-api-service.sh start

# Frontend web UI service
sudo systemctl enable beautyai-webui
sudo systemctl start beautyai-webui
```

### Nginx Configuration
SSL-ready configuration included for production deployment with proper WebSocket proxying.

## 📚 Documentation

### Detailed Documentation
- **[Backend Documentation](backend/README.md)**: API, CLI, services, development
- **[Frontend Documentation](frontend/README.md)**: Web UI, voice features, customization
- **API Reference**: http://localhost:8000/docs (when running)

### Key Resources
- **Model Management**: Add, load, configure inference models
- **Voice Integration**: WebSocket voice chat implementation
- **Performance Tuning**: Memory optimization and benchmarking
- **Production Setup**: Systemd services and SSL configuration

## 🔧 System Requirements

- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 4090 recommended)
- **Software**: Python 3.11+, CUDA drivers
- **Network**: Internet access for model downloads and Edge TTS
- **Audio**: Microphone and speakers/headphones for voice features

## 🐛 Quick Troubleshooting

```bash
# Check system status
beautyai system status

# Test API connectivity
curl http://localhost:8000/health

# Test voice features
curl http://localhost:8000/ws/voice-conversation/status

# View logs
./backend/unitTests_scripts/shell_scripts/manage-api-service.sh logs
```

## 🤝 Contributing

1. **Backend**: Follow service-oriented architecture, add comprehensive tests
2. **Frontend**: Maintain responsive design, test voice features across browsers  
3. **Documentation**: Update relevant README files, not the main one
4. **Testing**: Backend unit tests in `backend/unitTests_scripts/`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Quick Links**: [Backend Setup](backend/README.md) | [Frontend Setup](frontend/README.md) | [API Docs](http://localhost:8000/docs) | [Web UI](http://localhost:5001)

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

# Run the automated setup script for backend
cd backend
chmod +x unitTests_scripts/shell_scripts/setup_beautyai.sh
./unitTests_scripts/shell_scripts/setup_beautyai.sh

# Setup frontend (if you want the web UI)
cd ../frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The setup script will guide you through the installation process and ask if you want to install with vLLM support (recommended for better performance).

### Production Services Setup

BeautyAI includes systemd service files for production deployment:

```bash
# Install and start API service (backend)
cd backend/unitTests_scripts/shell_scripts
chmod +x manage-api-service.sh
./manage-api-service.sh install
./manage-api-service.sh start

# Install web UI service (frontend) 
sudo cp ../../beautyai-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable beautyai-webui
sudo systemctl start beautyai-webui
```

### Hugging Face Authentication

```bash
# Authenticate with Hugging Face (required for most models)
huggingface-cli login
```

Follow the prompts and enter your Hugging Face token from https://huggingface.co/settings/tokens

### Development Installation

```bash
# Backend development setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,vllm]"

# Frontend development setup  
cd ../frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## 🚀 Quick Start

### CLI Interface

After installation, activate the backend virtual environment and use the unified CLI:

```bash
# Navigate to backend and activate the virtual environment
cd backend
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

### Web UI Interface

Start the beautiful web interface for interactive chat:

```bash
# Navigate to frontend and activate environment
cd frontend
source venv/bin/activate

# Start the web UI server
python src/app.py

# Or run as development server
python src/app.py --dev
```

Access the web interface at:
- **Web UI**: http://localhost:5001
- **Features**: Animated 3D fractal background, voice controls, real-time chat
- **Voice Chat**: Built-in WebSocket voice conversation interface

### FastAPI Backend

Start the FastAPI web server for programmatic access:

```bash
# Navigate to backend and activate environment
cd backend  
source venv/bin/activate

# Start the API server (development mode)
python run_server.py

# Or using uvicorn directly
uvicorn beautyai_inference.api.app:app --reload --host 0.0.0.0 --port 8000
```

Access the API at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc  
- **Health Check**: http://localhost:8000/health
- **Simple Voice WebSocket**: ws://localhost:8000/ws/voice-conversation

### Voice Features Quick Start

Test voice-to-voice conversation through the API:

```bash
# Test simple voice WebSocket endpoint (ultra-fast <2s response)
curl -X GET "http://localhost:8000/ws/voice-conversation/status"

# Test voice endpoints
curl -X GET "http://localhost:8000/api/v1/voice/endpoints"

# Or use the Web UI for interactive voice chat
# Navigate to http://localhost:5001 and click the microphone button
```

**Simple Voice WebSocket Example** (for web apps):
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/voice-conversation?language=auto&voice_type=default');

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

For production deployment, BeautyAI includes complete systemd service management:

#### Backend API Service
```bash
# Navigate to the management script
cd backend/unitTests_scripts/shell_scripts

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

#### Frontend Web UI Service
```bash
# Install the web UI systemd service
sudo cp beautyai-webui.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start and enable the web UI service  
sudo systemctl start beautyai-webui
sudo systemctl enable beautyai-webui

# Check status
sudo systemctl status beautyai-webui
```

**Standard systemctl commands** also work after installation:
```bash
# Backend API
sudo systemctl start beautyai-api     # Start API service
sudo systemctl stop beautyai-api      # Stop API service
sudo systemctl status beautyai-api    # Check API status
sudo journalctl -u beautyai-api -f    # Follow API logs

# Frontend Web UI
sudo systemctl start beautyai-webui   # Start web UI service
sudo systemctl stop beautyai-webui    # Stop web UI service
sudo systemctl status beautyai-webui  # Check web UI status
sudo journalctl -u beautyai-webui -f  # Follow web UI logs
```

#### Production Features
- **Auto-restart on failure** for both services
- **Development mode** with `--reload` for automatic code reloading (API)
- **Proper security restrictions** and resource limits
- **Integration with Ubuntu system logging**
- **Easy start/stop** for development workflows
- **SSL support** via Nginx (configuration included)

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
beautyai run chat --model-name default \
                  --temperature 0.3 \
                  --top-p 0.95 \
                  --top-k 20 \
                  --repetition-penalty 1.1 \
                  --max-tokens 512

# Advanced sampling parameters
beautyai run chat --model-name default \
                  --min-p 0.05 \
                  --typical-p 1.0 \
                  --diversity-penalty 0.1 \
                  --no-repeat-ngram-size 3

# Thinking mode and content filtering
beautyai run chat --model-name default \
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
# Check simple voice WebSocket status
curl -X GET "http://localhost:8000/ws/voice-conversation/status"

# Check available voice endpoints  
curl -X GET "http://localhost:8000/api/v1/voice/endpoints"

# Test health of voice services
curl -X GET "http://localhost:8000/api/v1/health/voice"

# Or use the beautiful Web UI at http://localhost:5001
# Click the microphone button for instant voice chat
```

### Advanced Model Management

```bash
# List models with quantization info
beautyai model list --format detailed

# Add custom model configuration
beautyai model add --name custom-qwen \
                   --model-id "Qwen/Qwen3-14B" \
                   --engine transformers \
                   --quantization 4bit \
                   --description "Custom Qwen model"

# Load model with timer control
beautyai system load custom-qwen
beautyai model set-timer custom-qwen --minutes 60

# Monitor model performance
beautyai system status --format detailed
beautyai model show custom-qwen --include-timers
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

The BeautyAI framework provides a comprehensive FastAPI-based REST API for programmatic access to all functionality:

### API Endpoints Overview

**Health & Status**
- `GET /health` - Service health check
- `GET /api/v1/health/voice` - Voice services health check
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
- `GET /models/loaded` - List currently loaded models

**Simple Voice Features** (Ultra-fast <2s response)
- `WebSocket /ws/voice-conversation` - Simple voice chat with Edge TTS
- `GET /ws/voice-conversation/status` - Voice WebSocket status
- `GET /api/v1/voice/endpoints` - Available voice endpoints

**Enhanced Inference Operations**
- `POST /inference/chat` - Advanced chat with 25+ parameters, presets, thinking mode control
- `POST /inference/test` - Run model tests
- `POST /inference/benchmark` - Run performance benchmarks

**Configuration Management**
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `PUT /config` - Bulk update configuration
- `POST /config/validate` - Validate configuration

### Enhanced Chat API Examples

The chat API supports 25+ parameters with direct access (no nested JSON required):

```python
import requests

# API base URL
base_url = "http://localhost:8000"

# Simple chat with optimization-based preset
simple_chat = {
    "model_name": "default",
    "message": "What is artificial intelligence?",
    "preset": "qwen_optimized"  # Best settings from actual testing
}
response = requests.post(f"{base_url}/inference/chat", json=simple_chat)

# Advanced parameter control (direct field access)
advanced_chat = {
    "model_name": "default", 
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
```

### Simple Voice WebSocket API

Ultra-fast voice conversations with <2 second response times:

```python
import asyncio
import websockets
import json

async def voice_chat():
    uri = "ws://localhost:8000/ws/voice-conversation?language=auto&voice_type=default"
    
    async with websockets.connect(uri) as websocket:
        # Send audio data
        with open("audio.wav", "rb") as audio_file:
            await websocket.send(audio_file.read())
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "voice_response":
            # Save audio response
            with open("response.wav", "wb") as f:
                f.write(base64.b64decode(data["audio_base64"]))

asyncio.run(voice_chat())
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

# Get detailed model status
response = requests.get(f"{base_url}/models/custom-qwen-model")

# List currently loaded models
response = requests.get(f"{base_url}/models/loaded")
```

### Authentication

The API supports JWT-based authentication for production deployments:

```python
# Login to get access token (if authentication is enabled)
auth_data = {"username": "admin", "password": "secure_password"}
response = requests.post(f"{base_url}/auth/login", json=auth_data)
token = response.json()["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{base_url}/models", headers=headers)
```

### Web UI Integration

The web UI provides a complete interface for all framework features:

```bash
# Start the web UI (after starting the backend API)
cd frontend
source venv/bin/activate
python src/app.py

# Features available at http://localhost:5001:
# - Interactive chat interface with animated background
# - Real-time voice conversation with microphone button
# - Model management and status monitoring
# - Configuration management
# - Session management and history
# - Debug and testing interfaces
```

## 🔧 Supported Model Architectures

The framework supports a wide range of model architectures with intelligent backend selection:

### Causal Language Models (CLMs)
**Best for**: Chat, text generation, code completion, reasoning tasks

**Currently Supported Models**:
- **Qwen3 Series**: 
  - `default`: Qwen/Qwen3-14B (Transformers, 4-bit) - Default model
  - Custom configurations supported for various Qwen models
- **DeepSeek Series**:
  - DeepSeek R1 and other models with reasoning capabilities
- **Arabic Specialized Models**:
  - Arabic reasoning models and fine-tuned variants

### Speech and Audio Models
**Speech-to-Text (STT)**:
- Whisper models optimized for Arabic transcription
- Multilingual speech recognition support

**Text-to-Speech (TTS)**:
- **Edge TTS**: Microsoft Edge TTS (primary, ultra-fast cloud-based synthesis)
- **Coqui TTS**: Local TTS with voice cloning capabilities (optional)
- Multi-language support with automatic language detection

### Backend Engine Mapping
**Transformers Engine** (Primary):
- Full precision and BitsAndBytes quantization (4-bit, 8-bit)
- Broad model compatibility
- Production-ready with CUDA acceleration

**LlamaCpp Engine** (Alternative for GGUF models):
- Quantized GGUF models (Q4_K_S, Q4_K_M, Q6_K quantization)
- Memory efficient with fast inference
- CUDA acceleration support

**Specialized Engines**:
- **Edge TTS**: Ultra-fast cloud-based multilingual TTS (<2s response)
- **Coqui TTS**: High-quality local neural speech synthesis (optional)
- **Whisper/Transformers**: Speech recognition models

### Smart Model Selection
The framework automatically handles:
- **Quantization Selection**: 4-bit/8-bit for efficiency, full precision for compatibility
- **Memory Management**: Auto-selection based on available VRAM
- **Language Optimization**: Arabic-specific models when appropriate
- **Performance Tuning**: Edge TTS for speed, Coqui TTS for quality
- **Fallback Strategies**: Graceful degradation when preferred models unavailable

### Memory Requirements

**GPU Memory Usage Guidelines**:
```
Model Type          | 4-bit Transform | 8-bit Transform | Full Precision
Qwen3 14B          | ~10GB          | ~15GB          | ~28GB
DeepSeek R1 14B    | ~10GB          | ~15GB          | ~28GB  
Arabic Models      | ~8GB           | ~12GB          | ~25GB
```

**TTS Models**:
```
Edge TTS             | Cloud-based (no VRAM)
Coqui TTS Arabic     | ~1.2GB VRAM
Coqui TTS Multilingual | ~2.1GB VRAM
Whisper Large V3     | ~3.1GB VRAM
```

## ⚙️ Advanced Configuration

### Quantization Strategies

**Transformers Backend (Primary)**:
```bash
# 4-bit quantization with BitsAndBytes (recommended)
beautyai model add --name qwen-4bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 4bit

# 8-bit quantization for memory efficiency
beautyai model add --name qwen-8bit --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization 8bit

# Full precision for maximum quality (requires more VRAM)
beautyai model add --name qwen-fp16 --model-id Qwen/Qwen3-14B \
                   --engine transformers --quantization none
```

**LlamaCpp Engine (Alternative for GGUF models)**:
```bash
# Q4_K_S quantization (fastest, ~8.0GB)
beautyai model add --name qwen-q4ks --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp --quantization Q4_K_S \
                   --model-filename "Qwen3-14B-Q4_K_S.gguf"

# Q4_K_M quantization (balanced, ~8.4GB)
beautyai model add --name qwen-q4km --model-id "unsloth/Qwen3-14B-GGUF" \
                   --engine llama.cpp --quantization Q4_K_M \
                   --model-filename "Qwen3-14B-Q4_K_M.gguf"
```

**Voice Model Configuration**:
```bash
# Edge TTS (ultra-fast, recommended)
# No configuration needed - works out of the box

# Coqui TTS (optional, higher quality local synthesis)
beautyai model add --name arabic-tts \
                   --model-id "tts_models/ar/tn_arabicspeech/vits" \
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

### Dual-Stack Architecture

**Backend API Server** (`/backend/` - Port 8000):
```
📁 backend/
├── 🚀 run_server.py          # FastAPI server entry point
├── 📦 setup.py              # Package configuration and dependencies
├── 📋 requirements.txt      # Python dependencies
├── 🏗️ src/beautyai_inference/
│   ├── 🎯 cli/              # Unified CLI interface
│   ├── 🌐 api/              # FastAPI application and endpoints
│   ├── 🔧 services/         # 15+ specialized business logic services
│   ├── 🚀 inference_engines/ # Model inference backends
│   ├── 🏭 core/             # Factory patterns and model management
│   ├── ⚙️ config/           # Configuration management
│   └── 🛠️ utils/            # Utility functions and helpers
└── 🧪 unitTests_scripts/    # Testing and setup scripts
```

**Frontend Web UI** (`/frontend/` - Port 5001):
```
📁 frontend/
├── 📦 package.json          # Node.js-style project metadata
├── 📋 requirements.txt      # Python dependencies for Flask app
├── 🌐 src/
│   ├── 🎨 app.py           # Flask web server with beautiful UI
│   ├── 🎭 templates/       # HTML templates with 3D animations
│   ├── 🎨 static/          # CSS, JavaScript, and static assets
│   └── ⚙️ config.json      # Frontend configuration
└── 📚 docs/                # Frontend documentation
```

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
│   └── synthesis/        # Text-to-speech services (Edge TTS, Coqui TTS)
├── ⚙️ config/            # Configuration, validation, backup, and migration services
└── 💾 system/            # Memory monitoring, cache management, and status services
```

**API Layer**:
```
📁 api/
├── 🌐 endpoints/         # REST API route handlers
│   ├── models.py         # Model management endpoints
│   ├── inference.py      # Chat and inference endpoints
│   ├── system.py         # System monitoring and control
│   ├── websocket_simple_voice.py # Real-time voice conversation WebSocket
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
├── transformers_engine.py    # Hugging Face Transformers backend (primary)
├── llamacpp_engine.py        # LlamaCpp GGUF backend (alternative)
├── vllm_engine.py           # vLLM backend (optional)
└── voice_engines/           # Voice-specific engines
    ├── whisper_engine.py    # Speech recognition
    ├── edge_tts_engine.py   # Microsoft Edge TTS (primary)
    └── coqui_tts_engine.py  # Coqui TTS synthesis (optional)
```

**CLI Interface**:
```
📁 cli/
├── 🎯 unified_cli.py     # Main CLI entry point with command routing
├── ⚙️ argument_config.py # Standardized argument handling
└── 🔌 handlers/          # CLI adapters for service integration
    └── unified_cli_adapter.py # Unified adapter for all CLI commands
```

### Production Deployment

**System Services**:
```
📁 /etc/systemd/system/
├── 🔧 beautyai-api.service    # Backend API service (port 8000)
└── 🌐 beautyai-webui.service  # Frontend web UI service (port 5001)
```

**Nginx Configuration** (for SSL/production):
```
📁 nginx configuration/
├── 🔒 SSL termination for both services
├── 🌐 dev.gmai.sa → Frontend (port 5001)
├── 🔧 api.gmai.sa → Backend API (port 8000)
└── 🎤 WebSocket proxy for voice features
```

### Key Design Patterns

**1. Factory Pattern**: Intelligent model engine selection
- Automatic backend selection (Transformers vs LlamaCpp vs specialized engines)
- Architecture-aware quantization selection (4-bit/8-bit vs GGUF)
- Voice engine selection (Edge TTS vs Coqui TTS)
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
- Audio chunk processing with ultra-fast response (<2s)

## 🧪 Testing & Development

### Running Tests
```bash
# Backend tests
cd backend
source venv/bin/activate
python -m pytest unitTests_scripts/ -v

# Run specific test categories
python -m pytest unitTests_scripts/python_scripts/ -v

# Run with coverage reporting
python -m pytest unitTests_scripts/ --cov=beautyai_inference --cov-report=html
```

### Development Commands
```bash
# Backend development setup
cd backend
pip install -e ".[dev]"

# Frontend development setup
cd frontend
pip install -r requirements.txt

# Run code formatting (backend)
cd backend
black src/beautyai_inference/
isort src/beautyai_inference/

# Run linting
flake8 src/beautyai_inference/
mypy src/beautyai_inference/

# Run security checks
bandit -r src/beautyai_inference/
```

### Development Servers
```bash
# Start backend API in development mode (auto-reload)
cd backend
source venv/bin/activate
python run_server.py

# Start frontend web UI in development mode
cd frontend
source venv/bin/activate
python src/app.py --dev

# Both services together for full development stack
# Terminal 1: Backend
cd backend && source venv/bin/activate && python run_server.py

# Terminal 2: Frontend
cd frontend && source venv/bin/activate && python src/app.py
```

### Adding New Models

1. **Add to Model Registry**:
```bash
# Navigate to backend and activate environment
cd backend
source venv/bin/activate

# Add a new model configuration
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

### Adding New Frontend Features

1. **Frontend Structure**:
```bash
# Frontend templates and static files
frontend/src/
├── templates/
│   ├── index.html          # Main chat interface
│   ├── legacy.html         # Legacy interface
│   └── debug.html          # Debug interface
├── static/
│   ├── css/               # Stylesheets with 3D animations
│   ├── js/                # JavaScript for voice and chat
│   └── assets/            # Images and other static files
└── app.py                 # Flask application with all routes
```

2. **Adding New Routes**:
```python
# In frontend/src/app.py
@app.route('/new-feature')
def new_feature():
    return render_template('new_feature.html')
```

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use type hints, add comprehensive docstrings
2. **Architecture**: Use service-oriented patterns, maintain separation of concerns between backend and frontend
3. **Testing**: Add unit tests for new features, integration tests for CLI commands
4. **Documentation**: Update README and inline documentation for new features
5. **Backward Compatibility**: Maintain compatibility with existing CLI interfaces and API endpoints
6. **Frontend Development**: Follow Flask patterns, maintain responsive design, test voice features
7. **Backend Development**: Use service-oriented architecture, maintain API compatibility

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
# Navigate to backend first
cd backend
source venv/bin/activate

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
# Use 4-bit quantization for better performance
beautyai model update my-model --quantization 4bit

# Check GPU utilization
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Use optimized presets
beautyai run chat --preset speed_optimized
```

**4. Voice Processing Issues**
```bash
# Check simple voice WebSocket status
curl http://localhost:8000/ws/voice-conversation/status

# Check available voice endpoints
curl http://localhost:8000/api/v1/voice/endpoints

# Test voice health
curl http://localhost:8000/api/v1/health/voice

# Check if Edge TTS is working
python -c "import edge_tts; print('Edge TTS available')"

# Test web UI voice interface at http://localhost:5001
```

**5. WebSocket Voice Connection Issues**
```bash
# Test WebSocket connection manually
wscat -c "ws://localhost:8000/ws/voice-conversation"

# Check WebSocket status endpoint
curl http://localhost:8000/ws/voice-conversation/status

# Verify audio format support (WAV recommended)
# Convert audio to supported format if needed
ffmpeg -i input.mp3 output.wav

# Check web UI microphone permissions in browser
```

**6. API Connection Issues**
```bash
# Check backend API server status
curl http://localhost:8000/health

# Check if port is available
netstat -tulpn | grep :8000

# Check server logs
cd backend
sudo journalctl -u beautyai-api -f

# Start server manually for debugging
cd backend
source venv/bin/activate
python run_server.py
```

**7. Frontend Web UI Issues**
```bash
# Check frontend server status
curl http://localhost:5001/api/health

# Check if port is available
netstat -tulpn | grep :5001

# Start frontend manually for debugging
cd frontend
source venv/bin/activate
python src/app.py

# Check browser console for JavaScript errors
# Check backend API connectivity from frontend
```

**8. Service Management Issues**
```bash
# Backend service management
cd backend/unitTests_scripts/shell_scripts
./manage-api-service.sh status
./manage-api-service.sh logs

# Frontend service management
sudo systemctl status beautyai-webui
sudo journalctl -u beautyai-webui -f

# Restart both services
sudo systemctl restart beautyai-api
sudo systemctl restart beautyai-webui
```

### Performance Optimization

**Memory Management**:
- Use 4-bit quantization for GPU VRAM efficiency
- Monitor memory usage with `beautyai system status`
- Unload unused models with `beautyai system unload-all`
- Clear cache regularly with `beautyai system clear-cache --all`

**Inference Speed**:
- Use Transformers backend with 4-bit quantization for production workloads
- Enable GPU acceleration with proper CUDA installation
- Use optimization-based presets: `qwen_optimized`, `speed_optimized`
- Optimize `max_new_tokens` for your use case (256 for voice, 512 for text)

**Voice Performance**:
- Use Edge TTS for ultra-fast voice synthesis (<2 seconds)
- Keep audio files small (< 10MB) for WebSocket streaming
- Use WAV format for best compatibility and quality
- Test microphone permissions and audio device access in browsers

**Web UI Performance**:
- Use modern browsers with WebSocket and WebRTC support
- Enable hardware acceleration in browser settings
- Test with different audio input devices if voice features are slow
- Monitor browser console for JavaScript errors or warnings

**Network Performance**:
- Use local deployment for best voice latency
- Configure proper DNS resolution for domain names
- Use SSL/TLS termination with Nginx for production
- Monitor network bandwidth for large model downloads

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
# Enable debug logging for all backend commands
cd backend
source venv/bin/activate
beautyai --verbose system status
beautyai --log-level DEBUG run chat

# Start backend server with debug logging
python run_server.py --log-level debug

# Start frontend with debug mode
cd ../frontend
source venv/bin/activate
python src/app.py --debug

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
4. **Test Components**: Test backend API and frontend web UI separately
5. **Community Support**: Open an issue with system info and error logs
6. **Configuration Backup**: Always backup working configurations before changes

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
- [Hugging Face Transformers Models](https://huggingface.co/models)
- [Edge TTS Voices](https://github.com/rany2/edge-tts)
- [Coqui TTS Models](https://huggingface.co/coqui)

### Technical References
- [Transformers Quantization](https://huggingface.co/docs/transformers/quantization)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [WebSocket Protocol](https://tools.ietf.org/html/rfc6455)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**BeautyAI Inference Framework** - Empowering Arabic AI and multilingual language model deployment with professional-grade tools, advanced voice capabilities, beautiful web interface, and scalable dual-stack architecture.
