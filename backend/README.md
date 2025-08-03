# BeautyAI Backend API

FastAPI-based backend server providing inference capabilities for language models with voice features.

## 🚀 Quick Start

```bash
# Setup and installation
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the API server
python run_server.py
```

**API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
backend/
├── run_server.py                    # FastAPI server entry point
├── setup.py                        # Package configuration
├── requirements.txt                 # Dependencies
├── src/beautyai_inference/
│   ├── cli/                        # Unified CLI interface
│   ├── api/                        # FastAPI endpoints and schemas
│   ├── services/                   # Business logic services
│   ├── inference_engines/          # Model backends (Transformers, LlamaCpp)
│   ├── core/                       # Factory patterns and model management
│   └── config/                     # Configuration management
└── unitTests_scripts/              # Testing and setup utilities
```

## 🔧 Core Services

### Model Management
- **Registry Service**: Centralized model configuration
- **Lifecycle Management**: Loading, unloading, memory monitoring
- **Validation**: Model compatibility and configuration validation

### Inference Services
- **Chat Service**: Text-based conversations with 25+ parameters
- **Benchmark Service**: Performance testing and optimization
- **Session Management**: Conversation history and persistence

### Voice Services
- **Simple Voice Service**: Ultra-fast WebSocket voice chat (<2s response)
- **Transcription**: Whisper-based speech-to-text
- **Synthesis**: Edge TTS for fast multilingual speech synthesis

### System Services
- **Memory Monitoring**: GPU/CPU usage tracking
- **Cache Management**: Model cache cleanup and optimization
- **Configuration**: JSON-based config with validation and backup

## 🌐 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /inference/chat` - Text chat with advanced parameters
- `WebSocket /ws/voice-conversation` - Real-time voice chat

### Model Management
- `GET /models` - List available models
- `POST /models` - Add new model configuration
- `POST /models/{name}/load` - Load model into memory
- `DELETE /models/{name}/unload` - Unload model

### Voice Features
- `GET /api/v1/voice/endpoints` - Available voice endpoints
- `GET /api/v1/health/voice` - Voice services health check

## 🎯 CLI Interface

```bash
# Activate environment first
source venv/bin/activate

# Primary commands
beautyai --help                     # Show all commands
beautyai run chat                   # Interactive chat
beautyai run test                   # Model testing
beautyai model list                 # List models
beautyai system status              # System monitoring
```

## ⚙️ Configuration

### Model Registry (`src/model_registry.json`)
```json
{
  "default_model": "default",
  "models": {
    "default": {
      "model_id": "Qwen/Qwen3-14B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "description": "Default Qwen model with 4-bit quantization"
    }
  }
}
```

### Adding Custom Models
```bash
beautyai model add --name "custom-model" \
                   --model-id "organization/model-name" \
                   --engine transformers \
                   --quantization 4bit
```

## 🔧 Development

### Setup Development Environment
```bash
# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
python -m pytest unitTests_scripts/ -v
```

### Code Quality
```bash
# Formatting
black src/beautyai_inference/
isort src/beautyai_inference/

# Linting
flake8 src/beautyai_inference/
mypy src/beautyai_inference/
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Check GPU memory
nvidia-smi

# Verify model exists
beautyai model list

# Clear cache and retry
beautyai system clear-cache
```

**API Connection Issues**:
```bash
# Check server status
curl http://localhost:8000/health

# View server logs
python run_server.py --log-level debug
```

**Voice Service Issues**:
```bash
# Check voice endpoints
curl http://localhost:8000/api/v1/voice/endpoints

# Test voice health
curl http://localhost:8000/api/v1/health/voice
```

## 🚀 Production Deployment

### Systemd Service
```bash
# Install service
cd unitTests_scripts/shell_scripts
./manage-api-service.sh install
./manage-api-service.sh start

# Check status
./manage-api-service.sh status
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/beautyai/backend
```

## 📚 API Reference

- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add type hints and docstrings
3. Write unit tests for new features
4. Update API documentation
5. Test with multiple model configurations

---

For frontend documentation, see [`../frontend/README.md`](../frontend/README.md)  
For main project overview, see [`../README.md`](../README.md)
