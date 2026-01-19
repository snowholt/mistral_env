# BeautyAI Backend API

FastAPI-based backend server providing real-time voice-to-voice AI conversations using WebRTC.

## 🚀 Quick Start

```bash
# Setup environment
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian - required for WebRTC)
sudo apt-get update
sudo apt-get install -y libopus0 libopus-dev libvpx7 libvpx-dev \
    libsrtp2-1 libsrtp2-dev libssl-dev

# Start the server
python run_server.py
```

**API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
backend/
├── run_server.py           # Server entry point
├── setup.py                # Package configuration
├── requirements.txt        # Dependencies
├── alembic/                # Database migrations
├── rnnoise/                # Noise reduction library
├── src/beautyai_inference/
│   ├── api/                # FastAPI endpoints
│   │   └── endpoints/
│   │       └── webrtc_voice.py  # WebRTC voice (primary)
│   ├── services/           # Business logic
│   │   ├── voice/          # Voice processing
│   │   ├── inference/      # LLM inference
│   │   └── email/          # Email service
│   ├── inference_engines/  # Model backends
│   │   ├── llamacpp_engine.py
│   │   └── voice/stt/      # Faster-Whisper STT
│   ├── core/               # Model management
│   └── config/             # Configuration loaders
└── scripts/                # Utility scripts
```

## 🎤 Voice Pipeline

```
Browser Audio (48kHz) → WebRTC → Jitter Buffer → Resample (16kHz)
    → RNNoise → Silero VAD → Faster-Whisper STT
    → Qwen/Llama.cpp LLM → Edge TTS → Audio Response
```

**Latency**: <2s end-to-end response time

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/docs` | GET | Swagger documentation |
| `/api/v1/webrtc/voice` | POST | WebRTC voice endpoint |
| `/inference/chat` | POST | Text chat |
| `/models` | GET | List models |
| `/models/{name}/load` | POST | Load model |

## ⚙️ Configuration

Configuration files are located at the **project root** `config/` directory:

- `config/config.yaml` - Main configuration
- `config/models/model_registry.json` - Model definitions
- `config/models/voice_models_registry.json` - Voice model config
- `config/models/preload_config.json` - Startup preload config

## 🔧 Environment Variables

```bash
# Server
BEAUTYAI_HOST=0.0.0.0
BEAUTYAI_PORT=8000
BEAUTYAI_ENV=production
UVICORN_RELOAD=0

# Voice
VOICE_STREAMING_ENABLED=1
VOICE_MINIMAL_MODE=1

# Database
DATABASE_URL=postgresql://user:pass@localhost/beautyai

# Auth
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
```

## 🚀 Production Deployment

### Systemd Service
```bash
# The service file is at /etc/systemd/system/beautyai-api.service
sudo systemctl start beautyai-api
sudo systemctl enable beautyai-api
sudo systemctl status beautyai-api
```

### Database Migrations
```bash
cd backend
alembic upgrade head
```

## 🐛 Troubleshooting

```bash
# Check GPU memory
nvidia-smi

# Check service status
sudo systemctl status beautyai-api

# View logs
tail -f ../logs/api/api_app.jsonl

# Test health
curl http://localhost:8000/health
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Voice Guide**: `../docs/VOICE.md`
- **Architecture**: `../docs/ARCHITECTURE.md`

---

For frontend documentation, see [`../frontend/README.md`](../frontend/README.md)
