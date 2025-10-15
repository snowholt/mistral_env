# Configuration Guide

Comprehensive configuration options for the BeautyAI Inference Framework.

## 📋 Model Registry

The model registry (`backend/src/model_registry.json`) is the central configuration for all AI models.

### Default Configuration
```json
{
  "default_model": "default",
  "models": {
    "default": {
      "model_id": "Qwen/Qwen3-14B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "dtype": "float16",
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95,
      "do_sample": true,
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "name": "default",
      "description": "Default Qwen model with 4-bit quantization"
    }
  }
}
```

### Adding Custom Models
```bash
# Via CLI
beautyai model add --name "custom-model" \
                   --model-id "organization/model-name" \
                   --engine transformers \
                   --quantization 4bit \
                   --description "Custom model description"

# Via direct JSON editing
# Edit backend/src/model_registry.json
```

## ⚙️ Quantization Options

### Transformers Backend (Recommended)
- **4bit**: Best balance of speed and quality (`--quantization 4bit`)
- **8bit**: Higher quality, more memory (`--quantization 8bit`)
- **none**: Full precision, maximum memory (`--quantization none`)

### LlamaCpp Backend (Alternative)
- **Q4_K_S**: Fastest inference (~8GB)
- **Q4_K_M**: Balanced performance (~8.4GB)
- **Q6_K**: High quality (~12GB)

## 🎛️ Chat Parameters

### Core Parameters
```bash
# Temperature: Creativity (0.1-1.0)
beautyai run chat --temperature 0.3

# Top-p: Nucleus sampling (0.1-1.0)
beautyai run chat --top-p 0.95

# Max tokens: Response length
beautyai run chat --max-tokens 512
```

### Advanced Parameters
```bash
# All available parameters
beautyai run chat --help

# Example with multiple parameters
beautyai run chat --temperature 0.3 \
                  --top-p 0.95 \
                  --top-k 20 \
                  --repetition-penalty 1.1 \
                  --thinking-mode
```

### Optimization Presets
- `qwen_optimized`: Best tested settings (temp=0.3, top_p=0.95, top_k=20)
- `high_quality`: Maximum quality (temp=0.1, top_p=1.0, rep_penalty=1.15)
- `creative_optimized`: Creative but efficient (temp=0.5, top_p=1.0, top_k=80)
- `speed_optimized`: Fastest response
- `balanced`: Good balance of quality and speed

## 🎤 Voice Configuration

### Edge TTS (Primary)
```bash
# No configuration needed - works out of the box
# Supports 20+ languages including Arabic and English
# Ultra-fast response times (<2 seconds)
```

### Voice Parameters
```bash
# Language detection: auto, ar, en, etc.
# Voice type: default, male, female
# Used via WebSocket API or Web UI
```

## 🌐 API Configuration

### Backend API Settings
```python
# In backend/run_server.py or environment
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 1  # Single worker for development
RELOAD = True  # Auto-reload for development
```

### Frontend Web UI Settings
```python
# In frontend/src/config.json
{
  "api_url": "http://localhost:8000",
  "websocket_url": "ws://localhost:8000/ws/voice-conversation",
  "voice_enabled": true,
  "theme": "dark",
  "animations_enabled": true
}
```

### Environment Variables
```bash
# Backend
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/beautyai/backend

# Frontend
export BEAUTYAI_API_URL=http://localhost:8000
export BEAUTYAI_WS_URL=ws://localhost:8000/ws/voice-conversation
export PORT=5001
export FLASK_ENV=development
```

## 🚀 Production Configuration

### System Service Configuration
```ini
# beautyai-api.service
[Unit]
Description=BeautyAI Inference API Server
After=network.target

[Service]
Type=simple
User=lumi
WorkingDirectory=/home/lumi/beautyai/backend
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/lumi/beautyai/backend/venv/bin/python run_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Nginx Configuration
```nginx
# SSL configuration for production
server {
    listen 443 ssl http2;
    server_name api.gmai.sa;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🔧 Memory Optimization

### GPU Memory Settings
```bash
# Reduce GPU memory usage
beautyai config set gpu_memory_utilization 0.85

# Use CPU fallback for large models
beautyai run chat --force-cpu

# Clear cache regularly
beautyai system clear-cache --all
```

### Model Loading Optimization
```bash
# Load models on demand
beautyai system load my-model

# Unload unused models
beautyai system unload my-model

# Monitor memory usage
beautyai system status
```

## 📊 Performance Tuning

### Inference Speed
```bash
# Use 4-bit quantization
beautyai model update default --quantization 4bit

# Optimize token generation
beautyai run chat --max-tokens 256

# Use speed preset
beautyai run chat --preset speed_optimized
```

### Voice Performance
```bash
# Edge TTS is optimized by default
# For best latency:
# - Use local deployment
# - Ensure good network connection
# - Use modern browser with WebRTC support
```

## 🎙️ WebRTC Voice Configuration (Phase A)

### Environment Variables

**Backend (`config/config.yaml` or environment)**:
```yaml
webrtc:
  enabled: false  # Set to true to enable WebRTC mode
  max_utterance_sec: 10  # Maximum utterance duration (1-60 seconds)
  signaling_path: "/api/v1/webrtc/voice"
  
  # ICE Servers
  stun_servers:
    - "stun:stun.l.google.com:19302"
  turn_servers: []  # Add for production NAT traversal
  
  # VAD Configuration
  vad_webrtc_sensitivity: 3  # 0-3 (3 = most aggressive)
  vad_silero_sensitivity: 0.45  # English threshold (0-1)
  vad_silero_sensitivity_arabic: 0.5  # Arabic threshold (0-1)
  
  # Buffer Configuration
  buffer_pre_roll_ms: 300  # Pre-recording buffer
  buffer_post_roll_ms: 300  # Post-recording buffer
  
  # LLM Integration
  auto_no_think_prefix: true  # Auto-inject /no_think prefix
  
  # Debugging
  debug_logging: false  # Enable aiortc debug logs
```

**Frontend (`.env.webrtc.example`)**:
```bash
# Feature Toggle
REACT_APP_VOICE_WEBRTC_ENABLED=false

# Utterance Configuration
REACT_APP_VOICE_WEBRTC_MAX_UTTERANCE_SEC=10

# Debug Options
REACT_APP_VOICE_WEBRTC_DEBUG=false
```

### System Dependencies

WebRTC requires system libraries for audio/video codec support:

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y libopus0 libopus-dev \
                        libvpx7 libvpx-dev \
                        libsrtp2-1 libsrtp2-dev \
                        libssl-dev
```

**CentOS/RHEL**:
```bash
sudo yum install -y opus opus-devel \
                    libvpx libvpx-devel \
                    libsrtp libsrtp-devel \
                    openssl-devel
```

**macOS** (via Homebrew):
```bash
brew install opus libvpx srtp openssl
```

### Configuration Validation

```bash
# Check WebRTC configuration
python -c "from beautyai_inference.config import load_config; \
           cfg = load_config(); \
           print('WebRTC enabled:', cfg.get('webrtc', {}).get('enabled', False))"

# Verify system dependencies
python -c "import aiortc; print('aiortc version:', aiortc.__version__)"
```

### Enabling WebRTC Mode

1. **Install system dependencies** (see above)
2. **Update backend configuration**:
   ```bash
   # Edit config/config.yaml
   webrtc:
     enabled: true
     max_utterance_sec: 10
   ```
3. **Update frontend configuration**:
   ```bash
   # Create/edit frontend/.env.local
   echo "REACT_APP_VOICE_WEBRTC_ENABLED=true" >> frontend/.env.local
   ```
4. **Restart services**:
   ```bash
   ./backend/unitTests_scripts/shell_scripts/manage-api-service.sh restart
   ```

### Fallback Behavior

If WebRTC is not supported by the browser or fails to negotiate:
- Frontend automatically falls back to WebSocket mode
- Check browser console for: `[WebRTC] Browser does not support WebRTC, falling back to WebSocket`
- Monitor logs: `tail -f logs/api/webrtc_voice.jsonl`

## 🐛 Configuration Troubleshooting

### Common Issues

**Model Loading Fails**:
```bash
# Check model configuration
beautyai model show default

# Validate configuration
beautyai config validate

# Reset to defaults
beautyai config reset --confirm
```

**Out of Memory**:
```bash
# Use smaller quantization
beautyai model update default --quantization 4bit

# Reduce max tokens
beautyai run chat --max-tokens 256

# Check GPU memory
nvidia-smi
```

**API Connection Issues**:
```bash
# Check API status
curl http://localhost:8000/health

# Verify configuration
beautyai config show

# Check logs
./backend/unitTests_scripts/shell_scripts/manage-api-service.sh logs
```

---

**Next**: [API Documentation](API.md) | [Voice Features](VOICE.md)
