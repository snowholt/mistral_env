# Troubleshooting Guide

Common issues and solutions for the BeautyAI Inference Framework.

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system check
beautyai system-check

# Check individual components
beautyai model status
beautyai api status  
beautyai voice status

# Check logs
sudo journalctl -u beautyai-api -f
sudo journalctl -u beautyai-webui -f
```

### Common Commands
```bash
# Check running services
sudo systemctl status beautyai-api beautyai-webui

# Check port availability
sudo netstat -tlnp | grep -E ':(8000|5001)'

# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Check memory usage
free -h
ps aux --sort=-%mem | head -10
```

## 🔧 Model Issues

### Model Loading Problems

#### Issue: "Model not found in registry"
```bash
# Check available models
beautyai model list

# Add model to registry
beautyai model add qwen3-14b-instruct \
  --engine transformers \
  --quantization 4bit

# Verify model entry
beautyai model info qwen3-14b-instruct
```

#### Issue: "CUDA out of memory"
```python
# Solution 1: Enable quantization
{
  "quantization": {
    "enabled": true,
    "type": "4bit",
    "compute_dtype": "float16"
  }
}

# Solution 2: Reduce max_new_tokens
{
  "generation_config": {
    "max_new_tokens": 512  # Reduce from default 1024
  }
}

# Solution 3: Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

#### Issue: "Model loading timeout"
```bash
# Increase timeout in configuration
export BEAUTYAI_MODEL_TIMEOUT=300  # 5 minutes

# Or edit config file
{
  "model_loading": {
    "timeout_seconds": 300,
    "retry_attempts": 3
  }
}
```

### Model Performance Issues

#### Issue: Slow inference speed
```bash
# Check quantization status
beautyai model info --include-performance

# Enable performance optimizations
{
  "performance": {
    "torch_compile": true,
    "flash_attention": true,
    "use_cache": true
  }
}

# Monitor GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
```

#### Issue: Poor quality responses
```python
# Adjust generation parameters
{
  "generation_config": {
    "temperature": 0.7,        # Lower for more focused responses
    "top_p": 0.9,             # Nucleus sampling
    "top_k": 50,              # Top-k sampling
    "repetition_penalty": 1.1, # Reduce repetition
    "do_sample": true         # Enable sampling
  }
}
```

## 🌐 API Issues

### Connection Problems

#### Issue: "Connection refused to API"
```bash
# Check if API service is running
sudo systemctl status beautyai-api

# Check port binding
sudo netstat -tlnp | grep 8000

# Restart API service
sudo systemctl restart beautyai-api

# Check API logs
sudo journalctl -u beautyai-api --no-pager -l
```

#### Issue: "502 Bad Gateway" (Nginx)
```bash
# Check Nginx status
sudo systemctl status nginx

# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Test upstream connection
curl -I http://localhost:8000/api/health

# Reload Nginx configuration
sudo nginx -t && sudo systemctl reload nginx
```

#### Issue: "CORS errors in browser"
```python
# Update CORS configuration in backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Authentication Issues

#### Issue: "Unauthorized" errors
```bash
# Check API key configuration
echo $BEAUTYAI_API_KEY

# Test with curl
curl -H "Authorization: Bearer $BEAUTYAI_API_KEY" \
     http://localhost:8000/api/chat

# Regenerate API key if needed
beautyai auth generate-key
```

## 🎤 Voice Issues

### WebSocket Problems

#### Issue: "WebSocket connection failed"
```bash
# Check WebSocket endpoint
curl -I -N \
     -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: test" \
     http://localhost:8000/ws/voice

# Check Nginx WebSocket configuration
sudo nginx -T | grep -A 10 "location /ws"
```

#### Issue: "No audio input detected"
```javascript
// Browser console debugging
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => console.log("Microphone access granted"))
  .catch(err => console.error("Microphone access denied:", err));

// Check browser permissions
// Chrome: chrome://settings/content/microphone
// Firefox: about:preferences#privacy
```

### Speech Recognition Issues

#### Issue: "Speech not recognized"
```python
# Check audio format and quality
{
  "audio_config": {
    "sample_rate": 16000,
    "encoding": "webm",
    "channels": 1
  }
}

# Enable debugging
{
  "voice": {
    "debug_audio": true,
    "save_audio_files": true
  }
}
```

#### Issue: "Language detection errors"
```python
# Force specific language
{
  "voice_config": {
    "language": "ar",  # Force Arabic
    "auto_detect": false
  }
}

# Check supported languages
beautyai voice languages
```

### Text-to-Speech Issues

#### Issue: "No audio response"
```bash
# Check Edge TTS installation
pip install edge-tts

# Test TTS directly
edge-tts --text "Hello world" --write-media test.wav

# Check voice availability
edge-tts --list-voices | grep -i arabic
```

#### Issue: "Poor voice quality"
```python
# Adjust TTS settings
{
  "tts_config": {
    "voice": "ar-SA-ZariyahNeural",
    "rate": "+0%",
    "pitch": "+0Hz",
    "volume": "+0%"
  }
}
```

## 💻 Frontend Issues

### UI Loading Problems

#### Issue: "Frontend not loading"
```bash
# Check frontend service
sudo systemctl status beautyai-webui

# Check port binding
sudo netstat -tlnp | grep 5001

# Check frontend logs
sudo journalctl -u beautyai-webui --no-pager -l

# Test direct access
curl http://localhost:5001/
```

#### Issue: "Static files not loading"
```bash
# Check file permissions
ls -la frontend/static/

# Check Nginx static file configuration
sudo nginx -T | grep -A 5 "location /static"

# Verify file paths
find frontend/static/ -name "*.css" -o -name "*.js"
```

### JavaScript Errors

#### Issue: "WebSocket connection errors in browser"
```javascript
// Browser console debugging
const ws = new WebSocket('ws://localhost:8000/ws/voice');
ws.onopen = () => console.log('WebSocket connected');
ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = (event) => console.log('WebSocket closed:', event.code, event.reason);
```

#### Issue: "API calls failing"
```javascript
// Check API configuration
console.log('API URL:', window.CONFIG.api_url);

// Test API connection
fetch(window.CONFIG.api_url + '/api/health')
  .then(response => response.json())
  .then(data => console.log('API health:', data))
  .catch(error => console.error('API error:', error));
```

## 🔧 System Issues

### Performance Problems

#### Issue: "High CPU usage"
```bash
# Identify CPU-intensive processes
top -p $(pgrep -d',' python)

# Check thread usage
cat /proc/$(pgrep -f "run_server.py")/status | grep Threads

# Optimize CPU settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### Issue: "High memory usage"
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python run_server.py

# Configure memory limits
{
  "system": {
    "max_memory_gb": 16,
    "memory_cleanup_interval": 300
  }
}
```

#### Issue: "GPU memory leaks"
```python
# Monitor GPU memory
import torch
print(torch.cuda.memory_summary())

# Clear CUDA cache periodically
torch.cuda.empty_cache()

# Enable memory debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

### Storage Issues

#### Issue: "Disk space full"
```bash
# Check disk usage
df -h

# Find large files
find /opt/beautyai -type f -size +1G -exec ls -lh {} \;

# Clean model cache
rm -rf ~/.cache/huggingface/transformers/

# Clean logs
sudo journalctl --vacuum-time=7d
find /var/log -name "*.log" -type f -mtime +7 -delete
```

#### Issue: "Model download failures"
```bash
# Check internet connectivity
curl -I https://huggingface.co

# Check Hugging Face cache
ls -la ~/.cache/huggingface/

# Manual model download
git lfs clone https://huggingface.co/Qwen/Qwen1.5-14B-Chat

# Set download directory
export HF_HOME=/opt/beautyai/models
```

## 🛠️ Configuration Issues

### Environment Problems

#### Issue: "Python environment conflicts"
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|transformers|fastapi)"

# Create clean environment
python -m venv venv_clean
source venv_clean/bin/activate
pip install -r requirements.txt
```

#### Issue: "CUDA version mismatch"
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall correct PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration File Issues

#### Issue: "Configuration not loading"
```bash
# Check file permissions
ls -la backend/src/model_registry.json

# Validate JSON syntax
python -m json.tool backend/src/model_registry.json

# Check environment variables
env | grep BEAUTYAI
```

#### Issue: "Invalid model configuration"
```python
# Validate configuration
import json
with open('model_registry.json') as f:
    config = json.load(f)
    
# Check required fields
required_fields = ['model_name', 'engine_type', 'model_path']
for model in config.get('models', []):
    for field in required_fields:
        if field not in model:
            print(f"Missing field {field} in model {model.get('model_name', 'unknown')}")
```

## 🔍 Debugging Tools

### Logging Configuration
```python
# Enable detailed logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Component-specific logging
loggers = {
    'beautyai.model': logging.DEBUG,
    'beautyai.api': logging.INFO,
    'beautyai.voice': logging.DEBUG,
}
```

### Performance Profiling
```python
# Profile API endpoints
import cProfile
import pstats

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Memory Debugging
```python
# Track memory usage
import tracemalloc
import psutil

tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## 📞 Getting Help

### Log Collection
```bash
# Collect system logs
sudo journalctl -u beautyai-api -u beautyai-webui --since="1 hour ago" > beautyai_logs.txt

# Collect system information
{
    echo "=== System Info ==="
    uname -a
    python --version
    pip list
    nvidia-smi
    
    echo "=== Service Status ==="
    systemctl status beautyai-api beautyai-webui
    
    echo "=== Port Status ==="
    netstat -tlnp | grep -E ':(8000|5001)'
    
    echo "=== Recent Logs ==="
    journalctl -u beautyai-api --since="10 minutes ago" --no-pager
} > debug_info.txt
```

### Performance Report
```bash
# Generate performance report
beautyai benchmark --duration 60 --output performance_report.json

# System resource report
{
    echo "=== CPU Info ==="
    lscpu
    
    echo "=== Memory Info ==="
    free -h
    
    echo "=== GPU Info ==="
    nvidia-smi -q
    
    echo "=== Storage Info ==="
    df -h
} > system_report.txt
```

---

**Need more help?** Check the [Configuration Guide](CONFIGURATION.md) or [Performance Guide](PERFORMANCE.md)
