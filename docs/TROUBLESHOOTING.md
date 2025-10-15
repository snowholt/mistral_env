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

## 🌐 WebRTC Voice Issues

### WebRTC Connection Problems

#### Issue: "WebRTC connection failed" or stuck on "Connecting..."

**Possible Causes:**
- Network firewall blocking WebRTC traffic
- STUN server unreachable
- Browser permissions not granted
- Backend WebRTC endpoints not responding

**Solutions:**

1. **Check Browser Support:**
```javascript
// Run in browser console
console.log('RTCPeerConnection:', typeof RTCPeerConnection !== 'undefined');
console.log('getUserMedia:', navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function');
```

2. **Verify Backend Health:**
```bash
# Check WebRTC health endpoint
curl https://dev.gmai.sa/api/v1/webrtc/voice/health

# Expected response:
# {"status": "healthy", "webrtc_available": true}
```

3. **Test Signaling Endpoints:**
```bash
# Use WebRTC signaling probe tool
python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa --verbose

# This will test:
# - Health check
# - SDP offer/answer exchange
# - ICE candidate exchange
# - Connection status
# - Cleanup
```

4. **Check Firewall Rules:**
```bash
# WebRTC typically uses UDP ports for media
# Ensure UDP traffic is not blocked

# Check if STUN server is reachable
nc -zvu stun.l.google.com 19302

# Check if API endpoints are accessible
curl -I https://dev.gmai.sa/api/v1/webrtc/voice/offer
```

5. **Review Backend Logs:**
```bash
# Search for WebRTC-related errors
sudo journalctl -u beautyai-api | grep -i webrtc | tail -50

# Use enhanced service analyzer
python tools/service_analyzer.py --analyze --lines 500
```

#### Issue: ICE Negotiation Timeout

**Symptoms:** Connection gets stuck at "checking" or "gathering" state

**Solutions:**

1. **Check ICE Connection State:**
```javascript
// Enable debug overlay (add ?debug=1 to URL)
// Or run in browser console:
const stats = await webrtcClient.getConnectionStats();
console.log('ICE Connection State:', stats.connection.iceState);
console.log('ICE Gathering State:', stats.connection.gatheringState);
```

2. **Verify STUN Server Configuration:**
```yaml
# config/config.yaml
webrtc:
  stun_servers:
    - "stun:stun.l.google.com:19302"
    - "stun:stun1.l.google.com:19302"  # Backup STUN server
```

3. **Consider TURN Server (for restrictive networks):**
```yaml
# config/config.yaml
webrtc:
  turn_servers:
    - url: "turn:your-turn-server.com:3478"
      username: "your_username"
      credential: "your_password"
```

**Note:** TURN server setup is documented in `docs/DEPLOYMENT.md` (coturn installation).

#### Issue: No Audio Playback (TTS Silent)

**Symptoms:** Connection succeeds, but no TTS audio is heard

**Solutions:**

1. **Check Browser Audio:**
```javascript
// Run in browser console
const audioElement = document.getElementById('webrtc-remote-audio');
console.log('Audio element:', audioElement);
console.log('Audio source:', audioElement?.srcObject);
console.log('Audio volume:', audioElement?.volume);
console.log('Audio muted:', audioElement?.muted);
```

2. **Verify Remote Track Reception:**
```javascript
// Check for 'ontrack' event in browser console logs
// Should see: "[WebRTC] 🎵 Remote track received: audio"

// Check if remote stream has active tracks
console.log('Remote stream tracks:', webrtcClient.remoteStream?.getTracks());
```

3. **Check chrome://webrtc-internals:**
- Open new tab: `chrome://webrtc-internals`
- Look for "inbound-rtp" stats with `kind: audio`
- Verify `packetsReceived` is increasing
- Check `bytesReceived` is non-zero

4. **Test with WebSocket Mode:**
```javascript
// Switch to WebSocket mode to isolate WebRTC issue
// Use mode toggle in UI or set default mode in config
```

### WebRTC Audio Quality Issues

#### Issue: Choppy or Distorted Audio

**Possible Causes:**
- High packet loss
- Network jitter
- Insufficient bandwidth
- CPU overload

**Diagnostics:**

1. **Enable Debug Overlay:**
```
Add ?debug=1 to URL: https://dev.gmai.sa?debug=1
```

The debug overlay shows real-time stats:
- Packet loss rate (should be < 1%)
- Jitter (should be < 30ms)
- Bitrate (should be 32-48 kbps for Opus)

2. **Check Network Stats:**
```bash
# Check network latency
ping dev.gmai.sa

# Check bandwidth
speedtest-cli

# Monitor network during WebRTC session
iftop  # or nethogs
```

3. **Review Audio Processor Logs:**
```bash
# Search for audio quality warnings
sudo journalctl -u beautyai-api | grep -E "packet loss|jitter|bitrate" | tail -20
```

**Solutions:**

1. **Reduce Network Congestion:**
- Close bandwidth-intensive applications
- Use wired connection instead of WiFi
- Reduce number of concurrent WebRTC sessions

2. **Adjust Audio Constraints (if needed):**
```javascript
// In webrtc.config.js, adjust sample rate
audioConstraints: {
    sampleRate: 16000,  // Reduce from 48000 for lower bandwidth
    // ... other constraints
}
```

3. **Enable Adaptive Bitrate (future enhancement):**
Currently not implemented, tracked for Phase E+.

#### Issue: Echo or Feedback

**Note:** WebRTC uses browser's built-in AEC (Acoustic Echo Cancellation)

**Verify AEC is Enabled:**
```javascript
// In webrtc.config.js
audioConstraints: {
    echoCancellation: true,      // Must be true
    noiseSuppression: true,
    autoGainControl: true,
    // ...
}
```

**If Echo Persists:**
1. Use headphones instead of speakers
2. Reduce speaker volume
3. Check browser audio settings
4. Try different browser (Safari AEC differs from Chrome)

### WebRTC VAD (Voice Activity Detection) Issues

#### Issue: Speech Not Detected or False Positives

**Symptoms:**
- User speaks but no transcription happens
- Silent audio triggers transcription (false positives)

**Solutions:**

1. **Check VAD Thresholds:**
```yaml
# config/config.yaml
webrtc:
  vad:
    silero_threshold_arabic: 0.45  # Lower = more sensitive
    silero_threshold_english: 0.50
    webrtc_vad_aggressiveness: 2   # 0-3, higher = less sensitive
```

2. **Enable VAD Debug Logging:**
```yaml
# config/config.yaml
logging:
  level: DEBUG
  
webrtc:
  debug_logging: true
```

3. **Test with Different Microphones:**
- Built-in laptop mic vs external USB mic
- Different mic sensitivity settings
- Check mic permissions in browser

4. **Review VAD Statistics:**
```bash
# Check for VAD false positive count
python tools/service_analyzer.py --analyze | grep -A 5 "WEBRTC STATISTICS"
```

### WebRTC Debug Tools

#### Browser-Based Debugging

1. **Enable Debug Overlay:**
```
URL: https://dev.gmai.sa?debug=1
```

Features:
- Real-time connection state
- Audio bitrate, packets, jitter, packet loss
- ICE candidate information
- Performance metrics
- Quick access to chrome://webrtc-internals

2. **chrome://webrtc-internals (Chrome/Edge):**
```
Open in new tab: chrome://webrtc-internals
```

Provides:
- Complete WebRTC statistics
- ICE candidate negotiations
- Audio/video codec information
- Graphs for bitrate, packet loss, etc.

3. **about:webrtc (Firefox):**
```
Open in new tab: about:webrtc
```

Similar to Chrome's webrtc-internals.

#### Server-Side Debugging

1. **WebRTC Signaling Probe:**
```bash
# Test all endpoints
python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa

# Test specific endpoint
python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa --test offer

# Verbose mode
python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa --verbose
```

2. **Service Analyzer (Enhanced with WebRTC Stats):**
```bash
# Quick summary
python tools/service_analyzer.py --summary

# Detailed analysis with WebRTC stats
python tools/service_analyzer.py --analyze --lines 500
```

3. **Backend Logs:**
```bash
# Follow WebRTC-specific logs
sudo journalctl -u beautyai-api -f | grep -i "webrtc\|ice\|sdp"

# Export WebRTC logs
sudo journalctl -u beautyai-api --since="1 hour ago" | grep -i webrtc > webrtc_debug.log
```

### WebRTC Performance Tuning

#### Latency Optimization

**Target SLOs (90th percentile):**
- Round-trip: ≤ 6 seconds
- STT: ≤ 2 seconds
- LLM: ≤ 3 seconds (with `/no_think` prefix)
- TTS: ≤ 1 second

**Measure Current Latency:**
```javascript
// Use debug overlay (?debug=1) to view real-time latency

// Or export stats for analysis
// Click "Export Stats" in debug overlay
```

**Optimization Strategies:**

1. **Enable `/no_think` Prefix (Automatic):**
Already enabled in Phase C, reduces LLM latency by 40-60%.

2. **Use Quantized Models:**
```yaml
# config/config.yaml
models:
  qwen3_14b_instruct:
    quantization:
      enabled: true
      type: "4bit"  # Balance quality and speed
```

3. **Optimize TTS Voice Caching:**
```yaml
# config/config.yaml
tts:
  enable_voice_caching: true
  cache_size_mb: 500
```

4. **Monitor Processing Overhead:**
```bash
# Use performance monitor
python tools/service_analyzer.py --analyze
```

#### Memory Optimization

**Expected Memory Footprint per WebRTC Session:** ~40 MB

**Monitor Memory Usage:**
```bash
# Check backend memory
ps aux | grep "python.*beautyai" | awk '{print $6/1024 " MB"}'

# Detailed memory analysis
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**If Memory Usage Excessive:**
1. Check for memory leaks in logs
2. Restart service periodically
3. Reduce concurrent session limit
4. Enable aggressive garbage collection

### Common Error Messages

#### "aiortc not available"
```bash
# Install aiortc dependencies
pip install aiortc>=1.5.0

# If compilation fails, install system dependencies:
sudo apt-get update
sudo apt-get install -y libopus-dev libvpx-dev libsrtp2-dev

# Verify installation
python -c "import aiortc; print(aiortc.__version__)"
```

#### "WebRTC not enabled in configuration"
```yaml
# config/config.yaml
webrtc:
  enabled: true
```

#### "STUN server timeout"
```bash
# Test STUN connectivity
nc -zvu stun.l.google.com 19302

# If blocked, configure alternative STUN or use TURN
```

#### "Peer connection not found"
Backend cleaned up the peer connection (idle timeout or manual cleanup).

**Solution:** Reconnect from client.

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
