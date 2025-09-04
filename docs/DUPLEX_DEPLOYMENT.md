# Full Duplex Voice Streaming Deployment Guide

This guide provides comprehensive instructions for deploying, configuring, and troubleshooting the full duplex voice-to-voice streaming system in BeautyAI.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Testing & Validation](#testing--validation)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Monitoring](#monitoring)

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows 10+
- **Python**: 3.12 or higher
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for production
- **GPU**: CUDA-compatible GPU recommended for large models
- **Network**: Low-latency internet connection (< 100ms RTT for cloud deployment)

### Audio Hardware Requirements
- **Microphone**: USB or built-in microphone with echo cancellation support
- **Speakers/Headphones**: Separate audio output device (headphones strongly recommended)
- **Audio Interface**: ASIO-compatible audio interface for professional setups

### Dependencies
```bash
# Core dependencies
pip install edge-tts>=6.1.9
pip install scipy>=1.10.0
pip install websockets>=11.0.2
pip install numpy>=1.24.0

# Optional performance dependencies
pip install uvloop  # Linux/macOS only
pip install pyaudio  # Alternative audio backend
```

## Installation

### 1. Install Dependencies

```bash
# Navigate to backend directory
cd backend/

# Install Python dependencies
source venv/bin/activate
pip install -r requirements.txt

# Install additional duplex streaming dependencies
pip install edge-tts scipy websockets

# Optional: Install development dependencies for testing
pip install pytest pytest-asyncio websockets-test
```

### 2. Verify Installation

```bash
# Test echo detection utility
python -c "
from beautyai_inference.services.voice.utils import create_echo_detector
detector = create_echo_detector()
print('Echo detector initialized successfully')
"

# Test TTS engine
python -c "
import edge_tts
import asyncio
async def test():
    tts = edge_tts.Communicate('Hello world', 'en-US-AriaNeural')
    audio_data = b''
    async for chunk in tts.stream():
        if chunk['type'] == 'audio':
            audio_data += chunk['data']
    print(f'TTS generated {len(audio_data)} bytes')
asyncio.run(test())
"
```

### 3. Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend/

# No additional npm packages required - duplex components use vanilla JS
# Verify existing dependencies are installed
npm install
```

## Configuration

### 1. Backend Configuration

Create or update `backend/config/duplex_config.json`:

```json
{
  "duplex_streaming": {
    "enabled": true,
    "mode": "full",
    "echo_suppression": {
      "enabled": true,
      "correlation_threshold": 0.3,
      "spectral_threshold": 0.4,
      "vad_threshold": 0.02,
      "adaptive_threshold": true,
      "barge_in_delay_ms": 300,
      "resume_delay_ms": 500
    },
    "tts_streaming": {
      "chunk_size_ms": 30,
      "format": "opus",
      "voice": "en-US-AriaNeural",
      "sample_rate": 48000,
      "bitrate": 64000
    },
    "jitter_buffer": {
      "target_size_ms": 80,
      "max_size_ms": 200,
      "adaptive_sizing": true
    },
    "performance": {
      "max_concurrent_sessions": 10,
      "enable_metrics": true,
      "log_level": "INFO"
    }
  }
}
```

### 2. Environment Variables

Add to your `.env` file or export:

```bash
# Enable duplex streaming
VOICE_STREAMING_DUPLEX=1

# Enable enhanced metrics
VOICE_STREAMING_METRICS_JSON=1

# TTS configuration
EDGE_TTS_VOICE_DEFAULT="en-US-AriaNeural"
EDGE_TTS_VOICE_ARABIC="ar-SA-ZariyahNeural"

# Performance settings
DUPLEX_MAX_SESSIONS=10
DUPLEX_JITTER_BUFFER_MS=80

# Debug settings (development only)
DUPLEX_DEBUG_LOGGING=1
DUPLEX_SAVE_AUDIO_DEBUG=0
```

### 3. Audio Device Configuration

#### Linux (ALSA/PulseAudio)
```bash
# List available audio devices
aplay -l
arecord -l

# Configure PulseAudio for low latency
echo "default-sample-rate = 48000" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 16000" >> ~/.pulse/daemon.conf
echo "default-fragments = 2" >> ~/.pulse/daemon.conf
echo "default-fragment-size-msec = 5" >> ~/.pulse/daemon.conf

# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

#### macOS (Core Audio)
```bash
# Check available devices
system_profiler SPAudioDataType

# Set sample rate (optional, handled automatically)
sudo audio_utils set-sample-rate 48000
```

#### Windows (WASAPI)
- Use Windows Sound Settings to configure default devices
- Ensure exclusive mode is enabled for low latency
- Install ASIO4ALL for professional audio interfaces

## Deployment

### 1. Development Deployment

```bash
# Start backend with duplex support
cd backend/
export VOICE_STREAMING_DUPLEX=1
python run_server.py

# Or use VS Code task
# Task: "Dev: Run API (direct uvicorn script)"
```

### 2. Production Deployment

#### Using systemd (Linux)

Update `/etc/systemd/system/beautyai-api.service`:

```ini
[Unit]
Description=BeautyAI API Server with Duplex Streaming
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/beautyai/backend
Environment=VOICE_STREAMING_DUPLEX=1
Environment=VOICE_STREAMING_METRICS_JSON=1
Environment=EDGE_TTS_VOICE_DEFAULT=en-US-AriaNeural
ExecStart=/opt/beautyai/backend/venv/bin/python run_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Reload and restart service
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api
sudo systemctl enable beautyai-api
```

#### Using Docker

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  beautyai-api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - VOICE_STREAMING_DUPLEX=1
      - VOICE_STREAMING_METRICS_JSON=1
      - EDGE_TTS_VOICE_DEFAULT=en-US-AriaNeural
      - DUPLEX_MAX_SESSIONS=20
    volumes:
      - ./backend/logs:/app/logs
      - /dev/snd:/dev/snd  # Audio device access (Linux)
    devices:
      - /dev/snd  # Audio devices
    privileged: true  # Required for audio access
```

### 3. Nginx Configuration

Update nginx configuration for WebSocket support:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # WebSocket upgrade configuration
    location /api/v1/ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Duplex streaming optimizations
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        
        # Binary data support
        client_max_body_size 10M;
        proxy_request_buffering off;
    }

    # Static files
    location /static/ {
        alias /opt/beautyai/frontend/src/static/;
        expires 1d;
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Testing & Validation

### 1. Automated Tests

```bash
# Run duplex streaming tests
cd backend/
pytest tests/test_duplex_streaming.py -v

# Run specific test categories
pytest tests/test_duplex_streaming.py::TestEchoDetection -v
pytest tests/test_duplex_streaming.py::TestBinaryProtocol -v
pytest tests/test_duplex_streaming.py::TestDuplexStreamingIntegration -v
```

### 2. Manual Testing

#### Echo Test
1. Open debug streaming page: `http://localhost:8000/debug/streaming`
2. Select appropriate microphone and speaker devices
3. Enable echo cancellation and duplex mode
4. Click "Run Echo Test"
5. Verify correlation values are below 0.3

#### Full Duplex Test
1. Connect to streaming endpoint
2. Start speaking - should see live transcription
3. Wait for TTS response to start playing
4. Interrupt during TTS playback (barge-in test)
5. Verify TTS pauses and new transcription begins

#### Device Selection Test
1. Change microphone device in dropdown
2. Verify audio capture switches to new device
3. Change speaker device
4. Verify TTS playback routes to new device

### 3. Load Testing

```bash
# Install testing tools
pip install websockets-test locust

# Run WebSocket load test
python tests/load_test_duplex.py --concurrent-users 10 --duration 60

# Monitor metrics during load test
tail -f backend/logs/metrics.log | grep duplex
```

## Performance Tuning

### 1. Audio Latency Optimization

```bash
# Linux: Reduce audio buffer sizes
echo "pcm.!default {
    type hw
    card 0
    device 0
}" > ~/.asoundrc

# Set low-latency audio parameters
export PULSE_LATENCY_MSEC=5
export ALSA_PCM_PERIODS=2
export ALSA_PCM_BUFFER=256
```

### 2. Network Optimization

```json
{
  "network_optimization": {
    "websocket_compression": false,
    "binary_frame_compression": false,
    "tcp_nodelay": true,
    "keepalive_interval": 30,
    "max_message_size": 65536
  }
}
```

### 3. TTS Optimization

```json
{
  "tts_optimization": {
    "chunk_size_ms": 20,
    "prebuffer_chunks": 3,
    "concurrent_synthesis": true,
    "voice_cache_enabled": true,
    "sample_rate": 48000,
    "bitrate": 96000
  }
}
```

### 4. Echo Suppression Tuning

For different environments, adjust thresholds:

```json
{
  "echo_profiles": {
    "quiet_room": {
      "correlation_threshold": 0.25,
      "vad_threshold": 0.01,
      "adaptive_threshold": true
    },
    "noisy_environment": {
      "correlation_threshold": 0.4,
      "vad_threshold": 0.05,
      "adaptive_threshold": true,
      "spectral_threshold": 0.5
    },
    "professional_audio": {
      "correlation_threshold": 0.15,
      "vad_threshold": 0.005,
      "adaptive_threshold": false
    }
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Echo/Feedback Loop
**Symptoms**: Audio feedback, high correlation scores, poor user experience

**Solutions**:
- Use headphones instead of speakers
- Increase physical distance between microphone and speakers
- Enable hardware echo cancellation: `echoCancellation: true`
- Adjust echo suppression thresholds
- Check for audio loopback devices and disable them

```bash
# Debug echo detection
curl -X GET "http://localhost:8000/api/v1/debug/echo-metrics"

# Check device configuration
curl -X GET "http://localhost:8000/api/v1/debug/audio-devices"
```

#### 2. High Latency
**Symptoms**: Delayed TTS responses, poor real-time performance

**Solutions**:
- Reduce jitter buffer size: `target_size_ms: 40`
- Use smaller TTS chunk sizes: `chunk_size_ms: 20`
- Enable low-latency audio drivers
- Use local TTS models instead of cloud services
- Optimize network configuration

#### 3. Audio Dropouts/Stalls
**Symptoms**: Choppy audio, buffer underruns, playback interruptions

**Solutions**:
- Increase jitter buffer size: `max_size_ms: 300`
- Check CPU usage and reduce other processes
- Use higher quality audio devices
- Enable adaptive jitter buffer sizing
- Monitor network stability

#### 4. Device Selection Issues
**Symptoms**: No audio devices listed, wrong device selected

**Solutions**:
```bash
# Linux: Check ALSA/PulseAudio configuration
pulseaudio --check -v
alsamixer

# macOS: Check Core Audio
system_profiler SPAudioDataType

# Grant microphone permissions in browser
# Chrome: Settings > Privacy and security > Site Settings > Microphone
```

#### 5. WebSocket Connection Issues
**Symptoms**: Connection drops, binary frame errors

**Solutions**:
- Check firewall configuration
- Verify nginx WebSocket proxy configuration
- Monitor connection logs: `tail -f backend/logs/websocket.log`
- Test with different WebSocket libraries

### Debug Commands

```bash
# Check service status
systemctl status beautyai-api

# View logs
journalctl -u beautyai-api -f

# Check duplex metrics
curl "http://localhost:8000/api/v1/metrics/duplex" | jq .

# Test echo detection
curl -X POST "http://localhost:8000/api/v1/debug/echo-test" \
     -H "Content-Type: application/json" \
     -d '{"duration_seconds": 10}'

# Monitor WebSocket connections
curl "http://localhost:8000/api/v1/debug/websocket-status" | jq .
```

## Monitoring

### 1. Metrics Collection

Key metrics to monitor:

- **Audio Quality**: Echo correlation, VAD accuracy, audio dropouts
- **Performance**: TTS first byte latency, streaming duration, buffer sizes
- **User Experience**: Barge-in count, session duration, error rates
- **System**: CPU usage, memory usage, network throughput

### 2. Alerts & Thresholds

Configure alerts for:

```json
{
  "alerts": {
    "high_echo_correlation": {
      "threshold": 0.5,
      "action": "switch_to_half_duplex"
    },
    "high_latency": {
      "threshold": 500,
      "metric": "tts_first_byte_ms"
    },
    "frequent_dropouts": {
      "threshold": 5,
      "window": "1m",
      "metric": "buffer_underruns"
    }
  }
}
```

### 3. Dashboard Setup

Use Grafana or similar for monitoring:

```bash
# Export metrics to Prometheus format
curl "http://localhost:8000/metrics" | grep duplex

# Key dashboard panels:
# - Real-time latency graph
# - Echo correlation heatmap  
# - Active duplex sessions
# - Audio device usage
# - Error rate trends
```

### 4. Log Analysis

```bash
# Analyze performance logs
grep "duplex_metrics" backend/logs/metrics.log | jq .

# Monitor error patterns
grep "ERROR.*duplex" backend/logs/app.log | tail -20

# Track session statistics
grep "session_complete" backend/logs/sessions.log | \
  jq '.duplex.totals.duplex_session_duration_s' | \
  awk '{sum+=$1; n++} END {print "Average session: " sum/n "s"}'
```

## Best Practices

### 1. Audio Setup
- Always use headphones in duplex mode to prevent feedback
- Position microphone 6-12 inches from mouth
- Use USB headsets with built-in echo cancellation when possible
- Test audio setup before important sessions

### 2. Network Configuration
- Use wired connections when possible for stability
- Configure QoS to prioritize WebSocket traffic
- Monitor network latency and jitter regularly
- Implement connection retry logic with exponential backoff

### 3. Performance Optimization  
- Profile CPU usage during duplex sessions
- Monitor memory usage for audio buffer management
- Use audio device exclusivity when available
- Implement graceful degradation for overloaded systems

### 4. User Experience
- Provide clear feedback for duplex mode status
- Implement visual indicators for barge-in events
- Allow users to quickly switch between duplex modes
- Provide audio setup wizard for new users

## Conclusion

This deployment guide covers the essential aspects of setting up full duplex voice streaming in BeautyAI. Regular monitoring and performance tuning will ensure optimal user experience. For additional support, check the troubleshooting section or contact the development team.

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: BeautyAI Development Team