# Voice Features Guide

Complete guide to voice conversation capabilities in BeautyAI.

## 🎤 Overview

BeautyAI provides ultra-fast voice conversation with <2 second response times using Edge TTS and WebSocket streaming.

### Key Features
- **(New) Streaming Mode**: Low-latency incremental ASR + partial transcript events
- **Real-time Voice Chat**: WebSocket-based bidirectional communication
- **Language Support**: Arabic and English with automatic detection
- **Ultra-Fast Response**: <2 second typical first TTS packet; partial text <600 ms
- **Browser Integration**: Works with modern web browsers
- **API Access**: Both WebSocket and REST API interfaces
- **Graceful Fallback**: Legacy chunked voice endpoint retained during migration

> Deprecation: The legacy `/ws/voice-conversation` chunked + heuristic VAD path is now considered deprecated. The new streaming incremental endpoint `/api/v1/ws/streaming-voice` (feature-flag gated by `VOICE_STREAMING_ENABLED=1`) provides superior latency, transcript stability, and conversational flow. The old path will be removed after a stabilization period.

## 🚀 Quick Start

### Web UI Voice Chat
1. Start backend API: `cd backend && python run_server.py`
2. Start web UI: `cd frontend && python src/app.py`
3. Open http://localhost:5001
4. Click the microphone button and start talking

### WebSocket API (Legacy vs Streaming)
```javascript
// Legacy (deprecated) chunked conversation endpoint
const legacy = new WebSocket('ws://localhost:8000/ws/voice-conversation?language=auto&voice_type=default');

// New streaming incremental endpoint (enable with VOICE_STREAMING_ENABLED=1)
const streaming = new WebSocket('ws://localhost:8000/api/v1/ws/streaming-voice?language=ar');

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

## 🔧 Technical Architecture

### Voice Pipeline
```
Audio Input → Speech Recognition → Language Model → Text-to-Speech → Audio Output
     ↓              ↓                    ↓              ↓           ↓
  Browser      Whisper/OpenAI         Qwen/GPT      Edge TTS    Browser
```

### Response Flow
1. **Audio Capture**: Browser MediaRecorder API
2. **WebSocket Send**: Binary audio data to backend
3. **Speech-to-Text**: Whisper or OpenAI transcription
4. **Language Detection**: Automatic language identification
5. **Chat Generation**: AI model processes text input
6. **Text-to-Speech**: Edge TTS generates audio response
7. **WebSocket Return**: Base64 encoded audio to browser
8. **Audio Playback**: Browser plays response audio

## 🌐 API Endpoints

### WebSocket Endpoints
```bash
# Simple voice conversation (primary)
ws://localhost:8000/ws/voice-conversation
# Streaming (incremental ASR + partials + perf heartbeat)
ws://localhost:8000/api/v1/ws/streaming-voice

# Streaming status (Phase 12)
GET http://localhost:8000/api/v1/ws/streaming-voice/status


# With parameters
ws://localhost:8000/ws/voice-conversation?language=auto&voice_type=default
```

### REST Endpoints
```bash
# Check voice service status
GET /ws/voice-conversation/status

# Get available voice endpoints
GET /api/v1/voice/endpoints

# Voice service health check
GET /api/v1/health/voice
```

### WebSocket Message Format (Streaming)
```javascript
// Outgoing (browser -> server) for streaming endpoint:
// Send raw 16kHz mono little-endian Int16 PCM frames (recommended frame duration 20–40 ms)

// Incoming (server -> browser) event examples:
{
    "type": "ready",
    "session_id": "stream_f87...",
    "decode_interval_ms": 480,
    "window_seconds": 8.0
}
{
    "type": "partial_transcript",
    "text": "مرحبا بك",
    "stable": false,
    "stable_tokens": 2,
    "total_tokens": 3,
    "decode_ms": 92
}
{
    "type": "endpoint",
    "event": "start",
    "utterance_index": 0
}
{
    "type": "final_transcript",
    "utterance_index": 0,
    "text": "مرحبا بك في الوضع الجديد",
    "reason": "silence+stable",
    "decode_ms": 105
}
{
    "type": "tts_start", "utterance_index": 0 }
{
    "type": "tts_audio",
    "utterance_index": 0,
    "mime_type": "audio/wav",
    "encoding": "base64",
    "audio": "UklGR..."
}
{
    "type": "tts_complete", "utterance_index": 0, "processing_ms": 840 }
{
    "type": "perf_cycle",
    "decode_ms": 94,
    "cycle_latency_ms": 480,
    "tokens": 11
}
```

## 🎛️ Configuration Options

### Language Settings
- **auto**: Automatic language detection (recommended)
- **ar**: Force Arabic processing
- **en**: Force English processing
- **es**, **fr**, **de**: Other supported languages

### Voice Types
- **default**: Standard voice for detected language
- **male**: Male voice preference
- **female**: Female voice preference

### Audio Formats
- **Input**: WAV, MP3, WebM, OGG
- **Output**: WAV (high quality), WebM (streaming)
- **Recommended**: WAV for best compatibility

## 🎨 Web UI Integration

### Microphone Button
```html
<!-- Voice control button -->
<button id="voice-btn" onclick="toggleVoiceChat()">
    <i class="fas fa-microphone"></i>
</button>
```

### JavaScript Implementation
```javascript
let mediaRecorder;
let audioChunks = [];

async function startVoiceChat() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            websocket.send(event.data);
        }
    };
    
    mediaRecorder.start(1000); // Send chunks every second
}

function stopVoiceChat() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}
```

### Visual Feedback
- **Recording Indicator**: Animated microphone icon
- **Voice Activity**: Waveform visualization
- **Status Display**: Connection and processing status
- **Response Time**: Show actual response times

## 🔧 Development

### Testing Voice Features (Both Modes)
```bash
# Check WebSocket status
curl http://localhost:8000/ws/voice-conversation/status

# Streaming status
curl http://localhost:8000/api/v1/ws/streaming-voice/status

# Test voice endpoints
curl http://localhost:8000/api/v1/voice/endpoints

# Health check
curl http://localhost:8000/api/v1/health/voice
```

### Browser Testing
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/voice-conversation');
ws.onopen = () => console.log('Connected');
ws.onerror = (error) => console.error('Error:', error);
```

### Audio Testing
```bash
# Test audio file formats
ffmpeg -i input.mp3 -f wav output.wav

# Check audio properties
file audio.wav
```

## 🚀 Production Deployment

### SSL/WSS Configuration
```bash
# Environment variables for production
export BEAUTYAI_WS_URL=wss://api.gmai.sa/ws/voice-conversation
export USE_WSS=true
export BEAUTYAI_ENABLE_SSL=true
```

### Nginx WebSocket Proxy
```nginx
location /ws/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}
```

### Performance Optimization
```bash
# WebSocket configuration in run_server.py
ws_max_size=50 * 1024 * 1024  # 50MB max frame size
ws_ping_interval=20
ws_ping_timeout=20
```

## 🐛 Troubleshooting

### Common Issues

**Voice Not Working**:
```bash
# Check microphone permissions in browser
# Verify backend is running
curl http://localhost:8000/health

# Test WebSocket connection
curl http://localhost:8000/ws/voice-conversation/status
```

**Poor Audio Quality**:
```bash
# Use WAV format for input
# Check network bandwidth
# Verify Edge TTS connectivity
```

**Slow Response Times**:
```bash
# Check GPU availability
nvidia-smi

# Monitor network latency
ping api.gmai.sa

# Use local deployment for best performance
```

**WebSocket Connection Issues**:
```bash
# Check firewall settings
# Verify proxy configuration
# Test with different browsers
```

### Browser Compatibility
- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support with some limitations
- **Edge**: Full support
- **Mobile browsers**: Limited support

### Performance Monitoring (Streaming)
```javascript
// Track latency and decode performance via perf_cycle events
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/streaming-voice?language=ar');
let firstPartialAt = null;
ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'partial_transcript' && !firstPartialAt) {
        firstPartialAt = performance.now();
        console.log('First partial latency ms', firstPartialAt - startTs);
    }
    if (msg.type === 'final_transcript') {
        console.log('Final reason', msg.reason, 'decode_ms', msg.decode_ms);
    }
    if (msg.type === 'perf_cycle') {
        console.log('Cycle decode', msg.decode_ms, 'latency', msg.cycle_latency_ms);
    }
};
```

## 📱 Mobile Support

### Mobile Browser Considerations
- **Touch interface**: Large, touch-friendly buttons
- **Audio permissions**: Handle mobile permission requests
- **Battery optimization**: Efficient audio processing
- **Network conditions**: Handle variable connectivity

### iOS Safari Specific
```javascript
// Handle iOS audio context requirements
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

// Resume audio context on user interaction
document.addEventListener('touchstart', () => {
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
});
```

---

**Next**: [API Documentation](API.md) | [Performance Guide](PERFORMANCE.md)
