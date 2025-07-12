# Voice-to-Voice API Documentation 🎤

## Quick Reference for Frontend Implementation

### 📡 REST API Endpoint

```
POST /inference/voice-to-voice
Content-Type: multipart/form-data
```

### 📡 WebSocket Endpoint  

```
ws://localhost:8000/ws/voice-conversation
```

---

## 🔍 Default Configuration Analysis

Based on the current implementation, here are the answers to your questions:

### ✅ 1. Default Model Parameters (Temperature, etc.)
- **Temperature**: `null` by default (uses model defaults)
- **Top-p**: `null` by default  
- **Top-k**: `null` by default
- **Repetition Penalty**: `null` by default
- **Max New Tokens**: `null` by default

**Note**: Parameters are `null` by default, meaning the LLM uses its internal defaults unless explicitly set or a preset is used.

### ✅ 2. Content Filter Default
- **Content Filter**: `enable_content_filter=true` (ENABLED by default)
- **Filter Strictness**: `"balanced"` by default
- **Disable Override**: `disable_content_filter=false` by default

### ✅ 3. Thinking Mode Default  
- **Thinking Mode**: `false` (DISABLED by default) ✅
- **Auto \no_think**: YES - The service automatically adds `\no_think` prefix for voice conversations to ensure fast responses
- **Override Behavior**: 
  - If user says "/no_think" → `thinking_mode=false` 
  - If user says "/think" → `thinking_mode=true`
  - Default voice behavior → `thinking_mode=false` with auto `\no_think` prefix

---

## 📋 REST API Schema

### Required Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio_file` | `UploadFile` | ✅ | Audio file (wav, mp3, ogg, flac, m4a, wma, webm) |

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_language` | `string` | `"auto"` | Input language detection |
| `output_language` | `string` | `"auto"` | Output language matching |
| `stt_model_name` | `string` | `"whisper-large-v3-turbo-arabic"` | Speech-to-text model |
| `tts_model_name` | `string` | `"coqui-tts-arabic"` | Text-to-speech model |
| `chat_model_name` | `string` | `"qwen3-unsloth-q4ks"` | Chat model |
| `speaker_voice` | `string` | `"female"` | Voice type (female/male/neutral) |
| `thinking_mode` | `boolean` | `false` | Enable thinking mode |
| `disable_content_filter` | `boolean` | `false` | Disable content filtering |
| `content_filter_strictness` | `string` | `"balanced"` | Filter level (strict/balanced/relaxed) |

### Generation Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `null` | Randomness (0.0-2.0) |
| `top_p` | `float` | `null` | Nucleus sampling (0.0-1.0) |
| `top_k` | `int` | `null` | Top-k sampling |
| `repetition_penalty` | `float` | `null` | Repetition penalty (1.0-2.0) |
| `max_new_tokens` | `int` | `null` | Maximum tokens |
| `preset` | `string` | `null` | Preset name (qwen_optimized/high_quality/etc) |

---

## 📤 Response Schema

### Success Response

```json
{
  "success": true,
  "session_id": "session_123",
  "transcription": "Transcribed input text",
  "response_text": "Generated response text (thinking content removed)",
  "input_language": "ar",
  "response_language": "ar",
  "total_processing_time_ms": 5420.8,
  "audio_output_format": "wav",
  "audio_size_bytes": 245760,
  "models_used": {
    "stt": "whisper-large-v3-turbo-arabic",
    "chat": "qwen3-unsloth-q4ks", 
    "tts": "coqui-tts-arabic"
  },
  "preset_used": "qwen_optimized",
  "effective_config": {
    "temperature": 0.3,
    "top_p": 0.95
  },
  "speaker_voice_used": "female",
  "content_filter_applied": true,
  "content_filter_strictness": "balanced",
  "data": {
    "audio_output_path": "/path/to/audio.wav",
    "audio_output_base64": "UklGRiQAAABXQVZFZm10...",
    "language_auto_detected": true,
    "detected_input_language": "ar"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Primary error message",
  "transcription_error": null,
  "generation_error": "Model loading failed", 
  "tts_error": null,
  "errors": ["Detailed error messages"]
}
```

---

## 🌐 WebSocket API Schema

### Connection URL with Parameters

```
ws://localhost:8000/ws/voice-conversation?
  session_id=optional_session&
  input_language=auto&
  output_language=auto&
  thinking_mode=false&
  preset=qwen_optimized
```

### Message Types

#### 📨 Incoming (Client → Server)

**Binary Messages**: Audio data (WebM, WAV, MP3, etc.)
```javascript
// Send audio chunk
websocket.send(audioBlob);
```

**Text Messages**: Control commands (JSON)
```json
{
  "type": "ping"
}

{
  "type": "update_config",
  "config": {
    "thinking_mode": false,
    "temperature": 0.7
  }
}
```

#### 📤 Outgoing (Server → Client)

**Connection Established**
```json
{
  "type": "connection_established",
  "connection_id": "uuid",
  "session_id": "session_123",
  "timestamp": 1625097600.0,
  "message": "Voice conversation WebSocket connected successfully"
}
```

**Processing Started**
```json
{
  "type": "processing_started", 
  "timestamp": 1625097600.0,
  "message": "Processing your voice message..."
}
```

**Voice Response**
```json
{
  "type": "voice_response",
  "success": true,
  "timestamp": 1625097600.0,
  "session_id": "session_123",
  "transcription": "User's transcribed speech",
  "response_text": "AI response text (clean, no thinking)",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "audio_format": "wav",
  "audio_size_bytes": 245760,
  "processing_time_ms": 3420.5,
  "models_used": {
    "stt": "whisper-large-v3-turbo-arabic",
    "chat": "qwen3-unsloth-q4ks",
    "tts": "coqui-tts-arabic"
  },
  "metadata": {
    "thinking_mode": false,
    "content_filter_applied": true,
    "input_language": "ar",
    "output_language": "ar"
  }
}
```

**Error Response**
```json
{
  "type": "voice_response",
  "success": false,
  "timestamp": 1625097600.0,
  "error": "Processing error message"
}
```

---

## 🚀 Usage Examples

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://api.gmai.sa//ws/voice-conversation?thinking_mode=false&preset=qwen_optimized');

ws.onopen = () => {
    console.log('Connected to voice chat');
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    
    if (response.type === 'voice_response' && response.success) {
        // Play received audio
        if (response.audio_base64) {
            const audioBlob = base64ToBlob(response.audio_base64, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            new Audio(audioUrl).play();
        }
        
        // Display transcription and response
        console.log('User said:', response.transcription);
        console.log('AI replied:', response.response_text);
    }
};

// Send audio
function sendAudio(audioBlob) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(audioBlob);
    }
}

function base64ToBlob(base64, mimeType) {
    const bytes = atob(base64);
    const array = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
        array[i] = bytes.charCodeAt(i);
    }
    return new Blob([array], { type: mimeType });
}
```

### Python REST Client

```python
import requests

def voice_to_voice_request(audio_file_path):
    url = "http://localhost:8000/inference/voice-to-voice"
    
    with open(audio_file_path, 'rb') as f:
        files = {'audio_file': f}
        data = {
            'input_language': 'auto',
            'output_language': 'auto', 
            'thinking_mode': False,
            'disable_content_filter': True,
            'content_filter_strictness': 'balanced',
            'preset': 'qwen_optimized'
        }
        
        response = requests.post(url, files=files, data=data)
        return response.json()

# Usage
result = voice_to_voice_request('input_audio.wav')
if result['success']:
    print(f"Transcription: {result['transcription']}")
    print(f"Response: {result['response_text']}")
    # Audio is available in result['data']['audio_output_base64']
```

---

## ✅ Key Implementation Notes

1. **\no_think Prefix**: ✅ YES - The service automatically adds `\no_think` prefix to voice input by default for faster responses
2. **Content Filter**: ✅ DISABLED by default (`enable_content_filter=true`)
3. **Thinking Mode**: ✅ DISABLED by default (`thinking_mode=false`)
4. **Temperature**: Uses model defaults unless explicitly set or preset is used
5. **Language Detection**: Auto-detects input language and matches output language by default
6. **Audio Format**: Supports WebM (browser), WAV, MP3, OGG, FLAC, M4A
7. **Response Cleaning**: Thinking content is automatically removed from TTS output

---

## 🔧 Status Endpoint

```
GET /inference/voice-to-voice/status
```

Returns service availability, model status, and supported features.

---

This documentation covers the essential information needed for frontend implementation with the current voice-to-voice service configuration.
