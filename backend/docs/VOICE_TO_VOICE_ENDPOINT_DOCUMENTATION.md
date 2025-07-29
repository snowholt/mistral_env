# Voice-to-Voice Endpoint Documentation 🎤

## Overview

The **Voice-to-Voice** endpoint provides a complete audio-to-audio conversation pipeline that transforms speech input into intelligent audio responses. This endpoint implements the full **Audio Input → STT → LLM → TTS → Audio Output** pipeline with advanced parameter control and optimization features.

---

## 🔗 Endpoint Information

### Base Route
```
POST /inference/voice-to-voice
```

### Content Type
```
multipart/form-data
```

### Authentication
- **Required Permission**: `voice_to_voice`
- **Authentication**: Bearer token or API key (depending on configuration)

---

## 🎯 Pipeline Architecture

```mermaid
graph LR
    A[Audio Input] --> B[Speech-to-Text]
    B --> C[Content Filter]
    C --> D[Language Model]
    D --> E[Response Processing]
    E --> F[Text-to-Speech]
    F --> G[Audio Output]
```

### Pipeline Steps:
1. **🎤 Speech-to-Text (STT)**: Convert input audio to text using Whisper models
2. **🔒 Content Filtering**: Optional safety filtering with configurable strictness
3. **🧠 Language Model (LLM)**: Generate intelligent response using Qwen models
4. **✨ Response Processing**: Clean thinking content and prepare final response
5. **🎭 Text-to-Speech (TTS)**: Convert response to natural audio using Coqui TTS
6. **📤 Audio Output**: Return processed audio file

---

## 📋 Input Schema

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_file` | `UploadFile` | **Required**. Input audio file (wav, mp3, ogg, flac, m4a, wma, webm) |

### Core Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_language` | `string` | `"ar"` | Language of input audio (`ar`, `en`, `es`, `fr`, `de`, etc.) |
| `output_language` | `string` | `"ar"` | Language for output audio |
| `stt_model_name` | `string` | `"whisper-large-v3-turbo-arabic"` | Speech-to-text model identifier |
| `tts_model_name` | `string` | `"coqui-tts-arabic"` | Text-to-speech model identifier |
| `chat_model_name` | `string` | `"qwen3-unsloth-q4ks"` | Chat/language model identifier |

### Session Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `string` | `null` | Optional session ID for conversation continuity |
| `chat_history` | `string` | `null` | JSON string of previous conversation messages |

### Audio Output Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speaker_voice` | `string` | `"female"` | Voice type (`female`, `male`, `neutral`) |
| `emotion` | `string` | `"neutral"` | TTS emotion (`neutral`, `happy`, `sad`, `professional`) |
| `speech_speed` | `float` | `1.0` | Speech speed multiplier (0.5-2.0) |
| `audio_output_format` | `string` | `"wav"` | Output audio format (`wav`, `mp3`, `ogg`) |

### 🧠 AI Generation Parameters

#### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `null` | Randomness control (0.0-2.0) |
| `top_p` | `float` | `null` | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | `int` | `null` | Top-k sampling limit |
| `repetition_penalty` | `float` | `null` | Repetition penalty (1.0-2.0) |
| `max_new_tokens` | `int` | `null` | Maximum tokens to generate |
| `do_sample` | `bool` | `null` | Enable sampling vs greedy decoding |

#### Advanced Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_p` | `float` | `null` | Minimum probability threshold |
| `typical_p` | `float` | `null` | Typical sampling parameter |
| `epsilon_cutoff` | `float` | `null` | Epsilon cutoff for sampling |
| `eta_cutoff` | `float` | `null` | Eta cutoff for sampling |
| `diversity_penalty` | `float` | `null` | Diversity penalty for varied responses |
| `no_repeat_ngram_size` | `int` | `null` | N-gram repetition prevention |
| `encoder_repetition_penalty` | `float` | `null` | Encoder-specific repetition penalty |

#### Beam Search Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | `int` | `null` | Number of beams for beam search |
| `length_penalty` | `float` | `null` | Length penalty for beam search |
| `early_stopping` | `bool` | `null` | Enable early stopping in beam search |

### 🔒 Content Filtering & Safety

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_content_filter` | `bool` | `false` | Disable all content filtering |
| `content_filter_strictness` | `string` | `"balanced"` | Filter level (`strict`, `balanced`, `relaxed`, `disabled`) |
| `thinking_mode` | `bool` | `false` | Enable thinking mode for detailed reasoning |

### 🎨 Optimization Presets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | `string` | `null` | Optimization preset (see presets section below) |

---

## 🎨 Available Presets

Presets provide optimized parameter combinations based on actual performance testing:

### `qwen_optimized` ⭐ **Recommended**
- **Use Case**: Best overall performance for Qwen models
- **Parameters**: `temperature=0.3, top_p=0.95, top_k=20, repetition_penalty=1.1`
- **Ideal For**: General conversations, Arabic optimization

### `high_quality`
- **Use Case**: Maximum quality responses
- **Parameters**: `temperature=0.1, top_p=1.0, repetition_penalty=1.15, max_new_tokens=1024`
- **Ideal For**: Professional content, detailed explanations

### `creative_optimized`
- **Use Case**: Creative but efficient responses
- **Parameters**: `temperature=0.5, top_p=1.0, top_k=80, repetition_penalty=1.05`
- **Ideal For**: Creative writing, varied responses

### `speed_optimized`
- **Use Case**: Fast responses with good quality
- **Parameters**: `temperature=0.3, max_new_tokens=256, speech_speed=1.2`
- **Ideal For**: Quick interactions, real-time conversations

### `balanced`
- **Use Case**: Balanced quality and speed
- **Parameters**: `temperature=0.3, top_p=0.95, max_new_tokens=512`
- **Ideal For**: General purpose conversations

### `conservative`
- **Use Case**: Safe, predictable responses
- **Parameters**: `temperature=0.1, top_p=0.9, repetition_penalty=1.2`
- **Ideal For**: Formal contexts, consistent responses

---

## 📤 Output Schema

### Success Response (`VoiceToVoiceResponse`)

```json
{
  "success": true,
  "session_id": "session_123",
  
  // Input Processing
  "transcription": "مرحباً، كيف يمكنني مساعدتك اليوم؟",
  "input_language": "ar",
  "transcription_time_ms": 2340.5,
  
  // Response Generation
  "response_text": "أهلاً وسهلاً! يمكنني مساعدتك في العديد من الأمور...",
  "response_language": "ar",
  "generation_time_ms": 1850.2,
  
  // Output Audio
  "audio_output_format": "wav",
  "audio_size_bytes": 245760,
  "audio_generation_time_ms": 3200.1,
  
  // Model Information
  "models_used": {
    "stt": "whisper-large-v3-turbo-arabic",
    "chat": "qwen3-unsloth-q4ks",
    "tts": "coqui-tts-arabic"
  },
  
  // Performance Metrics
  "total_processing_time_ms": 7390.8,
  "pipeline_metrics": {
    "stt_latency": 2340.5,
    "llm_latency": 1850.2,
    "tts_latency": 3200.1
  },
  
  // Content Filtering
  "content_filter_applied": true,
  "content_filter_strictness": "balanced",
  "input_filtered": false,
  "output_filtered": false,
  
  // Configuration Used
  "effective_config": {
    "temperature": 0.3,
    "top_p": 0.95,
    "max_new_tokens": 512
  },
  "preset_used": "qwen_optimized",
  
  // TTS Settings Used
  "speaker_voice_used": "female",
  "emotion_used": "neutral",
  "speech_speed_used": 1.0
}
```

### Error Response

```json
{
  "success": false,
  "error": "Failed to initialize models: stt, chat, tts",
  "transcription_error": null,
  "generation_error": "Model loading failed",
  "tts_error": null,
  "errors": [
    "STT model initialization failed",
    "Chat model not available"
  ]
}
```

---

## 🌟 Special Features

### 🧠 Thinking Mode
- **Purpose**: Enable detailed reasoning in responses
- **Activation**: Set `thinking_mode=true` or use `/think` voice command
- **Behavior**: Model provides step-by-step reasoning before final answer
- **Processing**: Thinking content is automatically removed from TTS output

### 🗣️ Voice Commands
- **`/think`**: Enable thinking mode for current request
- **`/no_think`**: Disable thinking mode for current request
- **Processing**: Commands are detected in transcription and processed automatically

### 🔒 Content Filtering
- **Levels**: `strict`, `balanced`, `relaxed`, `disabled`
- **Coverage**: Both input transcription and output generation
- **Medical Context**: Optimized for beauty/cosmetic medical conversations
- **Bypass**: Use `disable_content_filter=true` for unrestricted mode

### 💬 Session Management
- **Continuity**: Maintain conversation context across requests
- **History**: Automatic conversation history tracking
- **Context**: Previous messages inform current responses

---

## 🚀 Usage Examples

### Basic Arabic Conversation
```bash
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "audio_file=@question_ar.wav" \
  -F "input_language=ar" \
  -F "output_language=ar" \
  -F "speaker_voice=female" \
  -F "preset=qwen_optimized"
```

### English with Thinking Mode
```bash
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "audio_file=@question_en.wav" \
  -F "input_language=en" \
  -F "output_language=en" \
  -F "thinking_mode=true" \
  -F "preset=high_quality"
```

### Creative Mode with Custom Parameters
```bash
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "audio_file=@creative_prompt.wav" \
  -F "temperature=0.7" \
  -F "diversity_penalty=0.3" \
  -F "disable_content_filter=true" \
  -F "emotion=happy"
```

### Session-based Conversation
```bash
curl -X POST "http://localhost:8000/inference/voice-to-voice" \
  -F "audio_file=@followup.wav" \
  -F "session_id=conversation_123" \
  -F "chat_history=[{\"role\":\"user\",\"content\":\"Previous question\"},{\"role\":\"assistant\",\"content\":\"Previous response\"}]"
```

---

## 📊 Performance Characteristics

### Typical Processing Times
| Component | Arabic (10s audio) | English (10s audio) |
|-----------|-------------------|-------------------|
| **STT (Whisper)** | 2-5 seconds | 2-4 seconds |
| **LLM (Qwen)** | 1-3 seconds | 1-3 seconds |
| **TTS (Coqui)** | 1-2 seconds | 1-2 seconds |
| **Total Pipeline** | 4-10 seconds | 4-9 seconds |

### Memory Requirements
- **Base Memory**: 2-4 GB for all models loaded
- **GPU Memory**: 4-6 GB with CUDA acceleration
- **Audio Processing**: Minimal additional overhead

### Supported Audio Formats
- **Input**: WAV, MP3, OGG, FLAC, M4A, WMA, WebM
- **Output**: WAV, MP3, OGG
- **Sample Rates**: Automatic resampling to 16kHz for STT
- **Channels**: Automatic conversion to mono

---

## 🔧 Status Endpoint

### Route
```
GET /inference/voice-to-voice/status
```

### Response (`VoiceToVoiceStatusResponse`)
```json
{
  "success": true,
  "service_available": true,
  "models_status": {
    "stt_model": {
      "available": true,
      "default": "whisper-large-v3-turbo-arabic",
      "loaded": false
    },
    "tts_model": {
      "available": true,
      "default": "coqui-tts-arabic",
      "engine": "Coqui TTS",
      "arabic_support": "native",
      "loaded": false
    },
    "chat_model": {
      "available": true,
      "default": "qwen3-unsloth-q4ks",
      "loaded": false
    }
  },
  "supported_languages": {
    "input": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"],
    "output": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"]
  },
  "supported_audio_formats": ["wav", "mp3", "ogg", "flac", "m4a", "wma", "webm"],
  "tts_library_available": true,
  "pipeline_stages": ["STT (Speech-to-Text)", "LLM (Language Model)", "TTS (Text-to-Speech)"],
  "estimated_setup_time_seconds": 60
}
```

---

## 🚨 Error Handling

### Common Error Codes

| HTTP Code | Description | Possible Causes |
|-----------|-------------|-----------------|
| `400` | Bad Request | Invalid audio format, missing required parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `413` | Payload Too Large | Audio file exceeds size limits |
| `422` | Unprocessable Entity | Invalid parameter values |
| `500` | Internal Server Error | Model initialization failure, processing errors |

### Error Response Format
```json
{
  "success": false,
  "error": "Primary error message",
  "transcription_error": "STT-specific error (if applicable)",
  "generation_error": "LLM-specific error (if applicable)",
  "tts_error": "TTS-specific error (if applicable)",
  "errors": ["Detailed error 1", "Detailed error 2"]
}
```

---

## 🛡️ Security & Rate Limiting

### Content Filtering
- **Input Validation**: Audio file size and format validation
- **Content Safety**: Configurable filtering for inappropriate content
- **Medical Context**: Optimized for beauty/cosmetic medical conversations

### Rate Limiting
- Standard API rate limits apply
- Audio processing may have additional time limits
- Large audio files may be subject to size restrictions

### Authentication
- **Required Permission**: `voice_to_voice`
- **Token Types**: Bearer token, API key (configurable)
- **Session Security**: Session IDs are validated and scoped per user

---

## 🎯 Best Practices

### Audio Input
- **Format**: Use WAV or FLAC for best quality
- **Duration**: 10-30 seconds optimal for responsiveness
- **Quality**: Clear speech, minimal background noise
- **Language**: Specify correct input language for better transcription

### Parameter Optimization
- **Start with Presets**: Use `qwen_optimized` for general use
- **Gradual Tuning**: Adjust one parameter at a time
- **Test Combinations**: Different languages may need different settings
- **Monitor Performance**: Track response times and quality

### Session Management
- **Conversation Flow**: Use consistent session IDs for multi-turn conversations
- **Context Length**: Monitor conversation history length for performance
- **Memory Management**: Clear old sessions periodically

### Error Handling
- **Graceful Degradation**: Handle partial failures appropriately
- **Retry Logic**: Implement exponential backoff for transient errors
- **Monitoring**: Log performance metrics and error rates

---

## 🔄 Integration Examples

### Python Client
```python
import requests

def voice_to_voice_request(audio_file_path, language="ar", preset="qwen_optimized"):
    with open(audio_file_path, 'rb') as audio_file:
        files = {'audio_file': audio_file}
        data = {
            'input_language': language,
            'output_language': language,
            'preset': preset,
            'thinking_mode': True
        }
        
        response = requests.post(
            'http://localhost:8000/inference/voice-to-voice',
            files=files,
            data=data,
            headers={'Authorization': 'Bearer YOUR_TOKEN'}
        )
        
        return response.json()
```

### JavaScript/Node.js Client
```javascript
const FormData = require('form-data');
const fs = require('fs');

async function voiceToVoice(audioFilePath, options = {}) {
    const form = new FormData();
    form.append('audio_file', fs.createReadStream(audioFilePath));
    form.append('input_language', options.language || 'ar');
    form.append('preset', options.preset || 'qwen_optimized');
    
    const response = await fetch('http://localhost:8000/inference/voice-to-voice', {
        method: 'POST',
        body: form,
        headers: {
            'Authorization': 'Bearer YOUR_TOKEN'
        }
    });
    
    return await response.json();
}
```

---

## 📈 Monitoring & Analytics

### Key Metrics to Track
- **Processing Times**: End-to-end latency by component
- **Success Rates**: Pipeline success/failure rates
- **Audio Quality**: Input/output audio characteristics
- **Model Performance**: Token generation speed, accuracy
- **Resource Usage**: CPU, GPU, memory utilization

### Logging
- Request parameters and model names
- Transcription success/failure and confidence scores
- Generation success/failure and token counts
- TTS success/failure and audio duration
- Performance timing data for each pipeline stage
- Error conditions and debugging information

---

## 🚀 Production Deployment

### Infrastructure Requirements
- **CPU**: 8+ cores recommended
- **RAM**: 16+ GB system memory
- **GPU**: NVIDIA GPU with 8+ GB VRAM (optional but recommended)
- **Storage**: SSD for model storage (20+ GB)
- **Network**: High bandwidth for audio file transfers

### Scaling Considerations
- **Model Loading**: Pre-load models for better response times
- **Concurrent Requests**: Limit concurrent processing based on resources
- **Audio Processing**: Consider async processing for large files
- **Caching**: Implement response caching for repeated requests

### Health Checks
- Use the `/voice-to-voice/status` endpoint for health monitoring
- Monitor individual pipeline component health
- Track resource usage and performance metrics
- Implement automated alerting for failures

---

This documentation provides comprehensive information about the Voice-to-Voice endpoint, covering all aspects from basic usage to advanced deployment considerations.
