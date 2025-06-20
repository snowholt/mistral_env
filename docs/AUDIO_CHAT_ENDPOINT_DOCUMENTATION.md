# Audio-Chat Endpoint Documentation

## Overview

The `/inference/audio-chat` endpoint provides a complete audio-to-text-to-chat pipeline that combines advanced speech transcription with intelligent chat response generation. This endpoint transcribes uploaded audio files using Whisper models and then generates chat responses using large language models.

**Endpoint URL:** `POST /inference/audio-chat`

**Content-Type:** `multipart/form-data`

**Response Model:** `AudioChatResponse`

---

## 🎯 Key Features

- **🎙️ Multi-format Audio Support**: WAV, MP3, OGG, FLAC, M4A, WMA
- **🌍 Multi-language Transcription**: Powered by Whisper models with language detection
- **🤖 Advanced Chat Generation**: Same 25+ parameters as the regular chat endpoint
- **🧠 Thinking Mode Support**: Automatic detection of `/no_think` commands in transcription
- **🔒 Content Filtering**: Configurable filtering for both transcription and response
- **⚡ Performance Metrics**: Detailed timing and token statistics
- **🎨 Optimization Presets**: Performance-tuned configurations
- **📊 Comprehensive Response**: Both transcription and chat response with full metrics

---

## 🔄 Workflow

1. **🎙️ Audio Upload**: Upload audio file in supported format
2. **📝 Transcription**: Convert audio to text using Whisper model
3. **🤖 Chat Generation**: Process transcription with chat model
4. **📊 Response**: Return both transcription and chat response with metrics

---

## 📥 Request Parameters

### **Required Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_file` | `File` | Audio file to transcribe (WAV, MP3, OGG, FLAC, M4A, WMA) |
| `model_name` | `str` | Chat model name for response generation |

### **Audio & Transcription Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `whisper_model_name` | `str` | `"whisper-large-v3-turbo-arabic"` | Whisper model for transcription |
| `audio_language` | `str` | `"ar"` | Language code for transcription (ar, en, fr, etc.) |

### **Session Management**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `null` | Session identifier for conversation continuity |
| `chat_history` | `str` | `null` | JSON string of previous conversation messages |

### **Core Generation Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | Preset-based | Randomness in generation (0.0-1.0) |
| `top_p` | `float` | Preset-based | Nucleus sampling threshold |
| `top_k` | `int` | Preset-based | Top-k sampling limit |
| `repetition_penalty` | `float` | Preset-based | Penalty for repeated tokens |
| `max_new_tokens` | `int` | Preset-based | Maximum tokens to generate |
| `min_new_tokens` | `int` | `null` | Minimum tokens to generate |
| `do_sample` | `bool` | Preset-based | Enable sampling vs greedy decoding |

### **Advanced Sampling Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_p` | `float` | `null` | Minimum probability threshold |
| `typical_p` | `float` | `null` | Typical sampling parameter |
| `epsilon_cutoff` | `float` | `null` | Epsilon cutoff for sampling |
| `eta_cutoff` | `float` | `null` | Eta cutoff for sampling |
| `diversity_penalty` | `float` | `null` | Penalty for repetitive content |
| `encoder_repetition_penalty` | `float` | `null` | Encoder repetition penalty |
| `no_repeat_ngram_size` | `int` | `null` | N-gram size to avoid repetition |

### **Beam Search Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | `int` | `null` | Number of beams for beam search |
| `num_beam_groups` | `int` | `null` | Number of beam groups |
| `length_penalty` | `float` | `null` | Length penalty for beam search |
| `early_stopping` | `bool` | `null` | Enable early stopping in beam search |

### **Thinking Mode Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_thinking` | `bool` | `null` | Override thinking mode (auto-detect if null) |
| `thinking_mode` | `str` | `null` | "auto", "force", "disable" |

### **Content Filtering**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_content_filter` | `bool` | `false` | Disable content filtering entirely |
| `content_filter_strictness` | `str` | `"balanced"` | "strict", "balanced", "relaxed", "disabled" |

### **Preset Configurations**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | `str` | `null` | Optimization preset name |

**Available Presets:**

| Preset | Use Case | Temperature | Top-P | Top-K | Max Tokens |
|--------|----------|-------------|--------|-------|------------|
| `conservative` | Safe, predictable responses | 0.1 | 0.9 | 10 | 256 |
| `balanced` | General purpose | 0.3 | 0.95 | 40 | 512 |
| `creative` | Creative, diverse responses | 0.7 | 1.0 | 100 | 1024 |
| `speed_optimized` | Fast responses | 0.2 | 0.9 | 20 | 256 |
| `qwen_optimized` | Optimized for Qwen models | 0.3 | 0.95 | 20 | 512 |
| `high_quality` | Maximum quality | 0.1 | 1.0 | 50 | 1024 |
| `creative_optimized` | Creative but efficient | 0.5 | 1.0 | 80 | 768 |

### **Legacy Support**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generation_config` | `str` | `null` | JSON string of generation parameters |
| `stream` | `bool` | `false` | Enable streaming response (not implemented) |

---

## 📤 Response Format

### **AudioChatResponse Structure**

```json
{
  "success": true,
  "response": "Generated chat response text",
  "session_id": "session_123",
  "model_name": "qwen3-unsloth-q4ks",
  
  // Transcription Information
  "transcription": "Transcribed text from audio",
  "whisper_model_used": "whisper-large-v3-turbo-arabic",
  "audio_language_detected": "ar",
  "transcription_time_ms": 477.7,
  
  // Generation Statistics
  "generation_stats": {
    "tokens_generated": 423,
    "tokens_per_second": 69.7,
    "generation_time_ms": 6066.2
  },
  
  // Parameter Information
  "effective_config": {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 512
  },
  "preset_used": "balanced",
  "thinking_enabled": true,
  
  // Content Filtering Information
  "content_filter_applied": false,
  "content_filter_strictness": "disabled",
  "content_filter_bypassed": true,
  
  // Performance Metrics
  "tokens_generated": 423,
  "generation_time_ms": 6066.2,
  "tokens_per_second": 69.7,
  "total_processing_time_ms": 6543.9,
  
  // Response Metadata (if thinking mode enabled)
  "thinking_content": "Internal reasoning process...",
  "final_content": "Final answer without thinking tags",
  
  // Error information (if any)
  "error": null,
  "transcription_error": null,
  "chat_error": null,
  
  // Standard API fields
  "timestamp": "2025-06-20T17:18:32.123456Z",
  "execution_time_ms": 6543.9
}
```

### **Response Fields Description**

#### **Core Response**
- `success`: Boolean indicating overall operation success
- `response`: Final chat response text (processed)
- `session_id`: Session identifier used
- `model_name`: Chat model used for generation

#### **Transcription Data**
- `transcription`: Complete transcribed text from audio
- `whisper_model_used`: Whisper model used for transcription
- `audio_language_detected`: Detected or specified language
- `transcription_time_ms`: Time taken for transcription

#### **Generation Statistics**
- `generation_stats`: Detailed generation metrics object
- `tokens_generated`: Number of tokens generated
- `generation_time_ms`: Time taken for chat generation
- `tokens_per_second`: Generation speed

#### **Configuration**
- `effective_config`: Actual parameters used (after preset/override resolution)
- `preset_used`: Preset configuration applied
- `thinking_enabled`: Whether thinking mode was active

#### **Content Filtering**
- `content_filter_applied`: Whether filtering was applied
- `content_filter_strictness`: Strictness level used
- `content_filter_bypassed`: Whether filtering was bypassed

#### **Performance Metrics**
- `total_processing_time_ms`: Total time (transcription + generation)

#### **Thinking Mode Data**
- `thinking_content`: Internal reasoning (if thinking enabled)
- `final_content`: Clean response without thinking tags

#### **Error Handling**
- `error`: General error message
- `transcription_error`: Transcription-specific error
- `chat_error`: Chat generation error

---

## 🎯 Usage Examples

### **Basic Audio Chat**

```bash
curl -X POST "http://localhost:8000/inference/audio-chat" \
  -F "audio_file=@example.mp3" \
  -F "model_name=qwen3-unsloth-q4ks"
```

### **Arabic Audio with Optimized Settings**

```bash
curl -X POST "http://localhost:8000/inference/audio-chat" \
  -F "audio_file=@arabic_question.wav" \
  -F "model_name=qwen3-unsloth-q4ks" \
  -F "whisper_model_name=whisper-large-v3-turbo-arabic" \
  -F "audio_language=ar" \
  -F "preset=qwen_optimized" \
  -F "disable_content_filter=true"
```

### **Custom Generation Parameters**

```bash
curl -X POST "http://localhost:8000/inference/audio-chat" \
  -F "audio_file=@question.mp3" \
  -F "model_name=qwen3-unsloth-q4ks" \
  -F "temperature=0.3" \
  -F "top_p=0.95" \
  -F "max_new_tokens=1024" \
  -F "repetition_penalty=1.1"
```

### **With Session Continuity**

```bash
curl -X POST "http://localhost:8000/inference/audio-chat" \
  -F "audio_file=@followup.mp3" \
  -F "model_name=qwen3-unsloth-q4ks" \
  -F "session_id=chat_session_123" \
  -F 'chat_history=[{"role":"user","content":"Previous question"},{"role":"assistant","content":"Previous answer"}]'
```

### **Python Example**

```python
import requests

url = "http://localhost:8000/inference/audio-chat"

files = {"audio_file": open("example.mp3", "rb")}
data = {
    "model_name": "qwen3-unsloth-q4ks",
    "whisper_model_name": "whisper-large-v3-turbo-arabic",
    "audio_language": "ar",
    "temperature": 0.7,
    "max_new_tokens": 512,
    "preset": "balanced",
    "disable_content_filter": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Transcription: {result['transcription']}")
print(f"Response: {result['response']}")
print(f"Generation time: {result['generation_time_ms']}ms")
```

---

## 🔧 Technical Details

### **Supported Audio Formats**
- **WAV**: Uncompressed, high quality
- **MP3**: Compressed, most common
- **OGG**: Open source alternative
- **FLAC**: Lossless compression
- **M4A**: Apple format
- **WMA**: Windows Media Audio

### **Audio Processing**
- Automatic resampling to 16kHz for Whisper compatibility
- Support for mono and stereo audio
- Automatic format detection from file extension
- Temporary file handling for processing

### **Whisper Models Available**
- `whisper-large-v3-turbo-arabic`: Optimized for Arabic (default)
- Other registered Whisper models in the model registry

### **Chat Models**
- Any registered chat model in the system
- `qwen3-unsloth-q4ks`: Fast Qwen model (recommended)
- `qwen3-model`: Standard Qwen model
- Other LLMs as configured

### **Performance Characteristics**
- **Transcription**: ~500ms for 10-second audio
- **Chat Generation**: 2-10 seconds depending on model and length
- **Memory Usage**: Varies by loaded models
- **Concurrent Requests**: Supported with model sharing

---

## ⚠️ Error Handling

### **Common Error Cases**

#### **400 Bad Request**
```json
{
  "detail": "Validation error details"
}
```

#### **Authentication Error**
```json
{
  "success": false,
  "error": "Missing required permissions: audio"
}
```

#### **Model Not Found**
```json
{
  "success": false,
  "error": "Model 'invalid-model' not found in registry"
}
```

#### **Audio Format Error**
```json
{
  "success": false,
  "error": "Unsupported audio format",
  "transcription_error": "Unsupported audio format: example.xyz"
}
```

#### **Transcription Failure**
```json
{
  "success": false,
  "transcription": "",
  "error": "Audio transcription failed",
  "transcription_error": "Failed to transcribe audio file"
}
```

#### **Chat Generation Error**
```json
{
  "success": false,
  "transcription": "Successful transcription",
  "response": "",
  "error": "Chat generation failed",
  "chat_error": "Model 'model-name' is not loaded"
}
```

### **Partial Success Cases**
The endpoint can return transcription even if chat generation fails:

```json
{
  "success": false,
  "transcription": "Successfully transcribed text",
  "transcription_time_ms": 477.7,
  "response": "",
  "error": "Chat generation failed",
  "chat_error": "Model timeout"
}
```

---

## 🔒 Security & Authentication

### **Required Permissions**
- `chat`: Permission for chat operations
- `audio`: Permission for audio processing

### **Content Filtering**
- **Strictness Levels**: `strict`, `balanced`, `relaxed`, `disabled`
- **Applied To**: Both transcription input and chat output
- **Bypass Option**: `disable_content_filter=true`

### **Rate Limiting**
- Standard API rate limits apply
- Audio processing may have additional time limits
- Large audio files may be subject to size restrictions

---

## 📊 Monitoring & Metrics

### **Performance Metrics Tracked**
- **Transcription Time**: Audio processing duration
- **Generation Time**: Chat response generation duration
- **Total Processing Time**: End-to-end request time
- **Token Metrics**: Generated tokens and generation speed
- **Model Usage**: Which models were used

### **Logging**
- Request parameters and model names
- Transcription success/failure
- Chat generation success/failure
- Performance timing data
- Error conditions and debugging info

---

## 🔄 Integration Guide

### **Frontend Integration**
```javascript
// HTML form example
const formData = new FormData();
formData.append('audio_file', audioFile);
formData.append('model_name', 'qwen3-unsloth-q4ks');
formData.append('preset', 'balanced');

fetch('/inference/audio-chat', {
  method: 'POST',
  body: formData
}).then(response => response.json())
  .then(data => {
    console.log('Transcription:', data.transcription);
    console.log('Response:', data.response);
  });
```

### **Backend Integration**
```python
# Service integration example
from beautyai_inference.api.endpoints.inference import audio_chat_completion

async def process_audio_chat(audio_data, model_name):
    response = await audio_chat_completion(
        audio_file=audio_data,
        model_name=model_name,
        preset="qwen_optimized"
    )
    return response
```

### **Mobile App Integration**
```swift
// iOS Swift example
let url = URL(string: "http://localhost:8000/inference/audio-chat")!
var request = URLRequest(url: url)
request.httpMethod = "POST"

let boundary = UUID().uuidString
request.setValue("multipart/form-data; boundary=\(boundary)", 
                forHTTPHeaderField: "Content-Type")

// Add audio file and parameters to multipart body
```

---

## 🚀 Best Practices

### **Audio Quality**
- **Recommended**: 16kHz, 16-bit, mono WAV files
- **Avoid**: Very noisy or low-quality audio
- **Duration**: 1 second to 5 minutes optimal
- **Language**: Match `audio_language` parameter to actual audio language

### **Performance Optimization**
- **Use Presets**: Leverage optimization presets for consistent performance
- **Model Loading**: Ensure chat models are pre-loaded for faster response
- **Caching**: Consider caching for repeated audio files
- **Batch Processing**: For multiple files, process sequentially

### **Error Handling**
- **Check Success**: Always check the `success` field
- **Partial Results**: Handle cases where transcription succeeds but chat fails
- **Retry Logic**: Implement retry for temporary failures
- **Fallbacks**: Have fallback strategies for model unavailability

### **Security**
- **Input Validation**: Validate audio file sizes and formats
- **Content Filtering**: Use appropriate content filtering levels
- **Rate Limiting**: Implement client-side rate limiting
- **Authentication**: Ensure proper permissions are granted

---

## 📈 Roadmap & Future Features

### **Planned Enhancements**
- **Streaming Support**: Real-time audio transcription and response
- **Speaker Diarization**: Multi-speaker audio support
- **Voice Activity Detection**: Automatic silence removal
- **Audio Preprocessing**: Noise reduction and audio enhancement
- **Multi-language Detection**: Automatic language detection
- **Response Audio**: Text-to-speech for responses

### **Model Improvements**
- **More Whisper Models**: Additional language-specific models
- **Custom Models**: Support for fine-tuned Whisper models
- **Model Optimization**: Faster transcription models
- **Quantization**: Smaller models for edge deployment

---

## 🆘 Troubleshooting

### **Common Issues**

#### **Empty Response**
- **Check Model Loading**: Ensure chat model is loaded
- **Verify Audio**: Test with known good audio file
- **Check Permissions**: Verify audio and chat permissions
- **Review Logs**: Check service logs for detailed errors

#### **Slow Performance**
- **Model Loading**: Pre-load models before first request
- **Audio Size**: Use smaller/shorter audio files
- **Parameters**: Use speed-optimized presets
- **Hardware**: Ensure adequate GPU/CPU resources

#### **Transcription Errors**
- **Audio Format**: Verify supported format
- **Audio Quality**: Check for corruption or noise
- **Language Match**: Ensure language parameter matches audio
- **File Size**: Check for size limitations

#### **Generation Errors**
- **Model Status**: Verify chat model is loaded and available
- **Parameters**: Check for invalid parameter combinations
- **Content Filter**: Try disabling content filtering
- **Resources**: Check system memory and GPU usage

### **Debug Steps**
1. **Test Basic Request**: Try with minimal parameters
2. **Check Service Health**: Verify API server status
3. **Review Logs**: Check both transcription and chat logs
4. **Test Components**: Try regular `/chat` endpoint separately
5. **Validate Audio**: Test audio file with other tools

---

## 📞 Support & Contact

For technical support, bug reports, or feature requests:

- **Documentation**: Check this document and API docs
- **Logs**: Include relevant service logs with issue reports
- **Test Cases**: Provide minimal reproducible examples
- **Environment**: Include system and model information

---

*Last Updated: June 20, 2025*
*Version: 1.0*
*BeautyAI Inference Framework*
