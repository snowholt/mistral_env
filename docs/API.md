# API Reference

Complete API documentation for the BeautyAI Inference Framework.

## 🌐 Base URLs

- **Development**: http://localhost:8000
- **Production**: https://api.gmai.sa
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🏥 Health & Status

### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-03T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "models": "healthy",
    "voice": "healthy"
  }
}
```

### Voice Service Health
```http
GET /api/v1/health/voice
```

## 🤖 Model Management

### List Models
```http
GET /models
```

**Response**:
```json
{
  "models": [
    {
      "name": "default",
      "model_id": "Qwen/Qwen3-14B",
      "engine_type": "transformers",
      "quantization": "4bit",
      "status": "loaded",
      "memory_usage": "8.2GB"
    }
  ],
  "default_model": "default"
}
```

### Add Model
```http
POST /models
Content-Type: application/json

{
  "name": "custom-model",
  "model_id": "organization/model-name",
  "engine_type": "transformers",
  "quantization": "4bit",
  "description": "Custom model configuration"
}
```

### Load Model
```http
POST /models/{name}/load
```

### Unload Model
```http
DELETE /models/{name}/unload
```

### Get Model Details
```http
GET /models/{name}
```

## 💬 Chat & Inference

### Chat Completion
```http
POST /inference/chat
Content-Type: application/json

{
  "model_name": "default",
  "message": "What is artificial intelligence?",
  "temperature": 0.3,
  "max_new_tokens": 512,
  "top_p": 0.95,
  "top_k": 20,
  "preset": "qwen_optimized"
}
```

**Response**:
```json
{
  "response": "Artificial intelligence (AI) is...",
  "model_used": "default",
  "processing_time": 2.1,
  "tokens_generated": 156,
  "parameters_used": {
    "temperature": 0.3,
    "max_new_tokens": 512,
    "top_p": 0.95
  }
}
```

### Available Chat Parameters
```json
{
  "model_name": "string",
  "message": "string",
  "preset": "qwen_optimized|high_quality|creative|balanced|speed_optimized",
  "temperature": 0.1-1.0,
  "top_p": 0.1-1.0,
  "top_k": 1-100,
  "max_new_tokens": 1-2048,
  "repetition_penalty": 1.0-2.0,
  "thinking_mode": true|false,
  "disable_content_filter": true|false,
  "response_language": "auto|en|ar|es|fr"
}
```

### Test Model
```http
POST /inference/test
```

### Benchmark Model
```http
POST /inference/benchmark
```

## 🎤 Voice Features

### WebSocket Voice Conversation
```
ws://localhost:8000/ws/voice-conversation?language=auto&voice_type=default
```

**Connection Parameters**:
- `language`: auto, ar, en, es, fr, de
- `voice_type`: default, male, female

**Message Format**:
```javascript
// Send: Raw audio blob
websocket.send(audioBlob);

// Receive: JSON response
{
  "type": "voice_response",
  "audio_base64": "UklGRnoGAABXQVZFZm10...",
  "transcript": "Your message",
  "response_text": "AI response",
  "language": "en",
  "processing_time": 1.2
}
```

### Voice Endpoints Info
```http
GET /api/v1/voice/endpoints
```

### Voice Service Status
```http
GET /ws/voice-conversation/status
```

## ⚙️ Configuration

### Get Configuration
```http
GET /config
```

### Update Configuration
```http
POST /config
Content-Type: application/json

{
  "default_model": "custom-model",
  "gpu_memory_utilization": 0.85,
  "voice_enabled": true
}
```

### Validate Configuration
```http
POST /config/validate
```

## 💾 System Management

### System Status
```http
GET /system/status
```

**Response**:
```json
{
  "status": "healthy",
  "gpu": {
    "available": true,
    "memory_used": "8.2GB",
    "memory_total": "24GB",
    "utilization": "34%"
  },
  "models": {
    "loaded": 1,
    "available": 5
  },
  "memory": {
    "system_used": "12.5GB",
    "system_total": "32GB"
  }
}
```

### Memory Status
```http
GET /system/memory
```

### Clear Cache
```http
POST /system/cache/clear
```

## 🔐 Authentication

### JWT Authentication (Production)
```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using JWT Token
```http
GET /models
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 📊 Response Formats

### Standard Success Response
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2025-08-03T10:30:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid-model' not found in registry",
    "details": { ... }
  },
  "timestamp": "2025-08-03T10:30:00Z"
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created (model added)
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (authentication required)
- `404`: Not Found (model/endpoint not found)
- `409`: Conflict (model already exists)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error
- `503`: Service Unavailable (model loading)

## 🚀 Rate Limiting

### Default Limits
- **Chat API**: 60 requests/minute per IP
- **Voice WebSocket**: 5 concurrent connections per IP
- **Model Management**: 10 requests/minute per IP

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1691058600
```

## 🔧 Python Client Example

```python
import requests
import json

class BeautyAIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def chat(self, message, model="default", **kwargs):
        data = {
            "model_name": model,
            "message": message,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/inference/chat", json=data)
        return response.json()
    
    def list_models(self):
        response = self.session.get(f"{self.base_url}/models")
        return response.json()
    
    def load_model(self, name):
        response = self.session.post(f"{self.base_url}/models/{name}/load")
        return response.json()

# Usage
client = BeautyAIClient()
response = client.chat("Hello, how are you?", preset="qwen_optimized")
print(response["response"])
```

## 🌐 JavaScript Client Example

```javascript
class BeautyAIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async chat(message, options = {}) {
        const response = await fetch(`${this.baseUrl}/inference/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: 'default',
                message: message,
                ...options
            })
        });
        return await response.json();
    }
    
    async listModels() {
        const response = await fetch(`${this.baseUrl}/models`);
        return await response.json();
    }
    
    connectVoice(onMessage) {
        const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/voice-conversation`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        return ws;
    }
}

// Usage
const client = new BeautyAIClient();
const response = await client.chat('Hello!', { preset: 'qwen_optimized' });
console.log(response.response);
```

---

**Next**: [Voice Features](VOICE.md) | [Configuration](CONFIGURATION.md)
