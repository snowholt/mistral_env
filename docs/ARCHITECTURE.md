# Architecture Guide

Technical architecture documentation for the BeautyAI Inference Framework.

## 🏗️ System Overview

BeautyAI follows a modern dual-stack architecture with clear separation between backend services and frontend interface.

```
┌─────────────────────────────────────────────────────────────┐
│                    BeautyAI Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Port 5001)          Backend (Port 8000)         │
│  ┌─────────────────────┐      ┌─────────────────────────┐   │
│  │   Flask Web UI      │◄────►│     FastAPI Server      │   │
│  │                     │      │                         │   │
│  │ • Chat Interface    │      │ • REST API              │   │
│  │ • Voice Controls    │      │ • WebSocket Voice       │   │
│  │ • 3D Animations     │      │ • Model Management      │   │
│  │ • Model Management  │      │ • Inference Engines     │   │
│  └─────────────────────┘      └─────────────────────────┘   │
│           │                             │                   │
│           └─── HTTP/WebSocket ──────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   AI Models       │
                    │ • Qwen3-14B       │
                    │ • Edge TTS        │
                    │ • Whisper         │
                    └───────────────────┘
```

## 🔧 Backend Architecture

### Service-Oriented Design

```
backend/src/beautyai_inference/
├── 🎯 cli/                    # Command Line Interface
│   ├── unified_cli.py         # Main CLI entry point
│   └── handlers/              # CLI command handlers
├── 🌐 api/                    # FastAPI Application
│   ├── app.py                 # FastAPI app configuration
│   ├── endpoints/             # API route handlers
│   ├── adapters/              # Service-to-API adapters
│   ├── schemas/               # Request/response models
│   └── middleware/            # Authentication, CORS, etc.
├── 🔧 services/               # Business Logic Services
│   ├── base/                  # Base service classes
│   ├── model/                 # Model management services
│   ├── inference/             # Chat and inference services
│   ├── voice/                 # Voice conversation services
│   ├── config/                # Configuration services
│   └── system/                # System monitoring services
├── 🚀 inference_engines/      # Model Backends
│   ├── transformers_engine.py # Hugging Face Transformers
│   ├── llamacpp_engine.py     # LlamaCpp for GGUF
│   └── voice_engines/         # Voice-specific engines
├── 🏭 core/                   # Core Infrastructure
│   ├── model_factory.py       # Factory pattern for engines
│   ├── model_manager.py       # Singleton model lifecycle
│   └── model_interface.py     # Abstract base classes
└── ⚙️ config/                 # Configuration Management
    └── voice_config_loader.py # Voice service configuration
```

### Design Patterns

#### 1. Factory Pattern
```python
class ModelFactory:
    @staticmethod
    def create_engine(engine_type: str, config: dict):
        if engine_type == "transformers":
            return TransformersEngine(config)
        elif engine_type == "llama.cpp":
            return LlamaCppEngine(config)
        else:
            raise ValueError(f"Unknown engine: {engine_type}")
```

#### 2. Singleton Pattern
```python
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

#### 3. Service Layer Pattern
```python
class BaseService:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def execute(self, *args, **kwargs):
        raise NotImplementedError
```

#### 4. Adapter Pattern
```python
class APIServiceAdapter:
    def __init__(self, service: BaseService):
        self.service = service
    
    async def handle_request(self, request: dict):
        # Convert API request to service call
        return await self.service.execute(**request)
```

## 🎨 Frontend Architecture

### Flask Application Structure

```
frontend/src/
├── app.py                     # Flask application
├── config.json                # Frontend configuration
├── templates/                 # Jinja2 templates
│   ├── index.html            # Main chat interface
│   ├── legacy.html           # Legacy interface
│   └── debug.html            # Debug interface
└── static/                   # Static assets
    ├── css/                  # Stylesheets
    │   └── main.css          # 3D animations & styling
    ├── js/                   # JavaScript
    │   ├── voice.js          # Voice functionality
    │   ├── chat.js           # Chat interface
    │   └── websocket.js      # WebSocket handling
    └── assets/               # Images, fonts, etc.
```

### Component Architecture

#### 1. Voice Component
```javascript
class VoiceManager {
    constructor(websocketUrl) {
        this.ws = new WebSocket(websocketUrl);
        this.mediaRecorder = null;
        this.audioChunks = [];
    }
    
    async startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);
        // ... implementation
    }
}
```

#### 2. Chat Component
```javascript
class ChatInterface {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.messages = [];
    }
    
    async sendMessage(message, options = {}) {
        // ... implementation
    }
}
```

## 🔄 Data Flow

### Chat Request Flow
```
┌─────────────┐    HTTP POST     ┌─────────────┐    Service Call    ┌─────────────┐
│   Frontend  │─────────────────►│   Backend   │───────────────────►│   Service   │
│   Web UI    │                  │   API       │                    │   Layer     │
└─────────────┘                  └─────────────┘                    └─────────────┘
       ▲                                │                                    │
       │                                │            Model Call              │
       │         JSON Response          │            ┌─────────────┐         │
       └────────────────────────────────┘            │   AI Model  │◄────────┘
                                                      │   Engine    │
                                                      └─────────────┘
```

### Voice Request Flow
```
┌─────────────┐  WebSocket/Audio  ┌─────────────┐   Voice Service   ┌─────────────┐
│   Browser   │──────────────────►│   Backend   │──────────────────►│ Voice Stack │
│   Web UI    │                   │  WebSocket  │                   │ STT→LLM→TTS │
└─────────────┘                   └─────────────┘                   └─────────────┘
       ▲                                 │                                  │
       │                                 │                                  │
       │      WebSocket/Audio Response   │                                  │
       └─────────────────────────────────┘◄─────────────────────────────────┘
```

## 🎤 Voice Architecture

### Voice Processing Pipeline
```
Audio Input (Browser)
        │
        ▼
WebSocket Connection
        │
        ▼
Audio Buffer/Chunking
        │
        ▼
Speech-to-Text (Whisper/OpenAI)
        │
        ▼
Language Detection
        │
        ▼
Text Processing (AI Model)
        │
        ▼
Text-to-Speech (Edge TTS)
        │
        ▼
Audio Response (Base64)
        │
        ▼
WebSocket Response
        │
        ▼
Browser Audio Playback
```

### Voice Service Architecture
```python
class SimpleVoiceService:
    def __init__(self):
        self.stt_engine = WhisperEngine()
        self.llm_engine = ModelManager().get_loaded_model()
        self.tts_engine = EdgeTTSEngine()
    
    async def process_voice(self, audio_data: bytes) -> dict:
        # 1. Speech-to-Text
        transcript = await self.stt_engine.transcribe(audio_data)
        
        # 2. Language Model
        response_text = await self.llm_engine.generate(transcript)
        
        # 3. Text-to-Speech
        audio_response = await self.tts_engine.synthesize(response_text)
        
        return {
            "type": "voice_response",
            "audio_base64": audio_response,
            "transcript": transcript,
            "response_text": response_text
        }
```

## 🔧 Configuration Architecture

### Hierarchical Configuration
```
Configuration Sources (Priority Order):
1. Command Line Arguments        (--temperature 0.3)
2. Environment Variables         (BEAUTYAI_API_URL)
3. Configuration Files           (config.json)
4. Model Registry               (model_registry.json)
5. Default Values               (built-in defaults)
```

### Configuration Flow
```python
class ConfigurationManager:
    def __init__(self):
        self.sources = [
            CLIArgumentSource(),
            EnvironmentSource(),
            FileSource("config.json"),
            ModelRegistrySource(),
            DefaultSource()
        ]
    
    def get_config(self, key: str):
        for source in self.sources:
            value = source.get(key)
            if value is not None:
                return value
        return None
```

## 🚀 Deployment Architecture

### Development Environment
```
Developer Machine
├── Backend Development Server
│   ├── Python Virtual Environment
│   ├── FastAPI with Auto-reload
│   ├── SQLite/JSON Configuration
│   └── Local GPU Access
└── Frontend Development Server
    ├── Flask Development Server
    ├── Auto-reload Templates
    └── Static File Serving
```

### Production Environment
```
Production Server
├── Systemd Services
│   ├── beautyai-api.service (Backend)
│   └── beautyai-webui.service (Frontend)
├── Nginx Reverse Proxy
│   ├── SSL Termination
│   ├── WebSocket Proxying
│   └── Static File Serving
├── GPU Resources
│   ├── CUDA Environment
│   ├── Model Storage
│   └── Memory Management
└── Monitoring & Logging
    ├── System Logs
    ├── Application Logs
    └── Performance Metrics
```

## 🔒 Security Architecture

### Authentication Flow
```
Client Request
      │
      ▼
JWT Token Validation
      │
      ▼
Role-based Authorization
      │
      ▼
Service Access Control
      │
      ▼
Resource Access
```

### Security Layers
1. **Network Security**: SSL/TLS, Firewall rules
2. **Authentication**: JWT tokens, API keys
3. **Authorization**: Role-based access control
4. **Input Validation**: Schema validation, sanitization
5. **Rate Limiting**: Per-IP and per-user limits
6. **Audit Logging**: Request/response logging

## 📊 Performance Considerations

### Scalability Patterns
- **Horizontal Scaling**: Multiple backend instances
- **Load Balancing**: Nginx upstream configuration
- **Caching**: Model caching, response caching
- **Resource Pooling**: GPU memory management

### Monitoring Points
- **Response Times**: API endpoint latency
- **Resource Usage**: GPU memory, CPU, RAM
- **Error Rates**: HTTP error status codes
- **Voice Latency**: End-to-end voice response time

---

**Next**: [Performance Guide](PERFORMANCE.md) | [Deployment Guide](DEPLOYMENT.md)
