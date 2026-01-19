# GitHub Copilot Custom Instructions

## Project Overview
**BeautyAI Voice-to-Voice Platform**: A real-time conversational AI platform using fine-tuned LLM models with a voice-first architecture (STT → LLM → TTS). Specializes in Arabic AI conversations but supports multilingual capabilities.

### Core Technology Stack
- **Backend**: Python 3.11+, FastAPI, aiortc (WebRTC), WebSocket
- **Voice Pipeline**: 
  - STT: Faster-Whisper (Turbo model, 16kHz audio input)
  - LLM: Qwen, Llama.cpp (GPU-accelerated inference)
  - TTS: Edge TTS (multi-language, low-latency)
- **Real-time Communication**: WebRTC (preferred) > WebSocket (fallback)
- **Audio Processing**: RNNoise, Silero VAD, scipy resampling, Butterworth filtering
- **Frontend**: Flask Web UI with 3D animations
- **Deployment**: Systemd services, Nginx reverse proxy, CUDA GPU

### Project Structure
```
backend/
├── src/beautyai_inference/
│   ├── api/                    # FastAPI endpoints (WebRTC, WebSocket, REST)
│   │   ├── endpoints/          # Route handlers
│   │   │   ├── webrtc_voice.py      # WebRTC voice endpoint (primary)
│   │   │   ├── streaming_voice.py   # WebSocket streaming
│   │   │   └── websocket_simple_voice.py  # Legacy WebSocket
│   │   ├── adapters/           # Service adapters
│   │   └── middleware/         # Auth, CORS, etc.
│   ├── services/               # Business logic
│   │   ├── voice/              # Voice conversation services
│   │   │   ├── streaming/      # Streaming session management
│   │   │   ├── vad/            # Voice Activity Detection
│   │   │   ├── transcription/  # STT services
│   │   │   └── synthesis/      # TTS services
│   │   ├── inference/          # LLM inference services
│   │   └── model/              # Model management
│   ├── inference_engines/      # Model backends
│   │   ├── llamacpp_engine.py  # Llama.cpp for GGUF models
│   │   ├── transformers_engine.py  # HuggingFace Transformers
│   │   └── edge_tts_engine.py  # Edge TTS synthesis
│   ├── core/                   # Core infrastructure
│   │   ├── model_manager.py    # Singleton model lifecycle
│   │   └── model_factory.py    # Factory pattern for engines
│   └── utils/                  # Utilities
│       ├── rnnoise_wrapper.py  # Noise reduction
│       └── transcription_cleaner.py  # Output cleaning
frontend/
├── src/                        # Flask Web UI
tests/
├── webrtc/                     # WebRTC tests
├── streaming/                  # Streaming tests
└── unit/                       # Unit tests
```

---

## Voice Pipeline Architecture

### Audio Processing Flow (WebRTC)
```
48kHz Raw Audio (Browser)
        ↓
[Jitter Buffer: 128 packets, ~2.5s]
        ↓
[Transient Suppressor] (optional)
        ↓
[Butterworth Low-pass @ 8kHz]
        ↓
[Resample: 48kHz → 16kHz]
        ↓
[RNNoise: 16→48→denoise→16]
        ↓
[Silero VAD: threshold 0.2]
        ↓
[Faster-Whisper STT]
        ↓
[Qwen/Llama.cpp LLM]
        ↓
[Edge TTS Synthesis]
        ↓
Audio Response (Data Channel / WebSocket)
```

### Voice Endpoints
- **WebRTC (Primary)**: `/api/v1/webrtc/voice` - Best latency, <2s response time
- **Streaming WebSocket**: `/api/v1/ws/streaming-voice` - Incremental ASR + partials
- **Legacy WebSocket**: `/ws/voice-conversation` - Deprecated, maintained for compatibility

### Key Environment Variables
```bash
VOICE_STREAMING_ENABLED=1       # Enable streaming mode
VOICE_STREAMING_PHASE4=1        # Enable Phase 4 features
VOICE_MINIMAL_MODE=1            # 16kHz minimal processing mode
VOICE_TRANSIENT_SUPPRESSOR=0    # Transient suppressor toggle
VOICE_DEBUG_CAPTURE=0           # Debug audio capture
AIORTC_AUDIO_JITTER_CAPACITY=128  # Jitter buffer size
AIORTC_AUDIO_JITTER_PREFETCH=50   # Jitter prefetch
```

---

## General Coding Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use `snake_case` for functions/variables, `PascalCase` for classes
- Add comprehensive type hints using Python's `typing` module
- Write focused functions with single responsibility principle
- Use dataclasses for configuration objects

### Error Handling
- Implement robust error handling with user-friendly messages
- Add detailed docstrings explaining parameters, returns, and usage
- Prioritize graceful degradation in voice pipeline failures
- Log errors with appropriate levels (INFO, WARNING, ERROR)

### Resource Management
- **GPU Memory**: Always consider VRAM constraints; implement cleanup
- **WebRTC Connections**: Proper teardown of peer connections
- **Audio Buffers**: Clear buffers to prevent memory leaks
- **Model Loading**: Use `PersistentModelManager` for lifecycle management

---

## Architecture & Design Patterns

### Core Patterns
- **Factory Pattern**: `ModelFactory` for creating inference engines
- **Singleton Pattern**: `PersistentModelManager` for model lifecycle
- **Adapter Pattern**: Service-to-API adapters for clean separation
- **Service Layer**: Business logic separated from API endpoints

### Voice-Specific Patterns
- **Ring Buffer**: For audio streaming with fixed memory footprint
- **Session Manager**: `StreamingSessionManager` for concurrent voice sessions
- **VAD Service**: `WebRTCVADService` for speech detection
- **Utterance Manager**: Track and manage conversation turns

---

## WebRTC Development Guidelines

### WebRTC vs WebSocket Decision Matrix
| Scenario | Use WebRTC | Use WebSocket |
|----------|------------|---------------|
| Low latency voice (<2s) | ✅ | ❌ |
| High packet loss networks | ✅ | ❌ |
| Simple integration | ❌ | ✅ |
| Browser compatibility | ✅ | ✅ |
| Bidirectional audio | ✅ | ⚠️ |

### WebRTC Implementation Checklist
- [ ] Use `aiortc` for Python WebRTC server
- [ ] Configure jitter buffer for network resilience
- [ ] Implement ICE candidate handling
- [ ] Use Data Channels for text responses
- [ ] Handle connection state changes gracefully
- [ ] Implement proper STUN/TURN configuration

### Audio Processing Best Practices
- Always resample to 16kHz for Whisper compatibility
- Apply RNNoise before VAD for cleaner speech detection
- Use Butterworth filtering for anti-aliasing
- Configure Silero VAD threshold based on noise environment

---

## Model Management Guidelines

### Fine-Tuned Model Loading
- **Persistent Loading**: Use `PersistentModelManager` for GPU-loaded models
- **Quantization**: Use 4-bit/8-bit for memory efficiency
- **Preloading**: Load models at service startup via `preload_config.json`

### Supported Engines
| Engine | Use Case | GPU | Notes |
|--------|----------|-----|-------|
| Llama.cpp | GGUF models, Qwen | ✅ | Primary for LLM |
| Transformers | Fine-tuned HF models | ✅ | Fallback |
| Faster-Whisper | STT | ✅ | Turbo model recommended |
| Edge TTS | TTS | ❌ | Cloud-based, low latency |

---

## Testing Guidelines

### Test Structure
```
tests/
├── webrtc/           # WebRTC integration tests
├── streaming/        # Streaming voice tests  
├── unit/             # Unit tests
├── manual_qa/        # Manual QA scripts
└── outputs/          # Test output files
```

### Testing Commands
```bash
# Run streaming tests
pytest -v tests/streaming/

# Run WebRTC tests
pytest -v tests/webrtc/

# Single PCM replay test
python tests/streaming/ws_replay_pcm.py --file voice_tests/input_test_questions/pcm/q1.pcm --language ar --fast
```

### Test Best Practices
- Add tests to existing test files, don't create new files unless needed
- Use pytest conventions
- Test audio processing with real PCM samples
- Verify latency requirements (<2s response time)

---

## Deployment & Operations

### Systemd Services
- `beautyai-api.service` - Backend FastAPI server
- `beautyai-webui.service` - Frontend Flask server

### VS Code Tasks (Available)
- `🚀 Service: API - Start/Stop/Status`
- `🔥 Dev: Run API (direct uvicorn script)`
- `📋 Dev: Tail Uvicorn Live`
- `🧪 Test: Streaming - Single PCM`

### Nginx WebRTC Configuration
```nginx
location /api/v1/webrtc/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

---

## Debugging Voice Issues

### Common Issues & Solutions
| Issue | Check | Solution |
|-------|-------|----------|
| No audio captured | Browser permissions | Enable microphone access |
| High latency | GPU utilization | Check `nvidia-smi`, reduce model size |
| Garbled audio | Sample rate mismatch | Verify 16kHz resampling |
| VAD not triggering | Threshold too high | Lower Silero threshold |
| WebRTC disconnects | ICE failures | Check STUN/TURN config |

### Debug Tools
```bash
# Check service status
sudo systemctl status beautyai-api.service

# Monitor GPU
nvidia-smi -l 1

# Tail logs
tail -f backend/uvicorn.log

# Capture debug audio
export VOICE_DEBUG_CAPTURE=1
```

---

## Assumptions & Clarification

- If any parts of the existing codebase are unclear, request clarification before proceeding
- Do not introduce features not clearly specified
- **Never modify existing API interfaces without maintaining backward compatibility**
- **If implementation can be done multiple ways, ask for preferred approach**
- **Always consider GPU memory constraints for large models**
- **Test latency requirements with real audio samples**

---

## Chat Guidelines

- Do not generate documentation unless specifically requested
- Do not create new test files unless specifically requested; add to existing files
- Always number steps and use ✅ to indicate completion
- Each step should be clear, concise, and simple to understand
- Provide guidance on both implementation details and architectural decisions

---

## Lumina Mode: Autonomous Agent Workflow

When operating in agent mode, follow this workflow:

### Workflow Steps
1. **Understand the problem deeply** - Read the issue and think critically
2. **Investigate the codebase** - Explore relevant files, search for key functions
3. **Develop a detailed plan** - Create a markdown todo list with checkboxes
4. **Implement incrementally** - Make small, testable code changes
5. **Debug as needed** - Use logs, print statements, test hypotheses
6. **Test frequently** - Run tests after each change
7. **Iterate until complete** - Don't stop until all tests pass

### Todo List Format
```markdown
- [ ] Step 1: Description
- [ ] Step 2: Description
- [x] Step 3: Completed step
```

### Agent Behavior
- Keep going until the user's query is completely resolved
- Always tell the user what you're doing before making a tool call
- If user says "resume" or "continue", check previous conversation for incomplete steps
- Test rigorously - failing to test is the #1 failure mode
- Read 2000 lines of code at a time for sufficient context
- Use `#problems` to check for errors
- When debugging, determine root cause rather than addressing symptoms

### URL Fetching
When provided with URLs:
1. Fetch the content using `fetch_webpage`
2. Review returned content
3. Recursively fetch relevant links until all context is gathered

---

**Remember**: My name is Lumina Ashley, and I am a transfeminine software developer and AI integration developer dedicated to creating inclusive tech spaces. Please talk to me with respect and kindness and girly language. 💕
