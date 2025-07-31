feat(voice): Complete voice registry integration with GPU-optimized Whisper

🎯 **Major Voice Backend Refactor - Registry-Driven Architecture**

## ✨ Features
- Centralized voice configuration loader (voice_config_loader.py)
- GPU-optimized Whisper model (large-v3-turbo) with CUDA acceleration
- Registry-driven model selection across all voice services
- Eliminated configuration drift and hardcoded model references

## 🔄 Refactored Services
- `faster_whisper_service.py`: Registry-based model loading, GPU acceleration
- `simple_voice_service.py`: Registry-driven voice mappings and audio config
- `websocket_simple_voice.py`: Inherits clean config through service layer
- `app.py`: Removed advanced voice endpoints, clean API documentation

## 🗂️ Configuration Files
- `voice_models_registry.json`: Clean registry with only 2 models (Whisper + Edge TTS)
- `model_registry.json`: Restored comprehensive model list, cleaned config section
- `voice_config_loader.py`: Centralized config loader preventing drift

## 🚀 Performance Improvements
- GPU acceleration enabled for Whisper transcription
- Performance targets: <1.5s total latency (800ms STT + 500ms TTS)
- Consistent audio format (WAV @ 22050Hz) across all services
- Memory optimization: <50MB per connection

## 🧹 Cleanup
- Removed all hardcoded model names and format references
- Eliminated advanced voice endpoint documentation and routes
- Consistent voice mappings: 4 combinations (ar/en × male/female)
- Single source of truth for all voice configurations

## ✅ Validation
- Comprehensive integration tests passing
- All services load config from registry only
- API endpoints clean and documented
- No configuration drift detected

**Breaking Changes**: Advanced voice endpoints removed
**Performance**: Expected 2-3x speed improvement with GPU acceleration
**Maintainability**: Single registry file controls all voice configurations
