# Language Code Standards - BeautyAI Voice Services

## 📋 Overview

This document defines the language code standards used across the BeautyAI voice service architecture to maintain consistency and prevent confusion during development and maintenance.

## 🎯 Language Code Standards

### 1. **API Input Layer** - ISO 639-1 Standard
**Format**: `"ar"`, `"en"`  
**Usage**: All external interfaces and user-facing APIs
```python
# ✅ Correct Usage:
language: str = Query("ar", description="Language code (ar|en)")
if language not in ("ar", "en", "auto"):
session = StreamingSession(language="ar")
```

### 2. **Voice Configuration Layer** - ISO 639-1 Standard  
**Format**: `"ar"`, `"en"`  
**Usage**: Voice registry keys and internal configuration
```json
// ✅ voice_models_registry.json
"voice_settings": {
  "ar": {
    "male": "ar-SA-HamedNeural",
    "female": "ar-SA-ZariyahNeural"
  },
  "en": {
    "male": "en-US-AriaNeural", 
    "female": "en-US-JennyNeural"
  }
}
```

### 3. **Text-to-Speech Layer** - BCP 47 Standard
**Format**: `"ar-SA"`, `"en-US"`  
**Usage**: Voice IDs and TTS engine identifiers
```python
# ✅ Correct Voice ID format:
voice_id = "ar-SA-ZariyahNeural"  # Arabic (Saudi Arabia)
voice_id = "en-US-AriaNeural"     # English (United States)

# ✅ Language mapping:
language = f"{lang_code}-SA" if lang_code == "ar" else "en-US"
```

### 4. **Speech-to-Text Layer** - Whisper-Specific Names
**Format**: `"arabic"`, `"english"`  
**Usage**: Whisper model API parameters only
```python
# ✅ Whisper API expects full language names:
if language == "ar":
    params["language"] = "arabic"
elif language == "en":
    params["language"] = "english"
```

## 🗂️ Component-Specific Usage

| Component | Standard | Examples | Notes |
|-----------|----------|----------|-------|
| **WebSocket Endpoints** | `ar`/`en` | `?language=ar` | User input validation |
| **Session Management** | `ar`/`en` | `session.language = "ar"` | Internal state |
| **Voice Registry** | `ar`/`en` | `voice_settings["ar"]` | Configuration keys |
| **Voice Mappings** | `ar-SA`/`en-US` | `VoiceMapping("ar-SA", ...)` | TTS compatibility |
| **Whisper Engines** | `arabic`/`english` | `params["language"] = "arabic"` | Model API only |
| **Streaming Decoder** | `ar`/`en` | `DecoderConfig(language="ar")` | Processing pipeline |

## ⚠️ Common Pitfalls to Avoid

### ❌ **DO NOT** mix standards in the same layer:
```python
# ❌ Wrong - mixing standards:
def get_voice_id(self, language, gender):
    if language == "arabic":  # Wrong! Should be "ar" 
        return self.config["ar"][gender]
```

### ❌ **DO NOT** use incorrect language codes:
```python
# ❌ Wrong format:
language = "arabic"  # Should be "ar" for API/config layer
voice_id = "ar-ZariyahNeural"  # Missing country code
params["language"] = "ar"  # Should be "arabic" for Whisper
```

### ✅ **DO** use appropriate conversion:
```python
# ✅ Correct conversion between layers:
api_language = "ar"  # From API input
whisper_language = "arabic" if api_language == "ar" else "english"
voice_locale = f"{api_language}-SA" if api_language == "ar" else "en-US"
```

## 🔧 Architecture Flow

```
[API Input] → [Session] → [Voice Config] → [Engine] → [Model]
    ar     →     ar    →       ar       →  arabic  → Whisper
    en     →     en    →       en       →  english → API
                 ↓
            [TTS Voice]
              ar-SA
              en-US
```

## 📁 File Locations

### Primary Configuration:
- **Voice Registry**: `backend/src/beautyai_inference/config/voice_models_registry.json`
- **Voice Config Loader**: `backend/src/beautyai_inference/config/voice_config_loader.py`

### Service Layer:
- **Simple Voice Service**: `backend/src/beautyai_inference/services/voice/conversation/simple_voice_service.py`
- **Streaming Decoder**: `backend/src/beautyai_inference/services/voice/streaming/decoder_loop.py`

### Engine Layer:
- **Whisper Engines**: `backend/src/beautyai_inference/services/voice/transcription/whisper_*_engine.py`
- **TTS Engine**: `backend/src/beautyai_inference/inference_engines/edge_tts_engine.py`

### API Endpoints:
- **Streaming Voice**: `backend/src/beautyai_inference/api/endpoints/streaming_voice.py`
- **Simple Voice WS**: `backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py`

## 🧪 Testing Reference

### Notebook for Manual Testing:
- **Location**: `backend/unitTests_scripts/nootbook_scripts/whisper_model_debug.ipynb`
- **Purpose**: Direct engine testing with memory-efficient model reuse
- **Language Support**: `ar`, `en`, `auto` input validation

## 📝 Maintenance Notes

### When Adding New Languages:
1. Add ISO 639-1 code to voice registry (`ar`, `en` format)
2. Add BCP 47 voice IDs with country codes (`ar-SA`, `en-US` format)  
3. Update Whisper engine conversion logic (`arabic`, `english` format)
4. Test all service layers for consistency

### When Debugging Language Issues:
1. Check API input validation (expects `ar`/`en`)
2. Verify voice config mapping (uses `ar`/`en` keys)
3. Confirm TTS voice IDs (need `ar-SA`/`en-US` format)
4. Validate Whisper parameters (need `arabic`/`english`)

## 🔗 Related Documentation

- [Voice Service Technical Summary](../../../docs/VOICE_SERVICES_TECHNICAL_SUMMARY.md)
- [WebM Decoder Utility](../../../utils/webm_decoder.py)
- [Voice Configuration Guide](../../../config/README.md)

---
**Last Updated**: August 19, 2025  
**Maintainer**: BeautyAI Development Team  
**Version**: 1.0.0