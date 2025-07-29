# 🎨 BeautyAI TTS Architecture Infographic

## 🏗️ Complete Service Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               🌐 WebSocket Layer                                │
│                          websocket_voice.py                                    │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  Client Audio   │────▶│            WebSocketVoiceManager                │   │
│  │   (WebM/WAV)    │    │  • Connection Management                        │   │
│  └─────────────────┘    │  • Session Tracking                             │   │
│                         │  • Real-time Streaming                          │   │
│                         └──────────────────┬───────────────────────────────┘   │
└────────────────────────────────────────────┼───────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          🎯 Voice-to-Voice Service Layer                        │
│                         voice_to_voice_service.py                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │      STT       │  │      LLM       │  │      TTS       │  │   Content    │  │
│  │   (Whisper)    │  │   (Qwen3)      │  │ (TextToSpeech  │  │   Filter     │  │
│  │                │  │                │  │   Service)     │  │              │  │
│  └────────────────┘  └────────────────┘  └───────┬────────┘  └──────────────┘  │
│           Audio → Text → LLM Response → TTS ──────┼─────── Audio Output        │
└───────────────────────────────────────────────────┼─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          🔊 Text-to-Speech Service Layer                        │
│                        text_to_speech_service.py                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ⚠️  CRITICAL OVERRIDE POINT                          │   │
│  │                                                                         │   │
│  │    language_model_mapping = {                                           │   │
│  │        "coqui-tts-arabic": "xtts_v2",      ←──── All roads lead here   │   │
│  │        "coqui-tts-english": "xtts_v2",     ←──── All roads lead here   │   │
│  │        "coqui-tts-multilingual": "xtts_v2" ←──── All roads lead here   │   │
│  │    }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                           │
│  ┌─────────────┐    ┌──────────────▼────────────┐    ┌─────────────────────┐   │
│  │   Engine    │    │     Model Selection       │    │    Speech           │   │
│  │  Selection  │    │   (Always XTTS v2!)       │    │   Generation        │   │
│  │             │    │                           │    │                     │   │
│  └─────────────┘    └───────────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────┼───────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🐸 Coqui TTS Engine Layer                             │
│                          coqui_tts_engine.py                                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    from TTS.api import TTS                              │   │
│  │                                                                         │   │
│  │    def load_model(self):                                                │   │
│  │        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"    │   │
│  │        self.tts = TTS(model_name=model_name, gpu=True)                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                           │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      🎭 Voice Features                                  │   │
│  │                                                                         │   │
│  │  ✅ 16 Languages (Including Arabic)     ✅ Voice Cloning               │   │
│  │  ✅ 24kHz High Quality Audio            ✅ <200ms Latency              │   │
│  │  ✅ GPU Acceleration                    ✅ Streaming Support            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────┼───────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           📦 Coqui TTS Library (Python Package)                 │
│                             pip install coqui-tts                              │
│                                                                                 │
│  Model Storage: ~/.local/share/tts/ (NOT HuggingFace cache!)                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  tts_models--multilingual--multi-dataset--xtts_v2/                     │   │
│  │  ├── model.pth           (1.8GB - Neural Network)                      │   │
│  │  ├── speakers_xtts.pth   (7.7MB - Speaker Embeddings)                 │   │
│  │  ├── vocab.json          (361KB - Tokenizer)                           │   │
│  │  └── config.json         (4KB - Configuration)                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Model Loading Decision Tree

```
🎯 User Request: "coqui-tts-arabic"
                │
                ▼
┌───────────────────────────────────┐
│       Model Registry Check        │
│   model_registry.json says:       │
│   "tts_models/ar/tn_arabic..."    │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│    TextToSpeechService Override   │
│   language_model_mapping says:    │
│   "tts_models/.../xtts_v2"        │ ← 🔧 OVERRIDE HERE
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│      CoquiTTSEngine Hardcode     │
│       Always loads:               │
│   "tts_models/.../xtts_v2"        │ ← 🔧 FINAL DECISION
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│        Coqui TTS Library          │
│     Downloads & Loads XTTS v2     │
│    from ~/.local/share/tts/       │
└───────────────────────────────────┘
```

## 📊 Model Comparison Matrix

```
┌─────────────────┬──────────────┬──────────────┬─────────────────┬──────────────┐
│     Feature     │   XTTS v2    │     VITS     │   Tacotron2     │  Edge TTS    │
├─────────────────┼──────────────┼──────────────┼─────────────────┼──────────────┤
│ Arabic Support  │      ✅      │      ❌      │       ❌        │      ✅      │
│ Voice Cloning   │      ✅      │      ❌      │       ❌        │      ❌      │
│ Multilingual    │  ✅ (16 Lang) │   ❌ (1 Lang) │    ❌ (1 Lang)   │  ✅ (Many)   │
│ Quality         │   🌟🌟🌟🌟    │   🌟🌟🌟     │     🌟🌟        │   🌟🌟🌟     │
│ Speed           │   🚀🚀🚀     │   🚀🚀🚀🚀    │     🚀🚀        │   🚀🚀🚀🚀   │
│ Model Size      │    1.8GB     │    ~100MB    │     ~50MB       │   Online     │
│ GPU Required    │   Optional   │   Optional   │    Optional     │      No      │
│ Offline         │      ✅      │      ✅      │       ✅        │      ❌      │
│ Used in BeautyAI│      ✅      │      ❌      │       ❌        │   Available  │
└─────────────────┴──────────────┴──────────────┴─────────────────┴──────────────┘
```

## 🎭 XTTS v2 Feature Showcase

```
                    🐸 XTTS v2 - The Universal TTS Model
                    
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Core Features                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🌍 Multilingual Support     🎭 Voice Cloning         ⚡ Real-time          │
│  ┌─────────────────────┐     ┌─────────────────────┐  ┌─────────────────────┐ │
│  │ Arabic (ar) ✅      │     │ 3+ seconds audio    │  │ <200ms latency      │ │
│  │ English (en) ✅     │     │ Clone any voice     │  │ Streaming support   │ │
│  │ Spanish (es) ✅     │     │ Cross-language      │  │ GPU acceleration    │ │
│  │ French (fr) ✅      │     │ Speaker adaptation  │  │ CPU fallback        │ │
│  │ German (de) ✅      │     │ Professional quality│  │ Memory efficient    │ │
│  │ + 11 more langs ✅  │     │ Emotional control   │  │ Batch processing    │ │
│  └─────────────────────┘     └─────────────────────┘  └─────────────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Technical Specs                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📏 Model Size: 1.8GB         🎵 Audio Quality: 24kHz                     │
│  🧠 Architecture: Transformer   🔧 Framework: PyTorch                      │
│  💾 RAM Usage: 2-3GB          ⚙️ GPU Memory: 1-2GB                        │
│  🏠 Storage: ~/.local/share/tts/                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔍 Registry vs Reality Comparison

```
📋 Model Registry Says:           🎯 Reality Check:
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ "coqui-tts-arabic": {       │   │ ALL MODELS ROUTE TO:        │
│   "model_id":               │ → │                             │
│   "tts_models/ar/tn_..."    │   │ "tts_models/multilingual/   │
│ }                           │   │  multi-dataset/xtts_v2"     │
│                             │   │                             │
│ "coqui-tts-multilingual": { │ → │ Why? Because XTTS v2:       │
│   "model_id":               │   │ ✅ Better Arabic support    │
│   "tts_models/multi..."     │   │ ✅ Voice cloning features   │
│ }                           │   │ ✅ Higher quality output    │
│                             │   │ ✅ Unified architecture     │
└─────────────────────────────┘   └─────────────────────────────┘

🚨 DISCREPANCY ALERT:
The model registry defines models that are NEVER actually loaded!
All TTS requests are overridden to use XTTS v2 in the service layer.
```

## 🔧 Service Integration Flow

```
🎤 Audio Input
     │
     ▼
┌─────────────────────────┐
│   WebSocket Endpoint   │ ── Receives WebM/WAV audio
│  websocket_voice.py    │    Manages real-time connection
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Voice-to-Voice Service │ ── Orchestrates full pipeline
│ voice_to_voice_service  │    STT → LLM → TTS → Audio
└───────────┬─────────────┘
            │
            ▼ tts_service.load_tts_model("coqui-tts-arabic")
┌─────────────────────────┐
│ Text-to-Speech Service  │ ── Maps to XTTS v2
│ text_to_speech_service  │    Creates CoquiTTSEngine
└───────────┬─────────────┘
            │
            ▼ CoquiTTSEngine(model_config)
┌─────────────────────────┐
│   Coqui TTS Engine     │ ── Loads XTTS v2 model
│  coqui_tts_engine.py   │    Generates speech audio
└───────────┬─────────────┘
            │
            ▼ TTS("tts_models/multilingual/multi-dataset/xtts_v2")
┌─────────────────────────┐
│   Coqui TTS Library    │ ── Downloads from GitHub releases
│   Python Package       │    Caches in ~/.local/share/tts/
└───────────┬─────────────┘
            │
            ▼
    🔊 Audio Output
```

## 🎯 Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           🎯 MAIN INSIGHTS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1️⃣ ONE MODEL RULES THEM ALL                                           │
│     Despite multiple model names, XTTS v2 handles everything           │
│                                                                         │
│  2️⃣ REGISTRY ≠ REALITY                                                 │
│     Model registry definitions are overridden by service logic         │
│                                                                         │
│  3️⃣ COQUI ≠ HUGGINGFACE                                                │
│     Models stored in ~/.local/share/tts/, not HuggingFace cache        │
│                                                                         │
│  4️⃣ XTTS v2 IS PART OF COQUI TTS                                       │
│     Not separate libraries - XTTS v2 is Coqui's flagship model         │
│                                                                         │
│  5️⃣ EXCELLENT ARABIC SUPPORT                                           │
│     Native Arabic synthesis with voice cloning capabilities            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📚 Quick Reference

```
🔗 Important Links:
├── GitHub: https://github.com/coqui-ai/TTS
├── Docs: https://docs.coqui.ai/
├── Models: https://huggingface.co/coqui
└── XTTS v2: https://docs.coqui.ai/en/latest/models/xtts.html

💾 Storage Locations:
├── Models: ~/.local/share/tts/
├── Cache: ~/.cache/tts/ (if exists)
├── Config: ~/.config/tts/ (if exists)
└── BeautyAI: /home/lumi/beautyai/voice_tests/

🛠️ Commands:
├── Install: pip install coqui-tts
├── List Models: tts --list_models
├── Clear Cache: rm -rf ~/.local/share/tts/
└── Test: tts --text "Hello" --model_name "xtts_v2"

🔧 Configuration Files:
├── Model Registry: beautyai_inference/config/model_registry.json
├── TTS Service: beautyai_inference/services/text_to_speech_service.py
├── Coqui Engine: beautyai_inference/inference_engines/coqui_tts_engine.py
└── WebSocket: beautyai_inference/api/endpoints/websocket_voice.py
```

---

*Infographic generated for BeautyAI TTS Architecture Analysis*  
*Visual guide to understanding service relationships and model usage*  
*July 19, 2025*
