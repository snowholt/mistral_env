# BeautyAI Inference Framework

Professional-grade multilingual (Arabic-focused) inference stack for text + real‑time voice. This root README is intentionally concise. Detailed operational & implementation docs now live in component READMEs.

## ✨ Core Highlights
| Area | Summary |
|------|---------|
| Backend | FastAPI + WebSocket ultra‑fast (<2s) voice + rich chat inference (25+ params) |
| Frontend | Flask UI + new Live Streaming & PCM Debug consoles |
| Models | Transformers (primary), LlamaCpp; quantized 4/8-bit; Arabic optimized |
| Voice | Auto speech detection, STT (Whisper optimized), TTS (Edge TTS) |
| Tooling | Unified CLI `beautyai`, benchmarking, memory + system services (systemd) |

## 🚀 Quick Start (Dev)
```bash
# Backend
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_server.py        # -> http://localhost:8000/docs

# Frontend (optional UI)
cd ../frontend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/app.py           # -> http://localhost:5001
```
CLI (after activating backend venv):
```bash
beautyai model list
beautyai run chat --preset qwen_optimized
beautyai system status
```

## 🎤 Live Voice Debug Tools (Current Dev Env)
| Tool | URL (example deployment) | Purpose |
|------|--------------------------|---------|
| Live Streaming Console | https://dev.gmai.sa/debug/streaming-live | Real mic → streaming transcription + model replies |
| Voice WebSocket Tester | https://dev.gmai.sa/debug/voice-websocket-tester | Test voice websockets with various audio files, frame-by-frame replay, latency metrics |

## ✅ Latest Weekly Progress (Aug 16–22, 2025)
Short summary (full details in backend & frontend READMEs):
- Added dual debug consoles with real-time metrics & export.
- Fixed conversation bleeding (buffer + decoder state resets per utterance).
- Implemented thinking-mode filtering & LlamaCpp parameter alignment.
- Achieved first-partial transcription latency ~65ms (audio worklet path).
- Roadmap next: Echo / self‑voice suppression → WebRTC low-latency → mobile optimizations.

## 🗂️ Where To Find Details
| Topic | Location |
|-------|----------|
| Backend API / Models / Voice pipeline | `backend/README.md` |
| Frontend UI / Debug consoles / Audio stack | `frontend/README.md` |
| Deployment (systemd, nginx, logging) | `docs/DEPLOYMENT.md` |
| Streaming, Voice internals | `docs/VOICE.md`, `docs/STREAMING_DEBUG_UI_ENHANCEMENT_VALIDATION.md` |

## 🧱 Minimal Concepts
- Model registry JSON defines engines + quantization.
- Unified CLI orchestrates loading, chat, benchmarks.
- WebSocket `/ws/voice-conversation` handles streaming audio + TTS.
- Auto language detection (Arabic / English) in streaming path.

## ⚙️ Requirements (Summary)
GPU (≥8GB VRAM recommended), Python 3.11+, CUDA, Hugging Face token (private models), microphone for voice.

## 🛠️ Production (Very Brief)
Systemd units: `beautyai-api.service`, `beautyai-webui.service` (see backend & frontend READMEs + deployment docs). Nginx handles SSL + WS proxy.

## 🤝 Contributing (Essentials)
PRs: keep root README lean; place depth in component docs. Maintain CLI & API backward compatibility. Add tests for new service logic.

## 📄 License
MIT – see `LICENSE`.

---
Need depth? Jump to: [Backend](backend/README.md) · [Frontend](frontend/README.md) · [API Docs (running)](http://localhost:8000/docs)
