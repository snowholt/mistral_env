# WebRTC MVP Migration Plan

Date: 2025-10-14  
Owner: BeautyAI Engineering  
Status: Planned (Ready for Implementation)  
Scope: Add parallel WebRTC voice-to-voice pipeline (keep existing WebSocket operational)  

---
## 1. Objective & Scope
- Deliver a browser-based WebRTC voice conversation flow that runs alongside the existing WebSocket implementation (WebSocket retained as fallback).
- Reuse existing voice pipeline: Whisper STT → Qwen LLM (with enforced `/no_think ` prefix) → Edge TTS.
- Enforce configurable 10-second utterance limit (client + server) via environment variables.
- Leverage WebRTC’s built‑in acoustic enhancements: AGC, Noise Suppression, Echo Cancellation.
- Remote, browser-access-only testing via `dev.gmai.sa` (no physical server access).

Out of Scope (MVP): multi-speaker diarization, barge‑in interruption, TURN relay, adaptive bitrate tuning, partial/streaming STT mid-utterance.

---
## 2. References
| Artifact | Path | Purpose |
|----------|------|---------|
| Analysis report | `reports/webrtc_migration_analysis.md` | Architectural rationale & benefits |
| Technical inventory | `reports/webrtc_migration_technical_analysis.md` (if added) | Component reuse map |
| Session decisions | `reports/WEBRTC_plan_temp_session.md` (if/when created) | Decision trace |
| Inspiration | KoljaB/RealtimeSTT | Dual VAD + buffering patterns |

---
## 3. Guiding Constraints & Non-Functional Requirements
- Simplicity: Single active user audio track; ignore new speech during TTS playback (document future barge‑in upgrade path).
- Latency SLO (p90): Round-trip ≤ 6000 ms (STT ≤ 2000 ms, LLM ≤ 3000 ms, TTS ≤ 1000 ms).
- Reliability: ≥99% successful signaling/connection establishment for supported browsers (Chrome ≥ 114, Edge ≥ 114, Safari ≥ 17 — fallback otherwise).
- Observability: Emit structured metrics with `webrtc_voice_*` prefix and log ICE / VAD / pipeline timing events.
- Security: HTTPS-only signaling; sanitize SDP/ICE JSON; no raw file writes from remote SDP.
- Feature flag: Ability to disable WebRTC globally without code removal.

---
## 4. Configuration Additions
Backend (add to `config/defaults.json`, surfacing to `.env`):
- `VOICE_WEBRTC_MAX_UTTERANCE_SEC` (int, default: 10)
- `VOICE_WEBRTC_FEATURE_ENABLED` (bool, default: true)
- `VOICE_WEBRTC_LANGUAGE_THRESHOLDS` (JSON string: `{ "english": 0.5, "arabic": 0.45 }`)
- `VOICE_WEBRTC_VAD_MODE` (int 0–3 for `webrtcvad`, default: 2)

Frontend (`frontend/.env.example`):
- `WEBRTC_MAX_UTTERANCE_SEC=10`
- `WEBRTC_FEATURE_ENABLED=true`

Docs: Update `docs/CONFIGURATION.md` and `docs/VOICE.md` accordingly.

---
## 5. High-Level Phases
| Phase | Name | Goal |
|-------|------|-----|
| A | Preparation | Dependencies, config, logging scaffold |
| B | Signaling & Session | SDP + ICE endpoints, session/connection pool |
| C | Media Processing | Track handling, dual VAD, buffering, utterance cap |
| D | Frontend Client | WebRTC client, UI toggle, playback handling |
| E | Integration & Metrics | End-to-end tests, instrumentation, tooling |
| F | Deployment | Nginx, systemd, feature flag, runtime validation |
| G | Documentation | Final docs, troubleshooting, runbook |

---
## 6. Detailed Work Breakdown
### Phase A — Preparation & Environment
1. Add dependencies to `backend/requirements.txt`: `aiortc`, `pyee`, `webrtcvad`.
2. System libs (document): `libopus`, `libvpx`, `libssl`, `libffi`, `libavformat` (FFmpeg if using advanced transforms). Provide install snippet (Ubuntu) in `docs/DEPLOYMENT.md`.
3. Add config keys (Section 4) + load into existing config loader.
4. Extend logging categories: `webrtc.signaling`, `webrtc.vad`, `webrtc.conn`, `webrtc.audio`.
5. Feature flag injection in app startup: skip router registration if disabled.

### Phase B — Backend Signaling & Session Lifecycle
New file: `backend/src/beautyai_inference/api/endpoints/webrtc_voice.py`
Endpoints:
- `POST /api/v1/webrtc/voice/offer` → Input: `{ "sdp": str, "type": "offer", "language": "en|ar"? }` → Output: `{ "sdp": answer, "type": "answer", "peer_id": str }`.
- `POST /api/v1/webrtc/voice/ice` → `{ "peer_id": str, "candidate": {...} }` → 202.
- `DELETE /api/v1/webrtc/voice/{peer_id}` → Cleanup (optional; also rely on timeout GC).
Supporting modules:
- `backend/src/beautyai_inference/core/webrtc_connection_pool.py`:
  - Data: `peer_id`, `RTCPeerConnection`, state, created_at, last_activity.
  - Methods: `create_peer(language)`, `get(peer_id)`, `add_ice(peer_id, candidate)`, `close(peer_id)`, `gc_expired(max_age=900s)`.
- `backend/src/beautyai_inference/core/webrtc_session_manager.py`:
  - Wraps existing `VoiceSessionManager` by mapping `peer_id` → session state.
  - Augments metadata: ICE state transitions, language, utterance counters.

Timeouts:
- Idle peer auto-close after 120s without audio frames.
- GC task runs every 60s (background `asyncio.create_task`).

### Phase C — Media Track Processing, VAD & Buffer
Files:
- `services/voice/webrtc_audio_processor.py`
  - Accept inbound `MediaStreamTrack` (audio) from aiortc.
  - Convert `AudioFrame` → 16 kHz mono PCM bytes (resample if needed from 48 kHz).
  - Append to circular buffer (pre-roll 300 ms).
  - Manage utterance window timer (stop accept after 10 s or VAD end + grace 500 ms).
  - Yield complete utterance to STT pipeline.
- `services/voice/vad/webrtc_vad_service.py`
  - Combines `webrtcvad.Vad` (fast) + existing Silero (already dependency) for confirmation.
  - States: `SILENCE`, `MAYBE_SPEECH`, `SPEECH_ACTIVE`, `POST_SPEECH`.
  - Emits events for metrics & debug.
- `core/webrtc_buffer_manager.py` (or subclass existing buffer integration):
  - APIs: `push(frame_bytes, timestamp_ms)`, `get_active_window()`, `drain_current_utterance()`.
  - Maintains `pre_roll_frames`, `speech_frames`, `post_roll_frames`.
Integration:
- Adapter to existing STT call site: ensure `/no_think ` prefix inserted before LLM invocation.
- Add utterance-level metrics: `duration_ms`, `speech_ratio`, `frames_dropped`.

### Phase D — Frontend WebRTC Client
File: `frontend/src/static/js/webrtcVoiceClient.js`
Responsibilities:
- Generate `RTCPeerConnection` with `iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]` (+ optional others).
- Acquire mic: `getUserMedia` with `{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }`.
- Create data channel `control` (optional, for VAD debug messages in later iteration — stub now).
- Send SDP offer → receive answer.
- Post ICE candidates as they appear.
- Track remote audio (TTS) via `ontrack` event → attach to hidden `<audio>` element.
- Enforce 10 s client recording window (stop tracks on timer or remote END event).
- Integrate existing `ImprovedVAD` for UI feedback only (server remains source of truth).
Feature flag:
- Add `isWebRTCEnabled()` helper in `frontend/src/static/js/config.js`.
- Modify HTML template to show WebRTC or WebSocket button set.

### Phase E — Integration, Metrics & Tooling
1. Add metrics emission in backend (existing performance monitoring service):
   - `webrtc_voice_stt_latency_ms`
   - `webrtc_voice_llm_latency_ms`
   - `webrtc_voice_tts_latency_ms`
   - `webrtc_voice_round_trip_latency_ms`
   - `webrtc_voice_vad_fp_count`
   - `webrtc_voice_utterance_seconds`
2. Extend `tools/service_analyzer.py` to summarize new metrics.
3. Add dev probe script: `tools/webrtc_signaling_probe.py` (curl-based or httpx) performing offer/answer handshake using dummy SDP.
4. Add end-to-end test harness (no real media) simulating frame injection (synthetic sine wave) to validate pipeline.

### Phase F — Deployment & Infrastructure
Nginx:
- Update site config: ensure `/api/v1/webrtc/voice` routes to FastAPI upstream (standard proxy_pass; no Upgrade headers needed).
Systemd:
- Add environment overrides for new vars in `beautyai-api.service` or drop an EnvironmentFile snippet.
- Confirm aiortc imports succeed (test service start).
Feature Flag Rollout:
- Stage 1: Disabled by default → manual enable in staging.
- Stage 2: Enable for internal testers.
- Stage 3: Public beta; maintain WebSocket fallback button for regression mitigation.

### Phase G — Documentation & Runbook
Update / create:
- `docs/VOICE.md`: New WebRTC architecture (sequence: Capture → PeerConn → Processor → STT → LLM → TTS → Remote Track).
- `docs/API.md`: Signaling endpoints (request/response examples + error codes).
- `docs/CONFIGURATION.md`: All new env keys.
- `docs/TROUBLESHOOTING.md`: ICE failure scenarios, `chrome://webrtc-internals`, when to introduce TURN.
- `docs/LOGGING.md`: New log categories and sample debug session.
- Add `docs/RUNBOOK_WEBRTC.md`: Operations tasks (restart, metric inspection, enabling/disabling flag).

---
## 7. Acceptance Criteria (Go/No-Go)
1. WebRTC feature flag ON: Browser session achieves 3 consecutive end-to-end voice turns with p90 < 6000 ms round-trip.
2. Flag OFF: Legacy WebSocket path unaffected and functional.
3. 10-second server and client utterance limit enforced (attempt >10.5 s truncated and logged).
4. `/no_think ` prefix present in all LLM prompts (verified via debug log sampling).
5. Metrics visible for all latency categories & VAD counters.
6. Proper cleanup: Peers closed within 5 s of disconnect; no lingering aiortc tasks (verified via logs/monitor).
7. Documentation updated (VOICE, API, CONFIGURATION, TROUBLESHOOTING, RUNBOOK_WEBRTC) and peer-reviewed.
8. No regressions in existing WebSocket streaming tests (`tests/streaming/`).

---
## 8. Testing Strategy
| Level | Tooling | Focus |
|-------|---------|-------|
| Unit | pytest | Signaling endpoints, buffer logic, VAD state transitions |
| Component | aiortc loopback | AudioFrame → PCM conversion & utterance boundary enforcement |
| Integration | Synthetic frame injector | Full pipeline timings & metrics emission |
| Manual | Browser (Chrome/Safari) | Real mic flow, latency measurement, UX correctness |
| Fallback | Existing WebSocket tests | Non-regression validation |

Synthetic Frame Test: Generate 3 s voiced + 1 s silence cycle repeated; assert single utterance segmentation per speech block.

---
## 9. Risk & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| ICE negotiation failure (NAT edge cases) | No audio | Add TURN later; provide fallback WebSocket button |
| aiortc CPU overhead | Latency spike | Profile early with 3 concurrent sessions; pin versions |
| VAD false positives in noisy Arabic environments | Extra STT calls | Tune thresholds; introduce noise floor adaptation |
| Long-running peer leaks | Memory growth | GC task + idle timeout + explicit DELETE endpoint |
| Regression in WebSocket path | Service disruption | Feature flag isolation & regression test suite |
| STT latency drift > 2s | Round-trip breach | Add preloaded model warmup; monitor GPU utilization |

---
## 10. Implementation Order (Granular Checklist)
A.1 Add deps  
A.2 Config keys  
A.3 Feature flag wiring  
B.1 Signaling endpoints (offer/ice)  
B.2 Connection pool  
B.3 Session manager  
C.1 Audio processor  
C.2 VAD service  
C.3 Buffer manager  
C.4 Connect to STT → LLM → TTS  
D.1 Frontend client (basic handshake)  
D.2 Remote audio playback  
D.3 10 s client limit  
D.4 VAD UI reuse  
E.1 Metrics emission  
E.2 Probe script  
E.3 E2E synthetic test  
F.1 Nginx path update  
F.2 Systemd env update  
F.3 Staged rollout  
G.1 Docs update  
G.2 Runbook  
G.3 Final acceptance review  

---
## 11. Post-MVP Roadmap (Not in current scope)
- TURN integration (coturn deployment & credential provisioning)
- Mid-utterance incremental STT (stream partials)
- Barge-in interruption & graceful cancel of TTS playback
- Adaptive bitrate & Opus DTX tuning
- Multi-language on-the-fly switching per utterance
- GPU offloading for real-time enhancement models (noise suppression upgrades)

---
## 12. Quick Start (After Merge)
1. Enable feature: set `VOICE_WEBRTC_FEATURE_ENABLED=true` in API environment.
2. Deploy updated backend & reload nginx.
3. Open `dev.gmai.sa` → toggle WebRTC mode → speak → observe logs & metrics.
4. Validate metrics: STT, LLM, TTS latencies + round-trip.

---
Prepared by: GitHub Copilot (Automated Plan Generation)
