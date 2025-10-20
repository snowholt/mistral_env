# WebRTC MVP Migration Plan

## 1. Objective & Scope
- Deliver a browser-based WebRTC voice-to-voice flow that replaces the current WebSocket MVP while keeping WebSocket as a fallback.
- Preserve the existing voice pipeline (Whisper STT → Qwen LLM → Edge TTS) with the new `"/no_think "` prefix enforcement to accelerate responses.
- Enforce a configurable 10-second utterance limit on both client and server sides, exposing the knob via `.env` files for the frontend and backend.
- Improve call quality by switching to WebRTC’s built-in AGC, Noise Suppression, and Echo Cancellation, validated end-to-end via dev.gmai.sa.

## 2. Reference Inputs
- Internal report: `reports/webrtc_migration_analysis.md` (architecture & benefits).
- Technical analysis: `reports/webrtc_migration_technical_analysis.md` (reusability split, file inventory).
- Session log: `reports/WEBRTC_plan_temp_session.md` (decision trail & clarifications).
- Open source inspiration: [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
  - Dual VAD pattern: [`RealtimeSTT/audio_recorder.py`](https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/audio_recorder.py#L150-L320) (`_is_voice_active`, `_is_silero_speech`, `_set_state`).
  - Buffering strategy: [`RealtimeSTT/audio_recorder.py`](https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/audio_recorder.py#L420-L515) (`feed_audio`, `audio_buffer`, `frames`).
  - Multiprocessing orchestration: [`RealtimeSTT/realtime.py`](https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/realtime.py#L45-L180) (`RealtimeTranscriber._realtime_worker`).

## 3. Guiding Constraints & NFRs
- MVP simplicity only: single-speaker, no diarization, no multi-party.
- Maintain session context parity with current WebSocket experience by introducing `WebRTCSessionManager` that wraps existing `VoiceSessionManager` primitives.
- Latency SLOs (90th percentile): round-trip ≤ 6s, STT ≤ 2s, LLM ≤ 3s, TTS ≤ 1s, connection success ≥ 99%.
- Interrupt handling (MVP): ignore new input while TTS response is still playing; document upgrade path for barge-in later.
- Security: TLS-only signaling, STUN-only for MVP (Google public STUN), TURN readiness noted for production.
- Testing must be browser-driven through `dev.gmai.sa` because hardware access to the server is unavailable.

## 4. Work Breakdown Structure

### Phase A — Preparation & Environment
1. Backend dependencies
   - Add `aiortc`, `pyee`, `webrtcvad` (server-side confirmation) to `backend/requirements.txt` and lockfiles if any.
   - Document system packages (e.g., `libopus`, `libssl`) in `backend/README.md` or `docs/DEPLOYMENT.md` for aiortc.
   - Verify GPU-related dependencies remain untouched.
2. Configuration scaffolding
   - Introduce `VOICE_WEBRTC_MAX_UTTERANCE_SEC` (backend) in `config/defaults.json`, `config/config.yaml`, and `.env` templates.
   - Add `REACT_APP_VOICE_WEBRTC_MAX_UTTERANCE_SEC` (frontend) in `frontend/.env.example` and usage docs.
   - Update `docs/CONFIGURATION.md` with new keys, default 10 seconds, override instructions.
3. Logging/monitoring baseline
   - Extend `docs/LOGGING.md` with WebRTC log categories and `aiortc` debug toggles.
   - Prepare metrics namespace (`webrtc_voice_*`) in performance monitor service for later phases.

### Phase B — Backend Signaling & Session Lifecycle
1. FastAPI endpoints
   - Create `backend/src/beautyai_inference/api/endpoints/webrtc_voice.py` (marked *Created for WebRTC*) exposing:
     - `POST /api/v1/webrtc/voice/offer` for SDP offer handling, returning SDP answer + `peer_id`.
     - `POST /api/v1/webrtc/voice/ice` for client ICE candidates.
     - Optional `DELETE /.../{peer_id}` for cleanup.
   - Reuse `fastapi.APIRouter` registration in `backend/src/beautyai_inference/api/router.py` alongside existing WebSocket routers.
2. Connection management
   - Introduce `backend/src/beautyai_inference/core/webrtc_connection_pool.py` to track `RTCPeerConnection` instances, state, timestamps, ICE info.
   - Ensure graceful shutdown via asyncio tasks and `finally` to close peers.
3. Session tracking
   - Create `backend/src/beautyai_inference/core/webrtc_session_manager.py` (wraps `VoiceSessionManager`, adds WebRTC-specific metadata such as peer ICE state, client capabilities, language).
   - Provide helper to retrieve or bootstrap sessions keyed by `peer_id` (mirrors WebSocket `connection_id`).
4. Configurable signaling routes
   - Use `VOICE_SIGNALING_PATH=/api/v1/webrtc/voice` env var to simplify Nginx wiring.
5. Tests
   - Add unit tests in `tests/openai_validation/test_webrtc_signaling.py` employing `asyncio` test client for offer/answer round-trip, invalid SDP rejection, and cleanup semantics.

### Phase C — Audio Track Processing, VAD & Buffer Service
1. Media pipeline module
   - Add `backend/src/beautyai_inference/services/voice/webrtc_audio_processor.py` handling `MediaStreamTrack` frames (created for WebRTC).
   - Responsibilities: convert `AudioFrame` → PCM (16 kHz mono), enforce server-side 10 s cap, push to voice pipeline.
2. Dual VAD module
   - Add `backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py` using browser hints + server verification.
   - Implement WebRTC VAD (`webrtcvad.Vad`) fast-path and Silero confirmation referencing RealtimeSTT functions (`_is_voice_active`, `_is_silero_speech`).
   - Provide metrics hooks (speech detected/silence ratio, false positives).
3. Buffer manager
   - Create `backend/src/beautyai_inference/core/webrtc_buffer_manager.py` (or extend existing `buffer_integration.py` with WebRTC subclass) adapting RealtimeSTT buffering strategy (`audio_buffer`, `frames`, 300 ms pre-roll, post-roll) to RTP frame timing.
4. Service wiring
   - Extend `SimpleVoiceService` or a thin adapter to inject `/no_think ` prefix at the LLM call boundary (before `generate_chat_response`).
   - Parameterize detection thresholds per language (Arabic vs English) using `.env` values; default to 0.45 and 0.5.
5. Tests
   - Add targeted unit tests under `tests/openai_validation/test_webrtc_audio_processor.py` to verify frame conversion, utterance limit enforcement, and VAD gating logic using canned PCM fixtures from `voice_tests/input_test_questions/pcm`.

### Phase D — Frontend WebRTC Client & UI Integration
1. Client implementation
   - Create `frontend/src/static/js/webrtcVoiceClient.js` (parallel to `simpleVoiceClient.js`, marked *Created for WebRTC*).
   - Responsibilities: manage `RTCPeerConnection`, negotiate SDP/ICE, add local tracks, receive remote TTS stream, enforce 10 s capture limit via timer.
   - Reuse `improvedVAD.js` for audio-level monitoring; dispatch VAD state across data channel (optional) for diagnostics.
2. UI wiring
   - Provide environment toggle in frontend config to select WebRTC vs WebSocket (feature flag in `frontend/src/static/js/config.js`).
   - Update HTML templates to load new client when `WEBRTC` mode is active; keep WebSocket buttons for fallback.
3. Audio pipeline
   - Ensure outgoing audio uses `getUserMedia` constraints enabling `echoCancellation`, `noiseSuppression`, `autoGainControl` (true) with sampleRate 48000.
   - Convert remote audio stream into a hidden `<audio>` element for playback; maintain existing queue logic.
4. `/no_think` guidance
   - Update frontend request composer to document that backend prefixes automatically; no client change required besides ensuring transcripts contain raw user speech.
5. Tests
   - Add Cypress (or Playwright) smoke test in `tests/streaming` (if available) to cover signaling handshake; otherwise document manual QA script.
   - Extend manual QA checklist in `docs/STREAMING_DEBUG_UI_ENHANCEMENT_VALIDATION.md` with WebRTC steps.

### Phase E — Integration, Observability & Tooling
1. End-to-end harness
   - Create `tests/streaming/test_webrtc_end_to_end.py` that feeds pre-recorded PCM through the audio processor to simulate track flow.
   - Provide script `tools/webrtc_signaling_probe.py` for local handshake testing.
2. Metrics & logging
   - Update `performance_monitor.py` to emit `webrtc_voice_latency_ms`, `webrtc_voice_vad_false_positive_count`, etc.
   - Extend `tools/service_analyzer.py` to include WebRTC stats and status.
3. Debugging aids
   - Add optional `?debug=1` query flag to expose WebRTC stats (ICE state, bitrate) in the frontend overlay.
   - Document usage of `chrome://webrtc-internals` in `docs/TROUBLESHOOTING.md`.

### Phase F — Deployment & Infrastructure
1. Nginx
   - Add signaling route `location /api/v1/webrtc/voice` forwarding to FastAPI (no WebSocket upgrade needed), ensure CORS mirrors existing API rules.
   - Confirm existing SSL certs cover the new path; reload nginx.
2. Systemd services
   - Update `beautyai-api.service` ExecStart environment to set `VOICE_WEBRTC_MAX_UTTERANCE_SEC` and enable aiortc logs.
   - Ensure `Restart=on-failure` remains for stability.
3. TURN readiness (post-MVP note)
   - Document steps for adding coturn (package install, `turnserver.conf`, credentials) in `docs/DEPLOYMENT.md`; mark as future enhancement.
4. Monitoring & alerting
   - Configure log ingestion to capture aiortc warnings; add synthetic check hitting `/webrtc/voice/health` once implemented.

### Phase G — Documentation & Enablement
1. Update `docs/VOICE.md` with WebRTC architecture diagrams (ASCII ok), call out fallback behavior.
2. Refresh `docs/API.md` with signaling endpoint descriptions and payload examples (offer/answer, ICE).
3. Provide operator runbook in `docs/TROUBLESHOOTING.md` detailing:
   - Common negotiation errors.
   - Using Google STUN, when to migrate to TURN.
   - Browser compatibility expectations.
4. Draft knowledge-transfer note for customer support referencing `/no_think` effect and 10 s limit messaging.

## 5. Acceptance Criteria (Go/No-Go)
- ✅ `reports/webrtc_migration_technical_analysis.md` annotated with references to completed work (footnotes or linked appendix).
- ✅ KoljaB/RealtimeSTT repo cloned locally, dual VAD path exercised, and insights documented in `copilot_resources/reports/webrtc_reuse_notes.md` (new file).
- ✅ Separate VAD + buffer services created under `backend/src/beautyai_inference/services/voice/vad/` and `.../core/` with unit tests achieving ≥ 90% branch coverage.
- ✅ Browser demo (dev.gmai.sa) shows speech → transcript → LLM (`/no_think` enforced) → Edge TTS playback with < 6 s round-trip across three consecutive trials.
- ✅ Documentation updated (API, Voice, Troubleshooting, Config) and reviewed.
- ✅ Performance metrics captured, stored, and compared against legacy WebSocket baseline; regression risks documented if metrics exceed targets.

## 6. Testing Strategy
- **Unit**: backend signal handling, audio frame conversion, VAD gating (pytest, `tests/openai_validation/`).
- **Integration**: simulate aiortc track via `RTCPeerConnection` in tests, verifying session lifecycle, conversation state persistence.
- **Manual QA**: remote browser sessions on Chrome + Safari performing Arabic and English utterances, monitoring `chrome://webrtc-internals`.
- **Load smoke**: limited concurrent session test (3-5 clients) to evaluate CPU impact.
- **Fallback**: verify WebSocket path remains functional when WebRTC feature flag disabled.

## 7. Timeline & Ownership (Suggested)
- Week 1: Phase A + initial signaling handshake (Phase B items 1-2).
- Week 2: Complete backend media pipeline (Phase C) and `/no_think` adjustments.
- Week 3: Frontend client integration (Phase D) + unit/integration tests (Phase E.1).
- Week 4: Deployment preparations, documentation, and acceptance testing (Phases F & G).
- Continuous: risk tracking, metric instrumentation, stakeholder demos.

## 8. Risk Register
- **ICE negotiation failures** (medium): Mitigation—provide detailed logs, fallback to WebSocket, document STUN/TURN setup.
- **Server CPU spikes from aiortc** (medium): Mitigation—profile with 3 concurrent sessions, consider native libs if needed.
- **Latency regressions > 6 s** (medium): Mitigation—instrument pipeline timings, adjust `/no_think` prefix logic, cache TTS voices.
- **Browser compatibility gaps** (low): Mitigation—feature detection, fallback messaging, maintain WebSocket mode.
- **Security misconfiguration** (low): Mitigation—reuse TLS endpoints, restrict signaling origin, sanitize SDP payloads.

## 9. Deliverables Checklist
- [ ] Backend: new WebRTC endpoints, connection pool, session manager, audio processor, VAD buffer modules.
- [ ] Frontend: WebRTC client, env toggles, updated UI.
- [ ] Config: env keys, docs, runbook.
- [ ] Tests: unit + integration suites, manual QA script.
- [ ] Metrics: telemetry in performance monitor, dashboard entry.
- [ ] Deployment: nginx/systemd updates, feature flag strategy.

---
Prepared by GitHub Copilot • October 14, 2025
