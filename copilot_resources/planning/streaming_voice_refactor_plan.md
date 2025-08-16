# Streaming Voice Refactor Plan

## 0. Overview
Hands-free, natural, low-latency bilingual (Arabic-first) voice conversation using streaming PCM + incremental Whisper decoding (large-v3-turbo baseline, with option to evaluate smaller models). Remove brittle Opus chunk + heuristic VAD path for new endpoint while keeping legacy as fallback (user chose hard-fail on new endpoint, but we will keep old code untouched during transition).

Target latency budget (from `voice_models_registry.json`):
- STT: <= 800 ms after end-of-speech
- TTS: <= 500 ms
- Total: <= 1500 ms
User acceptable end-of-utterance latency: < 1200 ms.

## 1. Model Comparison (Current vs Alternatives)
| Aspect | Current (whisper-large-v3-turbo) | Whisper Medium | Whisper Small | Quantized (int8 / faster-whisper) |
|--------|----------------------------------|----------------|---------------|----------------------------------|
| Params | ~809M (pruned 4 decoder layers) | 769M | 244M | Same arch, smaller weights | 
| Arabic Accuracy | High (near large-v2) | High | Moderate (WER higher) | Slight drop (1–3% WER) | 
| Speed (FP16 4090) | Fast (pruned) | Slightly slower | Much faster | Faster memory load | 
| VRAM (FP16) | ~3.2–3.6 GB | ~2.9–3.1 GB | ~1.2–1.4 GB | Reduced (int8) ~2.0–2.5 GB | 
| Real-time streaming suitability | Good | Good | Good (lower accuracy) | Good | 
| Arabic robustness (diacritics, code-switch) | Strong | Strong | Lower | Slight accuracy reduction | 

Conclusion: Stay with large-v3-turbo for accuracy; evaluate optional int8 quantization only if GPU contention arises.

## 2. Proposed Architecture
```
Browser (AudioWorklet) --PCM Int16 frames--> WS /api/v1/ws/streaming-voice
    └─ SessionState: ring_buffer, decoder_task, last_decode_tokens
          ├─ Decode loop (async interval / event-driven)
          │    ├─ Collect new frames -> assemble feature window -> run whisper
          │    ├─ Produce partial tokens -> diff -> emit partial_transcript
          │    ├─ Update stability metrics (no change duration, energy tail)
          │    └─ If endpoint criteria met -> final_transcript event
          └─ On final_transcript -> conversation pipeline -> TTS -> send audio -> playback complete -> auto re-arm
```

## 3. Endpoint Detection Strategy
- Adaptive noise floor: median RMS first 400 ms * factor (1.8) + margin
- Speech active when RMS > threshold for 120 ms cumulative
- Candidate end when (RMS < threshold for 500–700 ms) AND (no token change ≥ 600 ms) OR (utterance duration >= 12 s)
- Force finalize if provisional tokens exist and silence tail hits 1.2 s.

## 4. Events Schema (JSON over WS, plus binary frames)
```
Client → Server:
  {"type":"start", "language":"ar", "sample_rate":16000}
  Binary PCM Int16 frames (20–40 ms each)
  {"type":"stop"} (manual abort if needed)

Server → Client:
  {"type":"ready", "session_id": "..."}
  {"type":"partial_transcript", "text":"مرحبا", "stable":false, "avg_rms":0.02, "ts": 123.45}
  {"type":"partial_transcript", "text":"مرحبا كيف", "stable":false}
  {"type":"final_transcript", "text":"مرحبا كيف حالك", "utterance_id":"u1", "duration_ms": 2100}
  {"type":"tts_start", "utterance_id":"u1"}
  {"type":"tts_audio", "utterance_id":"u1", "url":"/cache/tts/u1.wav"} (initial impl)
  {"type":"tts_complete", "utterance_id":"u1"}
  {"type":"error", "message":"..."}
```

## 5. File / Module Plan
Backend (under `backend/src/beautyai_inference`):
- `api/endpoints/streaming_voice.py` (new) — WebSocket endpoint.
- `services/voice/streaming/streaming_session.py` — Session state dataclass (ring buffer, timestamps).
- `services/voice/streaming/ring_buffer.py` — Fixed-size PCM circular buffer.
- `services/voice/streaming/decoder_loop.py` — Orchestrates incremental decode (pluggable backend: faster-whisper transformer pipeline vs future alt engines).
- `services/voice/streaming/endpointing.py` — Endpoint detection functions.
- `services/voice/streaming/events.py` — Event serialization helpers.
- `services/voice/streaming/audio_worklet_spec.md` — Documentation for frame format.
- Reuse existing ModelManager to fetch whisper model (extend if needed with `get_streaming_whisper()` that ensures singleton load & caching of processor + model).
- Extend `voice_models_registry.json` only if adding quantized variants (phase later).

Frontend (`frontend/src/static/js`):
- `streaming-audio-worklet.js` — Worklet processor capturing audio, downsampling.
- `streamingVoiceClient.js` — Handles WS connection, partial transcript UI, mic bar.
- Integrate into existing `chat-interface.js` via feature flag `USE_STREAMING_VOICE`.
- UI components: mic level bar (horizontal), live transcript line.

Testing / Scripts:
- `backend/unitTests_scripts/python_scripts/test_ring_buffer.py`
- `backend/unitTests_scripts/python_scripts/test_endpointing.py`
- `backend/unitTests_scripts/python_scripts/test_decoder_loop_mock.py`
- Manual integration test script: `vad_testting_temp/streaming_integration_test.py` (receives events, logs timings).

Config / Flags:
- Env var: `VOICE_STREAMING_ENABLED=1`
- Optional: `VOICE_STREAMING_MODEL=whisper-large-v3-turbo`

## 6. Phase Breakdown with Technical Tasks
### Phase 1: Backend Scaffolding
Tasks:
- Create endpoint skeleton: accept WS, parse start message, send ready.
- Implement session registry (dict[connection_id] -> SessionState).
- Placeholder decode loop (no model) generating mock partials every 800 ms.
Verification:
- Connect via WebSocket manually, observe `ready` and mock partials.

### Phase 1.5: Minimal Client Audio Preprocessor
Purpose:
- Stabilize adaptive thresholding & partial token stability by reducing low-frequency rumble and normalizing loudness.
- Prevent spurious silence/guidance logic when streaming mode is active.

Scope (client-side AudioWorklet only — zero algorithmic latency):
1. High-pass filter @ ~25 Hz (1st order) to remove DC/infrasonic noise.
2. Rolling RMS (400 ms window) for level estimation.
3. Soft noise gate: attenuate (not mute) frames where frame_rms < baseline_rms * 1.45 (attenuation factor 0.3).
4. RMS normalization toward target loudness -23 dBFS with +/-6 dB clamp.
5. Optional peak limiter threshold 0.95 full scale (simple hard clip guard).
6. Debug metrics emission (dev mode): every 200 ms post {avg_rms, gain, gate_ratio} if `window.DEBUG_PREPROC_STATS` true.

Implementation Tasks:
- Add logic inside `streaming-audio-worklet.js` (or separate helper) before downsampling & Int16 conversion.
- Maintain causal processing (no look-ahead buffer). Ensure no extra frame buffering is introduced.
- Expose a simple configuration object in worklet parameters to allow tuning without rebuild.
- Provide a feature toggle flag `ENABLE_MINI_PREPROC` (default true in streaming mode).

Verification:
- Log difference between raw RMS and post-gate RMS (expect >2.5x speech/silence ratio on typical mic).
- Confirm no added latency (timestamps between capture and first frame send unchanged vs disabled state).
- Speak soft Arabic consonants (ح, خ) to verify they are still transcribed (no over-gating).

Success Criteria:
- No regression in transcription accuracy on sample Arabic phrases.
- Endpoint timing variability reduced (stddev of end detection over 5 trials decreases).
- CPU overhead <1% main thread (measured via Performance panel) and negligible in Worklet.

Future (Deferred):
- RNNoise / spectral gating only if real-world noise logs show persistent low SNR degradation.

Risks & Mitigations:
- Over attenuation of soft speech -> Keep attenuation partial (0.3) & threshold factor adjustable.
- Loudness pumping -> Slow attack (150 ms) / release (400 ms) smoothing in RMS normalization.

Rollback Plan:
- Single flag disable; frames bypass untouched if issues detected.

### Phase 2: Ring Buffer + Audio Frame Handling
Tasks:
- Implement Int16 frame ingestion; track sample count, compute rolling RMS.
- Unit test ring buffer write/read, overflow wrap.
Verification:
- Test file; log RMS values with synthetic sine + silence.

### Phase 3: Endpointing Module (Mock Tokens)
Tasks:
- Implement adaptive threshold + timers using mock token generator.
- Unit test endpoint detection state machine with synthetic sequences.
Verification:
- Controlled sequence triggers endpoint at expected frame index.

### Phase 4: Integrate Whisper Large-v3-Turbo (Streaming Approximation)
Tasks:
- Load model once (fp16, flash-attn if available).
- Implement incremental window decode: last N seconds (e.g. 8 s) of buffer -> decode -> diff tokens.
- Cache previous tokens to determine stable vs new tail.
Performance Considerations:
- Use `torch.inference_mode()` and `no_grad()`.
- Minimize re-tokenization overhead.
Verification:
- Real audio sample: confirm partial updates; measure decode cycle time.

### Phase 5: Real Endpoint Finalization
Tasks:
- Integrate endpoint criteria; on finalize issue final_transcript; clear token context appropriately.
- Prevent double finalization within cooldown (300 ms guard).
Verification:
- Speak short Arabic phrase; ensure single final_transcript emitted.

### Phase 6: LLM + TTS Integration
Tasks:
- New function `process_transcribed_text(text, language, session_id)` calling existing conversation service but skipping STT path.
- Ensure TTS saved to WAV per current audio_config.
- Emit tts_start/tts_audio/tts_complete events.
Verification:
- End-to-end: speak phrase → transcript → AI response → playback link.

### Phase 7: Frontend Streaming Client
Tasks:
- AudioWorklet: Downsample from 48 kHz to 16 kHz (linear interpolation or decimator) chunk size 20 ms.
- WebSocket binary frame send (ArrayBuffer Int16).
- Display mic bar (RMS scaled 0–1).
- Show partial transcript (update progressively; stable portion styled faint).
- Handle final_transcript → append to chat history.
Verification:
- Visual mic bar responds to voice; partial text appears while speaking.

### Phase 8: Auto Re-Arm + State Machine Cleanup
Tasks:
- After `tts_complete`, re-open mic automatically (unless user clicked stop).
- Remove old guidance overlay interactions when streaming mode active.
Verification:
- Multi-turn conversation without manual clicks.

### Phase 9: Performance & Arabic Accuracy Tuning
Tasks:
- Measure average decode latency (ms) and end-of-speech to final_transcript gap.
- Adjust window length, decode interval (e.g. 320 ms or 480 ms), beam size.
- Optionally apply `temperature=0` & `condition_on_prev_tokens=False`.
Verification:
- Logs show STT latency <= target 800 ms for typical utterances.

### Phase 10: Logging & Metrics
Tasks:
- Structured logs: event type, durations, tokens/sec, endpoint_reason.
- Error handling for model overload (backpressure: drop frame or skip cycle).
Verification:
- Log inspection shows clean lifecycle per utterance.

### Phase 11: Unit & Integration Tests Completion
Tasks:
- Finalize tests for ring buffer, endpointing, decode diff logic.
- Add regression test ensuring single finalization per utterance.
Verification:
- All tests pass in CI (future integration).

### Phase 12: Documentation & Cleanup
Tasks:
- Add README snippet to `docs/VOICE.md` describing streaming mode.
- Mark legacy VAD path as deprecated in comments (not removed yet).

### Phase 13 (Optional Roadmap): Streaming TTS & Barge-In
- Replace `tts_audio` URL event with chunked binary frames.
- Detect user speech energy during TTS; send interrupt to server.

## 7. Test Strategy
Unit Tests:
- Ring buffer (wrap-around, read windows).
- Endpoint detection (scenarios: fast phrase, long silence, max duration cap).
- Token diff stability classification.

Integration Tests:
- Simulated PCM injection script feeding Arabic sample; assert partial then final.
- End-to-end WS test producing TTS event chain.

Manual QA Checklist:
- Arabic short phrase recognition accuracy.
- Rapid back-to-back utterances (latency stable).
- Silence (no phantom transcripts).
- Long utterance truncated at 12 s.
- GPU memory stable over 10+ turns.

## 8. Noise Cancellation (Client-Side) Consideration
Approaches:
- Browser constraints: `echoCancellation`, `noiseSuppression`, `autoGainControl` (already basic).
- Advanced: WebRTC RNNoise / spectral gating in AudioWorklet (adds CPU + latency 5–15 ms); potential risk of removing weak consonants in Arabic (خ, ح) harming accuracy.
Recommendation:
- Start with default browser constraints ON.
- Defer custom RNNoise until baseline streaming stable; only add if background noise severely harms accuracy.

## 9. Dependencies
- Potential: `faster_whisper` (if not already) or remain with Transformers pipeline initially.
- Optional: `webrtcvad` (CPU fallback / hybrid) — low overhead.
- Ensure versions logged.

## 10. Security & Resource Management
- Per-session max buffered audio seconds (e.g. 40) to prevent memory abuse.
- Drop connection if no frames received for > 30 s.
- Validate frame byte size (multiple of 2, <= expected_max_frame_bytes).

## 11. Acceptance Criteria
- Hands-free multi-turn without guidance loops.
- Partial transcripts visible within <600 ms of speaking onset.
- Final transcript to TTS start < 1200 ms typical.
- Arabic accuracy comparable to current batch mode (spot-check phrases).
- Mic level bar reflects input; no freezes over 10-minute session.

## 12. Open Questions (If Any Before Coding)
None blocking; proceeding uses large-v3-turbo. Quantized evaluation can be added later as Phase 14.

---
Prepared for implementation. Await confirmation to begin Phase 1 scaffold.
