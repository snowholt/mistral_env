
<!-- =================================================================== -->
# Streaming Voice Refactor – Execution Template
<!-- =================================================================== -->

This file guides each implementation phase of the Streaming Voice Refactor (see `planning/streaming_voice_refactor_plan.md`). It enforces: clarity before code, minimal necessary changes, robust tests, and controlled knowledge capture.

## 1. Guiding Principles
1. Hands-free, natural, low latency Arabic-first experience.
2. Minimize code surface: introduce modules ONLY if clearly cohesive & reusable.
3. Single Responsibility: each new file has a narrow, documented purpose.
4. Feature Flag Driven: legacy path remains untouched; streaming path isolated.
5. Measurement Before Optimization: add lightweight timing logs first, optimize only when metrics show need.
6. Ask Clarifying Questions ONLY when an ambiguity would risk rework.
7. Reuse existing model management (no duplicate Whisper loads).

## 2. Phase Checklist (Top-Level)
Refer to detailed plan for content. Mark off here as phases complete.
```
[x] Phase 1   Backend Scaffolding
[x] Phase 1.5 Minimal Client Preprocessor
[x] Phase 2   Ring Buffer + Frame Handling
[x] Phase 3   Endpointing (Mock Tokens)
[x] Phase 4   Whisper Integration (Incremental)
[x] Phase 5   Endpoint Finalization Logic
[x] Phase 6   LLM + TTS Integration Path
[x] Phase 7   Frontend Streaming Client
[x] Phase 8   Auto Re-Arm & Cleanup
[x] Phase 9   Performance & Accuracy Tuning
[x] Phase 10  Logging & Metrics Hardening
[x] Phase 11  Unit & Integration Tests Finalization
[x] Phase 12  Documentation & Cleanup
[ ] Phase 13  (Optional) Streaming TTS & Barge-In Prep
```

## 3. Reporting Structure
Create directory: `planning/reports/streaming_voice/`
Each phase adds a markdown file:
`phase_<n>_<short>.md` including:
- Summary (1–2 sentences)
- Changes (bullet list of files + purpose)
- Metrics (if applicable: latency, tokens/sec)
- Open Issues / Follow-ups
- Decision Log entries appended centrally (see below)

Central cumulative file: `planning/reports/streaming_voice/progress.md` auto-appended per phase with ✅/⏳ icons.

## 4. Decision Log
File: `planning/reports/streaming_voice/decisions.md`
Entry template:
```
### <ID> <Short Title>
Context:
Decision:
Alternatives Considered:
Consequences:
Date:
```

Use for: model size shifts, endpoint thresholds, buffer lengths, quantization adoption.

## 5. Phase Execution Template
For EACH phase before coding:
```
### Phase N Execution
Goal:
Deliverables:
Non-Goals:
Key Risks & Mitigations:
Metrics to Capture (if any):
Clarifying Questions (ONLY if blocking):
Implementation Steps (ordered, atomic):
Test Plan (unit + integration):
Rollback Strategy:
```

After coding (before moving on): append a summary under `planning/reports/streaming_voice/phase_N_*.md` and update progress list.

## 6. Minimal Code Heuristics
- Prefer functions over classes until state truly needed.
- 1 ring buffer implementation only; no ad-hoc frame arrays in other modules.
- Endpointing logic pure & testable (no direct logging inside core detection function).
- Avoid early abstractions for multi-engine decode until a second engine is actually requested.
- Use dataclasses for simple state containers (SessionState, EndpointState) with explicit type hints.

## 7. Clarifying Question Protocol
Only ask when:
1. Specification conflict (e.g., 12 s cap vs. different constant elsewhere).
2. Missing input needed to choose irreversible structure (e.g., binary TTS streaming format).
3. Ambiguous event naming or ordering that affects clients.

Format questions in a block:
```
## Clarifying Questions (Phase N)
1. ...?
2. ...?
```
Then halt implementation awaiting answer.

## 8. Metrics & Telemetry (Initial Fields)
Log (structured JSON where feasible):
- `event`: decode_cycle|partial|final|endpoint_trigger|tts_start|tts_complete
- `session_id`
- `frames_ingested`
- `buffer_sec`
- `decode_ms`
- `tokens_emitted`
- `no_token_change_ms`
- `rms_speech` / `rms_silence`
- `endpoint_reason` (silence|no_change|max_duration)

## 9. Test Layering
Unit (fast): ring_buffer, endpointing, token diff.
Integration (moderate): simulated PCM → final transcript event.
System (manual): real mic loop multi-turn Arabic conversation.

## 10. Commit Message Format
First line ≤ 50 chars: `stream-voice: phase N scaffold`
Body (optional): bullet key changes, decisions IDs referenced.

## 11. Rollback Strategy
Feature flag gating ensures immediate disable by setting `VOICE_STREAMING_ENABLED=0`.
Critical failure path: fallback to legacy endpoint while retaining logs.

## 12. Phase 1 Pre-Fill (Example)
```
### Phase 1 Execution
Goal: Establish WS endpoint skeleton + session registry and mock decode loop.
Deliverables: streaming_voice.py, streaming_session.py (dataclass skeleton), events helpers, mock loop.
Non-Goals: Real decoding, endpoint detection, model loading.
Key Risks: Leaking tasks on disconnect → add on_close cleanup registry.
Metrics: cycle interval jitter (target < ±50 ms around planned period).
Implementation Steps:
 1. Create endpoint file with accept & protocol validation.
 2. Generate connection/session IDs, send `ready`.
 3. Schedule mock decode loop using asyncio.create_task.
 4. Emit synthetic partials until stop/disconnect.
Test Plan:
 - Connect via WS → expect `ready`.
 - Receive ≥1 partial within 1 s.
Rollback: Delete new files; flag off.
```

## 13. Phase 1.5 Pre-Fill (Example)
```
### Phase 1.5 Execution
Goal: Add minimal preprocessor (high-pass, gate, rms norm) in Worklet.
Deliverables: Updated streaming-audio-worklet.js (or new module), toggle flag.
Non-Goals: RNNoise, advanced denoise.
Risks: Over-gate soft Arabic fricatives → conservative threshold & attenuation.
Metrics: Speech/silence RMS ratio improvement.
Implementation Steps:
 1. Insert high-pass filter state (prev sample) per frame.
 2. Compute frame RMS & update rolling window.
 3. Apply gate attenuation & RMS normalization.
 4. Export debug metrics if debug flag.
Test Plan: Console log RMS values; verify no latency regression.
Rollback: Toggle disable; bypass processing.
```

---
Ready for Phase 1 execution once this template is accepted.
\n+### Phase 8 Execution (Completed)
Goal: Implement automatic re-arm after each assistant TTS response and robust session/task cleanup.
Deliverables: Updated frontend `streamingVoiceClient.js` with `autoRearm`, `suspend()`, `resume()`, backend ws disconnect cancellation.
Non-Goals: Performance tuning, adaptive silence parameters.
Key Risks & Mitigations:
 - Multiple concurrent WS connections → guard by checking OPEN state before reconnect.
 - Orphaned LLM/TTS tasks → ensure explicit cancel on disconnect.
Metrics: Qualitative latency unchanged (<1 frame difference); no memory growth across restarts.
Implementation Steps:
 1. Add flags (autoRearm, _suspended) + lifecycle methods.
 2. Fire synthetic `auto_rearm` event upon `tts_complete`.
 3. Backend: cancel llm_tts_task in disconnect path.
 4. Update progress + phase report.
Test Plan:
 - Open WS, speak, confirm TTS plays and client remains ready (no manual reload).
 - Call suspend(); verify capture ceases & WS closes.
 - Call resume(); verify new WS opens and audio resumes.
Rollback Strategy: Set autoRearm false or revert single JS file + backend minor block.