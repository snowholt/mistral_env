# Phase 3 – Endpointing (Mock Tokens)

## Summary
Introduced adaptive RMS + token stability endpoint detector (`endpointing.py`) with a pure functional state machine. While still using mock token emission, the detector now models the same transitions planned for real Whisper incremental decoding (start, active, final) enabling early validation of timing and UI event handling.

## Changes
- Added `backend/src/beautyai_inference/services/voice/streaming/endpointing.py` containing:
  - `EndpointConfig` (tunable thresholds)
  - `EndpointState` (mutable detection state)
  - `EndpointEvent` (start/final events)
  - `update_endpoint()` (advances state per audio frame + optional tokens)
- Extended ring buffer earlier (Phase 2) already provides RMS support; integration with decode loop deferred to Phase 4.
- Progress file updated to mark Phase 2 in progress; Phase 3 work initiated (logic in place, wiring to WS events will occur alongside Phase 4 real tokens).

## Detector Logic
1. Calibration window (default 400 ms) forms baseline RMS (exp avg).
2. Speech onset when cumulative voiced ≥ 120 ms (RMS > baseline * 1.8 + margin).
3. Endpoint candidates when:
   - Silence tail ≥ 600 ms AND no token change ≥ 600 ms, OR
   - Utterance duration ≥ 12 s (safety cap).
4. Emits `start` then later `final` events. State resets for next utterance.

## Metrics (Planned Wiring)
- Will log: `endpoint_start_ts`, `endpoint_final_ts`, `reason`, `voiced_ms`, `silence_ms`.
- UI may show interim energy bar once integrated.

## Open Issues / Follow-ups
1. Need continuous RMS frame feed (Phase 4 decode loop or separate analyzer task).
2. Token stability currently mock (synthetic tokens); real diff logic arrives with Whisper integration.
3. Silence / no-token thresholds will be tuned after empirical latency measurements.
4. Add optional decay on baseline RMS for changing noise floor (Phase 9 refinement).

## Risk Mitigation
- All thresholds in config dataclass → runtime tunable.
- Pure function allows fast unit test coverage before real audio hooking.

## Next Steps
Integrate with real decode (Phase 4) and emit `endpoint_trigger` telemetry event preceding `final_transcript` once Whisper provides stable token tails.

_Date: (auto)_
