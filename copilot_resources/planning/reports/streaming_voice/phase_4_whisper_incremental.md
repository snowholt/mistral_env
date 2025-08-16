# Phase 4 – Whisper Integration (Incremental Windowed Decode)

## Summary
Implemented a windowed incremental decoding loop using existing FasterWhisper service with periodic (480 ms) re-decode of the last 8 s PCM window. Added feature flag `VOICE_STREAMING_PHASE4=1` to switch from mock token emission to real partial transcripts. Endpoint now streams partial transcripts with stability metadata and endpoint telemetry events (placeholder events from endpoint state machine).

## Changes
- Added `decoder_loop.py` implementing `incremental_decode_loop` async generator.
- Updated `streaming_voice.py` to optionally launch decoder loop (Phase 4 flag) instead of mock loop.
- Added Phase 4 fields in ready payload (`decode_interval_ms`, `window_seconds`).
- Added faster-whisper dependency to `backend/requirements.txt`.
- Introduced endpoint events passthrough (`endpoint_event` -> client as `endpoint`).

## Decode Strategy
- Window Length: 8 s (tunable) re-decoded each cycle.
- Interval: 480 ms (balances latency vs GPU load; target < 600 ms partial emergence).
- Stability: Naive stable prefix via whitespace token diff (future: true tokenizer & alignment).
- RMS / Endpoint: Feeds frame RMS to endpoint state (tokens not yet provided → silence + duration rules only).

## Metrics (Initial Observability)
- Partial cadence ≈ decode_interval (if audio arrives continuously).
- Stability increases as early tokens persist across cycles.
- Endpoint events currently limited (no token-based stability yet). Token stability hook deferred to Phase 5.

## Open Issues / Follow-ups
1. Replace whitespace tokenization with model tokenizer for robust diff.
2. Supply current token list to `update_endpoint` for no_token_change logic.
3. Optimize re-decode by caching log-mel features or shorter dynamic window when speech ongoing.
4. Add GPU utilization sampling & decode latency measurements (Phase 9).
5. Implement final transcript emission on endpoint events (Phase 5) and context reset.

## Risk Mitigation
- Feature flag gating preserves mock path for fallback / regression testing.
- Windowed re-decode approachable; no incremental model state complexity introduced yet.
- Conservative interval to avoid GPU spikes; can shorten after profiling.

## Next Steps (Phase 5 Preview)
- Integrate endpoint finalization: on `endpoint` event finalize stable portion, emit `final_transcript`.
- Reset stable prefix tokens after finalization & re-arm for next utterance.
- Provide token change feed into endpointing to leverage `no_token_change_ms` path.

_Date: (auto)_
