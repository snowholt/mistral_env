# Phase 1.5 – Minimal Client Preprocessor

## Summary
Implemented minimal AudioWorklet-based preprocessing: high-pass (DC removal), rolling RMS baseline, soft noise gate, RMS normalization with clamped gain, peak limiter, optional debug metric emission. Added streaming voice client skeleton to load worklet and prepare future downsampling + frame sending.

## Changes
- Added `frontend/src/static/js/streaming-audio-worklet.js` (minimal preprocessor worklet).
- Added `frontend/src/static/js/streamingVoiceClient.js` (skeleton client w/ worklet loading & WS connect stub).
- Updated `planning/reports/streaming_voice/progress.md` marking Phase 1.5 in progress.

## Metrics (Planned Validation)
- Will compare speech/silence RMS ratio after integration with downsampling.
- Debug metrics available via `enableDebug` config (preproc_debug events).

## Open Issues / Follow-ups
1. Integrate downsampling (48k -> 16k) and Int16 framing (Phase 2) inside client.
2. Add feature flag wiring in UI (`USE_STREAMING_VOICE`) – not yet hooked into `chat-interface.js`.
3. Provide UI toggle to enable debug metrics overlay (future dev tool).
4. Evaluate attenuation factor & gain smoothing with real Arabic consonant samples.

## Risk Mitigation
- All processing bypassable by setting `enableProcessing=false` via port message.
- Gain clamped ±6 dB to avoid pumping; smoothing parameters conservative.

## Next Steps
Proceed to Phase 2: ring buffer + frame handling (server) and implement client-side downsampling & frame dispatch.

_Date: (auto)_
