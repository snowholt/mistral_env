# Phase 7 – Frontend Streaming Client

## Summary
Implemented functional browser client for streaming voice: captures mic, runs minimal preproc, downsamples 48k→16k, frames PCM Int16 (20ms), sends over WS, and handles partial/final transcripts plus LLM+TTS events with base64 WAV playback.

## Changes
- `streamingVoiceClient.js`: Upgraded from skeleton to full Phase 7 client (downsampling, framing, sending binary, event handling, audio playback).
- Added playback helper for base64 WAV (temporary until URL/static path or streaming audio frames introduced).

## Event Handling
- Inbound: ready, partial_transcript, final_transcript, endpoint, tts_start, tts_audio, tts_complete, error.
- Outbound: binary Int16 frames only (future: control messages e.g., stop / config adjust).

## Implementation Details
- Simple 3:1 average downsampler (48k→16k) acceptable MVP; future improvement: low-pass FIR for quality.
- 20ms frames (320 samples) aggregated and sent; leftover samples retained.
- Autoplay base64 WAV with optional injected audio element.
- Conversation continuity supported by server; client resets live partial on final.

## Metrics / Next Metrics
- Future capture: average frame send interval, jitter, downstream latency (partial receipt time vs frame capture time).

## Open Issues / Follow-ups
- Improve downsampling fidelity (FIR/linear interpolation) – low priority until quality issues observed.
- Add mic level UI (expose processed RMS from worklet via messages) – planned.
- Add user control to disable autoplay or switch to streaming audio chunks.
- Cleanup object URLs after playback (done for transient audio elements; persistent sink managed externally).

## Next Steps
- Phase 8: Auto re-arm & cleanup (auto microphone reactivation post TTS), ensure task cancellation robustness.

