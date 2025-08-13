# Phase 2 – Ring Buffer & Frame Handling

## Summary
Added server-side PCM Int16 ingestion path with per-connection ring buffer (≈40s capacity) and integrated session state (`StreamingSession`). Binary WebSocket frames are now treated as raw 16 kHz mono little-endian PCM and written into the async ring buffer. Mock decode loop augmented to surface buffer usage, dropped frame counts, and a crude audio level proxy for early frontend telemetry.

## Changes
- Modified `backend/src/beautyai_inference/api/endpoints/streaming_voice.py`:
  - Phase header updated to reflect Phase 2.
  - Added PCM frame ingestion (binary messages) writing to ring buffer via `StreamingSession`.
  - Added buffer usage, dropped frames, audio level fields in partial transcript mock events.
  - Ready payload now reports `pcm_sample_rate` and `ring_buffer_seconds`.
  - Status endpoint updated to phase=2 description.
  - Graceful ring buffer close on disconnect.
- Added `backend/src/beautyai_inference/services/voice/streaming/streaming_session.py` (session state w/ buffer + counters).
- Added `backend/src/beautyai_inference/services/voice/streaming/ring_buffer.py` (async ring buffer w/ metrics & overflow dropping).
- Updated `planning/reports/streaming_voice/progress.md` (Phase 1 & 1.5 done, Phase 2 in progress).

## Data Flow (Current)
Client (AudioWorklet Int16 frames @16kHz) → WebSocket binary frame → Endpoint → `StreamingSession.pcm_buffer.write()` → (future) incremental decoder consumer.

## Metrics (Initial)
- Buffer usage ratio exposed in mock events (`buffer_usage`).
- Drop count via `dropped` field (increments when capacity exceeded and oldest frames removed).
- Approximate audio level proxy derived from buffer usage (placeholder until RMS windows added in Phase 3/4).

## Open Issues / Follow-ups
1. Add consumer coroutine to drain ring buffer for endpointing (Phase 3).
2. Implement proper RMS / energy windows inside a lightweight analyzer (Phase 3) instead of buffer-usage proxy.
3. Add client downsampling confirmation & payload size validation (currently assumes correct rate/format).
4. Consider per-frame sequence numbering for loss diagnostics (optional).
5. Add health debug endpoint to list per-session buffer stats (Phase 3/4).

## Risk Mitigation
- Overflow handled by dropping oldest frames (prevents unbounded memory growth).
- 40s buffer window generous for early dev; will shrink (~8–12s) post endpointing tuning.
- If binary payload length is odd, it's logged (sanity check) but still appended; future strict validation may reject.

## Next Steps
Proceed to Phase 3: Endpointing (mock tokens) – introduce energy/RMS analyzer consuming the ring buffer and align emission of mock partial/final transcript events with detected speech segments.

## Rollback Strategy
Revert modifications to endpoint file and remove new streaming service modules; feature flag gating remains intact.

_Date: (auto)_
