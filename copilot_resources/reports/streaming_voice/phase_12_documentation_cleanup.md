# Phase 12 – Documentation & Cleanup

## Summary
Added streaming mode documentation, status endpoint, and deprecation notice for legacy chunked VAD path. Updated progress tracking and execution checklist.

## Changes
- `docs/VOICE.md`: Added streaming endpoint section, event schema examples, perf monitoring guidance, deprecation banner.
- `backend/src/beautyai_inference/api/endpoints/streaming_voice.py`: Added `/streaming-voice/status` probe.
- `planning/reports/streaming_voice/progress.md`: Marked Phase 12 complete.
- `planning/implementTheNextStep.md`: Checked Phase 12 in checklist.

## Metrics / Observability
- Status endpoint returns: `enabled`, `active_sessions`, `feature_flag`, `endpoint`.
- Encourages client perf tracking via `perf_cycle` events.

## Open Issues / Follow-ups
- Decide retention period before removing legacy `/ws/voice-conversation` (propose Phase 14 removal window after 2 weeks stable).
- Consider adding aggregated metrics endpoint (tokens/sec, mean decode_ms) in Phase 13 or separate enhancement.

## Decision Log Entries
- (Doc-001) Chose additive docs + deprecation notice instead of immediate removal to allow phased client migration.

## Next Steps
- Phase 13 (optional): Streaming TTS chunking + barge-in interrupt.
- Evaluate need for server-side auth gating on streaming endpoint.

