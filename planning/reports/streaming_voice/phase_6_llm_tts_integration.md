# Phase 6 – LLM + TTS Integration Path

## Summary
Wired LLM (ChatService) and Edge TTS (SimpleVoiceService) pipeline into streaming endpoint: on `final_transcript` events the server now generates a model response and returns synthesized audio (base64) plus lifecycle events.

## Changes
- `streaming_voice.py`: Added lazy initialization for ChatService & SimpleVoiceService, conversation history, final transcript handler, emits tts_start / tts_audio / tts_complete.
- `progress.md`: Phase 5 marked done; Phase 6 set to in progress.
- Added this phase report file.

## Event Flow Additions
1. final_transcript (from incremental decoder)
2. tts_start (ack & timing baseline)
3. tts_audio { audio: <base64 wav>, text: <assistant reply> }
4. tts_complete { processing_ms }

## Implementation Notes
- Base64 WAV used for initial simplicity (avoid introducing static file handler / URL path yet).
- Lazy service init prevents upfront load if user disconnects early.
- Conversation history stored per session enabling contextual responses.
- Temperature set to 0.3 for stability; max_new_tokens 256 to bound latency.
- Added guard ensuring single active llm_tts_task.

## Metrics (To Capture Later)
- STT finalization → tts_start latency
- LLM generation time
- TTS synthesis time
- Total voice reply latency

## Open Issues / Follow-ups
- Consider switching to streaming binary audio frames for long replies (Phase 13 roadmap).
- Add proper cleanup (await llm_tts_task) on disconnect.
- Add structured logging per stage with durations.
- Evaluate WebM/Opus output path to reduce payload size (currently WAV base64).

## Next Steps
- Validate end-to-end manually (speak phrase → hear AI response).
- Phase 7: Frontend enhancements to play base64 WAV quickly & show assistant transcript.

