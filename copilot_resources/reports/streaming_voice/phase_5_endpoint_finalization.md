# Phase 5 – Endpoint Finalization Logic

## Summary
Integrated real endpoint finalization: endpoint state machine now receives live token lists; emits `final_transcript` once per utterance with guard against duplicates.

## Changes
- `decoder_loop.py`: Added token-aware endpoint advancement, final transcript emission, stability reset.
- `progress.md`: Updated statuses (Phases 2–4 marked done, Phase 5 in progress).
- Added this report file.

## Logic Details
1. Decode window -> tokens.
2. Stable prefix diff maintained (whitespace token surrogate).
3. Endpoint update fed `current_tokens` per cycle (was None before).
4. On receiving `final` endpoint event, emit `final_transcript` (once) then reset decoder token state.
5. Guard via `last_final_utterance_index` prevents duplicate finals.

## Metrics (Initial Observations Placeholder)
(To capture after runtime validation with real audio) Expected:
- Single final per utterance.
- Final emitted within configured silence/token stability thresholds (silence >= 600 ms + no token change >= 600 ms or max duration).

## Open Issues / Follow-ups
- Replace whitespace tokenization with model-native tokens for higher stability.
- Add latency measurement (end-of-silence -> final_transcript timestamp).
- Add unit tests for double-final prevention.

## Next Steps
- Complete runtime validation (service restart & live test).
- Phase 6: Wire LLM + TTS pipeline trigger on `final_transcript`.

## Decision Log Entries
(None new – thresholds remain per earlier plan.)

