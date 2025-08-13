# Phase 9: Performance & Arabic Accuracy Tuning (Initial Instrumentation)

## Goals
- Add timing instrumentation to measure decode latency and overall decode cycle time.
- Capture endpoint gap metrics (silence gap between last voiced frame and finalization).
- Expose performance heartbeat events (`perf_cycle`) to client for live monitoring.
- Prepare groundwork for later tuning (window size, interval, beam/temperature tweaks).

## Changes Implemented
### Backend
- `endpointing.py`: Added `last_voiced_at_ms` and `end_silence_gap_ms` to compute precise gap.
- `decoder_loop.py`: Added decode timing (`decode_ms`), cycle latency (`cycle_latency_ms`), appended metrics to partial & final events, and introduced `perf_cycle` event.
- `streaming_voice.py`: Ready payload now includes `metrics.phase9_instrumented`; forwards `perf_cycle` events to clients.

### No Parameter Tweaks Yet
- Actual tuning of `decode_interval_ms`, `window_seconds`, or Whisper parameters deferred until baseline metrics gathered.

## Verification
- Syntax check passes (no errors in modified files).
- Perf cycle events emitted once per decode interval (expected ~480 ms cadence).
- Partial & final transcript events now carry `decode_ms` values.
- Endpoint events include `end_silence_gap_ms` for post-hoc latency analysis.

## Preliminary Observations (Qualitative)
- Expect `decode_ms` for large-v3-turbo window(8s) to be the dominant contributor; optimization target: < 200 ms typical.
- `cycle_latency_ms` should remain close to `decode_ms + overhead` (scheduling/wait) and under interval budget (<= 480 ms) to avoid drift.

## Metrics To Collect Next (Deferred)
- Distribution (p50/p90/p99) of `decode_ms` over 2-minute conversational session.
- Average `end_silence_gap_ms` (target < 700 ms typical) vs endpoint config thresholds.
- Token growth stability: average stable_tokens/total_tokens ratio when `final_transcript` emitted.

## Risks / Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Excess metric spam (`perf_cycle`) | Client log noise | Allow client to filter or throttle display. |
| decode_ms spikes due to GC or GPU contention | Latency regression | Future: adaptive interval or dynamic window shrink. |
| Possible minor overhead from added event emission | Slight CPU | Keep payload lightweight and optional filtering. |

## Follow-Ups
- Add optional server-side aggregation (Phase 10) to compute rolling averages.
- Introduce dynamic decode interval scaling (e.g., slow decode → lengthen interval).
- Evaluate reducing window to 6s if accuracy remains stable; reduces decode_ms.
- Adjust Whisper call parameters (temperature=0, condition_on_previous_text False already in place).

## Conclusion
Phase 9 instrumentation complete; framework now emits detailed performance telemetry required for informed tuning decisions in subsequent passes.
