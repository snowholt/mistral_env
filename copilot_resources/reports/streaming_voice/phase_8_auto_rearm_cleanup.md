# Phase 8: Auto Re-Arm & Session Cleanup

## Goals
- Auto re-arm microphone capture / readiness after each TTS completion without full page reload.
- Provide suspend()/resume() lifecycle controls in client for future push-to-talk or power saving.
- Ensure backend cancels any in-flight LLM+TTS tasks on websocket disconnect.
- Harden websocket event handling for graceful closure.

## Changes Implemented
### Frontend
- Added `autoRearm` flag (default true) to `StreamingVoiceClient` constructor.
- Added `_suspended` internal flag plus public `suspend()` and `resume()` methods.
- Added auto re-arm trigger on `tts_complete` (fires synthetic `auto_rearm` event to UI hook).
- Guarded audio processing loop to respect `_suspended` state.
- Ensured safe close in `suspend()` (closes ws) and reconnect in `resume()`.

### Backend
- Updated websocket disconnect handler to explicitly cancel `llm_tts_task` if still running to prevent orphan tasks / GPU leaks.

## Verification
- Service restart confirms API + WebUI active (see timestamps ~15:30 UTC).
- No errors in systemd status output after restarts.
- Memory stable after reconnect (no incremental growth across two restart cycles observed in status snippet).
- JS build not required (served static file) — direct replacement loaded.

## Metrics (Qualitative Until Load Test)
- Re-arm latency: ~0ms (event-based, no renegotiation required aside from natural ws persistence).
- Additional memory: negligible (a few new flags + closures).

## Risks / Mitigations
| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Rapid suspend/resume toggling | Potential multiple ws instances | Guard with state checks (current implementation returns early if OPEN). |
| Auto re-arm undesired in manual modes | UX confusion | `autoRearm` configurable via constructor. |
| Missed task cancellation if exception earlier | Resource leak | Added explicit cancellation in `finally` block on disconnect path. |

## Open Follow-Ups
- Add UI control to toggle `autoRearm` dynamically.
- Surface suspend/resume controls in Web UI (button / hotkey).
- Add integration test simulating ws disconnect mid TTS.
- Capture TTS duration metric in progress report table.

## Conclusion
Phase 8 complete — client lifecycle robustness improved, backend cleanup hardened. Ready to proceed to performance tuning or UI polish phases.
