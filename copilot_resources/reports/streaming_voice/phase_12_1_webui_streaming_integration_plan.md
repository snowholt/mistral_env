# WebUI Streaming Voice Integration Plan (Post Phase 12)

User Choices Incorporated:
- Toggle Placement: Replace legacy by default (fallback automatic)
- Auto-Rearm: ON by default
- Partial Transcript: Inline live typing bubble
- Language Change: Immediate reconnect
- Legacy VAD Controls: Fully hidden when streaming active
- Rollout Mode: Visible toggle default ON (will implement default ON; internal fallback only)
- Metrics Overlay: Developer-only toggle
- autoStop Checkbox: Hidden in streaming mode (autoRearm implicit)

## Objectives
Integrate `/api/v1/ws/streaming-voice` into existing `chat_ui.html` and `chat-interface.js` while preserving clean fallback to legacy `/ws/voice-conversation` without user confusion.

## Scope
IN: Frontend HTML/JS/CSS modifications, adapter, partial transcript UI, fallback logic, simple metrics overlay, localStorage persistence.
OUT: Removal of legacy code, mic level visualization, barge-in, accessibility pass.

## Architecture Additions
1. Feature Flag: `window.BEAUTYAI_STREAMING_VOICE = true` inline.
2. Mode State: `voiceMode = 'streaming' | 'legacy'` in localStorage.
3. Adapter: translate `StreamingVoiceClient` events to chat UI.
4. Body Class: `streaming-mode` for CSS toggles.
5. Live Partial Bubble: ephemeral user-live message replaced on final.
6. Metrics Overlay: developer-only panel (hidden unless debug enabled).

## Event Mapping
| Event | Action |
|-------|--------|
| ready | Status: Ready (Streaming) |
| partial_transcript | Update/create live bubble (italics) |
| final_transcript | Finalize bubble to permanent user msg |
| endpoint | Optional console debug |
| tts_start | Show synthesizing badge |
| tts_audio | Play audio (reuse playback) |
| tts_complete | Status update, auto re-arm (client auto) |
| perf_cycle | Append decode stats if overlay visible |
| error | Toast + fallback (if pre-first-partial) |

## Fallback Criteria
- Error/close before first `ready` or within 2s of connect.
- Policy close (1003) or feature disabled.

## CSS Targets
```
body.streaming-mode .vad-control,
body.streaming-mode [data-vad-only] { display:none !important; }
body.streaming-mode .streaming-badge { display:inline-flex; }
```

## Implementation Steps
1. Add inline flag + script tag for `streamingVoiceClient.js` in `chat_ui.html`.
2. Insert live partial container & streaming badge.
3. Add metrics overlay `<div id="streamingMetrics" class="hidden"></div>`.
4. Update `chat-interface.js`:
   - Determine initial mode (flag + localStorage) -> streaming.
   - Instantiate `StreamingVoiceClient` with `autoRearm:true`.
   - Implement `handleStreamingEvent` mapping.
   - Override `startRecording/stopRecording` logic when streaming mode active.
   - Hide legacy controls programmatically (backup to CSS).
   - Immediate reconnect on language change (debounced 500ms).
5. Add fallback helper `switchToLegacyFallback(reason)`.
6. Metrics overlay toggle via query param `?debug_audio=1` or `Ctrl+Alt+M`.
7. Latency logging (first partial vs ws_open).
8. Persist language/voice changes & reconnect streaming WS.
9. QA passes (see below) before commit.

## Test Plan (Manual + Scripted)
Manual:
- Load UI: streaming active; legacy controls hidden.
- Arabic phrase: partial <600ms.
- Consecutive turns: auto re-arm functioning.
- Language switch: reconnect & new partials show in new language.
- Disable server feature: fallback triggers.

Scripted (separate effort): PCM replay script for recorded prompts.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| AudioContext leak | Single instance reused |
| Duplicate final events | Track last utterance index |
| Fallback loops | Only fallback once per session; require manual refresh to retry streaming |
| UI confusion | Badge + hidden VAD controls |

## Rollout
R1 (this change): Streaming default ON, fallback silently.
R2: Remove legacy UI after stability window (later phase).

Prepared for implementation.
