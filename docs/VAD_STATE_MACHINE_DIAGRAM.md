# VAD State Machine Diagram

## Current Implementation - Dual VAD State Machine

This diagram shows the VAD state transitions and the conditions required for audio to reach the STT model.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VAD State Machine                                │
│                    (webrtc_vad_service.py)                               │
└─────────────────────────────────────────────────────────────────────────┘

                             START
                               │
                               ↓
                    ┌──────────────────┐
                    │    INACTIVE      │ ← Default state, pre-roll buffer active
                    │                  │
                    │  Dual VAD:       │
                    │  WebRTC=False    │
                    │  Silero=False    │
                    └────────┬─────────┘
                             │
                             │ Voice detected (WebRTC AND Silero)
                             │ require_silero_confirmation=True ⚠️
                             ↓
                    ┌──────────────────┐
                    │  VOICE_START     │ ← Speech onset detected
                    │                  │
                    │  Timer starts    │
                    │  Buffer:         │
                    │  Pre-roll copied │
                    │  to active       │
                    └────────┬─────────┘
                             │
                             │ Voice sustained for 300ms ⚠️
                             │ min_speech_duration_ms=300
                             │
      ┌──────────────────────┼──────────────────────┐
      │ ✗ Voice lost         │ ✓ 300ms reached      │
      │   before 300ms       │                      │
      ↓                      ↓                      │
┌──────────┐      ┌──────────────────┐            │
│ INACTIVE │ ← ✗  │  VOICE_ACTIVE    │ ← Recording │
│          │      │                  │            │
│ Reset    │      │  Buffer:         │            │
│ timer    │      │  Accumulating    │            │
└──────────┘      │  speech frames   │            │
                  └────────┬─────────┘            │
                           │                      │
                           │ Silence detected     │
                           ↓                      │
                  ┌──────────────────┐            │
                  │ VOICE_END_PENDING│            │
                  │                  │            │
                  │ Silence timer    │            │
                  │ starts           │            │
                  └────┬────────┬────┘            │
                       │        │                 │
   Voice resumed ──────┘        │ Silent 500ms ⚠️ │
   (back to ACTIVE)             │ post_speech_silence_ms=500
                                ↓                 │
                       ┌──────────────────┐       │
                       │   VOICE_END      │       │
                       │                  │       │
                       │  Post-roll       │       │
                       │  countdown       │       │
                       │  (300ms)         │       │
                       └────────┬─────────┘       │
                                │                 │
                                │ Post-roll done  │
                                ↓                 │
                       ┌──────────────────┐       │
                       │ SEGMENT READY    │ ◄─────┘
                       │                  │
                       │ Buffer finalized │
                       │ → Send to STT    │
                       └──────────────────┘
                                │
                                ↓
                       ┌──────────────────┐
                       │  STT Processing  │
                       │  (Whisper)       │
                       └──────────────────┘


═══════════════════════════════════════════════════════════════════════════

## Problem: Why Audio Doesn't Reach STT

ISSUE #1: Strict Dual VAD (Both must agree)
───────────────────────────────────────────

Chunk Timeline:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ #1  │ #2  │ #3  │ #4  │ #5  │ #6  │ #7  │ #8  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

WebRTC VAD:  ✓     ✓     ✗     ✓     ✓     ✓     ✓     ✓
Silero VAD:  ✗     ✓     ✓     ✓     ✗     ✓     ✓     ✓
             ───   ───   ───   ───   ───   ───   ───   ───
Final:       ✗     ✓     ✗     ✓     ✗     ✓     ✓     ✓
             └─────┴─────┴─────┴─────┘
                   INCONSISTENT
             Never sustains 300ms!

Result: INACTIVE → VOICE_START → INACTIVE → VOICE_START → INACTIVE
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Never reaches VOICE_ACTIVE, audio never buffered properly

═══════════════════════════════════════════════════════════════════════════

ISSUE #2: 300ms Minimum Duration
─────────────────────────────────

Short utterance: "Yes" (200ms total)

Time 0ms:    INACTIVE
Time 50ms:   VOICE_START (timer starts)
Time 250ms:  Voice ends (user stops)
             Only 200ms elapsed < 300ms threshold ⚠️
             Never reached VOICE_ACTIVE
             Buffer never properly accumulated

Result: Audio detected but discarded (didn't meet minimum)

═══════════════════════════════════════════════════════════════════════════

ISSUE #3: Continuous Speech (No Pauses)
────────────────────────────────────────

User speaks for 8 seconds without pausing:

Time 0s:     INACTIVE
Time 0.05s:  VOICE_START
Time 0.35s:  VOICE_ACTIVE (buffer accumulating)
Time 8.00s:  Still speaking...
             ^^^^^^^^^^^^^^
             No silence detected!
             State remains VOICE_ACTIVE
             Buffer accumulating but never finalized

Result: Audio buffered but never sent to STT (waiting for pause)

═══════════════════════════════════════════════════════════════════════════

ISSUE #4: 10-Second Utterance Limit
────────────────────────────────────

User speaks for 12 seconds:

Time 0s:     INACTIVE
Time 0.35s:  VOICE_ACTIVE
Time 10s:    Audio processor hits max_utterance_duration_sec
             Stops processing, tries to extract buffer
             BUT: is_recording=True, get_complete_segment() returns None
             
Result: 10 seconds of audio lost (buffer not properly finalized)

═══════════════════════════════════════════════════════════════════════════

## Buffer States During VAD Transitions

┌─────────────────────────────────────────────────────────────────────────┐
│                        Buffer States                                     │
└─────────────────────────────────────────────────────────────────────────┘

STATE: INACTIVE
───────────────
Pre-roll buffer:  [chunk, chunk, chunk] ← Rolling window (300ms)
Active buffer:    []                    ← Empty
Recording:        False

Audio: Buffered in pre-roll only, not accumulated

───────────────────────────────────────────────────────────────────────────

STATE: VOICE_START (< 300ms)
────────────────────────────
Pre-roll buffer:  [chunk, chunk, chunk] ← Still rolling
Active buffer:    [chunk, chunk]        ← Pre-roll COPIED + new chunks
Recording:        True

Audio: Being accumulated, but not confirmed yet

───────────────────────────────────────────────────────────────────────────

STATE: VOICE_ACTIVE (≥ 300ms)
──────────────────────────────
Pre-roll buffer:  [chunk, chunk, chunk] ← Still rolling
Active buffer:    [chunk, chunk, chunk, chunk, chunk, ...] ← Growing
Recording:        True

Audio: Confirmed speech, actively accumulating

───────────────────────────────────────────────────────────────────────────

STATE: VOICE_END_PENDING
─────────────────────────
Pre-roll buffer:  [chunk, chunk, chunk] ← Still rolling
Active buffer:    [chunk, ..., chunk]   ← Still adding (might resume)
Recording:        True

Audio: Silence detected, but waiting for confirmation

───────────────────────────────────────────────────────────────────────────

STATE: VOICE_END (post-roll)
─────────────────────────────
Pre-roll buffer:  [chunk, chunk, chunk] ← Still rolling
Active buffer:    [chunk, ..., chunk, chunk] ← Adding post-roll
Recording:        True (until post-roll done)
Post-roll count:  1, 2, 3, ... (up to ~10 frames for 300ms)

Audio: Finalizing, adding safety margin

───────────────────────────────────────────────────────────────────────────

POST-ROLL COMPLETE
──────────────────
Active buffer:    [ALL CHUNKS CONCATENATED] → b'...' (PCM bytes)
                  ↓
              STT Processing
Recording:        False (reset)
Active buffer:    [] (cleared)

Audio: Sent to Whisper for transcription

═══════════════════════════════════════════════════════════════════════════

## Solution: Recommended Configuration Changes

CURRENT CONFIG (Too Strict):
────────────────────────────
require_silero_confirmation = True   # Both VADs must agree ⚠️
min_speech_duration_ms = 300        # Too long for short utterances ⚠️
post_speech_silence_ms = 500        # Long wait for end detection ⚠️

RECOMMENDED CONFIG (Testing):
─────────────────────────────
require_silero_confirmation = False  # Allow either VAD ✓
min_speech_duration_ms = 100        # Faster VOICE_ACTIVE ✓
post_speech_silence_ms = 300        # Quicker finalization ✓

BENEFITS:
─────────
✓ Either VAD can trigger detection (more permissive)
✓ Short utterances ("yes", "no") can be detected
✓ Faster response time (600ms → 400ms)
✓ Better for conversational speech patterns

═══════════════════════════════════════════════════════════════════════════

## Dual VAD Decision Logic

CURRENT LOGIC:
──────────────
if require_silero_confirmation:
    voice_detected = webrtc_detected AND silero_detected  ⚠️
else:
    voice_detected = webrtc_detected OR silero_detected   ✓

Truth Table (Current, AND logic):
┌─────────┬─────────┬─────────┐
│ WebRTC  │ Silero  │ Result  │
├─────────┼─────────┼─────────┤
│  True   │  True   │  True   │ ✓ Both agree
│  True   │  False  │  False  │ ✗ WebRTC yes, Silero no
│  False  │  True   │  False  │ ✗ Silero yes, WebRTC no
│  False  │  False  │  False  │ ✗ Neither detected
└─────────┴─────────┴─────────┘

Only 25% of cases result in detection!

Truth Table (Recommended, OR logic):
┌─────────┬─────────┬─────────┐
│ WebRTC  │ Silero  │ Result  │
├─────────┼─────────┼─────────┤
│  True   │  True   │  True   │ ✓ Both agree
│  True   │  False  │  True   │ ✓ WebRTC detected
│  False  │  True   │  True   │ ✓ Silero detected
│  False  │  False  │  False  │ ✗ Neither detected
└─────────┴─────────┴─────────┘

75% of cases result in detection! (3x improvement)

═══════════════════════════════════════════════════════════════════════════
```

## Key Takeaways

1. **Dual VAD AND logic is too strict** - Reduces detection by 75%
2. **300ms minimum duration filters short utterances** - "Yes", "No", "Hello"
3. **500ms silence requirement causes delays** - User must pause significantly
4. **No streaming transcription** - Must wait for complete silence or 10s limit

## Recommended Actions

1. ✅ Change `require_silero_confirmation=False`
2. ✅ Reduce `min_speech_duration_ms=100`
3. ✅ Reduce `post_speech_silence_ms=300`
4. 🔧 Add streaming transcription for long utterances
5. 🔧 Fix 10-second limit handler to properly finalize buffer
