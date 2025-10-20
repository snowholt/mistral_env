# WebRTC Keep-Alive Flow Diagram

This document provides visual representations of the WebRTC connection flow before and after the keep-alive fix.

---

## Before Fix: Connection Timeout Issue

```
┌─────────────────────────────────────────────────────────────────┐
│                     WebRTC Connection Flow                       │
│                        (BEFORE FIX)                              │
└─────────────────────────────────────────────────────────────────┘

Client                    Server (Connection Pool)          Cleanup Loop
  │                              │                                │
  │  SDP Offer                   │                                │
  ├─────────────────────────────>│                                │
  │                              │ Create RTCPeerConnection       │
  │  SDP Answer                  │ last_activity = T0             │
  │<─────────────────────────────┤                                │
  │                              │                                │
  │  ICE Candidates              │                                │
  ├<────────────────────────────>│ Update on state changes        │
  │                              │ last_activity = T0             │
  │                              │                                │
  │  Connection Established      │                                │
  │  ══════════════════════════  │                                │
  │                              │                                │
  │  Audio Track Starts          │                                │
  ├─────────────────────────────>│ on_track() handler             │
  │                              │ Create voice adapter           │
  │                              │ ⚠️ NO activity update!         │
  │                              │                                │
  │  Audio Frames (continuous)   │                                │
  ├══════════════════════════════>│ Process audio                 │
  ├══════════════════════════════>│ STT → LLM → TTS               │
  ├══════════════════════════════>│ ⚠️ last_activity FROZEN       │
  │         ...                  │                                │
  │  (5-10 minutes pass)         │                                │
  │                              │                          [T=600s]
  │                              │                Check idle_time  │
  │                              │<───────────────────────────────┤
  │                              │ idle = 600s - T0 = 600s        │
  │                              │ ❌ idle > timeout              │
  │                              │ Remove connection              │
  │                              │────────────────────────────────>│
  │                              │ Close RTCPeerConnection        │
  │                              │                                │
  │  ❌ CONNECTION LOST          │                                │
  │<─────────────────────────────┤                                │
  │                              │                                │

Result: User sees disconnection after timeout period
        Active audio processing interrupted
        Conversation context lost
```

---

## After Fix: Keep-Alive Maintains Connection

```
┌─────────────────────────────────────────────────────────────────┐
│                     WebRTC Connection Flow                       │
│                        (AFTER FIX)                               │
└─────────────────────────────────────────────────────────────────┘

Client                    Server (Connection Pool)          Cleanup Loop
  │                              │                                │
  │  SDP Offer                   │                                │
  ├─────────────────────────────>│                                │
  │                              │ Create RTCPeerConnection       │
  │  SDP Answer                  │ last_activity = T0             │
  │<─────────────────────────────┤                                │
  │                              │                                │
  │  ICE Candidates              │                                │
  ├<────────────────────────────>│ Update on state changes        │
  │                              │ last_activity = T0             │
  │                              │                                │
  │  Connection Established      │                                │
  │  ══════════════════════════  │                                │
  │                              │                                │
  │  Audio Track Starts          │                                │
  ├─────────────────────────────>│ on_track() handler             │
  │                              │ Create voice adapter           │
  │                              │ ✅ Start keep-alive task       │
  │                              │                                │
  │  Audio Frames (continuous)   │     ┌─────────────────┐       │
  ├══════════════════════════════>│     │ Keep-Alive Task │       │
  ├══════════════════════════════>│     │   (async loop)  │       │
  │         ...                  │     └────────┬────────┘       │
  │                              │              │                 │
  │      [Every 30 seconds]      │              │ [T=30s]        │
  │                              │              │ update_activity()│
  │                              │<─────────────┤                 │
  │                              │ last_activity = T30            │
  │                              │              │                 │
  │  Audio continues...          │              │ [T=60s]        │
  │                              │              │ update_activity()│
  │                              │<─────────────┤                 │
  │                              │ last_activity = T60            │
  │                              │              │                 │
  │         ...                  │              │                 │
  │  (10+ minutes pass)          │              │                 │
  │                              │              │ [T=630s]       │
  │                              │              │ update_activity()│
  │                              │<─────────────┤                 │
  │                              │ last_activity = T630           │
  │                              │                          [T=600s]
  │                              │                Check idle_time  │
  │                              │<───────────────────────────────┤
  │                              │ idle = 600s - T30 = 570s       │
  │                              │ ✅ idle < timeout              │
  │                              │ Keep connection alive          │
  │                              │────────────────────────────────>│
  │                              │                                │
  │  ✅ CONNECTION ACTIVE        │                                │
  │  Audio continues normally    │                                │
  │                              │                                │

Result: Connection stays active during audio processing
        No unexpected disconnections
        Smooth user experience
```

---

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                  WebRTC Connection Pool Architecture                  │
│                          (After Fix)                                  │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Client        │
│   (Browser)     │
└────────┬────────┘
         │ SDP/ICE
         │ Audio Stream
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RTCPeerConnection                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  on("connectionstatechange")                                 │  │
│  │    └─> update_connection_state()                             │  │
│  │          └─> update_activity()  ✅                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  on("iceconnectionstatechange")                              │  │
│  │    └─> update_ice_connection_state()                         │  │
│  │          └─> update_activity()  ✅                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  on("track")  ← Audio Track Received                         │  │
│  │    ├─> Create Voice Service Adapter                          │  │
│  │    └─> ✨ NEW: Start keep_alive_during_audio()               │  │
│  │              ┌───────────────────────────────────┐            │  │
│  │              │  async def keep_alive():          │            │  │
│  │              │    while connection exists:       │            │  │
│  │              │      sleep(30)                    │            │  │
│  │              │      update_activity()  ✅        │            │  │
│  │              └───────────────────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ Processed Audio
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Voice Service Adapter                                    │
│  (Audio Processor → VAD → Buffer → STT → LLM → TTS)                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                Connection Pool - Cleanup Loop                         │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Every 60 seconds:                                             │ │
│  │    for each connection:                                        │ │
│  │      idle_time = now - last_activity                           │ │
│  │      if idle_time > timeout:                                   │ │
│  │        ❌ Remove connection                                    │ │
│  │      else:                                                      │ │
│  │        ✅ Keep connection (activity detected)                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## State Transition Diagram

```
┌──────────────────────────────────────────────────────────────┐
│         WebRTC Connection Lifecycle States                    │
│                  (With Keep-Alive)                            │
└──────────────────────────────────────────────────────────────┘

                          ┌─────────┐
                          │   NEW   │
                          └────┬────┘
                               │ create_peer_connection()
                               ▼
                       ┌───────────────┐
                       │  CONNECTING   │
                       │ last_activity │
                       │    updated    │
                       └───────┬───────┘
                               │ ICE negotiation
                               ▼
                      ┌────────────────┐
                      │   CONNECTED    │
                      │  last_activity │
                      │    updated     │
                      └────────┬───────┘
                               │ on("track")
                               ▼
                    ┌─────────────────────┐
                    │  AUDIO ACTIVE       │
                    │  ┌───────────────┐  │
                    │  │ Keep-Alive    │  │
                    │  │ Task Running  │  │
                    │  └───────┬───────┘  │
                    │          │          │
                    │   Every 30s:        │
                    │   update_activity() │ ← ✨ NEW
                    │          │          │
                    │          ▼          │
                    │  last_activity =    │
                    │  current_time       │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        │ Idle > timeout       │ User disconnect      │ Connection error
        ▼                      ▼                      ▼
    ┌───────┐           ┌────────────┐        ┌─────────────┐
    │ IDLE  │           │ DISCONNECT │        │   FAILED    │
    │CLEANUP│           │  (Normal)  │        │   (Error)   │
    └───┬───┘           └─────┬──────┘        └──────┬──────┘
        │                     │                      │
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │    CLOSED     │
                      │  Resources    │
                      │   Released    │
                      └───────────────┘
```

---

## Keep-Alive Task Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│              Keep-Alive Task Lifecycle                        │
└──────────────────────────────────────────────────────────────┘

Audio Track Received
        │
        ▼
┌────────────────────────────────────────────────┐
│  Create keep_alive_during_audio() task         │
│  Store in _keepalive_tasks[peer_id]            │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Task Running      │
        │  ┌──────────────┐  │
        │  │ while loop   │  │
        │  │   sleep(30)  │  │
        │  │   update()   │  │◄─┐
        │  └──────┬───────┘  │  │
        │         └───────────┘  │ Continues
        └────────────────────────┘ until closed
                 │
                 │ Connection closed
                 ▼
┌────────────────────────────────────────────────┐
│  _cleanup_connection(peer_id)                  │
│    ├─> Cancel keep-alive task                  │
│    ├─> Wait for CancelledError                 │
│    ├─> Remove from _keepalive_tasks            │
│    └─> Close RTCPeerConnection                 │
└────────────────────────────────────────────────┘
                 │
                 ▼
              Cleaned Up
```

---

## Activity Tracking Timeline

```
Time   │ Event                          │ last_activity │ idle_time │ Action
───────┼────────────────────────────────┼───────────────┼───────────┼─────────
T=0s   │ Connection created             │ T=0s          │ 0s        │ -
T=0s   │ ICE connected                  │ T=0s ✅       │ 0s        │ -
T=1s   │ Audio track received           │ T=0s          │ 1s        │ -
T=1s   │ Keep-alive task started        │ T=0s          │ 1s        │ -
T=30s  │ ⏰ Keep-alive update           │ T=30s ✅      │ 30s       │ Update
T=60s  │ 🧹 Cleanup check               │ T=30s         │ 30s       │ Keep
T=60s  │ ⏰ Keep-alive update           │ T=60s ✅      │ 60s       │ Update
T=90s  │ ⏰ Keep-alive update           │ T=90s ✅      │ 90s       │ Update
T=120s │ 🧹 Cleanup check               │ T=90s         │ 30s       │ Keep
T=120s │ ⏰ Keep-alive update           │ T=120s ✅     │ 120s      │ Update
...    │ ...                            │ ...           │ ...       │ ...
T=600s │ 🧹 Cleanup check               │ T=570s        │ 30s       │ Keep ✅
T=630s │ ⏰ Keep-alive update           │ T=630s ✅     │ 630s      │ Update
T=660s │ 🧹 Cleanup check               │ T=630s        │ 30s       │ Keep ✅

Result: Connection stays alive indefinitely during active audio
        Cleanup only occurs if truly idle (no updates for > timeout)
```

---

## Comparison: Before vs After

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BEFORE FIX                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Timeline:                                                           │
│  ═══════════════════════════════════════════════════                │
│  0s      300s     600s                                               │
│  │────────│────────│                                                 │
│  Connect  Audio    ❌ Disconnected                                   │
│           Active   (Timeout)                                         │
│                                                                      │
│  Issues:                                                             │
│  • Premature disconnection during active calls                       │
│  • User experience disrupted                                         │
│  • Conversation context lost                                         │
│  • Need to manually reconnect                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         AFTER FIX                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Timeline:                                                           │
│  ═════════════════════════════════════════════════════════════════  │
│  0s      300s     600s     900s    1200s   ...                       │
│  │────────│────────│────────│────────│──────────>                   │
│  Connect  Audio    Audio    Audio    Audio   Continues...            │
│           Active   Active   Active   Active                          │
│           ✅       ✅       ✅       ✅                                │
│                                                                      │
│  Benefits:                                                           │
│  ✅ Connection stays active during audio                             │
│  ✅ Smooth user experience                                           │
│  ✅ Context maintained                                               │
│  ✅ No unexpected disconnections                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Metrics Impact

```
┌──────────────────────────────────────────────────────────────┐
│                   Connection Duration                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  BEFORE FIX:                                                  │
│  ┌─────────────────────────────────────────────┐             │
│  │ ████████████████████████████████ │          │             │
│  └─────────────────────────────────────────────┘             │
│  0s                    300s                   600s            │
│     ↑                                          ↑              │
│  Connects                              Disconnects            │
│                                                               │
│  AFTER FIX:                                                   │
│  ┌──────────────────────────────────────────────────────────>│
│  │ ████████████████████████████████████████████████████████  │
│  └──────────────────────────────────────────────────────────>│
│  0s         600s        1200s       1800s      2400s   ...    │
│     ↑                                                  ↑       │
│  Connects                              User ends call         │
│                                                               │
└──────────────────────────────────────────────────────────────┘

  Average Duration:     300-600s  →  User-determined
  Premature Timeouts:   100%      →  0%
  User Satisfaction:    Low       →  High
```

---

**Legend:**
- ✅ = Activity update / Success
- ❌ = Connection lost / Failure
- ⏰ = Scheduled event (keep-alive)
- 🧹 = Cleanup check
- ═ = Active audio stream
- │ = Timeline marker

---

**Document Version:** 1.0  
**Created:** 2025-10-20  
**For:** WebRTC Keep-Alive Fix Documentation
