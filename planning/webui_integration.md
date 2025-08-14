### Clarifying Questions – Streaming UI Integration

1. Streaming Toggle Placement?
   - C) Replace legacy by default (no toggle; keep fallback automatic)
   Impact: A = safest; 

2. Auto-Rearm Behavior Default?
   - A) On (continues listening after each assistant response)

3. Partial Transcript Display Style?
   - A) Inline “live typing” bubble in conversation stream


4. Handling Language Change During Active Stream?
   - A) Immediate reconnect with new language


5. Legacy VAD Controls When Streaming Active?
   - A) Fully hide them

6. Initial Rollout Mode?
   - C) Visible toggle default ON
   Recommended path: B → C after QA.

7. Do we need a small metrics overlay (decode_ms, cycle_latency_ms) for dev?
   - A) Yes (developer-only toggle)

8. Should we repurpose existing “autoStop” checkbox to control autoRearm in streaming?

   - C) Hide in streaming mode, enabled autoRearm by default. 



1. Can you plan the webui integration based on your plan and my answers as we did for new endpoint? `planning`
2. Only for reminding :
```
1. Integration Plan for /api/v1/ws/streaming-voice
High-Level Goals
Introduce streaming incremental ASR path without breaking legacy VAD UX.
Allow A/B or per‑session opt‑in (feature flag + UI toggle).
Provide partial + final transcript display with minimal UI disruption.
Preserve existing voice settings where still relevant (language, TTS voice), hide obsolete VAD controls when streaming is active.
Graceful fallback to legacy endpoint if feature flag disabled or connection fails.
Step-by-Step Implementation Plan
Frontend Feature Flag & Bootstrapping

Add window.BEAUTYAI_STREAMING_VOICE = true (in a small inline script or pulled from a JSON config).
Inject <script src=".../streamingVoiceClient.js"></script> after chat-interface.js in chat_ui.html.
UI Toggle / Mode Selection

Add a “Streaming Voice (Beta)” switch in Voice Settings (or a radio: Legacy / Streaming).
Persist choice in localStorage (voiceMode=streaming|legacy).
On load: pick streaming if flag + stored preference.
Adapter Layer in chat-interface.js

Create a this.streamingClient (instance of StreamingVoiceClient) when streaming mode selected.
Map events:
ready → update status “Ready (Streaming)”
partial → live transcript area update
final → append final user transcript + trigger awaiting TTS already handled by backend events
tts_audio → play or hand off to existing playback logic (dedupe against legacy)
perf_cycle → (optional) small hidden performance log or console
Disable all legacy MediaRecorder / silence timers while streaming mode active (guard in methods like startRecording, stopRecording).
Live Transcript & UX Elements

Add a small DOM region (e.g., <div id="livePartialTranscript">) near voice overlay mic button.
Style to gray/italics for partial; replaced by normal styled bubble when final arrives.
Optional mic level later (not required initially—keep minimal).
Hide / Adjust Legacy Controls on Streaming

Add CSS class streaming-mode to <body> when active.
CSS hides: Auto-start (legacy VAD version), Auto-stop, Silence threshold sliders, any “(VAD mode)” labels.
Replace with an info badge: “Streaming incremental ASR active (no silence tuning needed).”
Language & Voice Binding

Reuse existing selectors (#language, #voice).
On language change while streaming: if connected → either reconnect with new ?language=... or queue change to next session (simpler: disconnect + reconnect).
Ensure no duplicate reconnection flood (debounce 500 ms).
Error Handling & Fallback

If WebSocket onerror or onclose occurs before a ready event → automatically:
Log error.
Show toast “Streaming unavailable – falling back to legacy mode.”
Switch UI back to legacy (remove streaming-mode, re-enable VAD controls).
If mid-session error after some success → offer “Reconnect” button (don’t force fallback immediately).
Session Lifecycle

Start: user presses mic → streamingClient.connect() then streamingClient.start() if not already capturing.
Stop: mic press again → streamingClient.suspend() (releasing WS, keeping audio graph warm) OR a full teardown method (destroy() later if needed).
Auto re‑arm uses existing autoRearm logic from client (optionally map to UI auto-start checkbox repurposed as “Auto re-arm after assistant reply”).
Code Organization / Minimal Changes

Keep modifications scoped to:
chat_ui.html (add script tag + partial transcript div + toggle + small inline script for flag).
chat-interface.js (adapter + conditionals).
A new tiny streaming_adapter.js if we want even cleaner separation (optional; can inline initially).
CSS additions for .streaming-mode.
Telemetry / Logging (Phase 12+ Adjacent)

Console log first-partial latency (timestamp difference between ws_open and first partial).
Optionally expose an internal debug panel if ?debug_audio=1—defer unless needed.
Progressive Rollout Strategy

Stage 1: Hidden toggle (manual enable).
Stage 2: Default ON for new sessions if stable.
Stage 3: Remove legacy VAD UI (scheduled after X days & metrics).
QA Checklist Before Default ON

Arabic short phrase → partial <600 ms.
Back-to-back utterances → no “stuck” partials.
TTS pipeline triggers only once per final.
Fallback path works (force reject by disabling feature flag).
Memory: no AudioContext leaks after 10 start/stop cycles.
Edge Cases & Handling
User switches language mid-utterance → apply on next reconnect.
User toggles streaming off while capturing → cleanly close WebSocket & stop sending frames.
Socket slow / decode backlog → handled server-side (we just show partial when available).
Multiple clicks on mic rapidly → debounce 300 ms to avoid connect storms.
Light Contract
Input: 16 kHz mono Int16 frames (~20 ms each).
Output events: ready, partial_transcript, endpoint, final_transcript, tts_*, perf_cycle.
Error modes: network close, feature disabled (1003 code), internal decode error -> error event.

```
3. I like this approach for testing, after planning to integration, please proceed to convert each webM file and then create test units to test new endpoint please. 
```
Recommended Real Audio Test Procedure
Convert Each WebM Test File to 16 kHz Mono PCM


ffmpeg -i q1.webm -ar 16000 -ac 1 -f s16le q1.pcm
Python Script to Stream PCM Frames Over WebSocket (20 ms / 320 samples)

Frame size: 320 samples * 2 bytes = 640 bytes per 20 ms.
Send with await asyncio.sleep(0.02) pacing to simulate real time (or faster for stress test).
Collect server events into log (JSON lines).
Metrics to Capture Per File

Time to first partial (ms)
Time to final transcript (ms)
decode_ms average & max (from perf_cycle)
Word accuracy (manual quick check or optional approximate WER baseline using reference transcript if available).
Script Outline (I will implement if you approve):

tests/streaming/manual/ws_replay_pcm.py
Args: --file q1.pcm --language ar --frame-ms 20 --loop 1
Output: stdout JSON lines & summary at end.
Optional: Add aggregator that runs all PCM files and produces a markdown summary.

Stability: Run each file twice back-to-back to confirm no resource leak; watch memory (optional logging of RSS if accessible).
```
- Please check the following transcription for the input voice `voice_tests/input_test_questions/webm`, I sorted the transcription based on the file name:
```
     "greeting.wav": "Hello, how are you today?",
  "greeting_ar.wav": "مرحبًا، كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصة بكم.",
  "botox.wav": "What is botox used for?",
  "laser_hair.wav": "How does laser hair removal work?",
  "q1.wav": "ما هو استخدام البوتوكس؟",
  "q2.wav": "كيف يعمل إزالة الشعر بالليزر؟",
  "q3.wav": "هل الحشوات الجلدية دائمة؟",
  "q4.wav": "ما هي الآثار الجانبية الشائعة للتقشير الكيميائي؟",
  "q5.wav": "هل الميزوثيرابي؟",
  "q6.wav": "كم تدوم نتائج جلسة تنظيف البشرة عادة؟",
  "q7.wav": "هل يمكن لأي شخص إجراء عملية تجميل الأنف غير الجراحية؟",
  "q8.wav": "ما هو الغرض من علاج البلازما الغنية بالصفائح الدموية PRP للبشرة؟",
  "q9.wav": "هل هناك فترة نقاهة بعد عملية شد الوجه بالخيوط؟",
  "q10.wav": "ما هي الفائدة الرئيسية لعلاج الضوء النبضي المكثف IPL؟"

```
 - Start tasks for unit tests, check their logs and check the `beautyai-api` and `beautyai-webui` services logs as well. 