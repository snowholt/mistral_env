I'll research relevant web sources about MediaRecorder WebM/Opus, streaming PCM ASR, and real-time Opus decoding to ensure accuracy. Below is an accurate, deeply reasoned, low‑jargon explanation of how your current streaming voice WebSocket works, why the WebM upload produced nonsense, and what “supporting WebM” would actually require. I’ve also separated what you have today vs. what would need to be added, so you can decide—and I am stopping after this explanation per your request.

--------------------------------------------------
1. Which WebSocket and what method are you using?
--------------------------------------------------
You are using the advanced streaming voice WebSocket (your debug page points to /api/v1/ws/streaming-voice). Internally it implements:

Method (conceptual name): Real‑time incremental streaming ASR with VAD and utterance endpointing.

Plain-language description:
1. You open a WebSocket.
2. You drip small slices (frames) of raw audio into it in order—each frame is a tiny piece of sound (e.g. 20 ms).
3. The server keeps a rolling buffer and every fixed short interval (e.g. 240–480 ms) runs a speech recognition decode on the current window.
4. As soon as it is reasonably confident about part of what you said, it sends you a partial transcript (so you see words forming early).
5. Voice Activity Detection (VAD) watches for speech + trailing silence. When it detects that one “spoken chunk” is complete and stable, it emits a final transcript and an endpoint event (utterance boundary).
6. After a final transcript, downstream steps (LLM reply, then TTS) start, and later the TTS audio is streamed or provided (currently in one chunk).
7. Connection stays open for more turns unless you or a timeout closes it.

Key traits:
- It does NOT wait for the whole audio file: it processes continuously (“incremental”).
- It expects UNCOMPRESSED PCM 16 kHz mono 16‑bit little‑endian frames (raw binary).
- If you feed compressed data (like WebM/Opus bytes) without decoding, it will misinterpret them as PCM sample values → garbage transcripts.

Simple analogy:
Think of the server as a clerk typing what you say while you’re still speaking, using short snippets of sound you hand over like flashcards. If you hand over scrambled encrypted cards (compressed audio) but the clerk thinks they are plain letters (PCM), the notes become nonsense.

--------------------------------------------------
2. Why did the WebM upload produce nonsense?
--------------------------------------------------
Because:
- WebM container + Opus codec are compressed, variable-length packets.
- Your current pipeline assumes each incoming chunk is raw PCM samples.
- Un-decoded Opus bytes look like random binary patterns → the ASR model “hears” digital noise → nonsensical repeated phrases.

Evidence that supports this (from broader ecosystem):
- Browsers’ MediaRecorder most often outputs WebM (Opus) at 48 kHz (MDN and community Q&A confirm typical Opus 48k default).
- Without a decode step (ffmpeg/libopus/etc.) you cannot directly treat those bytes as PCM.

--------------------------------------------------
3. Does your system currently support WebM?
--------------------------------------------------
Short answer: No (not in that streaming endpoint). It only supports raw PCM frames.

What “supporting WebM” would mean:
A. Detect format (e.g., first bytes show EBML header for WebM).
B. Extract Opus packets (demux).
C. Decode Opus → PCM (using libopus, ffmpeg, pyAV, or a wasm decoder).
D. Resample to 16 kHz mono if decode gives 48 kHz.
E. Feed resulting PCM frames into the existing incremental decode loop with correct timing.

If any of these steps (especially decoding) are missing, transcripts will be corrupted.

--------------------------------------------------
4. Can we do the “same streaming method” with WebM?
--------------------------------------------------
Yes, but you must add a real‑time decode layer.

Two architectural options:

Option 1: Client-Side Decode (Preferred for Lowest Latency)
- Use Web Audio API (AudioWorklet) to capture live microphone samples (float 32 at hardware / context rate).
- Downsample to 16 kHz.
- Convert to Int16.
- Send frames directly—no MediaRecorder, no compression.
Pros: Minimal added latency, simpler server, consistent sample rate.
Cons: A bit more JavaScript code (worklet), not just a simple MediaRecorder blob handler.

Option 2: Server-Side Decode (Opus/WebM ingestion)
- Keep using MediaRecorder.
- Send each recorded chunk (webm segment) over WebSocket OR a side channel.
- Server reassembles container fragments, decodes Opus frames as they arrive, yields PCM frames to the ASR pipeline.
Pros: Lowest frontend complexity.
Cons: Added server CPU, potential buffering delay (tens to low hundreds of ms), more complex error handling (partial container segments, jitter), need robust demux+decode in streaming mode.

Option 3: Hybrid
- Default to client PCM; fallback to server decode if browser lacks worklet features.

--------------------------------------------------
5. Latency considerations
Where delay comes from:
- Capture buffering (MediaRecorder often gives you chunks every 100–1000 ms).
- Opus decode time (small but non-zero).
- Resampling (if needed).
- ASR decode scheduling interval.

Why client PCM is often faster:
- You control frame size exactly (e.g., 20 ms).
- No container chunk latency; frames leave as soon as collected.
- No decode overhead server-side.

--------------------------------------------------
6. Is converting WebM to PCM “too slow”?
Not necessarily; real-time decode of Opus is inexpensive (Opus was designed for low-latency VoIP). The primary latency penalty usually appears because MediaRecorder batches data (e.g., ondataavailable every 250 ms or more), and you can’t get the samples earlier than when the browser emits them.

If you tune MediaRecorder timeslice to something small (e.g. 50–100 ms), you reduce batching but may increase overhead and possibly still get variability. You cannot get true 20 ms granularity easily with MediaRecorder. AudioWorklet gives that control.

--------------------------------------------------
7. Less-technical “visual” explanation (in words)
Imagine a conveyor belt:
- Current system wants single grapes (tiny raw PCM pieces).
- You were sending sealed jars filled with compressed grape juice concentrate (WebM/Opus).
- The machine expects grapes and tries to squeeze random bytes—so it outputs gibberish.
To fix this you either:
1. Open the jar yourself (client decode) and place individual grapes on the belt (raw PCM frames).
2. Add a jar-opening robot at the start of the belt (server decode).
Only then will the rest of the factory (VAD, transcription, TTS) produce correct results.

--------------------------------------------------
8. Decision summary
Requirement | Current | Possible | Recommendation
-----------|---------|----------|---------------
Raw PCM streaming | Supported | — | Keep
WebM/Opus direct | Not supported | Add decode layer | Only if you need MediaRecorder simplicity
Low latency | Good (once PCM) | Slightly worse with server decode | Use client worklet
Implementation complexity | Medium (already done) | Higher (demux + decode) | Start with client approach

--------------------------------------------------
9. Checklist if you choose to add WebM support later
Server side:
- Format sniffing
- Stream demux (libwebm/ffmpeg)
- Incremental Opus decode (libopus/ffmpeg)
- Resample to 16k mono (sox/ffmpeg or librosa/resampy)
- Frame slicing (exact 20 ms boundaries)
- Feed existing ASR loop
- Error handling + fallback logs

Client side (alternative):
- AudioWorklet capture
- Downsample & convert
- Frame pacing
- Backpressure / queue protection

--------------------------------------------------
10. Your direct questions answered succinctly
1. “Do we send small chunks and transcribe step by step without waiting for the whole file?”  
   Yes. You are already doing incremental streaming with partial and final transcripts. That’s exactly the design.
2. “Does the system support WebM?”  
   Not currently; only raw 16 kHz PCM frames. WebM requires additional decode logic that isn’t present.
3. “Is it possible to capture PCM at 16 kHz directly?”  
   Not directly via MediaRecorder. Yes via Web Audio API + AudioWorklet (capture float → downsample → Int16 → send).
4. “Will converting (decoding WebM → PCM) add delay?”  
   A little (tens of ms plus chunk batching). It’s often acceptable, but client-side PCM is still lower latency and simpler to reason about for timing.

--------------------------------------------------
11. Stopping here
You asked me to stop and let you decide. I have not proceeded with any code changes. Let me know which path you’d like:
A. Implement client AudioWorklet PCM streaming
B. Implement server-side WebM/Opus decode
C. Support both with an auto-detect
D. Something else

Just tell me the letter and I’ll take the next steps.