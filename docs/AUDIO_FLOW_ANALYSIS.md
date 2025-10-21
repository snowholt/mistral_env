# WebRTC Audio Processing Flow Analysis

## Complete Audio Pipeline: From Browser to STT

This document provides a detailed analysis of how audio flows through the BeautyAI WebRTC voice pipeline, from the browser client to the Speech-to-Text model.

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        WebRTC Client (Browser)                        │
│                                                                        │
│  getUserMedia() → MediaStream → RTCPeerConnection → RTP Packets       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ RTP/SRTP Audio Packets
                                 │ (encoded audio, e.g., Opus)
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     Server: RTCPeerConnection                         │
│                    (aiortc - Python WebRTC Stack)                     │
│                                                                        │
│  Decodes RTP → AudioFrame objects (PCM samples)                       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ @pc.on("track") event
                                 │ MediaStreamTrack (audio)
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     WebRTCAudioProcessor                              │
│              (webrtc_audio_processor.py:202-248)                      │
│                                                                        │
│  RESPONSIBILITIES:                                                     │
│  1. Read AudioFrame objects from track (track.recv())                │
│  2. Convert to numpy array (frame.to_ndarray())                      │
│  3. Resample to 16kHz if needed                                      │
│  4. Convert stereo → mono                                            │
│  5. Convert float32 → int16 PCM bytes                                │
│  6. Enforce 10-second utterance limit                                │
│  7. Calculate audio levels (RMS)                                     │
│                                                                        │
│  OUTPUT: PCM bytes (16kHz, mono, s16le) + metadata                   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ Callback: on_audio_chunk(pcm_bytes, metadata)
                                 │ Triggered for each audio frame (~20-40ms chunks)
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   WebRTCVoiceServiceAdapter                           │
│           (webrtc_voice_service_adapter.py:280-318)                   │
│                                                                        │
│  ORCHESTRATOR - Coordinates all components:                           │
│  • Audio Processor (receives chunks)                                  │
│  • VAD Service (detects speech)                                       │
│  • Buffer Manager (accumulates audio)                                 │
│  • Voice Service (STT/LLM/TTS)                                        │
│                                                                        │
│  FLOW:                                                                 │
│  1. Receive chunk from audio processor                                │
│  2. Pass to VAD for speech detection                                  │
│  3. Pass to buffer with VAD state                                     │
│  4. When segment complete → trigger STT                               │
└────────┬──────────────────────────┬─────────────────────────────────┘
         │                          │
         │ Step 2: VAD             │ Step 3: Buffering
         ↓                          ↓
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│  WebRTCVADService    │  │    WebRTCBufferManager                   │
│  (Dual VAD)          │  │    (Pre/Post-roll Buffering)             │
│                      │  │                                          │
│  ┌────────────────┐ │  │  ┌────────────────────────────────────┐ │
│  │ WebRTC VAD     │ │  │  │ Pre-roll Buffer (300ms rolling)    │ │
│  │ (Fast, C-based)│ │  │  │ - Continuously stores recent audio  │ │
│  │ 10/20/30ms     │ │  │  │ - Captures speech onset            │ │
│  │ frames         │ │  │  └────────────────────────────────────┘ │
│  └────────┬───────┘ │  │                                          │
│           │         │  │  ┌────────────────────────────────────┐ │
│           ↓         │  │  │ Active Buffer (during speech)      │ │
│  ┌────────────────┐ │  │  │ - Accumulates speech frames        │ │
│  │ Silero VAD     │ │  │  │ - Started when VAD → VOICE_START   │ │
│  │ (ML, PyTorch)  │ │  │  │ - Copies pre-roll first            │ │
│  │ Confirms speech│ │  │  └────────────────────────────────────┘ │
│  └────────┬───────┘ │  │                                          │
│           │         │  │  ┌────────────────────────────────────┐ │
│  ┌────────▼───────┐ │  │  │ Post-roll Buffer (300ms after end) │ │
│  │ State Machine  │ │  │  │ - Continues after VOICE_END        │ │
│  │ INACTIVE       │ │  │  │ - Prevents word clipping           │ │
│  │ VOICE_START    │ │  │  └────────────────────────────────────┘ │
│  │ VOICE_ACTIVE   │ │  │                                          │
│  │ VOICE_END_PEND │ │  │  STATE-DRIVEN BUFFERING:               │
│  │ VOICE_END      │ │  │  • INACTIVE: pre-roll only             │
│  └────────────────┘ │  │  • VOICE_START: copy pre-roll → active │
│                      │  │  • VOICE_ACTIVE: append to active      │
│  OUTPUT: VAD state   │  │  • VOICE_END: add post-roll            │
│  + probabilities     │  │                                          │
└──────────────────────┘  │  OUTPUT: segment_ready flag            │
                          └──────────────┬───────────────────────────┘
                                         │
                                         │ ONLY when segment_ready=True
                                         │ (all post-roll frames added)
                                         ↓
                          ┌──────────────────────────────────────────┐
                          │    Adapter._on_segment_ready()           │
                          │    (Complete audio segment assembled)    │
                          │                                          │
                          │    • Concatenate all buffer chunks       │
                          │    • Convert bytes → numpy array         │
                          │    • Prepare metadata                    │
                          └──────────────┬───────────────────────────┘
                                         │
                                         │ Complete audio segment
                                         │ (pre-roll + speech + post-roll)
                                         ↓
┌──────────────────────────────────────────────────────────────────────┐
│                       SimpleVoiceService                              │
│                  (Speech-to-Text Pipeline)                            │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. STT (Whisper)                                                │  │
│  │    - transcribe_audio(audio_bytes, language)                   │  │
│  │    - Returns: transcript text                                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 2. LLM (with auto /no_think prefix)                            │  │
│  │    - generate_chat_response("/no_think " + transcript)         │  │
│  │    - Returns: assistant response                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 3. TTS (Edge TTS)                                               │  │
│  │    - synthesize_speech(response_text, language)                │  │
│  │    - Returns: audio bytes                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ Callbacks to send via data channel
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      RTCDataChannel                                   │
│                  (Client ← Server Messaging)                          │
│                                                                        │
│  Messages sent to client:                                             │
│  • {"type": "transcription", "text": "..."}                          │
│  • {"type": "assistant_response", "text": "..."}                     │
│  • {"type": "tts_audio", "audio_base64": "..."}                      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ WebSocket-like messaging
                                 ↓
                        ┌────────────────────┐
                        │  Browser Client    │
                        │  Receives results  │
                        └────────────────────┘
```

---

## 2. Detailed Flow with Code References

### Stage 1: RTCPeerConnection Setup

**File:** `webrtc_connection_pool.py:338-533`

```python
@pc.on("track")
async def on_track(track):
    """
    Triggered when client sends audio track.
    
    This is the entry point for all audio processing.
    Without consuming this track, the WebRTC connection
    will timeout or disconnect.
    """
    logger.info(f"[WebRTC] Received {track.kind} track for peer {peer_id}")
    
    if track.kind == "audio":
        # Create voice adapter for this peer
        adapter = WebRTCVoiceServiceAdapter(...)
        await adapter.initialize()
        
        # Start processing the audio track
        await adapter.start_voice_session(track)
```

**Key Points:**
- Client creates `RTCPeerConnection` and adds audio track
- Server receives track via `@pc.on("track")` event
- Track is a `MediaStreamTrack` object from aiortc
- Must be actively consumed or connection times out

---

### Stage 2: Audio Frame Processing

**File:** `webrtc_audio_processor.py:202-248`

```python
async def _process_audio_track(self, audio_track: MediaStreamTrack):
    """
    Main loop that reads frames from WebRTC track.
    
    Each frame represents 20-40ms of audio (depending on codec).
    """
    while self.is_processing:
        # Read next frame from track
        frame: AudioFrame = await asyncio.wait_for(
            audio_track.recv(),
            timeout=1.0
        )
        
        # Process the frame
        await self._process_audio_frame(frame)
```

**Frame Processing Steps:**

```python
async def _process_audio_frame(self, frame: AudioFrame):
    """
    Convert AudioFrame → PCM bytes for VAD/STT.
    
    AudioFrame properties:
    - sample_rate: Usually 48kHz from browser
    - layout.channels: Usually stereo (2 channels)
    - format: float32 samples
    - samples: Raw audio data
    """
    
    # 1. Convert to numpy
    audio_array = frame.to_ndarray()  # Returns float32 array, range [-1, 1]
    
    # 2. Resample 48kHz → 16kHz (Whisper expects 16kHz)
    if frame.sample_rate != 16000:
        audio_array = self._resample_audio(audio_array, frame.sample_rate, 16000)
    
    # 3. Convert stereo → mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)  # Average channels
    
    # 4. Convert float32 → int16 PCM
    audio_array = np.clip(audio_array, -1.0, 1.0)
    pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
    
    # 5. Check utterance duration limit
    chunk_duration = len(pcm_bytes) / (16000 * 2)  # seconds
    self.current_utterance_duration += chunk_duration
    
    if self.current_utterance_duration > 10.0:
        # Stop processing, trigger forced transcription
        await self.stop_processing()
        return
    
    # 6. Send to adapter
    metadata = {
        "sample_rate": 16000,
        "channels": 1,
        "duration_sec": chunk_duration,
        "utterance_duration_sec": self.current_utterance_duration
    }
    
    await self._on_audio_chunk(pcm_bytes, metadata)
```

**Chunk Size Analysis:**
- Browser typically sends 20ms frames
- At 16kHz, 20ms = 320 samples = 640 bytes (16-bit mono)
- At 48kHz, 20ms = 960 samples → resampled to 320 samples
- Typical chunk size: **640-960 bytes** per callback

---

### Stage 3: VAD Processing

**File:** `webrtc_vad_service.py:236-331`

```python
async def process_audio_chunk(self, audio_data: bytes, metadata: Dict) -> Dict:
    """
    Dual VAD processing: WebRTC + Silero.
    
    Flow:
    1. Quick WebRTC VAD check (1-2ms)
    2. If WebRTC detects voice, confirm with Silero (10-20ms)
    3. Update state machine based on results
    """
    
    # Stage 1: WebRTC VAD (fast path)
    webrtc_detected = False
    if self.webrtc_vad and self.config.enable_browser_hints:
        webrtc_detected = self._is_voice_active_webrtc(audio_data)
    
    # Stage 2: Silero VAD (confirmation path)
    silero_detected = False
    silero_probability = 0.0
    
    if webrtc_detected or not self.config.enable_browser_hints:
        # Only run Silero if WebRTC detected voice (or WebRTC disabled)
        silero_detected, silero_probability = self._is_silero_speech(audio_data)
    
    # Determine final detection
    voice_detected = self._determine_voice_detection(
        webrtc_detected,
        silero_detected,
        silero_probability
    )
    
    # Update state machine
    new_state = await self._update_state(voice_detected, metadata)
    
    return {
        "success": True,
        "voice_detected": voice_detected,
        "voice_state": new_state,
        "webrtc_detected": webrtc_detected,
        "silero_detected": silero_detected,
        "silero_probability": silero_probability
    }
```

**State Machine Transitions:**

```python
async def _update_state(self, voice_detected: bool, metadata: Dict) -> VADState:
    """
    State machine for speech detection.
    
    States and transitions:
    
    INACTIVE ──voice──→ VOICE_START ──300ms──→ VOICE_ACTIVE
                          ↑                         │
                          │                         │
                          └────silence─────────────┘
                                (resume)
    
    VOICE_ACTIVE ──silence──→ VOICE_END_PENDING ──500ms──→ VOICE_END
                                      ↑                         │
                                      │                         │
                                      └────voice────────────────┘
                                         (resume)              (finalize)
    """
    
    if voice_detected:
        if self.current_state == VADState.INACTIVE:
            # Start of potential speech
            self.current_state = VADState.VOICE_START
            self.speech_start_time = time.time()
        
        elif self.current_state == VADState.VOICE_START:
            # Check if sustained for 300ms
            duration_ms = (time.time() - self.speech_start_time) * 1000
            if duration_ms >= 300:
                self.current_state = VADState.VOICE_ACTIVE
                # Trigger callback: on_voice_start()
        
        elif self.current_state == VADState.VOICE_END_PENDING:
            # Voice resumed, cancel silence
            self.current_state = VADState.VOICE_ACTIVE
    
    else:  # No voice detected
        if self.current_state in [VADState.VOICE_START, VADState.VOICE_ACTIVE]:
            # Start silence timer
            self.silence_start_time = time.time()
            self.current_state = VADState.VOICE_END_PENDING
        
        elif self.current_state == VADState.VOICE_END_PENDING:
            # Check if silent for 500ms
            silence_ms = (time.time() - self.silence_start_time) * 1000
            if silence_ms >= 500:
                self.current_state = VADState.VOICE_END
                # Trigger callback: on_voice_end()
                
                # Reset to INACTIVE
                self.current_state = VADState.INACTIVE
    
    return self.current_state
```

---

### Stage 4: Buffer Management

**File:** `webrtc_buffer_manager.py:159-290`

```python
async def feed_audio(
    self,
    audio_chunk: bytes,
    vad_state: str,
    metadata: Dict
) -> Dict:
    """
    State-driven buffering based on VAD state.
    
    Buffer Strategy:
    1. Pre-roll buffer: Always active, stores last 300ms
    2. Active buffer: Accumulates during speech
    3. Post-roll buffer: Continues 300ms after end
    """
    
    # Always feed to pre-roll (rolling window)
    self._pre_roll_buffer.append(audio_chunk)
    
    if vad_state == VADState.INACTIVE.value:
        # Just maintain pre-roll, don't record
        return {"segment_ready": False, "status": "buffering_pre_roll"}
    
    elif vad_state == VADState.VOICE_START.value:
        if not self.is_recording:
            # Copy pre-roll to active buffer
            self._active_buffer = list(self._pre_roll_buffer)
            self.is_recording = True
            logger.info(f"Speech started, copied {len(self._active_buffer)} pre-roll frames")
        
        # Add current chunk
        self._active_buffer.append(audio_chunk)
        return {"segment_ready": False, "status": "recording_speech"}
    
    elif vad_state == VADState.VOICE_ACTIVE.value:
        # Continue recording
        self._active_buffer.append(audio_chunk)
        return {"segment_ready": False, "status": "recording_speech"}
    
    elif vad_state == VADState.VOICE_END_PENDING.value:
        # Silence detected, but might resume
        self._active_buffer.append(audio_chunk)
        return {"segment_ready": False, "status": "silence_pending"}
    
    elif vad_state == VADState.VOICE_END.value:
        # Start post-roll countdown
        if not self.is_in_post_roll:
            self.is_in_post_roll = True
            self._post_roll_counter = 0
        
        # Add post-roll frames
        if self._post_roll_counter < self.post_roll_frames:
            self._active_buffer.append(audio_chunk)
            self._post_roll_counter += 1
            return {"segment_ready": False, "status": "post_roll"}
        else:
            # Post-roll complete, finalize segment
            segment_data = await self._finalize_segment(metadata)
            return {
                "segment_ready": True,
                "status": "segment_complete",
                **segment_data
            }
```

**Buffer Finalization:**

```python
async def _finalize_segment(self, metadata: Dict) -> Dict:
    """
    Concatenate all buffered chunks and prepare for STT.
    
    Buffer contents:
    - Pre-roll frames (copied when speech started)
    - Speech frames (accumulated during VOICE_ACTIVE)
    - Post-roll frames (300ms after VOICE_END)
    
    Total: ~300ms + speech_duration + ~300ms
    """
    
    # Concatenate all chunks
    complete_audio = b''.join(self._active_buffer)
    
    # Calculate duration
    duration_sec = len(complete_audio) / (16000 * 2)  # 16kHz, 16-bit mono
    
    logger.info(
        f"Segment finalized: {len(complete_audio)} bytes, "
        f"{duration_sec:.2f}s, {len(self._active_buffer)} frames"
    )
    
    # Trigger callback
    if self._on_segment_ready:
        self._on_segment_ready(self.peer_id, complete_audio, metadata)
    
    # Reset for next segment
    self._active_buffer.clear()
    self.is_recording = False
    self.is_in_post_roll = False
    
    return {
        "audio_data": complete_audio,
        "metadata": {
            "duration_sec": duration_sec,
            "num_frames": len(self._active_buffer),
            "total_bytes": len(complete_audio)
        }
    }
```

---

### Stage 5: Speech-to-Text Processing

**File:** `webrtc_voice_service_adapter.py:356-458`

```python
async def _process_voice_with_service(
    self,
    audio_array: np.ndarray,
    metadata: Dict
) -> Dict:
    """
    Complete voice pipeline: STT → LLM → TTS.
    
    This is triggered when buffer finalization completes,
    meaning we have a complete speech segment ready.
    """
    
    # 1. Transcription (Whisper)
    transcription_result = await self.voice_service.transcribe_audio(
        audio_data=audio_array.tobytes(),
        language=self.language  # "ar" or "en"
    )
    
    if not transcription_result.get("success"):
        logger.error("Transcription failed")
        return transcription_result
    
    transcript = transcription_result.get("transcription", "")
    logger.info(f"Transcription: {transcript}")
    
    # Send transcription to client via data channel
    if self._on_transcription:
        self._on_transcription(self.peer_id, transcript)
    
    # 2. LLM Response (with auto /no_think prefix)
    llm_input = "/no_think " + transcript  # Auto-inject for faster response
    
    llm_result = await self.voice_service.generate_chat_response(
        user_message=llm_input,
        session_id=self.session_id,
        language=self.language
    )
    
    if not llm_result.get("success"):
        logger.error("LLM generation failed")
        return llm_result
    
    llm_response = llm_result.get("response", "")
    logger.info(f"LLM response: {llm_response[:100]}...")
    
    # Send LLM response to client via data channel
    if self._on_llm_response:
        self._on_llm_response(self.peer_id, llm_response)
    
    # 3. TTS (Edge TTS)
    tts_result = await self.voice_service.synthesize_speech(
        text=llm_response,
        language=self.language,
        gender=self.config.default_gender
    )
    
    if not tts_result.get("success"):
        logger.error("TTS synthesis failed")
        return tts_result
    
    tts_audio = tts_result.get("audio_data")
    
    # Send TTS audio to client via data channel
    if self._on_tts_audio:
        self._on_tts_audio(self.peer_id, tts_audio)
    
    return {
        "success": True,
        "transcription": transcript,
        "llm_response": llm_response,
        "tts_audio_size": len(tts_audio)
    }
```

---

## 3. Critical Timing Analysis

### Latency Breakdown

**Total time from speech → transcription:**

```
User speaks → 300ms VAD confirmation → 500ms silence detection → STT processing
              (VOICE_START)            (VOICE_END)              (~2-5 seconds)

Total: ~3-6 seconds from end of speech to transcription result
```

**Breakdown:**

1. **Speech Onset Detection:** ~20-100ms
   - First chunk arrives: 20ms
   - WebRTC VAD detects: 1-2ms
   - Silero confirms: 10-20ms
   - State → VOICE_START: immediate

2. **Transition to VOICE_ACTIVE:** 300ms
   - Must sustain voice detection for 300ms
   - Prevents spurious detections
   - Buffer starts accumulating during VOICE_START

3. **User Speaks:** Variable (0.5 - 10 seconds)
   - Audio accumulates in active buffer
   - VAD runs on each chunk (~every 20ms)

4. **Silence Detection:** 500ms
   - User stops speaking
   - VAD detects silence
   - State → VOICE_END_PENDING
   - Waits 500ms to confirm end

5. **Post-roll Buffer:** 300ms
   - After VOICE_END confirmed
   - Adds 300ms more audio
   - Prevents word clipping

6. **Buffer Finalization:** ~10-50ms
   - Concatenates all chunks
   - Prepares metadata

7. **STT Processing:** 2-5 seconds
   - Whisper model inference
   - Depends on segment length
   - GPU accelerated

8. **LLM Processing:** 1-3 seconds
   - With /no_think prefix
   - Faster than normal mode

9. **TTS Processing:** 0.5-2 seconds
   - Edge TTS synthesis
   - Streaming possible

---

### Why Audio Might Not Reach STT

**Problem #1: State Never Reaches VOICE_ACTIVE**

```
Chunk 1: WebRTC=True, Silero=False → voice_detected=False (both must agree)
Chunk 2: WebRTC=True, Silero=True  → voice_detected=True  (state=VOICE_START)
Chunk 3: WebRTC=False, Silero=True → voice_detected=False (WebRTC rejected)
         ↑
         State resets to INACTIVE because voice_detected=False
         Speech timer never reaches 300ms threshold
         NEVER REACHES VOICE_ACTIVE
```

**Solution:** Set `require_silero_confirmation=False` or reduce `min_speech_duration_ms`

---

**Problem #2: User Never Pauses**

```
User speaks continuously for 8 seconds without pause:

Time 0s:   Speech starts → VOICE_START
Time 0.3s: Sustained 300ms → VOICE_ACTIVE
Time 8s:   User still speaking (no pause)
           Buffer accumulates 8 seconds of audio
           BUT segment never finalized because no VOICE_END detected
           
Outcome: Audio buffered but never sent to STT
```

**Solution:** Implement streaming transcription with periodic buffer flush

---

**Problem #3: 10-Second Limit Cuts Off Speech**

```
User speaks for 12 seconds:

Time 0s:    Speech starts → VOICE_START
Time 0.3s:  → VOICE_ACTIVE
Time 10s:   Audio processor hits limit, calls stop_processing()
            Tries to get buffered audio with get_complete_segment()
            BUT: is_recording=True, so returns None
            
Outcome: 10 seconds of audio lost, nothing sent to STT
```

**Solution:** Fix utterance limit handler to force finalization

---

## 4. Debugging Checkpoints

To trace where audio chunks get stuck, check these log points:

```python
# Checkpoint 1: Frame received
[PROCESSOR] Received first audio frame for peer_abc123

# Checkpoint 2: Chunk converted to PCM
[PROCESSOR] About to send chunk: 640 bytes, callback=True

# Checkpoint 3: Adapter receives chunk
[DEBUG-CHUNK] Audio chunk: 640 bytes for peer_abc123

# Checkpoint 4: VAD processing
[DEBUG-WEBRTC-VAD] Frames checked=1, speech_frames=1
[DEBUG-VAD] State=voice_start, detected=True, silero_prob=0.8234

# Checkpoint 5: Buffer receives chunk
[DEBUG-BUFFER] Status=recording_speech, buffer_size=15

# Checkpoint 6: Segment finalized
[WebRTC] Segment finalized for peer_abc123: 9600 bytes, 0.60s, 15 frames

# Checkpoint 7: STT processing
[ADAPTER] ✓ Transcription complete for peer_abc123: 'Hello, how are you?'

# Checkpoint 8: Data channel send
[WebRTC] ✓ Sent transcription to peer_abc123
```

**If audio chunks don't reach STT:**

- Missing Checkpoint 1 → Track not received
- Missing Checkpoint 2 → Audio processor not running
- Missing Checkpoint 3 → Callback not wired
- Missing Checkpoint 4 → VAD not processing
- Missing Checkpoint 5 → Buffer not accumulating
- Missing Checkpoint 6 → State never reaches VOICE_END
- Missing Checkpoint 7 → STT service failure
- Missing Checkpoint 8 → Data channel issue

---

## 5. Configuration Summary

### Current Settings

```python
# VAD Configuration
webrtc_sensitivity = 0  # Most aggressive
silero_threshold = 0.30  # Permissive
require_silero_confirmation = True  # Both must agree ⚠️

# Timing Configuration
min_speech_duration_ms = 300  # State transition delay ⚠️
post_speech_silence_ms = 500  # End detection delay
pre_speech_buffer_ms = 200   # Pre-roll
post_roll_duration_ms = 300  # Post-roll

# Limits
max_utterance_duration_sec = 10  # Hard cutoff ⚠️
```

### Recommended Testing Settings

```python
# More permissive for debugging
webrtc_sensitivity = 0  # Keep aggressive
silero_threshold = 0.20  # Lower threshold
require_silero_confirmation = False  # Allow either VAD ✓

# Faster state transitions
min_speech_duration_ms = 100  # Faster VOICE_ACTIVE ✓
post_speech_silence_ms = 300  # Shorter silence wait ✓

# Keep buffers
pre_speech_buffer_ms = 300   # Larger pre-roll
post_roll_duration_ms = 300  # Keep post-roll

# Increase limit for testing
max_utterance_duration_sec = 30  # Longer utterances
```

---

## 6. Summary

**Audio Flow Overview:**
1. Browser → WebRTC → Server receives AudioFrame objects
2. AudioProcessor converts frames → PCM bytes
3. Adapter orchestrates VAD + Buffer
4. VAD detects speech boundaries
5. Buffer accumulates complete segments
6. STT processes finalized segments

**Critical Bottleneck:**
- Audio only sent to STT when segment finalized (VOICE_END + post-roll)
- If VAD state machine never progresses → audio never finalized
- Dual VAD requirement (both must agree) is too strict

**Recommended Fixes:**
1. Set `require_silero_confirmation=False`
2. Reduce `min_speech_duration_ms=100`
3. Implement streaming transcription
4. Fix utterance limit handler
5. Add VAD diagnostics endpoint
