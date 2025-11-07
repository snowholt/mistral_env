### Issue:
- Investigate on my `backend/src/beautyai_inference/api/endpoints/webrtc_debug_capture.py` and relevant files and identify the issues or propose better approaches on this topics:
1. Combine the stereo to mono conversion and normalization steps into a single processing function to improve efficiency, without adding noise or latency.
2. Improving the VAD state machine to better handle brief pauses in speech, reducing fragmentation of transcriptions.
3. Improving the downsampling process to minimize audio quality loss, possibly by exploring alternative resampling algorithms or libraries.
4. Improving the noise gate thresholding to adapt dynamically based on ambient noise levels detected in the initial frames of audio.
5. Improving the noise gate algorithm to use a more sophisticated method, such as spectral gating, instead of simple RMS thresholding.



### Input: 

#### 📋 Complete File List: WebRTC → VAD → Whisper Pipeline

#### **1. API Entry Point & Routing**
1. **app.py**
   - Registers the debug capture router
   - Mounts WebRTC endpoints at `/api/v1/webrtc/debug/voice-capture`
   - Serves test HTML page

---

#### **2. Main Endpoint (WebRTC → VAD → Whisper Integration)**
2. **webrtc_debug_capture.py** ⭐ **PRIMARY FILE**
   - **WebRTC Connection**: RTCPeerConnection, SDP offer/answer, ICE candidates
   - **Audio Capture**: Receives audio frames from browser (48kHz stereo)
   - **Audio Processing**:
     - **Stereo → Mono conversion** (Lines ~490-500)
     - **Int16 → Float32 normalization** (Lines ~500-508)
     - **Two-stage resampling 48kHz → 24kHz → 16kHz** (Lines ~510-525)
     - **Noise reduction** (frame RMS check, Lines ~527-530)
   - **VAD Integration**: 
     - Initializes WebRTCVADService (Lines ~166-184)
     - Processes audio through VAD (Lines ~548-558)
     - State machine handling (Lines ~560-590)
   - **Whisper Integration**:
     - Loads persistent Whisper model (Lines ~146-162)
     - Dual transcription: Layer 4 (16kHz) + Layer 5 (48kHz→16kHz) (Lines ~603-665)
     - Comparison logging (Lines ~643-651)
   - **5-Layer Audio Capture**:
     - Layer 1: 48kHz raw mono int16
     - Layer 2: 48kHz float32 normalized
     - Layer 3: 16kHz resampled
     - Layer 4: 16kHz VAD-filtered (speech-only)
     - Layer 5: 48kHz VAD-filtered (speech-only)
   - **File Saving**: WAV file export for all layers (Lines ~750-890)

---

#### **3. VAD Service (Speech Detection)**
3. **webrtc_vad_service.py**
   - **`WebRTCVADConfig` class** (Lines ~68-115): Configuration for VAD parameters
     - `silero_sensitivity`: Neural network threshold (default 0.5)
     - `post_speech_silence_ms`: Silence duration before ending speech (default 1000ms)
     - `min_sustained_speech_frames`: Consecutive frames needed (default 3)
     - `language_thresholds`: Per-language probability thresholds
     - `warmup_filter_duration_ms`: Initial noise filtering
   - **`WebRTCVADService` class** (Lines ~117-900): Main VAD implementation
     - **Silero VAD model loading** (Lines ~220-250): torch.hub neural network
     - **Audio processing** (Lines ~400-700): 16kHz PCM processing
     - **State machine** (Lines ~730-800): 
       - `INACTIVE` → `VOICE_START` → `VOICE_ACTIVE` → `VOICE_END_PENDING` → `VOICE_END`
       - Respects `post_speech_silence_ms` to group words together
     - **Metrics tracking** (Lines ~800-850): Speech duration, segments, probabilities

4. **__init__.py**
   - Exports VAD classes and enums
   - Makes `WebRTCVADService`, `WebRTCVADConfig`, `VADState` importable

---

#### **4. Whisper Service (Transcription)**
5. **base_whisper_engine.py**
   - **`BaseWhisperEngine` abstract class** (Lines ~28-500)
   - **`transcribe_audio_bytes()` method** (Lines ~233-400): 
     - Converts bytes to audio array
     - Resamples to 16kHz (Whisper requirement)
     - Runs inference
     - Returns text transcription

6. **whisper_large_v3_turbo_engine.py**
   - **`WhisperLargeV3TurboEngine` class**: faster-whisper implementation
   - Model: `openai/whisper-large-v3-turbo`
   - Uses CTranslate2 for optimized inference
   - Supports 4bit/8bit quantization

7. **transcription_factory.py**
   - Factory pattern for creating Whisper engines
   - Protocol interface for type checking

---

#### **5. Model Management (Persistent Loading)**
8. **persistent_model_manager.py**
   - **`PersistentModelManager` singleton class** (Lines ~35-560)
   - **`get_persistent_model_manager()` function** (Lines ~558-570): Global accessor
   - **`preload_models()` method** (Lines ~100-250): Loads models from config
   - **`get_whisper_model()` method** (Lines ~400-450): Returns preloaded Whisper instance
   - **Lazy loading**: Models load on first use, stay loaded for subsequent requests
   - **Memory management**: Tracks GPU/CPU memory usage

9. **preload_config.json**
   - Whisper Turbo model configuration:
     ```json
     {
       "model_id": "openai/whisper-large-v3-turbo",
       "engine_type": "faster_whisper",
       "device": "cuda",
       "compute_type": "float16",
       "preload_on_startup": true,
       "priority": 1
     }
     ```

---

#### **6. Dependencies (Audio Processing Libraries)**
**Imported in webrtc_debug_capture.py:**
- **`numpy`**: Array manipulation, audio normalization
- **`scipy.signal.resample_poly`**: Two-stage resampling (48→24→16kHz)
- **`scipy.signal.butter, filtfilt`**: Audio filtering (currently minimal use)
- **`wave`**: WAV file writing
- **`aiortc`**: WebRTC implementation (RTCPeerConnection, MediaStreamTrack)
- **`torch`**: Used by Silero VAD model (in webrtc_vad_service.py)
- **`faster-whisper`**: Used by Whisper engine (CTranslate2 backend)

---

#### **7. Configuration & Settings**
10. **VAD Configuration**: Set in webrtc_debug_capture.py lines 168-176
    ```python
    vad_config.silero_sensitivity = 0.3
    vad_config.language_thresholds = {"ar": 0.1, "en": 0.1}
    vad_config.min_sustained_speech_frames = 2
    vad_config.min_speech_duration_ms = 50
    vad_config.post_speech_silence_ms = 700
    vad_config.warmup_filter_duration_ms = 200
    ```

---

#### **8. Frontend Test Page**
11. **test_webrtc_simple.html**
    - Browser-based WebRTC test interface
    - Captures microphone audio (48kHz stereo by default)
    - Sends via WebRTC to backend endpoint

---

#### 🔄 Complete Pipeline Flow

```
Browser Microphone (48kHz stereo)
          ↓
[test_webrtc_simple.html] WebRTC client
          ↓
[aiortc RTCPeerConnection] Receives frames
          ↓
[webrtc_debug_capture.py] Main processing:
          │
          ├─► Stereo → Mono (channel average)
          ├─► Int16 → Float32 (normalize -1.0 to 1.0)
          ├─► Resample 48kHz → 24kHz → 16kHz (scipy.resample_poly)
          ├─► Noise gate (frame RMS check)
          ↓
[WebRTCVADService] Speech detection:
          │
          ├─► Silero VAD neural network (torch model)
          ├─► State machine (INACTIVE → START → ACTIVE → END_PENDING → END)
          ├─► Respects post_speech_silence_ms (groups words)
          ↓
Speech segments identified
          │
          ├─► Layer 4: 16kHz VAD-filtered buffer
          ├─► Layer 5: 48kHz VAD-filtered buffer
          ↓
[PersistentModelManager] Get preloaded Whisper
          ↓
[WhisperLargeV3TurboEngine] Transcription:
          │
          ├─► Layer 4 (16kHz direct) → transcribe_audio_bytes()
          ├─► Layer 5 (48kHz → 16kHz resample) → transcribe_audio_bytes()
          ├─► Comparison logging (match/different)
          ↓
Results:
          ├─► Console logs ([WHISPER-L4], [WHISPER-L5], [WHISPER-COMPARE])
          ├─► JSON export (transcriptions.json with layer tags)
          └─► WAV files (all 5 layers saved)
```

---

#### 📂 Output Files Created

**Location:** debug_captures

For each session (peer_id = `debug_xxxxx`):
- `debug_capture_debug_xxxxx_layer1_48000hz_raw.wav` (48kHz raw mono)
- `debug_capture_debug_xxxxx_layer2_48000hz_float.wav` (48kHz normalized)
- `debug_capture_debug_xxxxx_layer3_16khz.wav` (16kHz resampled)
- `debug_capture_debug_xxxxx_layer4_16khz_vad_filtered.wav` (16kHz speech-only)
- `debug_capture_debug_xxxxx_layer5_48khz_vad_filtered.wav` (48kHz speech-only)
- `debug_capture_debug_xxxxx_transcriptions.json` (transcription results)

---


### Guidelines:
- Do not create or generate any code,
- Proceed wit hinvestigation, even search on the internet if needed,to find better solutions or approaches to the issues mentioned.
- Provide a detailed analysis of the issues mentioned, and suggest improvements or alternative approaches where applicable.
- Output should be in single code block in markdown format.

If something is unclear, ask me questions in **one single code block** using Markdown format as well.  
Make sure each question is:
- Clear and easy to understand  
- Includes a simple explanation (why you are asking it)  
- Provides examples where possible  
- Suggests possible answers if applicable  
