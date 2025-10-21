# VAD Investigation Summary

**Date:** 2025-10-21  
**Task:** Investigate VAD methods and audio processing issues  
**Status:** ✅ Investigation Complete

---

## Quick Reference

### VAD Implementation

**Library Used:** Dual-stage VAD
1. **WebRTC VAD** (`webrtcvad>=2.0.10`) - Fast initial detection
2. **Silero VAD** (PyTorch Hub: `snakers4/silero-vad`) - ML-based confirmation

**Inspired by:** KoljaB/RealtimeSTT dual VAD pattern

---

## Key Findings

### 1. VAD Method Identification ✅

The project uses a **dual-stage VAD system**:

```python
# Stage 1: WebRTC VAD (fast path)
import webrtcvad
vad = webrtcvad.Vad(sensitivity=0)  # 0-3, 0=most aggressive
is_speech = vad.is_speech(frame, sample_rate=16000)

# Stage 2: Silero VAD (confirmation)
silero_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
probability = silero_model(audio_tensor, sample_rate=16000).item()
is_speech = probability > threshold  # 0.30 for testing
```

**Location:**
- `/backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py`
- `/backend/requirements.txt` (webrtcvad>=2.0.10)
- `/backend/webrtc_requirements.txt` (webrtcvad==2.0.10)

---

### 2. Audio Processing Pipeline ✅

**Complete Flow:**
```
Browser → RTCPeerConnection → AudioFrame → AudioProcessor → 
  PCM bytes → VAD → Buffer → STT (Whisper) → LLM → TTS
```

**Key Components:**
1. **WebRTCAudioProcessor:** Converts RTP frames to PCM
2. **WebRTCVADService:** Dual VAD detection
3. **WebRTCBufferManager:** Pre/post-roll buffering
4. **WebRTCVoiceServiceAdapter:** Orchestrates all components
5. **SimpleVoiceService:** STT/LLM/TTS pipeline

---

### 3. Root Causes for Audio Not Reaching STT 🔴

#### **Primary Issue:** Strict Dual VAD Requirement

**Current Configuration:**
```python
require_silero_confirmation: bool = True  # Both VADs must agree
```

**Problem:**
- If WebRTC VAD detects but Silero rejects → no detection
- If Silero detects but WebRTC rejects → no detection
- Intermittent detection prevents state progression

**Impact:** Voice state never reaches `VOICE_ACTIVE`, so buffer never accumulates properly

---

#### **Secondary Issue:** State Transition Timing

**Current Configuration:**
```python
min_speech_duration_ms: int = 300  # Must sustain voice for 300ms
post_speech_silence_ms: int = 500  # Must pause for 500ms
```

**Problem:**
- Short utterances may not exceed 300ms threshold
- Continuous speech without pauses never triggers finalization
- Audio accumulates but never sent to STT

**Impact:** Segments never finalized, transcription never triggered

---

#### **Tertiary Issue:** 10-Second Utterance Limit

**Current Configuration:**
```python
max_utterance_duration_sec: int = 10  # Hard cutoff
```

**Problem:**
- Stops audio processing after 10 seconds
- Tries to extract buffered audio, but buffer may not be finalized
- `get_complete_segment()` returns `None` if `is_recording=True`

**Impact:** Long utterances cut off, audio lost

---

## Referenced Repositories Analysis

### ✅ py-webrtcvad (wiseman/py-webrtcvad)
- **Purpose:** Python bindings for Google's WebRTC VAD
- **Usage:** Fast initial detection (1-2ms per frame)
- **Implementation:** Our code uses this correctly for fast path

### ✅ RealtimeSTT (KoljaB/RealtimeSTT)
- **Purpose:** Real-time STT with dual VAD
- **Pattern:** Dual VAD (WebRTC + Silero), pre-roll buffering, state machine
- **Implementation:** Our code closely follows this pattern

### ✅ aiortc (aiortc/aiortc)
- **Purpose:** Python WebRTC implementation
- **Usage:** RTCPeerConnection, MediaStreamTrack, AudioFrame
- **Implementation:** Correctly integrated for WebRTC transport

---

## Recommended Fixes

### Immediate Fixes (Quick Wins)

#### Fix #1: Relax Dual VAD Requirement
```python
# File: webrtc_vad_service.py:WebRTCVADConfig
require_silero_confirmation: bool = False  # Change from True
```
**Impact:** Allows either VAD to trigger detection, more permissive

---

#### Fix #2: Reduce Minimum Speech Duration
```python
# File: webrtc_vad_service.py:WebRTCVADConfig
min_speech_duration_ms: int = 100  # Change from 300
```
**Impact:** Faster state transitions, detects shorter utterances

---

#### Fix #3: Reduce Silence Threshold
```python
# File: webrtc_vad_service.py:WebRTCVADConfig
post_speech_silence_ms: int = 300  # Change from 500
```
**Impact:** Faster end-of-speech detection

---

### Medium-Term Improvements

#### Fix #4: Implement Streaming Transcription
Add periodic buffer flush for long utterances (5+ seconds)

#### Fix #5: Fix Utterance Limit Handler
Properly finalize buffer when 10-second limit reached

#### Fix #6: Add VAD Diagnostics
Create endpoint to monitor VAD state in real-time

---

## Testing Checklist

### Test Cases

- [ ] **Short Utterance (< 1s):** "Hello"
  - Current: May fail due to 300ms minimum
  - Expected: Should transcribe with fix

- [ ] **Long Continuous Speech (10s+):** Paragraph
  - Current: Waits for pause or hits limit
  - Expected: Stream transcription every 5s

- [ ] **Multiple Short Utterances:** "Yes. No. Maybe."
  - Current: May miss due to strict VAD
  - Expected: Detect all three with fixes

- [ ] **Noisy Environment:** Speech + background noise
  - Current: Dual VAD may over-filter
  - Expected: Silero should filter false positives

---

## Configuration Recommendations

### For Testing/Debugging

```python
# VAD Configuration
webrtc_sensitivity = 0  # Most aggressive
silero_threshold = 0.20  # Lower threshold (more permissive)
require_silero_confirmation = False  # Allow either VAD

# Timing Configuration
min_speech_duration_ms = 100  # Faster transitions
post_speech_silence_ms = 300  # Shorter silence wait
pre_speech_buffer_ms = 300   # Capture more pre-roll
post_roll_duration_ms = 300  # Prevent clipping

# Limits
max_utterance_duration_sec = 30  # Longer for testing

# Logging
log_vad_decisions = True  # Enable detailed logging
```

### For Production

```python
# VAD Configuration
webrtc_sensitivity = 1  # Balanced (not too aggressive)
silero_threshold = 0.30  # Moderate threshold
require_silero_confirmation = False  # Permissive mode

# Timing Configuration
min_speech_duration_ms = 200  # Balance speed vs spurious
post_speech_silence_ms = 400  # Reasonable pause detection
pre_speech_buffer_ms = 300   # Capture onset
post_roll_duration_ms = 300  # Prevent clipping

# Limits
max_utterance_duration_sec = 30  # Allow longer speech

# Logging
log_vad_decisions = False  # Disable in production
```

---

## Files Created

1. **VAD_INVESTIGATION_REPORT.md** - Comprehensive investigation report
   - VAD implementation details
   - Audio pipeline architecture
   - Root cause analysis
   - Referenced repositories comparison
   - Recommended fixes

2. **AUDIO_FLOW_ANALYSIS.md** - Detailed audio flow documentation
   - Complete pipeline diagrams
   - Code references for each stage
   - Timing analysis
   - Debugging checkpoints

3. **VAD_INVESTIGATION_SUMMARY.md** (this file) - Quick reference
   - Key findings
   - Quick fixes
   - Testing checklist

---

## Next Steps

1. **Apply immediate fixes** to VAD configuration
2. **Test with q7.wav** audio file
3. **Monitor VAD state transitions** with debug logging
4. **Validate STT transcription** works correctly
5. **Implement streaming transcription** for long utterances
6. **Add VAD diagnostics endpoint** for monitoring

---

## Conclusion

**Root Cause Identified:** ✅  
Strict dual VAD requirement (`require_silero_confirmation=True`) combined with 300ms minimum duration prevents voice state from reaching `VOICE_ACTIVE`, causing audio to never be sent to STT.

**Solution:** ✅  
Relax VAD requirements and reduce timing thresholds to allow more permissive detection during testing phase.

**Verification Method:** ✅  
Test with provided audio files and monitor VAD state transitions via debug logging.

---

**Investigation completed by:** AI Agent  
**Date:** 2025-10-21  
**Status:** Ready for implementation
