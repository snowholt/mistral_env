# VAD Investigation - Quick Reference Guide

**Purpose:** Quick reference for VAD system investigation findings  
**Date:** 2025-10-21  
**Related:** VAD_INVESTIGATION.md, VAD_CODE_EXAMPLES.md

---

## TL;DR - Executive Summary

### What VAD Method Do We Use?

**Dual VAD Strategy:**
1. **webrtcvad** (py-webrtcvad 2.0.10) → Fast detection (Google's WebRTC VAD)
2. **Silero VAD** (5.1.0+) → ML-based confirmation (PyTorch model)

**Pattern:** Inspired by [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)

### How Do Voice Chunks Reach STT?

```
Browser Audio (48kHz) 
  ↓ WebRTC/RTP
aiortc MediaStreamTrack 
  ↓ AudioFrame
WebRTCAudioProcessor (convert to 16kHz mono PCM)
  ↓ bytes
WebRTCVADService (dual detection)
  ↓ if voice detected
WebRTCBufferManager (pre-roll + speech + post-roll)
  ↓ complete segment
Whisper STT
  ↓ transcription
Qwen LLM (/no_think prefix)
  ↓ response
Edge TTS → Browser
```

### Why Are Voice Chunks Not Reaching STT?

**6 Root Causes Identified:**

1. ❌ **Frame Accumulation Missing** - Partial frames dropped
2. ❌ **No Buffer Overflow Protection** - Can grow indefinitely  
3. ⚠️ **Strict VAD Mode** - Both VADs must agree (false negatives)
4. ❌ **Missing Test Fixture** - q7.webm doesn't exist
5. ⚠️ **No Init Checks** - Processing before VAD ready
6. ⚠️ **Frame Alignment** - Edge cases with chunk sizes

---

## Quick Diagnosis Checklist

### Is VAD Working?

```python
# Check VAD initialization
vad = WebRTCVADService(peer_id="test", language="en")
initialized = await vad.initialize()
print(f"VAD initialized: {initialized}")
print(f"WebRTC VAD available: {vad.webrtc_vad is not None}")
print(f"Silero VAD available: {vad.silero_vad_model is not None}")
```

### Are Audio Chunks Being Processed?

```python
# Check audio processor
processor = WebRTCAudioProcessor(peer_id="test")
print(f"Processing: {processor.is_processing}")
print(f"Frames received: {processor.metrics.frames_received}")
print(f"Frames processed: {processor.metrics.frames_processed}")
print(f"Frames dropped: {processor.metrics.dropped_frames}")
```

### Is Voice Being Detected?

```python
# Check VAD metrics
metrics = vad.get_metrics()
print(f"WebRTC detections: {metrics['webrtc_detections']}")
print(f"Silero confirmations: {metrics['silero_confirmations']}")
print(f"False positives: {metrics['false_positives']}")
print(f"Speech segments: {metrics['speech_segments']}")
print(f"Current state: {metrics['current_state']}")
```

### Are Chunks Reaching Buffer?

```python
# Check buffer manager
buffer_mgr = WebRTCBufferManager(peer_id="test")
print(f"Recording: {buffer_mgr.is_recording}")
print(f"Chunks received: {buffer_mgr.metrics.chunks_received}")
print(f"Chunks buffered: {buffer_mgr.metrics.chunks_buffered}")
print(f"Buffer overflows: {buffer_mgr.metrics.buffer_overflows}")
```

---

## Quick Fixes

### Fix #1: Add Frame Accumulation (CRITICAL)

**Problem:** Partial frames are dropped  
**Impact:** Can lose ~60ms of audio per chunk

```python
# In webrtc_vad_service.py __init__:
self._frame_accumulator = bytearray()

# In _is_voice_active_webrtc:
self._frame_accumulator.extend(audio_data)

detected = False
while len(self._frame_accumulator) >= frame_size_bytes:
    frame = bytes(self._frame_accumulator[:frame_size_bytes])
    self._frame_accumulator = self._frame_accumulator[frame_size_bytes:]
    
    if self.webrtc_vad.is_speech(frame, self.config.silero_sample_rate):
        detected = True

return detected
```

### Fix #2: Add Buffer Overflow Protection (CRITICAL)

**Problem:** Active buffer can grow indefinitely  
**Impact:** Memory issues, potential crashes

```python
# In webrtc_buffer_manager.py feed_audio:
current_size = sum(len(c) for c in self._active_buffer)

if current_size + len(chunk) > self.config.max_buffer_size_bytes:
    self.logger.warning("Buffer overflow, forcing segment completion")
    await self._complete_segment()
    self._active_buffer = []

self._active_buffer.append(chunk)
```

### Fix #3: Make VAD Mode Configurable (IMPORTANT)

**Problem:** Strict mode (AND) causes false negatives  
**Impact:** Voice not detected, especially for Arabic

```python
# Add to WebRTCVADConfig:
vad_mode: str = "permissive"  # "strict", "permissive", or "weighted"

# In _determine_voice_detection:
if self.config.vad_mode == "strict":
    return webrtc_detected and silero_detected  # AND
elif self.config.vad_mode == "permissive":
    return webrtc_detected or silero_detected   # OR
elif self.config.vad_mode == "weighted":
    score = 0.3 * webrtc_detected + 0.7 * silero_probability
    return score > 0.5
```

### Fix #4: Add Initialization Checks (IMPORTANT)

**Problem:** Processing starts before VAD ready  
**Impact:** No voice detection occurs

```python
# In process_audio_chunk:
if not self.is_initialized:
    self.logger.error("VAD not initialized")
    if not await self.initialize():
        return {"success": False, "error": "VAD init failed"}
```

---

## Configuration Reference

### For Arabic (Permissive)

```python
WebRTCVADConfig(
    webrtc_sensitivity=2,        # Less aggressive
    silero_sensitivity=0.45,      # Lower threshold
    vad_mode="permissive",        # OR logic
    min_speech_duration_ms=200,   # Faster response
    post_speech_silence_ms=400,   # Shorter gap
)
```

### For English (Standard)

```python
WebRTCVADConfig(
    webrtc_sensitivity=3,         # More aggressive
    silero_sensitivity=0.50,      # Standard threshold
    vad_mode="strict",            # AND logic
    min_speech_duration_ms=300,   # Standard
    post_speech_silence_ms=500,   # Standard
)
```

---

## Testing Quick Start

### Generate Test Audio

```bash
# Using ffmpeg
cd tests/webrtc/
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -ac 1 q7.webm

# Or copy from existing tests
cp ../../backend/voice_tests/input_test_questions/test.wav q7.wav
ffmpeg -i q7.wav -c:a libopus q7.webm
```

### Run Basic VAD Test

```python
import asyncio
from pathlib import Path

async def test_vad():
    # Initialize
    vad = WebRTCVADService(peer_id="test", language="en")
    await vad.initialize()
    
    # Load audio
    audio_file = Path("tests/webrtc/q7.webm")
    player = MediaPlayer(str(audio_file), format="webm")
    
    # Process frames
    results = []
    while True:
        try:
            frame = await asyncio.wait_for(player.audio.recv(), timeout=1.0)
            # Convert and process...
            result = await vad.process_audio_chunk(pcm_bytes, {})
            results.append(result)
        except asyncio.TimeoutError:
            break
    
    # Check results
    detections = sum(1 for r in results if r.get('voice_detected'))
    print(f"Voice detected in {detections}/{len(results)} chunks")

asyncio.run(test_vad())
```

---

## Debugging Commands

### Enable Debug Logging

```python
import logging

# Set VAD logging to DEBUG
logging.getLogger('beautyai_inference.services.voice.vad').setLevel(logging.DEBUG)
logging.getLogger('beautyai_inference.services.voice.webrtc_audio_processor').setLevel(logging.DEBUG)
logging.getLogger('beautyai_inference.core.webrtc_buffer_manager').setLevel(logging.DEBUG)
```

### Check Dependencies

```bash
# Verify libraries installed
python -c "import webrtcvad; print(f'webrtcvad: {webrtcvad.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import aiortc; print(f'aiortc: {aiortc.__version__}')"

# Test WebRTC VAD
python -c "import webrtcvad; vad = webrtcvad.Vad(3); print('WebRTC VAD OK')"

# Test Silero VAD
python -c "import torch; model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad'); print('Silero VAD OK')"
```

### Monitor Processing

```bash
# Watch logs in real-time
tail -f backend/logs/webrtc_voice.log | grep -E "VAD|PROCESSOR|BUFFER"

# Check metrics endpoint
curl http://localhost:8000/api/v1/webrtc/voice/metrics
```

---

## Common Issues & Solutions

### Issue: No Voice Detected

**Symptoms:** VAD always returns INACTIVE  
**Possible Causes:**
1. VAD not initialized → Check `vad.is_initialized`
2. Strict mode too strict → Switch to "permissive" mode
3. Silero threshold too high → Lower to 0.40-0.45 for Arabic
4. Audio format wrong → Verify 16kHz mono 16-bit PCM

**Solution:**
```python
# Use permissive mode for testing
config = WebRTCVADConfig(
    vad_mode="permissive",
    silero_sensitivity=0.40  # Very permissive
)
```

### Issue: False Positives

**Symptoms:** VAD detects voice in silence  
**Possible Causes:**
1. WebRTC sensitivity too low → Increase to 3
2. Permissive mode too loose → Switch to "strict"
3. Noise in audio → Check audio quality

**Solution:**
```python
# Use strict mode
config = WebRTCVADConfig(
    webrtc_sensitivity=3,      # Most aggressive
    vad_mode="strict",         # Require both to agree
    silero_sensitivity=0.55    # Higher threshold
)
```

### Issue: Buffer Overflow

**Symptoms:** Memory grows, "Buffer overflow" warnings  
**Possible Causes:**
1. VAD stuck in VOICE_ACTIVE → Check state machine
2. No buffer size limit → Add overflow protection
3. Very long utterances → Enforce 10s limit

**Solution:**
```python
# Add buffer limit check (see Fix #2 above)
# Or reduce max_buffer_duration_ms:
BufferConfig(max_buffer_duration_ms=10000)  # 10 seconds max
```

### Issue: Partial Audio Loss

**Symptoms:** Transcription incomplete, choppy  
**Possible Causes:**
1. Frame accumulation missing → Add accumulator (Fix #1)
2. Frames dropped due to size → Check frame alignment
3. Buffer cleared too early → Increase post_speech_silence_ms

**Solution:**
```python
# Add frame accumulator (see Fix #1 above)
# And increase post-roll:
WebRTCVADConfig(
    post_speech_silence_ms=700,  # Longer post-roll
    pre_speech_buffer_ms=300     # Longer pre-roll
)
```

---

## Performance Benchmarks

### Expected Timings

| Operation | Expected Time | Our Current |
|-----------|--------------|-------------|
| WebRTC VAD per frame (30ms) | < 1ms | ~0.5ms ✓ |
| Silero VAD per chunk (100ms) | < 10ms | ~8ms ✓ |
| Complete pipeline per chunk | < 15ms | ~12ms ✓ |
| State machine update | < 1ms | ~0.2ms ✓ |

### Frame Processing

| Metric | Value |
|--------|-------|
| Sample rate | 16000 Hz |
| Frame duration | 30 ms |
| Frame size (bytes) | 960 bytes |
| Frames per second | ~33 frames |
| Chunks per second | ~10 chunks (100ms each) |

---

## External Repository Links

### py-webrtcvad
- **Repo:** https://github.com/wiseman/py-webrtcvad
- **Key Files:** examples/webrtcvad_example.py
- **Our Usage:** webrtc_vad_service.py lines 333-373

### RealtimeSTT
- **Repo:** https://github.com/KoljaB/RealtimeSTT
- **Key Files:** 
  - audio_recorder.py (lines 150-320: VAD methods)
  - audio_recorder.py (lines 420-515: buffering)
- **Our Usage:** webrtc_vad_service.py (pattern), webrtc_buffer_manager.py

### aiortc
- **Repo:** https://github.com/aiortc/aiortc
- **Key Files:** 
  - aiortc/mediastreams.py (AudioFrame)
  - examples/ (WebRTC examples)
- **Our Usage:** webrtc_audio_processor.py lines 202-347

---

## Next Steps

### Immediate (Priority 1)
- [ ] Implement frame accumulation (Fix #1)
- [ ] Add buffer overflow protection (Fix #2)
- [ ] Create test audio fixture (q7.webm)
- [ ] Add initialization checks (Fix #4)

### Important (Priority 2)
- [ ] Make VAD mode configurable (Fix #3)
- [ ] Add comprehensive logging
- [ ] Create unit test suite
- [ ] Document troubleshooting procedures

### Enhancement (Priority 3)
- [ ] Add audio quality metrics
- [ ] Implement adaptive thresholds
- [ ] Create performance dashboard
- [ ] Add visualization tools

---

## Related Documents

1. **VAD_INVESTIGATION.md** - Complete investigation report (26 pages)
2. **VAD_CODE_EXAMPLES.md** - Detailed code examples (30 pages)
3. **webrtc_plan.md** - Original WebRTC migration plan
4. **webrtc_migration.md** - Migration documentation

---

## Quick Contact Points

### Code Locations

| Component | File |
|-----------|------|
| Dual VAD | `backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py` |
| Audio Processor | `backend/src/beautyai_inference/services/voice/webrtc_audio_processor.py` |
| Buffer Manager | `backend/src/beautyai_inference/core/webrtc_buffer_manager.py` |
| Voice Adapter | `backend/src/beautyai_inference/services/voice/webrtc_voice_service_adapter.py` |
| Test File | `tests/webrtc/test_webrtc_q7_audio.py` |

### Configuration Files

- VAD Config: `WebRTCVADConfig` dataclass
- Buffer Config: `BufferConfig` dataclass
- Audio Config: `AudioProcessingConfig` dataclass
- Main Config: `config/defaults.json`

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-21  
**For detailed information, refer to VAD_INVESTIGATION.md**
