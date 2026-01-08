# ✅ WebRTC Debug Audio Capture System - COMPLETE

## 🎉 What's Been Created

### 1. Backend Debug Endpoint ✅
**File**: `backend/src/beautyai_inference/api/endpoints/webrtc_debug_capture.py`

**Features**:
- **No STT/LLM overhead** - Pure audio capture only
- **3-layer capture**:
  - Layer 1: Raw 48kHz from WebRTC (as received from browser)
  - Layer 2: 48kHz after float32 normalization
  - Layer 3: 16kHz after downsampling (ready for Whisper)
- **Automatic WAV file creation** in `logs/webrtc/debug_captures/`
- **Frame-by-frame statistics** with detailed logging
- **Session management** with peer IDs for tracking multiple captures

**Endpoints**:
```
POST   /api/v1/webrtc/debug/voice-capture/offer    - Create debug session
POST   /api/v1/webrtc/debug/voice-capture/ice      - Handle ICE candidates
DELETE /api/v1/webrtc/debug/voice-capture/{peer_id} - Stop and save audio
```

**Verification**: ✅ Registered in OpenAPI schema at http://localhost:8000/docs

---

### 2. Frontend Test Page ✅
**File**: `frontend/src/templates/webrtc_voice_capture_test.html`

**Features**:
- **Real-time audio visualization** with Web Audio API
- **Connection status tracking** (Connecting → Connected → Disconnected)
- **Frame counter** showing capture progress
- **Debug logging** with color-coded messages (info/warn/error)
- **Microphone permission handling** with getUserMedia
- **WebRTC peer connection** setup with offer/answer/ICE
- **Clean disconnect** that triggers file saving

**Access**: Open in browser via SSH tunnel at `http://localhost:8000/webrtc_voice_capture_test.html`

---

### 3. Comprehensive Documentation ✅

#### Full Testing Guide
**File**: `docs/WEBRTC_DEBUG_CAPTURE_TESTING.md`

**Contents**:
- Complete setup instructions (SSH tunnel, API startup, browser testing)
- Detailed explanation of what each audio layer captures
- Expected vs actual behavior analysis
- Duration validation commands
- Comparison with test file behavior
- Troubleshooting common issues
- Next steps based on results

#### Quick Start Card
**File**: `docs/WEBRTC_DEBUG_CAPTURE_QUICK_START.md`

**Contents**:
- One-command testing workflow
- Quick diagnostics (duration checking)
- Visual comparison table (good vs bad results)
- Rapid troubleshooting guide

---

## 🎯 Purpose & Goals

### What This Solves
Currently, test file `laser_hair.wav` (24kHz, 2.41s) produces **garbled transcription** with **1.84x duration stretch**:
```
Input:  57,863 samples @ 24kHz = 2.41s
Output: 71,040 samples @ 16kHz = 4.44s ← WRONG (should be 2.41s)
```

This debug system helps determine:
1. **Is it test-file-specific?** (FileAudioTrack preprocessing issue)
2. **Or general pipeline bug?** (Core resampling logic broken)

### How It Works
By capturing **REAL microphone audio** from browser:
- Browser always sends 48kHz (WebRTC standard)
- No FileAudioTrack upsampling confusion
- Direct comparison with known-good input

**If real mic works** → Issue is test file preprocessing  
**If real mic fails** → Issue is core resampling logic

---

## 📊 Testing Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Start API Service                                    │
│    sudo systemctl start beautyai-api.service            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. SSH Tunnel from Laptop                               │
│    ssh -L 8000:localhost:8000 lumi@server               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Open Browser on Laptop                               │
│    http://localhost:8000/webrtc_voice_capture_test.html │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Capture Audio                                         │
│    • Click "Start Audio Capture"                         │
│    • Allow microphone access                             │
│    • Speak clearly for 5-10 seconds                      │
│    • Click "Stop Audio Capture"                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Analyze Results                                       │
│    cd /home/lumi/beautyai/logs/webrtc/debug_captures    │
│    • Check durations of all 3 layers                     │
│    • Copy files to laptop for playback                   │
│    • Compare with test file behavior                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 What to Check

### Duration Validation
All three layers should have **IDENTICAL duration**:
```bash
cd /home/lumi/beautyai/logs/webrtc/debug_captures

for f in debug_capture_*.wav; do
  python3 -c "import wave; w = wave.open('$f', 'rb'); print('$f:', w.getnframes() / w.getframerate(), 's')"
done
```

**Expected (✅ GOOD)**:
```
layer1_48khz_raw.wav:   5.00s
layer2_48khz_float.wav: 5.00s
layer3_16khz.wav:       5.00s
```

**Problem Indicator (❌ BAD)**:
```
layer1_48khz_raw.wav:   5.00s
layer2_48khz_float.wav: 5.00s
layer3_16khz.wav:       9.20s  ← 1.84x STRETCH!
```

### Audio Quality
Copy files to laptop and play back:
```bash
# From laptop
scp lumi@server:/home/lumi/beautyai/logs/webrtc/debug_captures/*.wav ~/Downloads/
```

Listen to each layer:
- **Layer 1 (48kHz raw)**: Should be clear, normal speed
- **Layer 2 (48kHz float)**: Should sound identical to Layer 1
- **Layer 3 (16kHz)**: Slight quality loss but **SAME SPEED**, no distortion

---

## 🐛 Debug Scenarios

### Scenario A: Real Mic Works, Test Fails
```
Real Microphone Capture:
  Layer 1 (48kHz): 5.00s ✅
  Layer 3 (16kHz): 5.00s ✅
  
Test File (laser_hair.wav):
  Input (24kHz):  2.41s
  Output (16kHz): 4.44s ❌ (1.84x stretch)
```

**Diagnosis**: Issue is in `FileAudioTrack` test utility
- Test is upsampling 24kHz → 48kHz incorrectly
- Sample rate metadata mismatch
- Real production code (browser input) works fine

**Fix**: Update test preprocessing to properly handle 24kHz files

---

### Scenario B: Both Fail (Real Mic + Test)
```
Real Microphone Capture:
  Layer 1 (48kHz): 5.00s ✅
  Layer 3 (16kHz): 9.20s ❌ (1.84x stretch)
  
Test File:
  Output (16kHz): 4.44s ❌ (1.84x stretch)
```

**Diagnosis**: Core resampling bug in `_resample_audio()`
- `scipy.signal.resample_poly` calculation wrong
- GCD ratio computation error
- Affects ALL audio (production broken)

**Fix**: Review downsampling logic in `webrtc_audio_processor.py`

---

### Scenario C: Different Stretch Ratios
```
Real Microphone: 1.5x stretch
Test File:       1.84x stretch
```

**Diagnosis**: Multiple issues
- One bug in test preprocessing
- Another bug in resampling logic
- Both need fixing

---

## 📁 Files Created

```
backend/src/beautyai_inference/api/endpoints/
  └── webrtc_debug_capture.py         (280 lines) - Debug endpoint

frontend/src/templates/
  └── webrtc_voice_capture_test.html  (360 lines) - Test UI

docs/
  ├── WEBRTC_DEBUG_CAPTURE_TESTING.md (450 lines) - Full guide
  ├── WEBRTC_DEBUG_CAPTURE_QUICK_START.md (80 lines) - Quick ref
  └── WEBRTC_DEBUG_CAPTURE_COMPLETE.md (this file)

backend/src/beautyai_inference/api/app.py
  └── (modified) - Added debug router registration
```

---

## 🎯 Next Steps

### 1. Run the Test
Follow the quick start guide:
```bash
# Start API
sudo systemctl start beautyai-api.service

# From laptop - SSH tunnel
ssh -L 8000:localhost:8000 lumi@server

# Browser
http://localhost:8000/webrtc_voice_capture_test.html
```

### 2. Capture Real Audio
- Speak the **exact same phrase** as test file: "How does laser hair removal work?"
- This allows direct comparison

### 3. Analyze Results
Check if real microphone has same 1.84x stretch issue:
- **YES** → Core pipeline bug (Scenario B)
- **NO** → Test file preprocessing bug (Scenario A)

### 4. Report Findings
Provide:
- Duration output from all 3 layers
- Audio quality assessment (clear vs garbled)
- Comparison with test file behavior
- Debug logs: `sudo journalctl -u beautyai-api.service | grep "DEBUG-CAPTURE"`

---

## 🔗 Related Documentation

- **[WEBRTC_VAD_SAMPLE_FLOW.md](./WEBRTC_VAD_SAMPLE_FLOW.md)** - Complete audio pipeline trace
- **[WEBRTC_DEBUG_CAPTURE_TESTING.md](./WEBRTC_DEBUG_CAPTURE_TESTING.md)** - Full testing guide
- **[WEBRTC_DEBUG_CAPTURE_QUICK_START.md](./WEBRTC_DEBUG_CAPTURE_QUICK_START.md)** - Quick reference
- **[VOICE.md](./VOICE.md)** - Voice streaming architecture

---

## ✨ Summary

You now have a **complete debugging system** to isolate WebRTC audio processing issues:

✅ **Backend endpoint** capturing audio at 3 distinct layers  
✅ **Frontend test page** with real-time visualization  
✅ **Comprehensive documentation** with step-by-step guides  
✅ **Analysis framework** for comparing real mic vs test files  
✅ **Decision tree** for determining root cause (test vs pipeline)  

**Ready to test!** 🎤🐛🔍

All endpoints verified and registered. System is operational.
