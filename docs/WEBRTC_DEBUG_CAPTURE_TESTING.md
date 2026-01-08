# WebRTC Debug Audio Capture Testing

## 🎯 Purpose
Capture **REAL microphone audio** from your laptop browser through the WebRTC pipeline to diagnose sample rate and resampling issues. This bypasses STT/LLM processing to isolate audio processing problems.

## 📁 What Gets Captured
The debug endpoint saves audio at **three distinct layers**:

### Layer 1: Raw 48kHz from WebRTC
- **File**: `debug_capture_{peer_id}_layer1_48khz_raw.wav`
- **Description**: Unprocessed audio frames directly from `aiortc` RTCPeerConnection
- **Format**: 48kHz, mono, int16 (native WebRTC format)
- **Purpose**: Verify what WebRTC receives from browser

### Layer 2: 48kHz Float Normalized
- **File**: `debug_capture_{peer_id}_layer2_48khz_float.wav`
- **Description**: Audio after conversion to float32 and normalization
- **Format**: 48kHz, mono, float32 → int16 for storage
- **Purpose**: Check normalization/clipping before resampling

### Layer 3: 16kHz Downsampled
- **File**: `debug_capture_{peer_id}_layer3_16khz.wav`
- **Description**: Audio after `scipy.signal.resample_poly` to 16kHz
- **Format**: 16kHz, mono, int16 (ready for Whisper STT)
- **Purpose**: Validate final resampling output quality

**Files saved to**: `/home/lumi/beautyai/logs/webrtc/debug_captures/`

---

## 🔧 Setup Instructions

### Step 1: Ensure API is Running
Check if the API service is running:
```bash
sudo systemctl status beautyai-api.service
```

If not running:
```bash
sudo systemctl start beautyai-api.service
```

Or for development mode (direct uvicorn):
```bash
cd /home/lumi/beautyai
source backend/venv/bin/activate
export VOICE_STREAMING_ENABLED=1 VOICE_STREAMING_PHASE4=1
python backend/run_server.py
```

Verify the debug endpoint is registered by checking startup logs:
```bash
sudo journalctl -u beautyai-api.service -n 50 | grep "WebRTC debug"
```

Expected output:
```
INFO WebRTC debug capture endpoints registered at /api/v1/webrtc/debug/voice-capture
```

---

### Step 2: SSH Tunnel from Laptop
From your **laptop** (not the server), create an SSH tunnel to forward port 8000:

```bash
ssh -L 8000:localhost:8000 lumi@<your-server-ip>
```

**What this does**:
- Forwards `http://localhost:8000` on your laptop to the server's port 8000
- Allows browser on laptop to access server as if it's local
- Required for WebRTC to access microphone (HTTPS/localhost requirement)

**Keep this terminal open** while testing!

---

### Step 3: Open Test Page in Browser
On your **laptop**, open your browser (Chrome/Firefox recommended) and navigate to:

```
http://localhost:8000/webrtc_voice_capture_test.html
```

---

## 🎤 Testing Procedure

### 1. Grant Microphone Permission
When you click **"Start Audio Capture"**, browser will request microphone access:
- Click **"Allow"**
- If denied, click the 🔒 lock icon in address bar and enable microphone

### 2. Start Capture
Click **"Start Audio Capture"** button:
- Status should change to **"Connected"**
- Peer ID will be displayed (e.g., `debug_a1b2c3d4`)
- Audio visualization will show waveform in real-time
- Frame counter will increment as audio is captured

### 3. Speak into Microphone
Say a clear test phrase, for example:
- **"How does laser hair removal work?"** (matches our test audio)
- **"Testing one two three four five"**
- **"مرحبا كيف حالك"** (Arabic greeting)

Speak for **5-10 seconds** to capture sufficient audio.

### 4. Stop Capture
Click **"Stop Audio Capture"** button:
- Connection closes
- Audio files are saved
- Log will show: `✅ Cleanup complete. Check logs/webrtc/debug_capture_<peer_id>_*.wav`

---

## 📊 Analyzing Results

### On the Server
SSH back to server and check captured files:
```bash
cd /home/lumi/beautyai/logs/webrtc/debug_captures
ls -lh debug_capture_*
```

Expected output:
```
-rw-r--r-- 1 lumi lumi  480K Nov 29 15:30 debug_capture_debug_a1b2c3d4_layer1_48khz_raw.wav
-rw-r--r-- 1 lumi lumi  480K Nov 29 15:30 debug_capture_debug_a1b2c3d4_layer2_48khz_float.wav
-rw-r--r-- 1 lumi lumi  160K Nov 29 15:30 debug_capture_debug_a1b2c3d4_layer3_16khz.wav
```

### Validate Audio Duration
Check each layer's duration:
```bash
cd /home/lumi/beautyai

# Layer 1: 48kHz raw
python3 -c "
import wave
with wave.open('logs/webrtc/debug_captures/debug_capture_debug_a1b2c3d4_layer1_48khz_raw.wav', 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    duration = frames / rate
    print(f'Layer 1: {frames} samples @ {rate} Hz = {duration:.2f}s')
"

# Layer 3: 16kHz resampled
python3 -c "
import wave
with wave.open('logs/webrtc/debug_captures/debug_capture_debug_a1b2c3d4_layer3_16khz.wav', 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    duration = frames / rate
    print(f'Layer 3: {frames} samples @ {rate} Hz = {duration:.2f}s')
"
```

**Expected behavior**: Duration should be identical across all layers (±0.01s tolerance).

**Problem indicators**:
- ❌ Duration mismatch between layers (1.84x stretch suggests sample rate error)
- ❌ Layer 3 significantly longer/shorter than Layer 1
- ❌ Garbled audio when playing back Layer 3

### Listen to Audio Files
Copy files to your laptop for playback:
```bash
# From your laptop
scp lumi@<server-ip>:/home/lumi/beautyai/logs/webrtc/debug_captures/debug_capture_debug_a1b2c3d4_*.wav ~/Downloads/
```

Play each file and compare:
- **Layer 1** (48kHz raw): Should sound clear and normal
- **Layer 2** (48kHz float): Should sound identical to Layer 1
- **Layer 3** (16kHz): Should sound slightly lower quality but SAME SPEED and CLEAR

**Red flags**:
- Layer 3 sounds slower/faster than Layer 1
- Layer 3 is garbled/robotic/distorted
- Any layer has crackling/static

---

## 🔍 Comparing with Test File
Compare the **real microphone capture** with the **test file** we've been debugging:

### Test File Behavior
```bash
cd /home/lumi/beautyai
python3 tests/webrtc/test_webrtc_audio_processor.py
```

Check test output for `laser_hair.wav`:
- Expected duration: ~2.41s @ 24kHz (57,863 samples)
- Actual output: 4.44s @ 16kHz (71,040 samples)
- Issue: **1.84x duration stretch** (garbled output)

### Real Microphone Comparison
After capturing 5 seconds of speech:
- Layer 1 (48kHz): Should be exactly 5.0s
- Layer 3 (16kHz): Should be exactly 5.0s

**If durations match** → Issue is specific to test file preprocessing  
**If Layer 3 is stretched** → Issue is in resampling logic (affects all audio)

---

## 🧪 Expected vs Actual Analysis

### ✅ Correct Behavior
```
Layer 1 (48kHz raw):   240,000 samples @ 48kHz = 5.00s
Layer 2 (48kHz float): 240,000 samples @ 48kHz = 5.00s
Layer 3 (16kHz):        80,000 samples @ 16kHz = 5.00s
```

**Resampling ratio**: 48kHz → 16kHz = 3:1 downsample
- Input samples: 240,000
- Output samples: 240,000 ÷ 3 = 80,000 ✅

### ❌ Current Test File Behavior
```
Input:  57,863 samples @ 24kHz = 2.41s
Output: 71,040 samples @ 16kHz = 4.44s  ← WRONG!
```

**Expected output** for 24kHz → 16kHz (3:2 downsample):
```
57,863 ÷ (24000/16000) = 57,863 × (2/3) = 38,575 samples @ 16kHz = 2.41s ✅
```

**But we get**: 71,040 samples = 4.44s (1.84x stretch)

---

## 🐛 What We're Debugging

### Hypothesis 1: Test File Preprocessing Issue
The `FileAudioTrack` in `test_webrtc_audio_processor.py` might be:
- Incorrectly reporting sample rate (reports 48kHz when it's 24kHz)
- Upsampling 24kHz → 48kHz WITHOUT adjusting reported rate
- Causing processor to downsample from wrong base rate

**Test**: Real microphone capture should NOT have this issue (browser always sends 48kHz).

### Hypothesis 2: General Resampling Bug
The `scipy.signal.resample_poly` might be:
- Using incorrect up/down ratios
- Not handling all sample rate combinations correctly

**Test**: If real mic has same 1.84x stretch, resampling is fundamentally broken.

### Hypothesis 3: Sample Rate Detection Failure
The `frame.sample_rate` might be:
- Returning wrong value for some audio sources
- Missing from frame object (defaulting incorrectly)

**Test**: Check debug logs for "Detected frame sample rate" messages during capture.

---

## 📝 What to Report

After testing, provide:

1. **Captured audio durations** (all 3 layers)
2. **Audio quality assessment** (clear, garbled, stretched?)
3. **Any duration mismatches** between layers
4. **Comparison** with test file behavior
5. **Debug logs** from backend during capture:
   ```bash
   sudo journalctl -u beautyai-api.service -f | grep "DEBUG-CAPTURE"
   ```

---

## 🔧 Troubleshooting

### Browser Can't Access Microphone
**Error**: `NotAllowedError: Permission denied`

**Solution**:
1. Check browser permissions: Settings → Privacy → Microphone
2. Ensure you're using `http://localhost:8000` (not IP address)
3. Try Chrome/Firefox (Safari has stricter WebRTC requirements)

### SSH Tunnel Connection Refused
**Error**: `channel 3: open failed: connect failed: Connection refused`

**Solution**:
1. Verify API is running on server: `sudo systemctl status beautyai-api.service`
2. Check port 8000 is listening: `sudo netstat -tlnp | grep 8000`
3. Try reconnecting SSH tunnel

### WebRTC Connection Fails
**Status**: "Connecting..." indefinitely

**Solution**:
1. Check browser console for errors (F12 → Console)
2. Verify ICE candidates are being sent: Look for "ICE candidate sent" in logs
3. Check backend logs: `sudo journalctl -u beautyai-api.service -n 100`

### No Audio Files Saved
**Issue**: Stop button clicked but no WAV files generated

**Solution**:
1. Check debug logs: `grep "DEBUG-CAPTURE" /var/log/syslog`
2. Verify capture directory exists: `ls -la /home/lumi/beautyai/logs/webrtc/`
3. Check permissions: `sudo chown -R lumi:lumi /home/lumi/beautyai/logs/`

---

## 🎯 Next Steps

After capturing and analyzing real microphone audio:

### If Duration Matches (No Stretch)
✅ **Issue is test file preprocessing specific**
- Focus on fixing `FileAudioTrack` in test utilities
- Update test to properly handle 24kHz → 48kHz upsampling
- Real-world usage is fine

### If Duration Still Stretches
❌ **Issue is in core resampling logic**
- Review `scipy.signal.resample_poly` parameters
- Check GCD ratio calculation in `_resample_audio()`
- Validate `frame.sample_rate` detection accuracy
- May need to switch resampling method

---

## 📚 Related Documentation
- [WEBRTC_VAD_SAMPLE_FLOW.md](./WEBRTC_VAD_SAMPLE_FLOW.md) - Complete audio pipeline flow
- [VOICE.md](./VOICE.md) - Voice streaming architecture
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - General debugging guide

---

**Remember**: This tool bypasses STT/LLM to isolate audio processing issues. Any problems found here are in the audio pipeline BEFORE transcription.
