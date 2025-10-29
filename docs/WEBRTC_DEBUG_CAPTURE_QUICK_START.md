# 🎤 WebRTC Debug Capture - Quick Start

## One-Command Testing

### 1. Start API (if not running)
```bash
sudo systemctl start beautyai-api.service
# OR for dev mode:
cd /home/lumi/beautyai && source backend/venv/bin/activate && python backend/run_server.py
```

### 2. SSH Tunnel (from laptop)
```bash
ssh -L 8000:localhost:8000 lumi@<your-server-ip>
```

### 3. Open Browser (on laptop)
```
http://localhost:8000/webrtc_voice_capture_test.html
```

### 4. Capture Audio
1. Click **"Start Audio Capture"**
2. Allow microphone access
3. Speak clearly for 5-10 seconds
4. Click **"Stop Audio Capture"**

### 5. Check Results (on server)
```bash
cd /home/lumi/beautyai/logs/webrtc/debug_captures
ls -lh debug_capture_*

# Check durations
for f in debug_capture_*.wav; do
  python3 -c "import wave; w = wave.open('$f', 'rb'); print('$f:', w.getnframes() / w.getframerate(), 's')"
done
```

### 6. Copy to Laptop
```bash
# From laptop
scp lumi@<server-ip>:/home/lumi/beautyai/logs/webrtc/debug_captures/debug_capture_*.wav ~/Downloads/
```

---

## What to Look For

### ✅ Good (durations match)
```
layer1_48khz_raw.wav:   5.00s
layer2_48khz_float.wav: 5.00s
layer3_16khz.wav:       5.00s
```

### ❌ Bad (duration mismatch)
```
layer1_48khz_raw.wav:   5.00s
layer3_16khz.wav:       9.20s  ← 1.84x STRETCHED!
```

---

## Compare with Test File

### Real Mic (should work correctly)
```bash
# Durations should match across layers
```

### Test File (currently broken)
```bash
cd /home/lumi/beautyai
python3 tests/webrtc/test_webrtc_audio_processor.py

# Shows 1.84x duration stretch:
# Input:  2.41s @ 24kHz
# Output: 4.44s @ 16kHz ← WRONG
```

**If real mic works but test fails** → Issue is test file preprocessing  
**If both fail** → Issue is core resampling logic

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Microphone not detected | Use Chrome/Firefox, ensure localhost:8000 |
| SSH tunnel fails | Check API is running on port 8000 |
| No audio files saved | Check `sudo journalctl -u beautyai-api.service` |
| WebRTC won't connect | Check browser console (F12) for errors |

---

## Debug Logs
```bash
# Watch capture in real-time
sudo journalctl -u beautyai-api.service -f | grep "DEBUG-CAPTURE"

# Check for errors
sudo journalctl -u beautyai-api.service -n 200 | grep -i error
```

---

**Full docs**: [WEBRTC_DEBUG_CAPTURE_TESTING.md](./WEBRTC_DEBUG_CAPTURE_TESTING.md)
