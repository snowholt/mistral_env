# WebRTC Simple Test Tool - Fixed ✅

## What Was Fixed

### 1. **Audio Wasn't Being Captured** ❌ → ✅

**Problems:**
- The original HTML stopped too quickly (before server captured enough frames)
- No feedback on whether audio was actually flowing
- No way to verify microphone was working
- Server needs time to accumulate frames before saving

**Solutions:**
- ✅ Added **5-second minimum capture time** (enforced in UI)
- ✅ Added **real-time audio visualization** (waveform display)
- ✅ Added **duration counter** so you know how long you've been recording
- ✅ Added **audio level meter** to confirm mic is picking up sound

---

### 2. **No Visual Feedback** ❌ → ✅

**Problems:**
- Couldn't tell if microphone was working
- Couldn't tell if audio was streaming
- No way to debug connection issues

**Solutions:**
- ✅ **Status panel** showing:
  - Peer ID
  - Connection state (Disconnected → Connecting → Connected)
  - Audio level percentage (real-time)
  - Recording duration
- ✅ **Canvas waveform visualization** (green waveform on black background)
- ✅ **Color-coded log messages** (info=blue, success=green, warn=orange, error=red)

---

### 3. **No Confirmation of File Saving** ❌ → ✅

**Problems:**
- Didn't know where files were saved
- Didn't know if save succeeded
- No feedback on what files were created

**Solutions:**
- ✅ **Download section** appears after successful save
- ✅ Shows exact file paths on server
- ✅ Lists all 3 layer files created:
  - `layer1_48khz_raw.wav` - Raw audio from WebRTC
  - `layer2_48khz_float.wav` - Normalized 48kHz
  - `layer3_16khz.wav` - Downsampled for Whisper
- ✅ Shows frame count captured

---

### 4. **Poor Audio Quality Settings** ❌ → ✅

**Problems:**
- Used browser defaults (may enable echo cancellation, noise suppression)
- These can distort audio for debugging purposes

**Solutions:**
- ✅ Disabled echo cancellation
- ✅ Disabled noise suppression
- ✅ Disabled auto gain control
- ✅ Explicitly requested 48kHz, mono audio

---

## How to Use

### Step 1: Open the File
```bash
# From your local machine, open in browser:
https://188.48.209.107:8443/test_webrtc_simple.html
```

Or if you've set up SSH tunnel:
```bash
# On local machine:
ssh -L 8443:localhost:8443 lumi@188.48.209.107

# Then open in browser:
https://localhost:8443/test_webrtc_simple.html
```

### Step 2: Start Capture
1. Click **"🎤 Start Capture"**
2. Allow microphone access when prompted
3. Wait for **"🎉 CONNECTED!"** message
4. You should see:
   - Green waveform moving on canvas
   - Audio level showing percentage
   - Duration counter increasing

### Step 3: Record for at Least 5 Seconds
- The tool enforces a **5-second minimum** to ensure server captures enough frames
- You'll see: `"⏰ Please capture for at least 5 seconds before stopping"`
- If you try to stop early, it will warn you

### Step 4: Stop and Save
1. After 5+ seconds, click **"🛑 Stop Capture"**
2. Server processes and saves audio files
3. Green download section appears with file locations
4. Check the log for confirmation

### Step 5: Retrieve Files
```bash
# On server, check the files:
cd /home/lumi/beautyai/logs/webrtc/debug_captures/
ls -la debug_capture_*

# Listen to them:
ffplay debug_capture_{peer_id}_layer1_48khz_raw.wav
ffplay debug_capture_{peer_id}_layer3_16khz.wav

# Or copy to local machine:
scp lumi@188.48.209.107:/home/lumi/beautyai/logs/webrtc/debug_captures/debug_capture_*.wav .
```

---

## What Gets Captured

The server saves **3 audio layers** for debugging:

| File | Sample Rate | Format | Purpose |
|------|-------------|--------|---------|
| `layer1_48khz_raw.wav` | 48 kHz | int16 | Raw audio as received from WebRTC browser |
| `layer2_48khz_float.wav` | 48 kHz | float32→int16 | After normalization |
| `layer3_16khz.wav` | 16 kHz | float32→int16 | Downsampled for Whisper STT |

This lets you debug **exactly where** audio degradation happens in the pipeline.

---

## Troubleshooting

### ❌ "Connection failed: failed"
**Problem:** ICE connection couldn't establish

**Solutions:**
1. Check server is running: `sudo systemctl status beautyai-api.service`
2. Check TURN server: `sudo systemctl status coturn`
3. Check nginx is proxying correctly: `sudo nginx -t && sudo systemctl status nginx`
4. Check browser console (F12) for ICE candidate errors

### ❌ "Audio level" shows 0% or very low
**Problem:** Microphone not working or too quiet

**Solutions:**
1. Check browser has microphone permission (click lock icon in address bar)
2. Check system microphone is not muted
3. Speak louder or move closer to mic
4. Try different microphone in browser settings

### ❌ Waveform is flat (straight line)
**Problem:** No audio signal reaching browser

**Solutions:**
1. Refresh page and allow microphone again
2. Check browser console for `getUserMedia` errors
3. Try different browser (Chrome/Edge recommended for WebRTC)
4. Check if another app is using microphone

### ❌ "No frames captured" or very few frames
**Problem:** Connection established but audio not flowing

**Solutions:**
1. Record for **at least 10 seconds** (not just 5)
2. Check server logs: `sudo journalctl -u beautyai-api.service -n 100`
3. Look for `[DEBUG-CAPTURE] received frame` messages
4. If no frames, might be codec negotiation issue (check SDP in logs)

---

## File Location

**Tool:** `/home/lumi/beautyai/test_webrtc_simple.html`  
**Captured Audio:** `/home/lumi/beautyai/logs/webrtc/debug_captures/`  
**Backend Endpoint:** `/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/webrtc_debug_capture.py`

---

## Key Improvements Summary

✅ **Real-time audio visualization** - See if mic is working  
✅ **5-second minimum capture** - Ensures server has time to save  
✅ **Status indicators** - Know exactly what's happening  
✅ **File path confirmation** - Know where audio was saved  
✅ **Better audio settings** - Disable processing that distorts audio  
✅ **Duration counter** - Know how long you're recording  
✅ **TURN server support** - Better NAT traversal  
✅ **Color-coded logs** - Easy to spot errors vs success  

---

## Next Steps for Debugging WebRTC Issues

1. **Use this tool to capture audio** from your VPN-connected browser
2. **Download the 3 layer files** to local machine
3. **Compare waveforms** using Audacity or similar:
   - Check if layer1 (raw WebRTC) already has issues → WebRTC codec problem
   - Check if layer2 has issues → Normalization problem
   - Check if layer3 has issues → Resampling problem
4. **Adjust WebRTC parameters** in `webrtc_debug_capture.py` if needed:
   - Opus codec settings
   - Jitter buffer size
   - Packet loss concealment
   - Sample rate conversion method

Good luck debugging, honey! 💚✨

---

**Created:** November 4, 2025  
**Author:** Lumina Ashley (with GitHub Copilot)
