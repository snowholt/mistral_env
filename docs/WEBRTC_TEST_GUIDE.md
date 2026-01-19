# 🎤 WebRTC Testing Guide - BeautyAI

**Last Updated:** November 25, 2024  
**Author:** Lumina Ashley ✨

---

## 📋 Quick Access URLs

All test pages are accessible through HTTPS on your production server:

### 🎯 **Primary Test Pages**

| Test Page | URL | Purpose |
|-----------|-----|---------|
| **Lean Capture Test** | `https://192.168.100.39:8443/test_lean_capture.html` | Production-ready hardened architecture with RT scheduling |
| **Enhanced Debug Tool** | `https://192.168.100.39:8443/test_simple.html` | Full debug suite with noise suppression & live transcription |
| **WebRTC Simple Test** | `https://192.168.100.39:8443/test_webrtc_simple.html` | Lightweight WebRTC test (alternate version) |

### 📍 **File Locations**

All test pages are now in: `/home/lumi/beautyai/backend/src/beautyai_inference/api/static/`

```
backend/src/beautyai_inference/api/static/
├── test_lean_capture.html       ✅ Moved from root!
├── test_simple.html
├── test_webrtc_simple.html
└── test_ws.html
```

---

## 🧪 Test Page Details

### 1. 🚀 **Lean Capture Test** (Production Architecture)

**URL:** `https://192.168.100.39:8443/test_lean_capture.html`

**Features:**
- ✅ Real-time scheduling (Round-Robin Priority 20)
- ✅ Bounded queue (5 frames, drop-oldest policy)
- ✅ RNNoise denoiser preset (`lean_rnnoise`)
- ✅ Acceptance criteria validation built-in
- ✅ <1% underrun rate, <2ms recv p99

**Testing Steps:**
```bash
# 1. Open the test page in browser
https://192.168.100.39:8443/test_lean_capture.html

# 2. Click "Connect & Capture"
# 3. Allow microphone access
# 4. Speak naturally for ~30 seconds (fan on for noise testing!)
# 5. Click "Disconnect" or wait for auto-completion

# 6. Validate results
cd /home/lumi/beautyai
./validate_lean_capture.sh
```

**What It Tests:**
- Queue underrun rates
- Receive latency (p99)
- Worker processing time (p99)
- Frame drops
- RT scheduling activation
- RNNoise denoising effectiveness

---

### 2. 🔧 **Enhanced Debug Tool** (Full Diagnostics)

**URL:** `https://192.168.100.39:8443/test_simple.html`

**Features:**
- ✅ 4-layer audio capture (raw → normalized → resampled → VAD-filtered)
- ✅ Live transcription feed with Whisper latency stats
- ✅ Browser noise suppression toggle
- ✅ Real-time VAD state monitoring
- ✅ Audio waveform visualizer
- ✅ Hard refresh button (cache clearing)

**Testing Steps:**
```bash
# 1. Open the debug tool
https://192.168.100.39:8443/test_simple.html

# 2. Click "Start Capture"
# 3. Monitor the status panel (left sidebar):
#    - Connection state
#    - VAD state (Silero probability)
#    - Live transcriptions
#    - Whisper latency stats

# 4. Speak the test script provided on the page
# 5. Click "Stop Capture"

# 6. Check saved audio files
ls -lh backend/logs/webrtc/debug_captures/
```

**Saved Audio Files:**
- `debug_capture_{peer_id}_layer1_48khz_raw.wav` - Raw WebRTC input
- `debug_capture_{peer_id}_layer2_48khz_float.wav` - Normalized audio
- `debug_capture_{peer_id}_layer3_16khz.wav` - Downsampled (all audio)
- `debug_capture_{peer_id}_layer4_16khz_vad_filtered.wav` - Speech only!

**Analysis:**
```bash
# Analyze noise reduction
python tools/analyze_audio_noise.py --compare --visualize

# Expected results:
# - <100 crackles @ 48kHz (was 7488 before RNNoise)
# - 80 Hz hum eliminated (visible in spectrograms)
```

---

### 3. 🎙️ **WebRTC Simple Test** (Lightweight)

**URL:** `https://192.168.100.39:8443/test_webrtc_simple.html`

**Features:**
- ✅ Simplified version of test_simple.html
- ✅ Same 4-layer capture
- ✅ Live transcription
- ✅ Minimal UI for quick testing

**Use When:**
- Quick sanity checks
- Low-resource testing
- Debugging specific WebRTC issues

---

## 🔗 API Endpoints Being Tested

### **Lean Capture Endpoint**
```
POST /api/v1/webrtc/lean/voice-capture/offer
```
**Body:**
```json
{
  "sdp": "<webrtc_sdp_offer>",
  "type": "offer",
  "preset": "lean_rnnoise"
}
```

**Features:**
- Hardened architecture with bounded queues
- RT scheduling (if CAP_SYS_NICE available)
- Drop-oldest policy (no blocking)
- RNNoise denoising preset

### **Debug Capture Endpoint**
```
POST /api/v1/webrtc/voice-capture/offer
```
**Features:**
- 4-layer audio debugging
- VAD (Voice Activity Detection) with Silero
- Live Whisper transcription
- Detailed metrics logging

---

## 📊 Validation & Metrics

### **Lean Capture Validation Script**

**Location:** `/home/lumi/beautyai/validate_lean_capture.sh`

**Checks:**
1. ✅ Underrun rate < 1%
2. ✅ Recv p99 < 2ms (warning if <5ms)
3. ✅ Worker p99 < 20ms
4. ✅ Queue peak ≤ 8 frames
5. ✅ Dropped frames < 1% of total
6. ✅ Total frames > 1000 (30s capture)
7. ✅ RT scheduling active

**Usage:**
```bash
cd /home/lumi/beautyai
./validate_lean_capture.sh
```

**Reports Location:**
```
reports/debug/webrtc/
├── queue_stats.json
├── pipeline_stats.json
└── <audio_files>
```

---

## 🛠️ Troubleshooting

### **Issue: Test page not loading**
```bash
# Check if API is running
sudo systemctl status beautyai-api.service

# Check nginx routing
sudo nginx -t
sudo systemctl status nginx

# Check API logs
sudo journalctl -u beautyai-api.service -f
```

### **Issue: 404 on test_lean_capture.html**
**Fixed!** ✅ The file has been moved to the static folder and the route updated.

```bash
# Verify file location
ls -lh /home/lumi/beautyai/backend/src/beautyai_inference/api/static/test_lean_capture.html

# Restart API to apply changes
sudo systemctl restart beautyai-api.service
```

### **Issue: High latency or underruns**
```bash
# Check system load
htop

# Reduce logging verbosity
export WEBRTC_DEBUG_VERBOSE=0

# Try minimal preset
# Edit test page: preset: "minimal"
```

### **Issue: Microphone not accessible**
- Ensure HTTPS is used (required for WebRTC)
- Check browser permissions
- Verify SSL certificate is trusted

---

## 🎯 Acceptance Criteria (Production Ready)

### **Lean Capture Must Meet:**
- [ ] Underrun rate < 1%
- [ ] Recv p99 < 2ms
- [ ] Worker p99 < 20ms
- [ ] Queue peak ≤ 8 frames
- [ ] Dropped frames < 1% of total
- [ ] Total frames > 1000 per 30s
- [ ] RT scheduling active (if possible)

### **Audio Quality Must Meet:**
- [ ] <100 crackles @ 48kHz (RNNoise target)
- [ ] 80 Hz hum eliminated (visual inspection)
- [ ] VAD correctly identifies speech segments
- [ ] Whisper transcription latency <500ms avg

---

## 📚 Related Documentation

- **Architecture:** `docs/WEBRTC_LEAN_ARCHITECTURE.md`
- **Quick Start:** `docs/WEBRTC_LEAN_QUICKSTART.md`
- **Debug Logs:** `docs/WEBRTC_DEBUG_LOGS_GUIDE.md`
- **Voice Pipeline:** `docs/VOICE.md`

---

## 🚦 Testing Workflow (Recommended)

### **Step 1: Quick Sanity Check**
```bash
# Use simple test for basic WebRTC functionality
open https://192.168.100.39:8443/test_simple.html
```

### **Step 2: Debug Session (if issues found)**
```bash
# Use enhanced debug tool with 4-layer capture
open https://192.168.100.39:8443/test_simple.html

# Analyze captured audio
python tools/analyze_audio_noise.py --compare --visualize
```

### **Step 3: Production Validation**
```bash
# Use lean capture test with hardened architecture
open https://192.168.100.39:8443/test_lean_capture.html

# Validate acceptance criteria
./validate_lean_capture.sh

# If all pass → Deploy to production! 🎉
```

---

## 💡 Tips for Testing

1. **Use Real Noise:** Run tests with fan/AC on to test denoising
2. **Test Edge Cases:** Try soft whispers, loud speech, long pauses
3. **Monitor System:** Watch CPU/GPU usage during tests
4. **Compare Layers:** Listen to layer3 vs layer4 to hear VAD filtering
5. **Check Transcriptions:** Verify Whisper latency is acceptable

---

## 🎉 Success Indicators

### **You're production-ready when:**
- ✅ All validation criteria pass
- ✅ Audio is clean (no crackles, no hum)
- ✅ VAD correctly filters silence
- ✅ Whisper latency is acceptable (<500ms avg)
- ✅ No dropped frames or queue overruns
- ✅ RT scheduling is active (if CAP_SYS_NICE available)

---

**Happy testing, babe!** 💜✨

For questions or issues, check the logs or reach out to the dev team!
