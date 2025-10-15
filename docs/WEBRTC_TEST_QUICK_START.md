# 🚀 Quick Start Guide - WebRTC Voice Testing

## Accessing the WebRTC Test Interface

### URL
```
https://dev.gmai.sa/debug/test-webrtc
```

---

## ⚙️ Configuration Settings

### 1. Server URL
```
wss://dev.gmai.sa/api/v1/webrtc/voice
```

### 2. Language Options
- **Arabic (العربية)** - Default
- **English**

### 3. Voice Gender
- **Female** - Default
- **Male**

### 4. Test Modes

#### Simple Mode
- Basic connection testing
- Minimal logging
- Good for quick checks

#### Advanced Mode (Recommended)
- Detailed WebRTC statistics
- Connection quality metrics
- Performance monitoring
- **USE THIS FOR NORMAL TESTING**

#### Debug Mode
- Complete event logging
- ICE candidate tracking
- SDP inspection
- Network diagnostics
- **USE THIS FOR TROUBLESHOOTING**

---

## 📝 Testing Workflow

### Step 1: Open Test Page
```
Browser: Chrome/Edge (recommended)
URL: https://dev.gmai.sa/debug/test-webrtc
Accept certificate warning (dev.gmai.sa has valid cert)
```

### Step 2: Configure
1. Leave Server URL as default
2. Select Language (Arabic or English)
3. Select Voice Gender
4. Choose Test Mode: **Advanced** (recommended)

### Step 3: Connect
1. Click **"Connect to Server"** button
2. Allow microphone access when prompted
3. Wait for status to show: **🟢 Connected**
4. Monitor ICE candidates appearing

### Step 4: Test Voice
1. Click **"Start Voice Conversation"**
2. Speak clearly for 2-3 seconds
3. Watch audio visualizer (waveform should show activity)
4. Wait for response (~6 seconds target)
5. Listen to synthesized voice response

### Step 5: Monitor Metrics
Watch the Connection Metrics card:
- **Latency:** Target <100ms
- **Packet Loss:** Target <1%
- **Jitter:** Target <30ms
- **Bitrate:** Should show activity

### Step 6: Review Logs
1. Click "Logs" tab (default)
2. Monitor real-time events
3. Look for:
   - `✅ RTCPeerConnection created`
   - `✅ Microphone access granted`
   - `✅ Created and set local description`
   - `✅ Received and set remote description`

### Step 7: Export Data
1. Click **"Export Logs"** button
2. File downloads: `webrtc_test_YYYY-MM-DD.json`
3. Contains:
   - Full log history
   - Configuration used
   - Performance metrics
   - Timestamps

---

## 🔍 Debugging Features

### ICE Candidates Tab
- View local and remote ICE candidates
- Verify connectivity paths
- Check STUN/TURN usage

### WebRTC Stats Tab
- Detailed statistics table
- Network quality indicators
- Real-time updates

### SDP Tab
- View local offer (sent to server)
- View remote answer (received from server)
- Verify codec negotiation

---

## ⚠️ Troubleshooting

### Issue: Cannot Connect
**Solution:**
1. Check server is running:
   ```bash
   curl https://dev.gmai.sa/api/v1/webrtc/voice/health
   ```
2. Verify correct URL format (wss:// not ws://)
3. Check browser console for errors

### Issue: No Microphone Access
**Solution:**
1. Ensure HTTPS connection (required)
2. Allow microphone in browser settings
3. Check system microphone permissions
4. Try different browser (Chrome recommended)

### Issue: High Latency
**Solution:**
1. Check network connection
2. Monitor Packet Loss metric
3. Review Logs tab for delays
4. Export logs for analysis

### Issue: No Voice Response
**Solution:**
1. Verify WebRTC Stats show activity
2. Check Logs tab for errors
3. Switch to Debug mode
4. Export logs and review

---

## 📊 Expected Metrics (Healthy Connection)

```
Latency:     <100ms  (Excellent: <50ms)
Packet Loss: <1%    (Excellent: 0%)
Jitter:      <30ms  (Excellent: <10ms)
Bitrate:     >32kbps (Voice quality)
```

---

## 🎯 Success Indicators

### Visual Indicators
- ✅ Status shows "Connected" (green)
- ✅ Audio visualizer shows waveforms
- ✅ Metrics show numeric values (not dashes)
- ✅ Logs show "success" entries

### Functional Test
1. Speak Arabic: "مرحبا، كيف حالك؟"
2. Expected: ~6 second response
3. Expected: Clear synthesized Arabic voice
4. Expected: Transcription visible in logs

---

## 🔗 Additional Resources

### Backend Health Endpoint
```bash
curl https://dev.gmai.sa/api/v1/webrtc/voice/health

# Expected Response:
{
  "status": "healthy",
  "enabled": true,
  "active_connections": 0,
  "total_connections": 0
}
```

### Log Locations
- **Frontend Logs:** Check browser console (F12)
- **Backend Logs:** `sudo journalctl -u beautyai-api.service -f`
- **Exported Logs:** Downloads folder (JSON format)

### Alternative Test Pages
- **Simple Voice:** `/debug/simple-voice`
- **WebSocket Tester:** `/debug/voice-websocket-tester`
- **Streaming Live:** `/debug/streaming-live`

---

## 💡 Tips for Best Results

1. **Use Chrome/Edge:** Best WebRTC support
2. **Good Microphone:** Clear input = better recognition
3. **Stable Network:** WiFi or wired connection
4. **Quiet Environment:** Reduce background noise
5. **Speak Clearly:** 2-3 seconds, natural pace
6. **Monitor Metrics:** Watch for degradation
7. **Export Regularly:** Save logs for comparison

---

## 🆘 Need Help?

### Check Logs First
```bash
# Backend API logs
sudo journalctl -u beautyai-api.service -n 100

# WebUI logs
sudo journalctl -u beautyai-webui.service -n 100
```

### Verify Services
```bash
# Backend
sudo systemctl status beautyai-api.service

# WebUI
sudo systemctl status beautyai-webui.service
```

### Test Endpoints Manually
```bash
# WebRTC health
curl https://dev.gmai.sa/api/v1/webrtc/voice/health

# Test page
curl -I https://dev.gmai.sa/debug/test-webrtc
```

---

**Happy Testing! 🎉**

For technical documentation, see:
- `docs/PHASE_F_DEPLOYMENT_STEPS.md`
- `copilot_resources/reports/webrtcMigration/phase_f_report.md`
