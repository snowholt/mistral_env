# WebRTC Debug Capture - Nginx Configuration Complete ✅

**Date**: October 29, 2025  
**Version**: nginx config v2.1  
**Status**: Production Ready

---

## Summary

WebRTC debug capture endpoints have been successfully added to the production nginx configuration. These routes enable proper network access for WebRTC testing, bypassing SSH tunnel limitations that prevented ICE connectivity.

---

## What Was Added

### Debug Capture Endpoints (Added to Both Server Blocks)

#### 1. **SDP Offer Exchange**
```nginx
location /api/v1/webrtc/debug/voice-capture/offer {
    proxy_pass http://localhost:8000/api/v1/webrtc/debug/voice-capture/offer;
    # ... proxy headers ...
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    client_max_body_size 10M;
}
```
- **Purpose**: Exchange WebRTC session descriptions (SDP offer/answer)
- **Method**: POST
- **Timeout**: 30 seconds (sufficient for session setup)

#### 2. **ICE Candidate Exchange**
```nginx
location /api/v1/webrtc/debug/voice-capture/ice {
    proxy_pass http://localhost:8000/api/v1/webrtc/debug/voice-capture/ice;
    # ... proxy headers ...
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
}
```
- **Purpose**: Exchange ICE candidates for connection establishment
- **Method**: POST
- **Timeout**: 30 seconds

#### 3. **Session Cleanup**
```nginx
location ~ ^/api/v1/webrtc/debug/voice-capture/([a-zA-Z0-9_-]+)$ {
    proxy_pass http://localhost:8000/api/v1/webrtc/debug/voice-capture/$1;
    # ... proxy headers ...
    proxy_read_timeout 60s;
    proxy_send_timeout 60s;
}
```
- **Purpose**: Delete session and save captured audio files
- **Method**: DELETE
- **Timeout**: 60 seconds (allows time for file I/O)
- **URL Pattern**: `/api/v1/webrtc/debug/voice-capture/{peer_id}`

---

## Configuration Details

### Server Blocks Updated

1. **`dev.gmai.sa` (Frontend Server)**
   - Added debug capture routes BEFORE generic `/api/` location
   - Maintains proper route priority
   - Preserves existing WebRTC voice routes

2. **`api.gmai.sa` (API Server)**
   - Added same debug capture routes
   - Includes CORS headers for cross-origin access
   - Maintains backward compatibility with existing endpoints

### Route Priority Order

```
Critical Order (from highest to lowest priority):
1. WebRTC Debug Capture routes (/api/v1/webrtc/debug/voice-capture/*)
2. WebRTC Voice routes (/api/v1/webrtc/voice/*)
3. WebSocket routes (/api/v1/ws/*)
4. Generic API routes (/api/*)
5. Frontend routes (/)
```

**Why This Matters**: Nginx matches locations in order. More specific routes MUST come before generic catch-all routes to avoid being overridden.

---

## Testing Instructions

### Prerequisites
- API backend running (`sudo systemctl status beautyai-api.service`)
- Frontend running (`sudo systemctl status beautyai-webui.service`)
- Nginx reloaded (`sudo systemctl reload nginx`)

### Test Access Path

**Option 1: Via Frontend Domain**
```
https://dev.gmai.sa/webrtc_voice_capture_test.html
```

**Option 2: Via API Domain**
```
https://api.gmai.sa/webrtc_voice_capture_test.html
```

### What Should Happen

1. **Page Load**: Test interface appears with microphone access prompt
2. **Connection**: WebRTC connection establishes (green status indicator)
3. **Audio Capture**: Real-time waveform visualization shows mic input
4. **Session End**: Click "Stop & Save" → 3 WAV files generated
5. **Results**: Debug captures saved to `/home/lumi/beautyai/logs/webrtc/debug_captures/`

---

## Expected Output

After successful capture, you'll see:

```
Capture started! Speaking into microphone...
Connection State: connected
Audio Track State: live
Frames Captured: 142

Stop & Save clicked...
Session cleanup complete
Saved 3 audio files:
- debug_{peer_id}_48000Hz_raw.wav
- debug_{peer_id}_48000Hz_float.wav  
- debug_{peer_id}_16000Hz_resampled.wav
```

---

## Network Topology

### Previous Issue (SSH Tunnel)
```
Browser → localhost:8000 (SSH tunnel) → Server (192.168.100.39)
        ❌ ICE candidates unreachable through tunnel
```

### Current Solution (Proper Domain)
```
Browser → https://dev.gmai.sa → nginx → localhost:8000 → aiortc
        ✅ Direct network path for WebRTC
```

---

## Verification Commands

### Check Nginx Config
```bash
sudo nginx -t
```

### Reload Nginx (if needed)
```bash
sudo systemctl reload nginx
```

### View API Logs
```bash
sudo journalctl -u beautyai-api.service -f
```

### Check Debug Captures Directory
```bash
ls -lh /home/lumi/beautyai/logs/webrtc/debug_captures/
```

### Verify Route Accessibility
```bash
# Test health endpoint
curl -k https://dev.gmai.sa/api/v1/webrtc/debug/voice-capture/health

# Or via API domain
curl -k https://api.gmai.sa/api/v1/webrtc/debug/voice-capture/health
```

---

## Backup Information

- **Backup Created**: `gmai.sa.backup.20251029_200740`
- **Location**: `/home/lumi/beautyai/gmai.sa.backup.20251029_200740`
- **Purpose**: Rollback point if issues arise

### Rollback Command (if needed)
```bash
sudo cp /home/lumi/beautyai/gmai.sa.backup.20251029_200740 /etc/nginx/sites-enabled/gmai.sa
sudo nginx -t && sudo systemctl reload nginx
```

---

## Next Steps

### 1. Test Via Browser
Navigate to: `https://dev.gmai.sa/webrtc_voice_capture_test.html`

### 2. Capture Real Microphone Audio
- Allow microphone access
- Speak into mic for 5-10 seconds
- Click "Stop & Save"

### 3. Analyze Captured Files
```bash
cd /home/lumi/beautyai/logs/webrtc/debug_captures/
ls -lh debug_*

# Inspect with soxi (if installed)
soxi debug_*_48000Hz_raw.wav
soxi debug_*_16000Hz_resampled.wav
```

### 4. Compare Durations
- Expected: Real-time duration matches recording time
- Bug Case: 1.84x stretch (e.g., 5s recording → 9.2s file)

### 5. Update Documentation
After testing, document findings in:
- `WEBRTC_DEBUG_CAPTURE_TESTING.md` (add real mic results)
- `WEBRTC_VAD_SAMPLE_FLOW.md` (update with root cause if found)

---

## Related Documentation

- **Backend Code**: `/home/lumi/beautyai/backend/src/api/voice/webrtc/webrtc_debug_capture.py`
- **Frontend Test UI**: `/home/lumi/beautyai/backend/src/api/voice/webrtc/webrtc_voice_capture_test.html`
- **Quick Start Guide**: `/home/lumi/beautyai/docs/WEBRTC_DEBUG_CAPTURE_QUICK_START.md`
- **Complete Testing Guide**: `/home/lumi/beautyai/docs/WEBRTC_DEBUG_CAPTURE_TESTING.md`
- **Audio Flow Documentation**: `/home/lumi/beautyai/docs/WEBRTC_VAD_SAMPLE_FLOW.md`

---

## Technical Notes

### Why Nginx Routes Are Required

WebRTC requires:
1. **Direct IP Reachability**: ICE candidates must connect directly
2. **Proper TLS Context**: HTTPS required for getUserMedia()
3. **CORS Support**: Cross-origin requests for signaling

SSH tunnels fail because:
- Server advertises local IP (192.168.100.39) in ICE candidates
- Browser can't reach local server IP through tunnel
- STUN/TURN servers not configured (host-only candidates)

### Configuration Philosophy

- **Specific Routes First**: Debug capture routes before generic `/api/`
- **Appropriate Timeouts**: 30s for signaling, 60s for cleanup
- **Buffer Control**: `proxy_buffering off` for real-time data
- **Size Limits**: 10M for SDP payloads (generous for future growth)

---

## Troubleshooting

### Issue: 404 Not Found
**Cause**: Route not matching correctly  
**Fix**: Check nginx route order, ensure debug routes come before `/api/`

### Issue: 502 Bad Gateway
**Cause**: Backend not running  
**Fix**: `sudo systemctl start beautyai-api.service`

### Issue: Connection Timeout
**Cause**: Firewall blocking HTTPS or backend unreachable  
**Fix**: Check nginx error logs: `sudo tail -f /var/log/nginx/error.log`

### Issue: ICE Disconnect After 15s (Still Happening)
**Cause**: Need STUN/TURN servers for non-local connections  
**Solution**: Add STUN server configuration to aiortc RTCPeerConnection (future enhancement)

---

## Success Criteria

✅ Nginx config validates: `nginx -t` passes  
✅ Nginx reloaded successfully  
✅ API backend running  
✅ Frontend accessible at dev.gmai.sa  
✅ Debug test page loads without errors  
⏳ **Next**: Browser test with real microphone  

---

## Conclusion

The nginx configuration now properly routes WebRTC debug capture requests to the backend API, enabling real-world microphone testing without SSH tunnel limitations. This provides the necessary infrastructure to diagnose the 1.84x audio duration stretch issue with actual microphone input rather than pre-recorded test files.

**Status**: Ready for Browser Testing 🎤

---

**Author**: Lumina Ashley  
**For**: BeautyAI Inference Framework  
**Phase**: F - WebRTC Production Deployment (Debug Enhancement)
