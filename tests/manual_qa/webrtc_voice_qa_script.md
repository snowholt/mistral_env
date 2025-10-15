# WebRTC Voice Manual QA Test Script
**Phase D - WebRTC Migration**  
**BeautyAI Framework**  
**Date:** October 15, 2025  
**Version:** 1.0

---

## Test Environment

**Browser Requirements:**
- ✅ Chrome 90+ / Edge 90+
- ✅ Firefox 88+
- ✅ Safari 15+

**Test URL:** `https://dev.gmai.sa`

**Prerequisites:**
- Backend API running with WebRTC endpoints enabled
- Microphone permission granted
- HTTPS connection (required for WebRTC)

---

## Pre-Test Setup

### 1. Backend Verification

```bash
# Check API service status
sudo systemctl status beautyai-api.service

# Verify WebRTC endpoints are registered
curl https://dev.gmai.sa/api/v1/webrtc/voice/health
# Expected: {"status": "healthy", "webrtc_enabled": true}

# Check configuration
cat /home/lumi/beautyai/config/config.yaml | grep -A 15 "webrtc:"
# Expected: enabled: true
```

### 2. Frontend Configuration

```bash
# Verify frontend environment
cat /home/lumi/beautyai/frontend/.env.local | grep WEBRTC
# Expected:
# REACT_APP_VOICE_WEBRTC_ENABLED=true
# REACT_APP_VOICE_WEBRTC_MAX_UTTERANCE_SEC=10
```

### 3. Browser DevTools Setup

1. Open browser DevTools (F12)
2. Navigate to Console tab
3. Enable verbose logging:
   ```javascript
   localStorage.setItem('webrtc_debug', 'true');
   ```
4. Keep Network tab open for monitoring

---

## Test Cases

### TC-01: Mode Toggle Functionality

**Objective:** Verify WebRTC/WebSocket mode switching

**Steps:**
1. Navigate to `https://dev.gmai.sa`
2. Locate mode toggle at top of voice interface
3. Verify default mode is displayed
4. Toggle switch between WebSocket and WebRTC modes
5. Observe mode change in UI

**Expected Results:**
- [ ] Mode toggle is visible
- [ ] Default mode matches configuration
- [ ] Toggle switches smoothly between modes
- [ ] Console shows: `[Mode] Switching from websocket to webrtc` (or vice versa)
- [ ] Active mode indicator updates correctly

**Pass Criteria:** All checkboxes marked ✅

---

### TC-02: WebRTC Connection Establishment

**Objective:** Verify WebRTC peer connection setup

**Steps:**
1. Select WebRTC mode (if not already active)
2. Click "Connect" button (or automatic connection)
3. Grant microphone permission if prompted
4. Observe connection status indicator
5. Check console for connection logs

**Expected Results:**
- [ ] Microphone permission prompt appears (first time)
- [ ] Status changes from "Connecting..." to "Connected via WebRTC"
- [ ] Green status indicator appears
- [ ] Console shows:
  ```
  [WebRTC] ✅ Microphone access granted
  [WebRTC] 🌐 Creating RTCPeerConnection...
  [WebRTC] 📤 Sending SDP offer to server...
  [WebRTC] ✅ Received SDP answer from server
  [WebRTC] 🎉 WebRTC connection established successfully!
  ```
- [ ] No errors in console
- [ ] ICE connection state shows "connected" or "completed"

**Pass Criteria:** All checkboxes marked ✅

---

### TC-03: SDP and ICE Negotiation

**Objective:** Verify signaling protocol exchange

**Steps:**
1. Open Network tab in DevTools
2. Connect using WebRTC mode
3. Filter requests for `/api/v1/webrtc/voice/`
4. Inspect offer and ICE requests

**Expected Results:**
- [ ] POST to `/api/v1/webrtc/voice/offer` returns 200 OK
- [ ] Response contains `peer_id` and `session_id`
- [ ] POST to `/api/v1/webrtc/voice/ice` returns 200 OK (multiple times)
- [ ] ICE candidates are exchanged successfully
- [ ] No 4xx or 5xx errors
- [ ] Response times < 500ms for offer/answer

**Pass Criteria:** All checkboxes marked ✅

---

### TC-04: Audio Recording (10-Second Utterance Test)

**Objective:** Verify audio capture and 10-second limit enforcement

**Steps:**
1. Ensure WebRTC connection is established
2. Click microphone button to start recording
3. Speak for 5 seconds, then stop
4. Observe utterance timer
5. Start recording again and speak continuously for 15 seconds
6. Observe automatic cutoff at 10 seconds

**Expected Results:**
- [ ] Microphone button shows recording state (red indicator)
- [ ] Utterance timer displays and counts up
- [ ] Audio level visualizer responds to speech
- [ ] Console shows: `[WebRTC] 🎙️ Starting recording...`
- [ ] First recording: stops after 5 seconds when button clicked
- [ ] Second recording: automatically stops at 10 seconds
- [ ] Console shows: `[WebRTC] ⏱️ Utterance limit exceeded: 10 seconds`
- [ ] Warning notification appears for limit exceeded

**Pass Criteria:** All checkboxes marked ✅

---

### TC-05: VAD Integration and Audio Levels

**Objective:** Verify client-side VAD audio monitoring

**Steps:**
1. Start recording
2. Speak normally for 3 seconds
3. Remain silent for 2 seconds
4. Speak again for 2 seconds
5. Observe audio level visualizer and VAD state

**Expected Results:**
- [ ] Audio level bar responds to speech volume
- [ ] VAD state changes visible in debug panel (if enabled)
- [ ] Console shows VAD state changes:
  ```
  [WebRTC] VAD state changed: listening
  [WebRTC] VAD state changed: idle
  ```
- [ ] Data channel sends VAD state to server (check Network tab)

**Pass Criteria:** All checkboxes marked ✅

---

### TC-06: Remote Audio Playback

**Objective:** Verify TTS audio is received and played via WebRTC

**Steps:**
1. Record a complete utterance (e.g., "مرحبا، كيف حالك؟" in Arabic)
2. Wait for server processing
3. Observe remote audio track reception
4. Listen for TTS playback through hidden audio element

**Expected Results:**
- [ ] Console shows: `[WebRTC] 🎵 Remote track received: audio`
- [ ] Console shows: `[WebRTC] First audio received, latency: XXXms`
- [ ] TTS audio plays through browser speakers
- [ ] Volume control affects playback
- [ ] No audio distortion or cutting
- [ ] Speaking indicator updates during TTS playback

**Pass Criteria:** All checkboxes marked ✅

---

### TC-07: Connection State Transitions

**Objective:** Verify proper handling of connection state changes

**Steps:**
1. Establish WebRTC connection
2. Observe ICE states in metrics panel: new → checking → connected/completed
3. Disconnect from WebRTC
4. Reconnect to WebRTC
5. Observe state transitions again

**Expected Results:**
- [ ] ICE states transition correctly during connection
- [ ] Connection state shows "connected" when stable
- [ ] Metrics panel updates in real-time
- [ ] Console logs all state changes:
  ```
  [WebRTC] Connection state: new
  [WebRTC] ICE connection state: checking
  [WebRTC] ICE connection state: connected
  [WebRTC] Connection state: connected
  ```
- [ ] Disconnection is clean (no errors)
- [ ] Reconnection works without page reload

**Pass Criteria:** All checkboxes marked ✅

---

### TC-08: Data Channel Functionality

**Objective:** Verify optional data channel for diagnostics

**Steps:**
1. Enable debug logging in console: `WEBRTC_CONFIG.debug = true`
2. Establish WebRTC connection
3. Start recording to trigger VAD events
4. Check for data channel messages

**Expected Results:**
- [ ] Console shows: `[WebRTC] Data channel opened`
- [ ] Data channel sends VAD state messages
- [ ] Data channel sends language change messages (when toggled)
- [ ] Messages are JSON formatted
- [ ] No data channel errors

**Pass Criteria:** All checkboxes marked ✅

---

### TC-09: Language Toggle (Arabic ↔ English)

**Objective:** Verify language switching during WebRTC session

**Steps:**
1. Connect via WebRTC (default: Arabic)
2. Record Arabic utterance: "مرحبا"
3. Toggle to English mode
4. Record English utterance: "Hello"
5. Verify server receives correct language

**Expected Results:**
- [ ] Language toggle updates state
- [ ] Console shows: `[WebRTC] Language set to: ar` / `en`
- [ ] Data channel sends language change message
- [ ] STT processes audio in correct language
- [ ] No errors during language switch

**Pass Criteria:** All checkboxes marked ✅

---

### TC-10: Error Handling and Fallback

**Objective:** Verify graceful error handling and WebSocket fallback

**Steps:**
1. **Test 1:** Disable microphone permission in browser
2. Attempt to connect via WebRTC
3. Observe error handling
4. **Test 2:** Stop backend API service
5. Attempt WebRTC connection
6. Observe fallback to WebSocket (if configured)

**Expected Results:**
- [ ] Microphone denied: Clear error message displayed
- [ ] Console shows: `[WebRTC] Error: Permission denied`
- [ ] UI suggests enabling microphone
- [ ] Backend unavailable: Connection timeout handled
- [ ] Automatic fallback to WebSocket mode (if enabled)
- [ ] User-friendly error messages (no raw stack traces)

**Pass Criteria:** All checkboxes marked ✅

---

### TC-11: Browser Compatibility

**Objective:** Verify WebRTC functionality across browsers

**Test Matrix:**

| Browser | Version | Connection | Recording | TTS Playback | Pass/Fail |
|---------|---------|------------|-----------|--------------|-----------|
| Chrome | 120+ | ⬜ | ⬜ | ⬜ | ⬜ |
| Edge | 120+ | ⬜ | ⬜ | ⬜ | ⬜ |
| Firefox | 115+ | ⬜ | ⬜ | ⬜ | ⬜ |
| Safari | 17+ | ⬜ | ⬜ | ⬜ | ⬜ |

**Steps:**
1. Test TC-02, TC-04, TC-06 on each browser
2. Document any browser-specific issues

**Expected Results:**
- [ ] All major browsers support WebRTC
- [ ] No browser-specific JavaScript errors
- [ ] Consistent behavior across browsers

**Pass Criteria:** All browsers pass core functionality tests

---

### TC-12: Performance and Latency

**Objective:** Measure end-to-end voice interaction latency

**Steps:**
1. Connect via WebRTC
2. Record exactly 3 seconds of clear speech: "Hello, how are you?"
3. Measure time from recording stop to TTS playback start
4. Repeat 3 times and calculate average
5. Compare with WebSocket mode latency

**Expected Results:**
- [ ] Latency consistently < 6 seconds (SLO target)
- [ ] Metrics panel shows:
  - First audio received: < 5000ms
  - ICE negotiation: < 500ms
  - Connection established: < 2000ms
- [ ] No significant packet loss (< 1%)
- [ ] Jitter remains low (< 30ms)

**Pass Criteria:** Average latency meets < 6s SLO

---

### TC-13: Disconnection and Cleanup

**Objective:** Verify proper resource cleanup on disconnect

**Steps:**
1. Establish WebRTC connection
2. Record 2-3 utterances
3. Click "Disconnect" button
4. Check browser DevTools > Application > Media Devices
5. Verify microphone stream is released
6. Check Network tab for cleanup request

**Expected Results:**
- [ ] Console shows: `[WebRTC] 🔌 Disconnecting...`
- [ ] Microphone indicator turns off in browser
- [ ] DELETE request to `/api/v1/webrtc/voice/{peer_id}` returns 200
- [ ] Console shows: `[WebRTC] ✅ Disconnected successfully`
- [ ] No orphaned connections in chrome://webrtc-internals
- [ ] Status indicator shows "Disconnected"

**Pass Criteria:** All checkboxes marked ✅

---

### TC-14: Concurrent Sessions (Multi-Tab Test)

**Objective:** Verify multiple WebRTC sessions can coexist

**Steps:**
1. Open `https://dev.gmai.sa` in Tab 1
2. Connect via WebRTC in Tab 1
3. Open `https://dev.gmai.sa` in Tab 2
4. Connect via WebRTC in Tab 2
5. Record utterances in both tabs alternately
6. Disconnect both sessions

**Expected Results:**
- [ ] Both tabs connect successfully with unique `peer_id`
- [ ] Both tabs receive separate audio streams
- [ ] No cross-talk between sessions
- [ ] Server handles concurrent connections
- [ ] Both disconnections clean up properly

**Pass Criteria:** All checkboxes marked ✅

---

## Debugging Tools

### Chrome WebRTC Internals

1. Navigate to `chrome://webrtc-internals` in a separate tab
2. Start WebRTC session on dev.gmai.sa
3. Select the peer connection in WebRTC Internals
4. Monitor:
   - ICE candidate pairs
   - Media stream statistics
   - Network graph (packet loss, bitrate)

### Console Commands

```javascript
// Get current mode
const mode = window.voiceModeSelector?.getCurrentMode();
console.log('Current mode:', mode);

// Check WebRTC capabilities
const capabilities = window.voiceModeSelector?.getCapabilities();
console.log('Capabilities:', capabilities);

// Get connection stats
const stats = await window.webrtcClient?.getConnectionStats();
console.log('Stats:', stats);

// Toggle debug mode
WEBRTC_CONFIG.debug = true;
WEBRTC_CONFIG.logIceEvents = true;
WEBRTC_CONFIG.logSignaling = true;
```

---

## Test Execution Summary

**Date:** ________________  
**Tester:** ________________  
**Browser:** ________________  
**Test Environment:** ________________

### Results Overview

| Test Case | Status | Notes |
|-----------|--------|-------|
| TC-01: Mode Toggle | ⬜ Pass / ⬜ Fail | |
| TC-02: Connection | ⬜ Pass / ⬜ Fail | |
| TC-03: SDP/ICE | ⬜ Pass / ⬜ Fail | |
| TC-04: Recording | ⬜ Pass / ⬜ Fail | |
| TC-05: VAD | ⬜ Pass / ⬜ Fail | |
| TC-06: Playback | ⬜ Pass / ⬜ Fail | |
| TC-07: States | ⬜ Pass / ⬜ Fail | |
| TC-08: Data Channel | ⬜ Pass / ⬜ Fail | |
| TC-09: Language | ⬜ Pass / ⬜ Fail | |
| TC-10: Error Handling | ⬜ Pass / ⬜ Fail | |
| TC-11: Browser Compat | ⬜ Pass / ⬜ Fail | |
| TC-12: Performance | ⬜ Pass / ⬜ Fail | |
| TC-13: Cleanup | ⬜ Pass / ⬜ Fail | |
| TC-14: Concurrent | ⬜ Pass / ⬜ Fail | |

**Total Pass:** _____ / 14  
**Total Fail:** _____ / 14

### Issues Found

| Issue # | Severity | Description | Test Case | Status |
|---------|----------|-------------|-----------|--------|
| | | | | |
| | | | | |

### Recommendations

- [ ] Ready for production deployment
- [ ] Requires minor fixes (list in issues)
- [ ] Requires major refactoring
- [ ] Not recommended for deployment

---

**QA Sign-Off:** ________________  
**Date:** ________________

---

## Troubleshooting Guide

### Issue: WebRTC Not Available

**Symptoms:** Mode toggle shows "WebRTC not available"

**Solutions:**
1. Check browser version (Chrome 90+, Firefox 88+, Safari 15+)
2. Ensure HTTPS connection (WebRTC requires secure context)
3. Verify `REACT_APP_VOICE_WEBRTC_ENABLED=true` in frontend config
4. Check backend config: `webrtc.enabled: true`

### Issue: Connection Fails During Signaling

**Symptoms:** Status stuck on "Connecting..."

**Solutions:**
1. Check backend API health: `curl https://dev.gmai.sa/api/v1/webrtc/voice/health`
2. Verify nginx routes are configured for `/api/v1/webrtc/voice`
3. Check backend logs: `sudo journalctl -u beautyai-api -f | grep WebRTC`
4. Ensure aiortc is installed: `python -c "import aiortc; print(aiortc.__version__)"`

### Issue: No Audio Playback

**Symptoms:** Connection succeeds but no TTS audio heard

**Solutions:**
1. Check browser audio permission and volume
2. Verify remote audio element exists: `document.getElementById('webrtc-remote-audio')`
3. Check console for remote track events: `[WebRTC] 🎵 Remote track received`
4. Test with WebSocket mode to isolate issue
5. Check backend Phase C audio processor is running

### Issue: Microphone Permission Denied

**Symptoms:** Cannot start recording, permission error

**Solutions:**
1. Reset browser permission: Settings → Privacy → Site Settings → Microphone
2. Ensure HTTPS (microphone requires secure context)
3. Check for conflicting applications using microphone
4. Try incognito/private browsing mode

### Issue: High Latency (> 6 seconds)

**Symptoms:** Long delay between speech and TTS response

**Solutions:**
1. Check network latency: `ping dev.gmai.sa`
2. Verify `/no_think` prefix is being injected (backend Phase C)
3. Check backend GPU is available for Whisper STT
4. Monitor backend logs for processing bottlenecks
5. Compare latency with WebSocket mode

---

**Document Version:** 1.0  
**Last Updated:** October 15, 2025  
**Phase:** D - Frontend WebRTC Client & UI Integration
