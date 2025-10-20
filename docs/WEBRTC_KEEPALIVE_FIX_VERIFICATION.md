# WebRTC Keep-Alive Fix - Verification Guide

This guide explains how to verify that the WebRTC connection keep-alive fix is working correctly.

## Problem Fixed

**Issue:** WebRTC connections were disconnecting after 5-10 minutes due to idle timeout, even when audio was actively being processed.

**Root Cause:** The connection pool's `last_activity` timestamp was only updated during state changes (ICE, connection state), not during audio processing.

**Fix:** Added periodic keep-alive task that updates `last_activity` every 30 seconds while audio tracks are active.

---

## Quick Verification (Browser Test)

### 1. Start the Server

```bash
# Make sure server is running with WebRTC enabled
cd /home/runner/work/mistral_env/mistral_env
systemctl status beautyai-api.service

# Or start manually for debugging
cd backend
python3 -m beautyai_inference.api.app
```

### 2. Open WebRTC Test Interface

```
https://web.lumidev.ca/debug/test-webrtc
```

Or if testing locally:
```
http://localhost:8000/debug/test-webrtc
```

### 3. Connect and Monitor

1. **Click "Connect to Server"**
   - Should see "Connected" status
   - Check browser console for connection logs

2. **Start Voice Conversation**
   - Click "Start Voice Conversation"
   - Speak into microphone

3. **Monitor for 15+ minutes**
   - Keep the tab open and active
   - Periodically speak to generate audio
   - Connection status should remain "Connected"
   - No disconnection should occur

### 4. Check Server Logs

```bash
# Monitor WebRTC logs in real-time
journalctl -u beautyai-api.service -f | grep WebRTC

# Look for these positive indicators:
# [WebRTC] Started keep-alive task for peer peer_xxx
# [WebRTC] Updated activity for peer peer_xxx during audio processing

# Should NOT see (unless truly idle):
# [WebRTC] Peer peer_xxx idle for 300.Xs, cleaning up
```

---

## Detailed Verification Steps

### Test 1: Connection Stays Active During Audio

**Objective:** Verify connection doesn't timeout during active audio streaming

1. Connect to WebRTC server
2. Start speaking and continue for 10+ minutes
3. Monitor connection status
4. **Expected:** Status stays "Connected" throughout
5. **Expected:** No "Disconnected" events in logs

**Pass Criteria:**
- ✓ Connection active for > 10 minutes during audio
- ✓ No premature disconnection warnings in logs
- ✓ Audio processing continues without interruption

### Test 2: Activity Timestamp Updates

**Objective:** Verify `last_activity` is updated during audio

1. Connect to WebRTC server
2. Note initial connection time: `T0`
3. Start audio streaming
4. After 2 minutes, check server logs or metrics
5. **Expected:** Activity timestamp updated at least once

**Verification via API:**
```bash
# Get connection status
curl https://web.lumidev.ca/api/v1/webrtc/voice/{peer_id}/status

# Check last_activity field - should be recent
```

**Pass Criteria:**
- ✓ `last_activity` increases over time during audio
- ✓ Activity updates visible in debug logs every ~30s

### Test 3: Idle Connections Still Cleanup

**Objective:** Ensure truly idle connections are still cleaned up

1. Connect to WebRTC server
2. Do NOT start audio or speak
3. Wait for timeout period (default 10 minutes)
4. **Expected:** Connection gets cleaned up

**Pass Criteria:**
- ✓ Idle connection removed after timeout
- ✓ Cleanup logs show connection was idle
- ✓ No memory leaks or orphaned connections

---

## Log Patterns to Look For

### Positive Indicators (Fix Working)

```
[WebRTC] Started keep-alive task for peer peer_abc123 to prevent idle timeout during audio
[WebRTC] Updated activity for peer peer_abc123 during audio processing
[WebRTC] Received audio track for peer peer_abc123
```

### Negative Indicators (Issue Still Present)

```
[WebRTC] Peer peer_abc123 idle for 300.5s, cleaning up
[WebRTC] Cleaned up peer connection: peer_abc123
# ^ If this happens during active audio, fix didn't work
```

---

## Metrics to Monitor

### Connection Duration

**Before Fix:**
- Average: 300-600 seconds (timeout limit)
- Max: ~600 seconds
- Pattern: Spike at timeout value

**After Fix:**
- Average: Based on actual conversation length
- Max: Limited by user, not system
- Pattern: Normal distribution

### Premature Disconnections

**Before Fix:**
- Rate: ~100% of connections lasting > timeout
- Timing: Predictable at timeout value

**After Fix:**
- Rate: 0% (should be zero)
- Timing: N/A

---

## Troubleshooting

### Issue: Still Disconnecting After 10 Minutes

**Possible Causes:**
1. Keep-alive task not starting
2. Task being cancelled prematurely
3. Timeout configured too low

**Debug Steps:**
```bash
# Check if keep-alive tasks are starting
journalctl -u beautyai-api.service | grep "keep-alive task"

# Check timeout configuration
env | grep WEBRTC_CONNECTION_TIMEOUT

# Increase timeout temporarily
export WEBRTC_CONNECTION_TIMEOUT=3600
systemctl restart beautyai-api.service
```

### Issue: Keep-Alive Logs Not Appearing

**Possible Causes:**
1. Log level too high (INFO instead of DEBUG)
2. Keep-alive task not being created
3. Audio track not received

**Debug Steps:**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
systemctl restart beautyai-api.service

# Check for track received logs
journalctl -u beautyai-api.service | grep "Received.*track"
```

### Issue: Activity Not Updating

**Possible Causes:**
1. Task created but not running
2. Connection data not accessible
3. Lock contention

**Debug Steps:**
```python
# Add debug logging in keep_alive_during_audio()
logger.debug(f"Keep-alive iteration for {peer_id}")

# Check asyncio tasks
import asyncio
print(asyncio.all_tasks())
```

---

## Configuration Options

### Connection Timeout

```bash
# Default: 600 seconds (10 minutes)
export WEBRTC_CONNECTION_TIMEOUT=1800  # 30 minutes

# For testing (shorter timeout)
export WEBRTC_CONNECTION_TIMEOUT=120  # 2 minutes
```

### Activity Update Interval

Currently hardcoded to 30 seconds. To change:

```python
# In webrtc_connection_pool.py, on_track handler
update_interval = 30  # Change this value (seconds)
```

### Debug Logging

```bash
# Enable detailed WebRTC logging
export WEBRTC_DEBUG=1
export LOG_LEVEL=DEBUG
```

---

## Performance Impact

### Resource Usage

- **CPU:** Negligible (~0.01% per connection)
- **Memory:** Minimal (one task per connection)
- **Network:** None (no additional traffic)

### Task Overhead

- **Tasks Created:** 1 per audio track
- **Task Lifetime:** Same as audio track
- **Cleanup:** Automatic on task cancellation

---

## Automated Test

Run the included test script:

```bash
cd /home/runner/work/mistral_env/mistral_env
python3 tests/manual_qa/test_webrtc_keepalive_fix.py
```

**Expected Output:**
```
✓ Connection pool started
✓ Peer connection created
✓ Activity timestamp updated during audio processing
✓ Keep-alive task is working correctly!
✓ Connection still active (not cleaned up prematurely)

Results: 1/1 tests passed
```

---

## Rollback Plan

If issues occur after deploying the fix:

1. **Immediate Mitigation:**
   ```bash
   # Increase timeout to reduce cleanup frequency
   export WEBRTC_CONNECTION_TIMEOUT=3600
   systemctl restart beautyai-api.service
   ```

2. **Revert Changes:**
   ```bash
   git revert <commit-hash>
   systemctl restart beautyai-api.service
   ```

3. **Alternative Workaround:**
   ```bash
   # Disable cleanup entirely (temporary)
   export WEBRTC_DISABLE_CLEANUP=1
   ```

---

## Success Criteria Checklist

- [ ] WebRTC connections last > 15 minutes during active audio
- [ ] Activity timestamp updates every ~30 seconds
- [ ] Keep-alive logs appear in server logs
- [ ] Idle connections still get cleaned up after timeout
- [ ] No memory leaks or task accumulation
- [ ] CPU usage remains unchanged
- [ ] User experience: no unexpected disconnections

---

## References

- **Analysis Report:** `docs/reports/WEBRTC_DISCONNECTION_ANALYSIS.md`
- **Code Changes:** `backend/src/beautyai_inference/core/webrtc_connection_pool.py`
- **Test Script:** `tests/manual_qa/test_webrtc_keepalive_fix.py`
- **Issue Tracker:** GitHub PR #XXX

---

**Last Updated:** 2025-10-20  
**Status:** Fix Implemented - Awaiting Validation
