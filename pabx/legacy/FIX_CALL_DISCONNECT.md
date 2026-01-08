# Fix HT813 Call Disconnection After 10 Seconds

## Problem
Calls connect but disconnect after exactly 10 seconds.

## Root Cause
This is typically caused by:
1. **Session Timer** - SIP session expires and neither side refreshes it
2. **Missing RTP** - No audio flowing, device thinks call failed
3. **NAT timeout** - Connection lost due to NAT

## Solution - Disable Session Timer

### Go to HT813 Web Interface: `http://192.168.100.96`

### For FXS Port:
1. Scroll to **"Enable Session Timer"** section
2. Change: `Enable Session Timer: Yes` → **`No`** ⚠️
3. Click **Apply**

### For FXO Port:
1. Go to FXO PORT tab
2. Find **"Session Expiration"** section
3. There's no "Enable" toggle, but set:
   - **Session Expiration**: `1800` (30 minutes instead of 180)
   - **Min-SE**: `90`
   - **Caller Request Timer**: `No`
   - **Callee Request Timer**: `No`
   - **Force Timer**: `No`
4. Click **Apply**

### Additional Settings to Check:

#### Call Timeout Settings:
- **Ring Timeout**: `60` seconds (default, OK)
- **No Key Entry Timeout**: `4` seconds (default, OK)

#### Keep-Alive Settings:
- **Enable SIP OPTIONS Keep Alive**: `Yes` ✅
- **SIP OPTIONS Keep Alive Interval**: `30` seconds
- **SIP OPTIONS Keep Alive Max Lost**: `3`

#### RTP Settings:
- **Enable RTCP**: `No` (simpler for testing)
- **Symmetric RTP**: `Yes` ✅

### After Making Changes:
1. Click **Apply** on each port
2. **Reboot** the device (Maintenance → Reboot)
3. Wait 1-2 minutes
4. Try the call again

## Testing After Fix:

1. **Start SIP server** (if not running):
   ```bash
   cd /home/lumi/beautyai/pabx
   sudo python3 simple_sip_server.py
   ```

2. **Start audio capture**:
   ```bash
   cd /home/lumi/beautyai/pabx
   sudo venv/bin/python3 ht813_audio_capture.py -d 120
   ```

3. **Make a call** and talk for at least 30 seconds

4. **You should see**:
   - SIP server: INVITE → 200 OK → ACK → (call stays active)
   - Capture: RTP packets flowing continuously
   - No disconnection after 10 seconds

## What the Logs Should Show:

### SIP Server (good):
```
📞 Incoming INVITE from 192.168.100.96:XXXX
✅ Call established! RTP should flow on port 12000
   Waiting for ACK and RTP packets...
📨 Received from 192.168.100.96:XXXX
   ACK sip:...
```

### Capture Script (good):
```
📞 New RTP session detected:
📊 Session XXXXX: 100 packets, 0 lost, 2.0s
📊 Session XXXXX: 500 packets, 0 lost, 10.0s
📊 Session XXXXX: 1000 packets, 0 lost, 20.0s
📊 Session XXXXX: 1500 packets, 0 lost, 30.0s
```

## Alternative: Update SIP Server

If you can't change HT813 settings, I can update the SIP server to handle session timers properly by responding to re-INVITEs.
