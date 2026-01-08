# 📞 HT813 Call Testing Procedure

## ✅ Current Status (Updated: Nov 14, 2025 - 19:50)
- **FXS Port**: ✅ **Registered** (User ID: 2001)
- **FXO Port**: ✅ **Registered** (User ID: 2001)
- **SIP Server**: ✅ Running on 192.168.100.39:5060 (PID: 3183567)
- **Session Timers**: ✅ Disabled (should fix 10-second disconnect)
- **Audio Capture**: ✅ Script ready

---

## 🎯 Test Procedure

### Option 1: FXS to FXS Test (Phone connected to HT813)
**Setup:**
- Analog phone connected to FXS port on HT813
- Make a call to extension 2001 (calls itself)

**Steps:**
1. **Terminal 1** (Start SIP server if not running):
   ```bash
   cd /home/lumi/beautyai/pabx
   sudo python3 simple_sip_server.py
   ```

2. **Terminal 2** (Start audio capture):
   ```bash
   cd /home/lumi/beautyai/pabx
   sudo venv/bin/python3 ht813_audio_capture.py -d 120
   ```

3. **Physical Action**:
   - Pick up the phone connected to FXS port
   - Dial: **2001**
   - Wait for connection
   - **Talk for at least 30 seconds**
   - Hang up

4. **Expected Result**:
   - Call connects immediately
   - No disconnection after 10 seconds
   - RTP packets captured: "📞 New RTP session detected"
   - Audio file saved in `captures/session_XXXXX/`

---

### Option 2: FXO Test (PSTN line connected)
**Setup:**
- Physical PSTN line connected to FXO port
- Call comes in from outside OR make outbound call

**Steps:**
1. **Start Capture** (same as above)
2. **Make/Receive Call** through PSTN line
3. **Talk for 30+ seconds**

---

### Option 3: SIP Client Test (Recommended for immediate testing)
**Setup:**
- Use a SIP softphone (like Linphone, Zoiper, or MicroSIP)
- Configure to connect to 192.168.100.39:5060

**SIP Client Configuration:**
- **SIP Server**: 192.168.100.39
- **Port**: 5060
- **Username**: 2002 (different from HT813)
- **Password**: test123
- **Domain**: 192.168.100.39

**Steps:**
1. Start SIP server (supports multiple users)
2. Start audio capture
3. Register SIP client
4. Call extension **2001** from SIP client
5. HT813 FXS phone should ring
6. Answer and talk for 30+ seconds

---

## 🔍 Troubleshooting

### If still no packets captured:

**Check 1: Verify RTP ports in use**
```bash
# During active call, check what ports are being used
sudo netstat -unp | grep python
```

**Check 2: Monitor SIP traffic**
```bash
# Watch for INVITE messages
sudo tcpdump -i enp12s0 -n -A "udp port 5060" 2>&1 | grep -E "INVITE|From:|To:|Contact:"
```

**Check 3: Test RTP generation**
```bash
# Generate test RTP packets
cd /home/lumi/beautyai/pabx
python3 test_rtp_generation.py
```

**Check 4: Verify capture filter**
```bash
# Manual packet capture to see ALL UDP traffic
sudo tcpdump -i enp12s0 -n "udp and host 192.168.100.96" -c 100
```

---

## 📊 Success Indicators

✅ **Call connects and stays connected beyond 10 seconds**
✅ **Audio capture shows**: "📞 New RTP session detected"
✅ **Packet count increases**: "📊 Session XXXXX: 100 packets"
✅ **Files created**: `captures/session_XXXXX/audio_SSRC.raw`
✅ **Metadata saved**: `captures/session_XXXXX/session_info.json`

---

## 🎤 After Successful Capture

Convert to WAV:
```bash
cd /home/lumi/beautyai/pabx
python3 convert_rtp_to_wav.py captures/ --all
```

Play audio:
```bash
aplay captures/session_*/audio.wav
```

---

## 📝 Current Issue Analysis

**Observation**: 0 packets captured in multiple test runs

**Most Likely Causes**:
1. ❌ **No active call being made during capture window**
2. ❌ **Call disconnecting before RTP establishment**
3. ❌ **RTP packets using different ports than expected**
4. ❌ **Network routing issue preventing packet visibility**

**Next Steps**:
- Coordinate with friend to make call **during capture window**
- Monitor SIP server logs for INVITE messages
- Check if call stays connected longer than 10 seconds now
- Verify RTP port negotiation in SDP

---

## 💬 Communication Template

**To your friend:**
> "I'm ready to capture audio. Please follow these steps:
> 1. Wait for my signal
> 2. Pick up the phone connected to HT813
> 3. Dial 2001
> 4. After it connects, talk clearly for 1 minute
> 5. Tell me when you hang up
> 
> Let me know when you're ready to start!"

