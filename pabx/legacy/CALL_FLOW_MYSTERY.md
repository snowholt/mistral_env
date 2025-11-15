# 🔍 HT813 Call Flow Mystery - Diagnosis

## Current Status:
- **FXS Port**: On Hook, User 2001, **Registered**
- **FXO Port**: **Using**, User 2001, **Registered**

## 🤔 Critical Question:

**"FXO Using"** means the FXO port (physical PSTN/phone line connection) is currently in use!

### Possible Scenarios:

### Scenario A: Incoming PSTN Call
```
PSTN Line → FXO Port → HT813 → FXS Port → Your Phone
```
- Someone calling FROM outside (PSTN) TO your phone
- This would NOT involve our SIP server at all!
- This is a direct analog connection
- **No SIP/VoIP involved** ❌

### Scenario B: Outgoing PSTN Call  
```
Your Phone → FXS Port → HT813 → FXO Port → PSTN Line
```
- You calling FROM your phone TO external PSTN number
- This would also NOT involve our SIP server!
- Direct analog pass-through
- **No SIP/VoIP involved** ❌

### Scenario C: VoIP Call (What we WANT to test)
```
Your Phone → FXS Port → HT813 → SIP Server (192.168.100.39) → Back to FXS
```
- Dialing extension 2001 (calling yourself via VoIP)
- This REQUIRES SIP server
- Should generate SIP traffic
- **SIP/VoIP involved** ✅

## 🎯 The Mystery:

You report:
1. ✅ Device shows "Registered" (both ports)
2. ✅ Call shows "Ringing" status
3. ❌ We see ZERO SIP traffic on network
4. ❌ Call disconnects after short time

**This suggests the device is NOT actually using our SIP server!**

## 💡 Most Likely Cause:

**The configuration changes were NOT saved/applied properly!**

The HT813 might be:
- Still cached with old server (192.168.100.99)
- Configuration not applied (needs reboot)
- Using different server entirely
- Or calls are going through FXO/PSTN (analog), not SIP!

## ✅ Verification Steps:

### Step 1: Verify SIP Server Address
In HT813 web UI:
1. Go to **FXS Port** config
2. Check **"Primary SIP Server"**: Should be `192.168.100.39`
3. Go to **FXO Port** config  
4. Check **"Primary SIP Server"**: Should be `192.168.100.39`
5. If anything shows `192.168.100.99` → Change it and **click Apply**

### Step 2: Clear and Re-register
1. In web UI, click **"Reboot"** button
2. Wait 2-3 minutes for full reboot
3. Check status page - both should show "Registered"
4. Try test call again

### Step 3: Verify Call Type
**When making test call:**
- Dial: **2001** (extension number)
- Should hear: SIP dialing tone (not PSTN dial tone)
- Should ring on same phone (loopback call)

**Do NOT dial:**
- External numbers (9+number)
- PSTN access code (*00)
- These would use FXO/PSTN, not SIP!

### Step 4: Monitor During Call
While on call, run:
```bash
sudo tcpdump -i enp12s0 -n "udp port 5060" -c 10
```
Should see INVITE, 200 OK, ACK messages!

## 🚨 Action Required:

Please verify in web UI that **Primary SIP Server** is set to `192.168.100.39` for BOTH ports!

If it still shows `192.168.100.99` or something else, that's why we see no traffic! 💕
