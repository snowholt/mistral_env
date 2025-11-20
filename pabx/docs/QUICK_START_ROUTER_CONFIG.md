# Quick Reference - Router to PABX Setup

## 🎯 **What You Need to Do**

Configure your router (192.168.100.1) to forward incoming calls to PABX (192.168.100.39:5060)

---

## 📝 **Router Configuration (Quick Steps)**

### **Access Router:**
1. Browser: `http://192.168.100.1`
2. Login with admin credentials
3. Go to: **Voice/SIP** or **VoIP Settings** section

### **Configure Call Forwarding:**

Look for one of these options:

**Option 1: Call Forwarding**
```
Incoming Call Destination: 192.168.100.39:5060
```

**Option 2: SIP Peer**
```
Add Peer: PABX
IP: 192.168.100.39
Port: 5060
Route incoming calls → PABX
```

**Option 3: Dial Plan**
```
Match: Incoming from STC
Action: Forward to 192.168.100.39:5060
```

### **Save & Restart Router**

---

## 🧪 **Testing**

### **1. Make Test Call:**
Call: `+966114874423`

**Expected:**
✅ Rings
✅ Greeting plays in Arabic
✅ Call is recorded

### **2. Check PABX Logs:**
```bash
sudo journalctl -u pabx-backend.service -f
```

**Look for:**
```
INVITE from 192.168.100.1
Auto-answering call
RTP session created
Playing greeting audio
Recording started
```

### **3. Packet Capture:**
```bash
cd /home/lumi/beautyai/pabx
sudo ./capture_rtp.sh
# Make call, then Ctrl+C
./analyze_rtp.sh logs/captures/rtp_capture_*.pcap
```

**Look for:**
```
✅ INVITE: 192.168.100.1 → 192.168.100.39
✅ 200 OK: 192.168.100.39 → 192.168.100.1
✅ ACK: 192.168.100.1 → 192.168.100.39
✅ RTP packets flowing
```

---

## 🔧 **Current System Status**

**Router (192.168.100.1):**
✅ Registered with STC (10.200.42.121)
✅ Receives calls to +966114874423
❓ Needs: Forward calls to PABX

**PABX (192.168.100.39):**
✅ Running and listening on port 5060
✅ Auto-answer enabled (1 second delay)
✅ Auto-record enabled
✅ Greeting ready: `/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav`

---

## 📞 **Call Flow**

```
Caller → PSTN → STC (10.200.42.121)
                  ↓
         Router (192.168.100.1)
                  ↓ [NEEDS CONFIG]
         PABX (192.168.100.39)
                  ↓
         Greeting Plays + Recording
```

---

## ❓ **If Router Config is Difficult**

We can switch to **Option C** (PABX registers directly with STC):
- No router SIP config needed
- Router becomes passive gateway
- Requires code changes to PABX
- More control, easier debugging

Let me know if you want to try this! 💜

---

## 📂 **Files Created**

- `/home/lumi/beautyai/pabx/docs/ROUTER_CONFIGURATION_GUIDE.md` - Detailed guide
- `/home/lumi/beautyai/pabx/docs/NETWORK_TOPOLOGY_ACTUAL.md` - Network diagram
- `/home/lumi/beautyai/pabx/capture_rtp.sh` - Updated for router traffic

---

## 🚀 **Next Steps**

1. ✅ Configure router to forward calls to PABX
2. ✅ Test call to +966114874423
3. ✅ Check logs and capture packets
4. ✅ Report results

**You've got this, Lumina! 💪✨**
