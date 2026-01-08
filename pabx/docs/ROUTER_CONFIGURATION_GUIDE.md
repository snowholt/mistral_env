# Router Configuration Guide - Forward Calls to PABX

## 🎯 **Objective**
Configure your router/modem (192.168.100.1) to forward incoming SIP calls to PABX server (192.168.100.39:5060)

---

## ✅ **Current Status**

**Router/Modem (192.168.100.1):**
- ✅ Registered with STC provider as `+966114874423@fmc.stc.com.sa`
- ✅ Receives incoming calls from STC (10.200.42.121)
- ❓ **Needs configuration:** Forward calls to PABX

**PABX Server (192.168.100.39):**
- ✅ Running and listening on port 5060 (UDP)
- ✅ Ready to receive INVITE messages
- ✅ Greeting audio configured: `/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav`
- ✅ Auto-answer and auto-record enabled

---

## 🔧 **Router Configuration Steps**

### **Step 1: Access Router Web Interface**

1. Open browser: `http://192.168.100.1`
2. Login with admin credentials
3. Navigate to **Voice/SIP** or **VoIP Settings**

---

### **Step 2: Configure Call Forwarding**

Look for one of these sections (depends on router model):

#### **Option 1: Call Forwarding Rules**
```
Setting: Incoming Call Destination
Value: 192.168.100.39:5060

OR

Setting: Call Forward to SIP URI
Value: sip:192.168.100.39:5060
```

#### **Option 2: SIP Peer/Trunk Configuration**
```
Add SIP Peer:
─────────────
• Name: PABX_Server
• IP Address: 192.168.100.39
• Port: 5060
• Transport: UDP
• Enabled: Yes

Call Routing:
─────────────
• Incoming calls to +966114874423 → Forward to PABX_Server
```

#### **Option 3: Dial Plan / Routing Rules**
```
Rule: Route incoming calls to local SIP server
─────────────────────────────────────────────
• Match: Incoming calls from STC (10.200.42.121)
• Destination: 192.168.100.39:5060
• Action: Forward
```

---

### **Step 3: Verify SIP Domain Settings**

Make sure these settings are correct:

| Setting | Current Value | Notes |
|---------|--------------|-------|
| **SIP domain name** | fmc.stc.com.sa | ✅ Correct |
| **SIP Outbound Proxy** | 10.200.42.121 | ✅ Correct |
| **Registration Expire Timeout** | 3600 | ✅ Correct (1 hour) |
| **SIP Transport protocol** | UDP | ✅ Correct |

---

### **Step 4: Configure NAT/Firewall (if needed)**

Some routers may need explicit routing rules:

```
NAT/Port Forwarding:
───────────────────
• External Port: 5060 (UDP) → Internal: 192.168.100.39:5060
• RTP Ports: 10000-20000 (UDP) → Internal: 192.168.100.39:10000-20000
```

**Note:** Only needed if router is very strict with internal routing.

---

## 📋 **Expected Call Flow After Configuration**

```
1. Caller → PSTN → STC Provider (10.200.42.121)
                        ↓
2. STC sends INVITE to Router (192.168.100.1)
   • To: +966114874423@fmc.stc.com.sa
                        ↓
3. Router receives INVITE and forwards to PABX
   • To: 192.168.100.39:5060
   • INVITE sip:+966114874423@192.168.100.39:5060
                        ↓
4. PABX receives INVITE
   • Auto-answers after 1 second
   • Sends 200 OK with SDP (RTP info)
                        ↓
5. Router relays 200 OK back to STC
                        ↓
6. RTP audio stream established:
   Caller ↔ Router ↔ PABX
                        ↓
7. PABX plays greeting audio (greeting_ar.wav)
   • Caller hears: "مرحبا بك في بيوتي AI..."
                        ↓
8. PABX records entire conversation
   • Saved to: /home/lumi/beautyai/pabx/logs/recordings/
```

---

## 🧪 **Testing After Configuration**

### **Test 1: Check Router Logs**

After applying configuration:
1. Router web interface → **System Logs** or **VoIP Logs**
2. Look for: "Call forwarding configured to 192.168.100.39"

### **Test 2: Make Test Call**

1. Call `+966114874423` from your mobile
2. Wait for greeting to play

**Expected behavior:**
```
✅ Phone rings
✅ Call connects
✅ You hear greeting audio in Arabic
✅ Conversation is recorded
```

### **Test 3: Check PABX Logs**

```bash
# Watch PABX logs in real-time
sudo journalctl -u pabx-backend.service -f

# Check recent call activity
sudo journalctl -u pabx-backend.service --since "5 minutes ago" | grep INVITE
```

**Look for:**
```
INFO SIP server: INVITE from 192.168.100.1:XXXXX
INFO SIP server: Auto-answering call
INFO RTP session created
INFO Playing greeting audio
INFO Recording started
```

### **Test 4: Packet Capture**

```bash
cd /home/lumi/beautyai/pabx
sudo ./capture_rtp.sh
# Make test call
# Ctrl+C to stop
./analyze_rtp.sh logs/captures/rtp_capture_*.pcap
```

**Look for:**
```
✅ INVITE from 192.168.100.1 → 192.168.100.39
✅ 200 OK from 192.168.100.39 → 192.168.100.1
✅ ACK from 192.168.100.1 → 192.168.100.39
✅ RTP packets flowing both directions
```

---

## 🔍 **Troubleshooting**

### **Problem: Call rings but no connection**

**Possible causes:**
1. Router not forwarding INVITE to PABX
2. Firewall blocking port 5060

**Solution:**
```bash
# Check if PABX is receiving any SIP traffic
sudo tcpdump -i any -n port 5060 and host 192.168.100.1

# Make test call and watch output
# Should see: INVITE, 200 OK, ACK messages
```

---

### **Problem: Call connects but no audio**

**Possible causes:**
1. RTP ports (10000-20000) blocked
2. Router not forwarding RTP packets
3. Symmetric NAT issues

**Solution:**
```bash
# Check RTP traffic
sudo tcpdump -i any -n 'udp and portrange 10000-20000' and host 192.168.100.1

# Should see bidirectional RTP packets
```

---

### **Problem: Router doesn't have call forwarding option**

**Alternative Solution:**
Register PABX as an extension on the router:

```
Router Settings:
───────────────
• Add SIP Extension:
  - Extension: 2000
  - IP: 192.168.100.39
  - Port: 5060
  
• Incoming Call Rule:
  - When +966114874423 is called → Ring extension 2000
```

Then update PABX configuration:
```yaml
# /home/lumi/beautyai/pabx/config/settings.yaml

sip:
  trunk:
    enabled: true
    sip_server: "192.168.100.1"  # Register with router
    user_id: "2000"
    auth_id: "2000"
    auth_password: "your_extension_password"
    register: true
```

---

## 📞 **Alternative: Direct SIP Forwarding Without Router Config**

If router configuration is too complex, we can implement **Option C** instead:
- PABX registers directly with STC (10.200.42.121)
- Router becomes passive gateway (just NAT)
- Requires adding SIP Client code to PABX

Let me know if you want to try this approach! 💜

---

## 🎉 **Success Criteria**

When everything works correctly:

✅ Router receives INVITE from STC
✅ Router forwards INVITE to PABX (192.168.100.39:5060)
✅ PABX sends 200 OK back through router
✅ RTP stream established between caller and PABX
✅ Greeting audio plays to caller
✅ Conversation is recorded
✅ Call ends cleanly with BYE message

---

## 📝 **Next Steps**

1. **Apply router configuration** (one of the options above)
2. **Restart router** (if required after config change)
3. **Verify PABX is running**: `sudo systemctl status pabx-backend.service`
4. **Make test call** to +966114874423
5. **Check logs** and **capture packets** to verify
6. **Report results** so we can debug if needed

**Good luck, dear Lumina! 🌸 You've got this!** 💪✨
