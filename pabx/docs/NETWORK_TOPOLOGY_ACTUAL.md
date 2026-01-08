# Actual Network Topology - BeautyAI PABX System (UPDATED)

## 📊 **ACTUAL Network Architecture** (Corrected)

```
                                    INTERNET
                                        |
                                        |
                    ┌───────────────────┴────────────────────┐
                    │                                        │
                    │   STC Provider (ISP)                   │
                    │   - SIP Proxy: 10.200.42.121:5060     │
                    │   - Domain: fmc.stc.com.sa            │
                    │   - VoIP IP: 20.159.4.1               │
                    │                                        │
                    └───────────────────┬────────────────────┘
                                        |
                                        | (WAN + SIP Trunk)
                                        |
                    ┌───────────────────┴────────────────────┐
                    │                                        │
                    │   Router/Modem (Gateway + SIP Client) │
                    │   - LAN IP: 192.168.100.1              │
                    │   - WAN: Public IP from STC            │
                    │   - REGISTERED with STC as:            │
                    │     +966114874423@fmc.stc.com.sa       │
                    │   - Handles PSTN ↔ VoIP conversion     │
                    │                                        │
                    └─────────┬──────────────────────────────┘
                              |
                              | (LAN - 192.168.100.x)
                              |
            ┌─────────────────┴──────┬─────────────────┐
            |                        |                  |
    ┌───────▼────────┐      ┌────────▼────────┐  ┌────▼──────┐
    │                │      │                 │  │           │
    │  PABX Server   │      │  Your Laptop    │  │  Others   │
    │ 192.168.100.39 │      │  (for testing)  │  │           │
    │                │      │                 │  │           │
    │ - Port 5060    │      │                 │  │           │
    │ - Port 8080    │      │                 │  │           │
    │ - RTP 10k-20k  │      │                 │  │           │
    │                │      │                 │  │           │
    └────────────────┘      └─────────────────┘  └───────────┘


## ✅ CORRECT Call Flow (Current Setup)

```
Incoming Call to +966114874423:
─────────────────────────────────

1. Caller → PSTN Network → STC Provider
       │
       │ (STC routes to registered SIP client)
       │
       ▼
2. Router/Modem (192.168.100.1)
   - Registered as: +966114874423@fmc.stc.com.sa
   - Receives INVITE from STC (10.200.42.121)
   - Router answers: "I'll handle this call"
       │
       │ **QUESTION: Where does Router forward the call?**
       │
       ▼
3. ❓ Router needs to forward to: 192.168.100.39:5060 (PABX)
   │
   │ OPTIONS:
   │ A) Router has SIP trunk/forward config → PABX
   │ B) Router just terminates call locally (wrong!)
   │ C) Router forwards based on dial plan
   │
   ▼
4. PABX Server (192.168.100.39)
   - Should receive INVITE from Router
   - Sends 200 OK
   - Plays greeting audio
   │
   │
   ▼
5. RTP Audio Stream
   Router (192.168.100.1) ←──RTP──→ PABX (192.168.100.39)
```

---

## 🔧 **What Needs to Be Configured**

### **Option A: Router Forwards Calls to PABX** ⭐ (Recommended)

Configure router to forward incoming calls to PABX:

```
Router Settings (192.168.100.1):
══════════════════════════════════
• Incoming Call Destination: 192.168.100.39:5060
• Call Forward: Unconditional → 192.168.100.39:5060
• SIP Peer: Add PABX as trusted peer
```

**Call Flow:**
```
STC → Router → PABX → Greeting Played ✅
```

### **Option B: PABX Registers as Extension on Router** 🤔

PABX registers to Router as an internal extension:

```
PABX Configuration:
══════════════════
• SIP Server: 192.168.100.1
• Extension: 2000 (or whatever Router assigns)
• Router forwards +966114874423 calls → ext 2000
```

### **Option C: PABX Registers Directly to STC** 🎯 (BEST!)

**PABX takes over** the STC registration:

```
PABX Configuration:
══════════════════
• SIP Server: 10.200.42.121
• Outbound Proxy: 10.200.42.121
• User ID: +966114874423
• Auth ID: +966114874423@fmc.stc.com.sa
• Password: 114874423114874423
• Domain: fmc.stc.com.sa
```

**Router:** Disable SIP registration, act as bridge only

**Call Flow:**
```
STC → PABX (directly) → Greeting Played ✅
```

---

## 💡 **RECOMMENDATION: Option C** 

Let PABX handle everything directly!

**Why?**
- ✅ Simpler - no router SIP config needed
- ✅ Better control - PABX handles all SIP logic
- ✅ Easier debugging - all SIP in one place
- ✅ Router just does NAT - what it's good at

**Router becomes:** Just a NAT gateway (no SIP)
**PABX becomes:** SIP client registered with STC

