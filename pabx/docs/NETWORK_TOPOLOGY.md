# Network Topology - BeautyAI PABX System

## 📊 Current Network Architecture

```
                                    INTERNET
                                        |
                                        |
                    ┌───────────────────┴────────────────────┐
                    │                                        │
                    │   STC Provider (ISP)                   │
                    │   - SIP Proxy: 10.200.42.121:5060     │
                    │   - Domain: fmc.stc.com.sa            │
                    │                                        │
                    └───────────────────┬────────────────────┘
                                        |
                                        | (WAN Connection)
                                        |
                    ┌───────────────────┴────────────────────┐
                    │                                        │
                    │   Router/Modem (Gateway)               │
                    │   - LAN IP: 192.168.100.1              │
                    │   - WAN: Public IP from STC            │
                    │   - DHCP Server for LAN                │
                    │                                        │
                    └─────────┬──────────────────┬───────────┘
                              |                  |
                              |                  |
            ┌─────────────────┴──────┐  ┌───────┴──────────────────┐
            |                        |  |                           |
            | (LAN)                  |  | (LAN)                     |
            |                        |  |                           |
    ┌───────▼────────┐      ┌────────▼──▼────────┐       ┌─────────▼─────────┐
    │                │      │                     │       │                   │
    │  PABX Server   │      │   HT813 Gateway     │       │  Your Laptop      │
    │ 192.168.100.39 │      │  192.168.100.96     │       │ (for testing)     │
    │                │      │                     │       │                   │
    │ - Port 5060    │      │  [FXS] [FXO] [LAN]  │       │                   │
    │ - Port 8080    │      │    │     │     │    │       │                   │
    │   (Web UI)     │      │    │     │     │    │       │                   │
    │                │      │    │     │     └────┴───────┴─── To Router      │
    └────────────────┘      │    │     │                                      │
                            │    │     │                                      │
                            │    │     └─────────── PSTN Line ───────────────┤
                            │    │                  +966114874423            │
                            │    │                                           │
                            │    └── Analog Phone                            │
                            │        (Optional)                              │
                            │                                                │
                            └────────────────────────────────────────────────┘


PSTN Line Details:
┌────────────────────────────────────────────────────────┐
│  Phone Number: +966114874423                           │
│  Provider: STC (Saudi Telecom Company)                 │
│  Line Type: Traditional PSTN (analog)                  │
│  Physically connected to HT813 FXO port                │
└────────────────────────────────────────────────────────┘
```

---

## 🔄 Current Call Flow (WRONG Configuration)

```
Incoming PSTN Call to +966114874423:
─────────────────────────────────────

1. PSTN Network
       │
       │ (Analog signal via copper wire)
       │
       ▼
2. HT813 FXO Port
       │
       │ (Detects ring, converts analog → SIP)
       │
       ▼
3. HT813 tries to REGISTER to: 192.168.100.39 (PABX Server)
   ❌ PROBLEM: Should register to STC SIP Proxy instead!
       │
       │
       ▼
4. HT813 gets confused because:
   - "Unconditional Call Forward to VOIP: 1001@192.168.100.39:5060"
   - This creates a loop or blocks the call
       │
       │
       ▼
5. ❌ NO INVITE SENT TO PABX
   ❌ Call fails or goes nowhere
```

---

## ✅ CORRECT Call Flow (After Configuration Fix)

```
Incoming PSTN Call to +966114874423:
─────────────────────────────────────

1. Caller dials +966114874423
       │
       │ (Routes through PSTN network)
       │
       ▼
2. STC Provider receives call
       │
       │ (Knows this number belongs to your SIP account)
       │
       ▼
3. STC SIP Proxy (10.200.42.121)
       │
       │ (Sends SIP INVITE to registered device)
       │
       ▼
4. HT813 FXO Port (registered with STC)
   - Registered as: +966114874423@fmc.stc.com.sa
   - Receives INVITE from STC
       │
       │ (HT813 answers the SIP call)
       │
       ▼
5. HT813 forwards call to PABX Server
   - Sends INVITE to: 192.168.100.39:5060
   - From: 1002 (FXO extension)
       │
       │
       ▼
6. PABX Server (192.168.100.39)
   - Receives INVITE
   - Sends 200 OK with SDP
   - Auto-answers the call
       │
       │
       ▼
7. RTP Audio Stream Established
   ┌─────────────────────────────────────┐
   │  HT813 (1002) ←──RTP──→ PABX Server │
   │  192.168.100.96:5012 ←→ 192.168.100.39:10000-20000
   └─────────────────────────────────────┘
       │
       │
       ▼
8. PABX plays greeting audio
   - greeting_ar.wav → encoded → RTP packets
       │
       │
       ▼
9. Caller hears greeting! ✅
```

---

## 🔧 Configuration Requirements

### **HT813 FXO Port Must Do TWO Things:**

```
┌──────────────────────────────────────────────────────────────┐
│                    HT813 FXO Configuration                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PRIMARY REGISTRATION (to STC Provider):                     │
│  ═══════════════════════════════════════                     │
│  • Primary SIP Server: 10.200.42.121                         │
│  • Outbound Proxy: 10.200.42.121                             │
│  • SIP User ID: +966114874423                                │
│  • Authenticate ID: +966114874423@fmc.stc.com.sa             │
│  • Password: 114874423114874423                              │
│  • Local SIP Port: 5062                                      │
│                                                               │
│  ─────────────────────────────────────────────────────────── │
│                                                               │
│  CALL FORWARDING (to PABX):                                  │
│  ═══════════════════════════                                 │
│  • When FXO receives call from STC                           │
│  • Forward to: 1001@192.168.100.39:5060                      │
│  • Method: Use dial plan or auto-dial                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📱 Device Details

### **PABX Server (192.168.100.39)**
```
┌─────────────────────────────────────────┐
│  Hardware: Ubuntu Server                │
│  LAN IP: 192.168.100.39                 │
│  Services:                               │
│    • SIP Server: Port 5060               │
│    • RTP Range: 10000-20000              │
│    • Web UI: Port 8080                   │
│    • WebSocket: ws://192.168.100.39/ws   │
│                                          │
│  Registered Extensions:                  │
│    • 1001 (FXS from HT813)              │
│    • 1002 (FXO from HT813)              │
└─────────────────────────────────────────┘
```

### **HT813 Gateway (192.168.100.96)**
```
┌─────────────────────────────────────────┐
│  Model: Grandstream HT813               │
│  LAN IP: 192.168.100.96                 │
│  MAC: ec:74:d7:62:4e:34                 │
│                                          │
│  Ports:                                  │
│    • FXS (Phone): Port 5060             │
│      └─ Extension 1001 → PABX           │
│      └─ RTP: 5004                       │
│                                          │
│    • FXO (PSTN): Port 5062              │
│      └─ Extension 1002 → PABX           │
│      └─ RTP: 5012                       │
│      └─ PSTN Line: +966114874423        │
│                                          │
│    • LAN: Connected to Router           │
└─────────────────────────────────────────┘
```

### **STC Provider (External)**
```
┌─────────────────────────────────────────┐
│  Provider: Saudi Telecom Company (STC)  │
│  SIP Proxy: 10.200.42.121:5060          │
│  Domain: fmc.stc.com.sa                 │
│  VoIP IP: 20.159.4.1                    │
│                                          │
│  Your Account:                           │
│    • Phone: +966114874423                │
│    • User: +966114874423@fmc.stc.com.sa │
│    • Pass: 114874423114874423            │
└─────────────────────────────────────────┘
```

---

## 🎯 The Problem Visualized

```
CURRENT (WRONG):
════════════════

PSTN → HT813 FXO → ❌ Tries to register with PABX
                   ❌ "Unconditional forward" blocks calls
                   ❌ No connection to STC SIP Proxy


CORRECT:
════════

PSTN → HT813 FXO ──┬─→ ✅ Registers with STC (10.200.42.121)
                   │
                   └─→ ✅ Forwards incoming calls to PABX (192.168.100.39)
                       └─→ ✅ PABX answers & plays greeting
```

---

## 📋 Quick Fix Summary

**Change these HT813 settings:**

1. **FXO PORT** → Point to STC instead of PABX
2. **BASIC SETTINGS** → Clear "Unconditional Call Forward to VOIP"
3. **FXO PORT** → Set call forwarding to PABX for incoming calls

**Result:**
- HT813 FXO registers with STC ✅
- Incoming PSTN calls route to PABX ✅
- PABX answers and plays greeting ✅
- RTP audio works ✅

