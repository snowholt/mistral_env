# WebRTC Debug Logs Guide

This guide explains the different types of logs available in the WebRTC test interface and which ones to use for debugging.

---

## Log Types Overview

### 1. 📋 Logs Tab (Human-Readable Timeline)

**What it is:** A chronological list of events that happened during your WebRTC session.

**Contains:**
- Connection establishment events
- ICE negotiation progress
- Audio track events
- Error messages
- State changes

**Example entries:**
```
17:30:15.123 INFO  RTCPeerConnection created
17:30:15.456 SUCCESS ICE candidate accepted
17:30:16.789 ERROR Connection timeout
```

**When to use:**
- Quick overview of what happened
- Spotting errors at a glance
- Understanding the event timeline
- Real-time monitoring

---

### 2. 🌐 ICE Candidates Tab

**What it is:** List of network paths (candidates) that WebRTC tested to establish the connection.

**Contains:**
- Local candidates (your network paths)
- Remote candidates (server network paths)
- Candidate types (host, srflx, relay)
- IP addresses and ports

**Example:**
```json
{
  "candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host",
  "sdpMid": "0",
  "sdpMLineIndex": 0
}
```

**When to use:**
- Diagnosing connection failures
- Understanding NAT traversal issues
- Checking if STUN/TURN servers are working
- Network troubleshooting

---

### 3. 📊 WebRTC Stats Tab

**What it is:** Real-time connection quality metrics.

**Contains:**
- Latency (round-trip time)
- Packet loss percentage
- Jitter (timing variation)
- Bitrate (data rate)
- Bytes sent/received
- Packets sent/received

**Example:**
```
latency: 45 ms
packetLoss: 0.2%
jitter: 12 ms
bitrate: 128 kbps
```

**When to use:**
- Monitoring connection quality
- Detecting audio quality issues
- Performance tuning
- Identifying network problems

---

### 4. 📄 SDP Tab (Session Description Protocol)

**What it is:** Technical description of the media session negotiated between client and server.

**Contains:**
- Media types (audio/video)
- Codecs supported
- Encryption details
- Network information
- Media capabilities

**Example:**
```
v=0
o=- 123456789 2 IN IP4 127.0.0.1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
```

**When to use:**
- Understanding codec negotiation
- Debugging media format issues
- Analyzing capability exchange
- Deep technical troubleshooting

---

### 5. 📦 JSON Log Tab (Complete Structured Data)

**What it is:** Complete export of ALL debugging data in machine-readable JSON format.

**Contains:**
- All logs from Logs tab
- All ICE candidates
- All WebRTC stats
- Both SDP (local and remote)
- Configuration settings
- Connection state
- Timestamps

**Example:**
```json
{
  "timestamp": "2025-10-20T17:30:00.000Z",
  "testMode": "advanced",
  "configuration": {
    "serverUrl": "https://web.lumidev.ca/api/v1/webrtc/voice",
    "language": "ar",
    "voiceGender": "female"
  },
  "logs": [...],
  "iceCandidates": [...],
  "sdp": {...},
  "metrics": {...},
  "webrtcStats": {...}
}
```

**When to use:**
- **PRIMARY USE:** Sharing with developers for debugging
- Archiving session data
- Automated analysis
- Bug reports
- Performance analysis

---

## Which Logs to Use for Debugging?

### ✅ Best Practice: Use JSON Log

**For sharing with developers/support, always use the JSON Log because:**

1. ✅ **Complete** - Contains ALL other logs in one file
2. ✅ **Structured** - Easy to parse and analyze programmatically
3. ✅ **Self-contained** - No need to export multiple files
4. ✅ **Timestamped** - Exact timing of all events
5. ✅ **Context-rich** - Includes configuration and state

### Quick Reference by Use Case

| Use Case | Best Log Type |
|----------|---------------|
| 🐛 Bug Report | **JSON Log** (includes everything) |
| 📧 Email to Support | **JSON Log** (attach file or copy text) |
| 🔍 Quick Check | **Logs Tab** (human-readable) |
| 🌐 Connection Issues | **ICE Candidates + JSON Log** |
| 📉 Quality Issues | **WebRTC Stats + JSON Log** |
| 🎤 Audio Problems | **SDP + JSON Log** |
| 💾 Archive Session | **JSON Log** (save file) |

---

## How to Share Logs

### Method 1: Copy JSON to Clipboard (Easiest)
1. Open WebRTC test interface
2. Go to "JSON Log" tab
3. Click "Copy to Clipboard" button
4. Paste into email/ticket/chat

### Method 2: Download JSON File
1. Click "Export Logs" button (top right)
2. Save the `.json` file
3. Attach file to email/ticket

### Method 3: Screenshot (For Quick Issues)
1. Take screenshot of relevant tab
2. Include in report for visual reference
3. **Still include JSON log for complete data**

---

## Understanding the Differences

### Timeline vs Snapshot

- **Logs Tab** = Timeline (events over time)
- **ICE/Stats/SDP/JSON** = Snapshots (state at specific moments)

### Human vs Machine Readable

- **Human-Readable:** Logs Tab
- **Machine-Readable:** JSON Log
- **Mixed:** ICE, Stats, SDP tabs (both readable)

### Diagnostic Purpose

| Log Type | Primary Purpose |
|----------|----------------|
| Logs | Event sequence and errors |
| ICE Candidates | Network connectivity |
| WebRTC Stats | Connection quality |
| SDP | Media negotiation |
| JSON Log | Complete debugging data |

---

## Auto-Update Behavior

- **Logs Tab:** Real-time (instant)
- **ICE Candidates:** On new candidate (event-driven)
- **WebRTC Stats:** Every 1 second
- **SDP:** On negotiation (one-time per connection)
- **JSON Log:** Every 5 seconds + manual refresh

---

## Tips for Effective Debugging

1. **Always start with JSON Log** - It has everything
2. **Check Logs tab first** - Quick error identification
3. **Use Stats tab** - Monitor quality in real-time
4. **Export early and often** - Capture issues as they happen
5. **Include timestamps** - JSON log has automatic timestamps
6. **Test duration** - Let it run 15+ minutes for timeout issues

---

## Example Debugging Workflow

### Connection Fails to Establish
1. Check **Logs Tab** for errors
2. Review **ICE Candidates** for network issues
3. Export **JSON Log** and share with support

### Connection Drops After Time
1. Monitor **WebRTC Stats** for quality degradation
2. Check **Logs Tab** for disconnect events
3. Export **JSON Log** capturing the moment of disconnect

### Audio Quality Issues
1. Check **WebRTC Stats** for packet loss/jitter
2. Review **SDP** for codec mismatches
3. Export **JSON Log** during poor quality period

### Complete Failure Investigation
1. Export **JSON Log** immediately
2. Review all tabs for context
3. Share JSON log with complete state

---

## Summary

**🎯 Key Takeaway:** When in doubt, use the **JSON Log** tab - it includes everything you need for debugging and sharing!

**📦 JSON Log includes:**
- ✅ All human-readable logs
- ✅ All ICE candidates
- ✅ All WebRTC statistics
- ✅ Both SDP (local and remote)
- ✅ Configuration and state
- ✅ Timestamps for everything

**For Developers:** JSON Log is your one-stop solution for debugging WebRTC issues.

---

**Last Updated:** 2025-10-20  
**Related:** [WebRTC Fix Documentation](WEBRTC_FIX_README.md)
