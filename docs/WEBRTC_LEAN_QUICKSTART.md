# WebRTC Lean Capture: Quick Start Guide

**Date**: November 13, 2025  
**Goal**: Test hardened architecture with <1% buffer underrun rate

---

## 🚀 Quick Deploy (5 Minutes)

### 1. Install & Activate ✅

```bash
cd /home/lumi/beautyai

# Deploy new systemd service with RT scheduling
sudo cp beautyai-api-lean.service /etc/systemd/system/beautyai-api.service
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api.service

# Verify RT scheduling active
ps -eo pid,comm,cls,rtprio,ni | grep python
# Expected: CLS=RR, RTPRIO=20, NI=-5

# Check logs for successful startup
sudo journalctl -u beautyai-api.service -f --since="1 minute ago"
# Look for: "WebRTC lean capture endpoints registered"
```

---

### 2. Run 30-Second Test Capture 🎤

**Option A: Browser-Based (Recommended)**

1. Open browser: `https://192.168.100.39:8443/test_simple.html`
2. **Modify JavaScript** (one-time change):
   ```javascript
   // Change line ~30:
   // const response = await fetch('/api/v1/webrtc/debug/voice-capture/offer', {
   const response = await fetch('/api/v1/webrtc/lean/voice-capture/offer', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           sdp: offer.sdp,
           type: offer.type,
           preset: "lean_rnnoise"  // Add this line
       })
   });
   ```
3. Connect → Speak for 30 seconds → Wait for "✅ capture complete"

**Option B: Automated Test Script**

```bash
cd /home/lumi/beautyai
source backend/venv/bin/activate

# Create test script
cat > test_lean_capture.py << 'EOF'
import asyncio
import aiohttp
import json

async def test_lean_capture():
    async with aiohttp.ClientSession() as session:
        # Simple SDP offer (browser generates this normally)
        offer_sdp = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp-mux
a=sendonly
a=mid:0
a=rtpmap:111 opus/48000/2
"""
        
        # Send offer to lean endpoint
        async with session.post(
            'http://localhost:8000/api/v1/webrtc/lean/voice-capture/offer',
            json={
                'sdp': offer_sdp,
                'type': 'offer',
                'preset': 'lean_rnnoise'
            }
        ) as resp:
            answer = await resp.json()
            print(f"✅ Session created: {answer['peer_id']}")
            print(f"   Preset: {answer['preset']}")
            print(f"   SDP: {answer['sdp'][:100]}...")

asyncio.run(test_lean_capture())
EOF

python test_lean_capture.py
```

---

### 3. Check Results 📊

```bash
cd /home/lumi/beautyai/reports/debug/webrtc

# Quick metrics check
echo "=== Queue Statistics ==="
cat queue_stats.json | jq '{
  underrun_rate: .underrun_rate_percent,
  recv_p99_ms: .recv_delta_p99_ms,
  worker_p99_ms: .worker_service_p99_ms,
  queue_peak: .peak_depth,
  total_frames: .enqueued
}'

echo ""
echo "=== Pipeline Statistics ==="
cat pipeline_stats.json | jq '{
  preset: .denoiser_type,
  limiter_activations: .limiter_activations,
  comb_active_frames: .comb_active_frames,
  gate_closed_frames: .gate_closed_frames
}'

# Success criteria
echo ""
echo "=== Success Criteria ==="
UNDERRUN_RATE=$(cat queue_stats.json | jq -r '.underrun_rate_percent')
RECV_P99=$(cat queue_stats.json | jq -r '.recv_delta_p99_ms')
WORKER_P99=$(cat queue_stats.json | jq -r '.worker_service_p99_ms')

echo "Underrun rate: $UNDERRUN_RATE% (target <1%)"
echo "Recv p99: $RECV_P99 ms (target <2ms)"
echo "Worker p99: $WORKER_P99 ms (target <20ms)"

# Auto-check
if (( $(echo "$UNDERRUN_RATE < 1.0" | bc -l) )); then
  echo "✅ PASS: Underrun rate <1%"
else
  echo "❌ FAIL: Underrun rate >1%"
fi
```

---

### 4. Compare with Analyzer 🔬

```bash
cd /home/lumi/beautyai
source backend/venv/bin/activate

# Run analyzer on new capture
python tools/analyze_audio_noise.py --compare --visualize

# Expected improvements:
echo "=== Expected Metrics (Before → After) ==="
echo "Crackles @ 48kHz: 7488 (240/s) → <100 (<3/s)"
echo "Crackles @ 16kHz: 2336 (80/s) → <30 (<1/s)"
echo "Underrun rate: 21% → <1%"
echo "80 Hz hum: Present → Eliminated (if underruns fixed)"
```

---

## 🎯 Success Criteria Checklist

Run after 30s capture:

```bash
cd /home/lumi/beautyai/reports/debug/webrtc

# Auto-verify all criteria
python3 << 'EOF'
import json

with open('queue_stats.json') as f:
    queue = json.load(f)

with open('pipeline_stats.json') as f:
    pipeline = json.load(f)

criteria = {
    "Underrun rate < 1%": queue['underrun_rate_percent'] < 1.0,
    "Recv p99 < 2ms": queue['recv_delta_p99_ms'] < 2.0,
    "Worker p99 < 20ms": queue['worker_service_p99_ms'] < 20.0,
    "Queue peak ≤ 8 frames": queue['peak_depth'] <= 8,
    "No dropped frames": queue['dropped'] == 0,
    "Total frames > 1000": queue['enqueued'] > 1000,
}

print("=" * 50)
print("SUCCESS CRITERIA VERIFICATION")
print("=" * 50)
for name, passed in criteria.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")

all_passed = all(criteria.values())
print("=" * 50)
if all_passed:
    print("🎉 ALL CRITERIA MET - READY FOR PRODUCTION")
else:
    print("⚠️  SOME CRITERIA FAILED - REVIEW METRICS")
print("=" * 50)
EOF
```

---

## 🔧 Troubleshooting Fast Fixes

### **Problem**: Underrun rate still >1%

```bash
# Quick fix 1: Increase queue depth
nano backend/src/beautyai_inference/api/endpoints/webrtc_lean_capture.py
# Line ~118: Change BoundedFrameQueue(max_size=5) → max_size=8

# Quick fix 2: Reduce worker count
# Line ~128: Change LeanPipeline(..., max_workers=2) → max_workers=1

# Restart
sudo systemctl restart beautyai-api.service
```

### **Problem**: Recv loop p99 >2ms

```bash
# Disable verbose logging
export WEBRTC_DEBUG_VERBOSE=0

# Restart
sudo systemctl restart beautyai-api.service
```

### **Problem**: No RT scheduling active

```bash
# Check capabilities
getcap /home/lumi/beautyai/backend/venv/bin/python3

# If empty, add capability:
sudo setcap 'cap_sys_nice=eip' /home/lumi/beautyai/backend/venv/bin/python3

# Verify
ps -eo pid,comm,cls,rtprio,ni | grep python
# Should show: CLS=RR, RTPRIO=20
```

---

## 📈 Before/After Comparison Table

| Metric | Before (Legacy) | After (Lean) | Target | Status |
|--------|----------------|--------------|---------|--------|
| **Underrun rate** | 21.06% | ❓ | <1% | ⏳ Run test |
| **Recv p99** | ~14ms | ❓ | <2ms | ⏳ Run test |
| **Worker p99** | N/A | ❓ | <20ms | ⏳ Run test |
| **Crackles 48kHz** | 7488 (240/s) | ❓ | <100 (<3/s) | ⏳ Run analyzer |
| **Crackles 16kHz** | 2336 (80/s) | ❓ | <30 (<1/s) | ⏳ Run analyzer |
| **80 Hz hum** | Present | ❓ | Eliminated | ⏳ Run analyzer |
| **CPU usage** | 1.33% avg | ❓ | <10% avg | ⏳ Monitor htop |

---

## 🔄 Rollback (If Needed)

```bash
# Restore original service file
sudo cp /home/lumi/beautyai/servicesBackups/beautyai-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api.service

# Verify rollback
curl http://localhost:8000/health
```

---

## 📝 Next Steps After Successful Test

1. ✅ **Document results** in `reports/` with before/after table
2. ✅ **Update production config** if all criteria met
3. ✅ **Monitor production** for 24 hours
4. ⏳ **Optional**: Test `lean_dtln` and `minimal` presets
5. ⏳ **Optional**: Fine-tune hum detector thresholds

---

**Author**: BeautyAI Framework  
**Date**: November 13, 2025  
**Status**: Ready for Testing 🚀

**Estimated Time**: 5 minutes deploy + 30 seconds capture + 2 minutes analysis = **~10 minutes total**
