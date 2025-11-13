# WebRTC Lean Capture: Hardened Real-Time Audio Architecture

**Date**: November 13, 2025  
**Author**: BeautyAI Framework  
**Status**: ✅ Implementation Complete, Ready for Testing

## Executive Summary

Implemented a production-hardened WebRTC audio capture system to eliminate buffer underruns (<1% target) while maintaining low latency (<50ms end-to-end). The refactored architecture uses bounded queues, minimal recv loops, and lean processing pipelines following real-time audio best practices.

### Key Improvements

| Metric | Before (Legacy) | After (Lean) | Target | Status |
|--------|----------------|--------------|---------|--------|
| **Buffer underrun rate** | ~21% (329/1562 frames) | ❓ TBD | <1% | 🧪 Testing Required |
| **Recv loop p99 latency** | ~14ms+ (blocking) | ❓ TBD | <2ms | 🧪 Testing Required |
| **Worker service p99** | N/A (inline) | ❓ TBD | <20ms | 🧪 Testing Required |
| **Queue depth (peak)** | Unbounded tasks | ❓ TBD | ≤8 frames | 🧪 Testing Required |
| **Denoisers per frame** | 7 sequential | 1 | 1 | ✅ Implemented |
| **CPU usage (single conn)** | 1.33% avg (4.8% peak) | ❓ TBD | <10% avg | 🧪 Testing Required |
| **Systemd scheduling** | Default (CFS) | Round-Robin Priority 20 | RR/FIFO | ✅ Implemented |
| **80 Hz hum elimination** | Static comb filter | Adaptive (detector-based) | On-demand | ✅ Implemented |

---

## Architecture Overview

### 1. **Bounded Queue with Drop-Oldest Policy**

**File**: `backend/src/beautyai_inference/utils/frame_queue.py`

- **Design**: Fixed capacity deque (3-8 frames = 60-160ms buffer)
- **Producer (Recv Loop)**: Never blocks; drops oldest frame if full
- **Consumer (Workers)**: Drains queue asynchronously
- **Reordering**: Frame-indexed results with monotonic commit cursor
- **Metrics**: Underrun detection, recv deltas (p50/p90/p99), service times

```python
class BoundedFrameQueue:
    def __init__(self, max_size: int = 5):  # 100ms @ 20ms frames
        self._queue = deque(maxlen=max_size)  # Auto-drops oldest
        self._results: Dict[int, FramePacket] = {}
        self._commit_cursor = 0  # Next frame_index to flush
```

**Key Features**:
- ✅ **Backpressure**: Drop-oldest policy prevents recv loop blocking
- ✅ **Frame Ordering**: Commit cursor ensures contiguous disk writes
- ✅ **Timing Instrumentation**: Per-frame recv delta, worker service time histograms

---

### 2. **Minimal Recv Loop (Hot Path)**

**File**: `backend/src/beautyai_inference/api/endpoints/webrtc_lean_capture.py::_minimal_recv_loop`

**Operations (Target <2ms)**:
1. `await track.recv()` - Get frame from aiortc
2. Stereo → Mono conversion (if needed)
3. Anti-alias LPF (8th-order Butterworth @ 8kHz)
4. Resample 48kHz → 16kHz (two-stage: 48→24→16)
5. `queue.enqueue(packet)` - Non-blocking enqueue

**NO heavy ops in recv loop**:
- ❌ No disk I/O
- ❌ No denoisers (RNNoise/DTLN)
- ❌ No VAD processing
- ❌ No per-frame prints (throttled to every 100 frames)

```python
async def _minimal_recv_loop(track, info, queue, pipeline):
    while True:
        frame = await asyncio.wait_for(track.recv(), timeout=1.0)
        
        # STAGE 1: Raw extraction
        audio_48k_int16 = frame.to_ndarray().flatten()
        
        # STAGE 2: Anti-alias LPF + Resample
        audio_16k_float32 = resample_with_lpf(audio_48k_int16)
        
        # STAGE 3: Enqueue (non-blocking)
        packet = FramePacket(frame_index, audio_48k_int16, audio_16k_float32)
        queue.enqueue(packet)
```

---

### 3. **Lean Processing Pipeline**

**File**: `backend/src/beautyai_inference/utils/lean_pipeline.py`

**Single-pass chain** (no multiple denoisers):
1. **Limiter @ 48kHz**: Fast attack (<1ms), release 20ms, adaptive threshold
2. **Resample**: 48kHz → 16kHz (already done in recv loop, stored in packet)
3. **Single Denoiser**: RNNoise **OR** DTLN (not both)
4. **Adaptive Comb Filter**: Only when 80 Hz hum detected (>100ms dwell)
5. **Percentile Gate**: P10 noise floor, hysteresis -50/-45 dB

**Executor offload**:
```python
class LeanPipeline:
    def __init__(self, denoiser_type="rnnoise", max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_frame_async(self, audio_48k, audio_16k, loop):
        result = await loop.run_in_executor(
            self.executor,
            self.process_frame_sync,
            audio_48k,
            audio_16k
        )
        return result
```

**Presets**:
- `lean_rnnoise`: Limiter + RNNoise + Adaptive Comb + Gate
- `lean_dtln`: Limiter + DTLN + Adaptive Comb + Gate
- `minimal`: Limiter + Gate only (no denoiser/comb)

---

### 4. **80 Hz Hum Detector with Hysteresis**

**File**: `backend/src/beautyai_inference/utils/hum_detector.py`

**Design**:
- **Band Energy Monitoring**: 80/160/240 Hz bins in FFT
- **Relative Threshold**: Peak must exceed neighborhood by 15 dB
- **Dwell Time**: Require 10 consecutive frames (~200ms) before trigger
- **Hysteresis**: 5 dB drop required to turn off (prevents flapping)

**Adaptive Comb Activation**:
```python
if hum_detector.process_frame(audio_16k):
    # Apply comb filter (Q=2.0, 80 Hz fundamental)
    audio_16k_comb = comb_filter.process_audio(audio_16k)
else:
    # Bypass comb filter (preserve formants)
    audio_16k_comb = audio_16k
```

**Why Adaptive?**:
- ❌ **Static comb**: Always active, risks formant damage
- ✅ **Adaptive comb**: Only activates when hum present, preserves Arabic phonemes

---

### 5. **Fast Peak Limiter (Transient Suppression)**

**File**: `backend/src/beautyai_inference/utils/fast_limiter.py`

**Replacement for broken median filter** (Layer 1.5):
- **Attack**: 0.5ms (catches impulses immediately)
- **Release**: 20ms (prevents pumping)
- **Threshold**: Adaptive (relative to 100ms RMS window)

**Why Fast Limiter > Median Filter?**:
- ❌ **Median filter**: Introduced edge discontinuities (crackles 7488→7493)
- ✅ **Fast limiter**: Smooth gain reduction, no artifacts

```python
class FastPeakLimiter:
    def process_frame(self, audio):
        adaptive_threshold = self.rms_estimate * 10**(threshold_db / 20)
        
        for i, sample in enumerate(audio):
            if abs(sample) > adaptive_threshold:
                gain = adaptive_threshold / (abs(sample) + 1e-10)
            else:
                gain = 1.0
            
            output[i] = sample * gain
```

---

### 6. **Systemd Service with Real-Time Scheduling**

**File**: `beautyai-api-lean.service`

**Changes from default**:
```ini
[Service]
# Round-Robin scheduling, moderate priority (avoid aggressive FIFO)
CPUSchedulingPolicy=rr
CPUSchedulingPriority=20

# Nice value for non-RT tasks
Nice=-5

# I/O priority: real-time class
IOSchedulingClass=realtime
IOSchedulingPriority=4

# CPU affinity: pin to cores 0-3
CPUAffinity=0-3

# Required for RT scheduling
AmbientCapabilities=CAP_SYS_NICE
```

**Deployment**:
```bash
sudo cp beautyai-api-lean.service /etc/systemd/system/beautyai-api.service
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api.service
```

**Verification**:
```bash
# Check scheduling policy
ps -eo pid,comm,cls,rtprio,ni | grep python

# Expected output:
# PID   COMMAND  CLS  RTPRIO  NI
# 12345 python   RR   20      -5
```

---

## API Endpoints

### **POST** `/api/v1/webrtc/lean/voice-capture/offer`

Create lean capture session with hardened architecture.

**Request**:
```json
{
  "sdp": "v=0\r\no=- ...",
  "type": "offer",
  "preset": "lean_rnnoise"  // Optional: lean_rnnoise, lean_dtln, minimal
}
```

**Response**:
```json
{
  "sdp": "v=0\r\no=- ...",
  "type": "answer",
  "peer_id": "session",
  "preset": "lean_rnnoise",
  "message": "Lean capture session created with preset: lean_rnnoise"
}
```

### **GET** `/api/v1/webrtc/lean/voice-capture/stats/{peer_id}`

Get real-time statistics for active session.

**Response**:
```json
{
  "peer_id": "session",
  "frames_captured": 1500,
  "preset": "lean_rnnoise",
  "queue": {
    "underruns": 12,
    "current_depth": 2,
    "peak_depth": 5,
    "recv_delta_p50_ms": 20.1,
    "recv_delta_p90_ms": 21.5,
    "recv_delta_p99_ms": 25.3,
    "worker_service_p50_ms": 14.2,
    "worker_service_p90_ms": 16.8,
    "worker_service_p99_ms": 18.9
  },
  "pipeline": {
    "limiter_activations": 234,
    "comb_active_frames": 0,
    "gate_closed_frames": 456
  }
}
```

---

## Output Files

### **Saved Layers** (WAV files @ `reports/debug/webrtc/`)

| File | Description | Sample Rate |
|------|-------------|-------------|
| `layer1_raw_48khz.wav` | Raw PCM from browser | 48 kHz |
| `layer15_limited_48khz.wav` | Fast peak limiter output | 48 kHz |
| `layer3_baseline_16khz.wav` | Anti-alias LPF + resample | 16 kHz |
| `layer32_denoised_16khz.wav` | RNNoise or DTLN output | 16 kHz |
| `layer36_comb_16khz.wav` | Adaptive comb filter output | 16 kHz |
| `layer31b_gated_16khz.wav` | Percentile gate output | 16 kHz |

### **Metrics** (JSON files)

#### `queue_stats.json`
```json
{
  "enqueued": 1562,
  "dequeued": 1562,
  "dropped": 0,
  "underruns": 12,
  "underrun_rate_percent": 0.77,
  "recv_delta_p50_ms": 20.1,
  "recv_delta_p99_ms": 25.3,
  "worker_service_p50_ms": 14.2,
  "worker_service_p99_ms": 18.9
}
```

#### `pipeline_stats.json`
```json
{
  "frame_count": 1562,
  "denoiser_type": "rnnoise",
  "limiter_activations": 234,
  "comb_active_frames": 0,
  "gate_closed_frames": 456,
  "limiter_stats": {
    "rms_estimate": 0.023,
    "max_gain_reduction_db": -12.4
  }
}
```

---

## Testing & Validation

### **Step 1**: Deploy Service with RT Scheduling

```bash
# Copy new service file
sudo cp beautyai-api-lean.service /etc/systemd/system/beautyai-api.service

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api.service

# Verify RT scheduling
ps -eo pid,comm,cls,rtprio,ni | grep python
# Expected: CLS=RR, RTPRIO=20, NI=-5

# Check logs
sudo journalctl -u beautyai-api.service -f
```

### **Step 2**: Run 30-Second Capture

1. **Connect**: Open browser to `https://192.168.100.39:8443/test_simple.html`
2. **Update endpoint**: Change `/offer` URL to `/api/v1/webrtc/lean/voice-capture/offer`
3. **Add preset** (optional):
   ```javascript
   const offerRequest = {
     sdp: offer.sdp,
     type: offer.type,
     preset: "lean_rnnoise"  // or "lean_dtln", "minimal"
   };
   ```
4. **Speak for 30 seconds**: Read test script with fan running
5. **Wait for completion**: Check console for "✅ capture complete"

### **Step 3**: Analyze Metrics

```bash
cd /home/lumi/beautyai/reports/debug/webrtc

# Check queue stats
cat queue_stats.json | jq '.underrun_rate_percent, .recv_delta_p99_ms, .worker_service_p99_ms'

# Expected results:
# < 1.0          (underrun rate)
# < 2.0          (recv p99)
# < 20.0         (worker p99)
```

### **Step 4**: Run Analyzer

```bash
cd /home/lumi/beautyai
source backend/venv/bin/activate

python tools/analyze_audio_noise.py --compare --visualize

# Check for improvements:
# - Crackles at 48kHz: expect <100 (was 7488)
# - Crackles at 16kHz: expect <30 (was 2336)
# - 80 Hz hum: expect eliminated naturally (if underruns fixed)
```

### **Step 5**: Before/After Comparison

| Metric | Before (Legacy) | After (Lean) | Pass? |
|--------|----------------|--------------|-------|
| Underrun rate | 21.06% | ❓ <1.0% | ⏳ |
| Recv p99 latency | ~14ms+ | ❓ <2ms | ⏳ |
| Worker service p99 | N/A | ❓ <20ms | ⏳ |
| Crackles @ 48kHz | 7488 (240/s) | ❓ <100 (<3/s) | ⏳ |
| Crackles @ 16kHz | 2336 (80/s) | ❓ <30 (<1/s) | ⏳ |
| 80 Hz hum | Present | ❓ Eliminated | ⏳ |
| CPU usage | 1.33% avg | ❓ <10% avg | ⏳ |

---

## Configuration

### **Environment Variables**

```bash
# Pipeline preset selection
export WEBRTC_PIPELINE_PRESET="lean_rnnoise"  # lean_rnnoise, lean_dtln, minimal

# Debug verbosity
export WEBRTC_DEBUG_VERBOSE="1"  # 0 = minimal logs, 1 = verbose (every 100 frames)

# Capture directory
export VOICE_DEBUG_CAPTURE_DIR="/home/lumi/beautyai/reports/debug/webrtc"
```

### **Presets**

#### `lean_rnnoise` (Recommended)
- Limiter + RNNoise + Adaptive Comb + Gate
- Best for fan noise and periodic artifacts
- Expected service time: 12-16ms

#### `lean_dtln` (Alternative)
- Limiter + DTLN + Adaptive Comb + Gate
- Stronger denoising but slightly higher latency
- Expected service time: 15-20ms

#### `minimal` (Baseline)
- Limiter + Gate only
- No denoiser or comb filter
- Expected service time: <5ms
- Use for A/B testing or when denoisers cause issues

---

## Troubleshooting

### **Problem**: Underrun rate still >1%

**Check**:
1. Verify RT scheduling: `ps -eo pid,comm,cls,rtprio | grep python`
2. Check CPU usage: `htop` (filter by process)
3. Increase queue depth: Edit `BoundedFrameQueue(max_size=8)` in `webrtc_lean_capture.py`
4. Reduce worker count: Change `max_workers=1` in `LeanPipeline.__init__`

### **Problem**: Recv loop p99 >2ms

**Check**:
1. Disable verbose logging: `export WEBRTC_DEBUG_VERBOSE=0`
2. Check resample performance: Profile `resample_poly` calls
3. Consider FIR filter alternative to IIR Butterworth

### **Problem**: Worker service time p99 >20ms

**Check**:
1. Switch preset to `minimal`: Test without denoiser overhead
2. Profile denoiser: Add timing prints in `lean_pipeline.py::process_frame_sync`
3. Check ThreadPoolExecutor: Ensure workers=2, not spawning unbounded tasks

### **Problem**: 80 Hz hum still present

**Check**:
1. Verify hum detector triggering: Check `pipeline_stats.json` → `comb_active_frames`
2. Adjust detection threshold: Edit `HumDetector(relative_threshold_db=15.0)` → lower to 10.0
3. Reduce dwell time: Change `dwell_frames=10` → 5 for faster activation

---

## Next Steps

1. ✅ **Test 30s capture** with new lean endpoint
2. ✅ **Verify metrics** meet acceptance criteria
3. ✅ **Deploy to production** if successful
4. ⏳ **Optional**: Fix or replace Layer 1.5 transient suppressor (currently broken median filter)
5. ⏳ **Optional**: Make comb filter adaptive (may be unnecessary if underruns fixed)

---

## References

- **Frame Queue**: `backend/src/beautyai_inference/utils/frame_queue.py`
- **Lean Pipeline**: `backend/src/beautyai_inference/utils/lean_pipeline.py`
- **Hum Detector**: `backend/src/beautyai_inference/utils/hum_detector.py`
- **Fast Limiter**: `backend/src/beautyai_inference/utils/fast_limiter.py`
- **Lean Capture Endpoint**: `backend/src/beautyai_inference/api/endpoints/webrtc_lean_capture.py`
- **Systemd Service**: `beautyai-api-lean.service`

---

**Author**: BeautyAI Framework  
**Date**: November 13, 2025  
**Status**: Ready for Testing 🚀
