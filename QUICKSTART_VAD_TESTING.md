# WebRTC-VAD Branch - Quick Start Guide

## ✅ Branch Created: `WebRTC-VAD`

All changes are safely isolated on the `WebRTC-VAD` branch. Your `local-development` branch is untouched.

---

## 🚀 Quick Start

### 1. **Run VAD Test** (Recommended First Step)

```bash
# Make sure you're on the WebRTC-VAD branch
git branch  # Should show * WebRTC-VAD

# Run the automated test
./tests/webrtc/run_vad_test.sh
```

This will:
- ✅ Start test server with mock models (no GPU needed)
- ✅ Stream `laser_hair.wav` via WebRTC
- ✅ Monitor VAD detection (dual mode: WebRTC + Silero)
- ✅ Generate test report with VAD metrics
- ✅ Save debug audio files

**Expected Duration**: ~1-2 minutes

---

## 📋 What Was Added

### New Files (5 total):

1. **`backend/run_vad_test_server.py`**
   - Lightweight FastAPI server for VAD testing
   - Uses mock models (no Whisper/LLM/TTS loading)
   - Quick startup (<1 second)

2. **`backend/src/beautyai_inference/services/voice/mock_models.py`**
   - MockWhisperModel (dummy transcription)
   - MockLLMModel (dummy responses)
   - MockTTSModel (dummy audio)

3. **`config/config.vad_test.yaml`**
   - WebRTC enabled, models disabled
   - VAD configuration (dual mode settings)
   - Debug logging enabled

4. **`tests/webrtc/run_vad_test.sh`**
   - Automated test runner
   - Starts server, runs tests, collects metrics
   - Generates test reports

5. **`tests/webrtc/README_VAD_TESTING.md`**
   - Comprehensive testing documentation
   - Debugging guide
   - Architecture overview

### Existing Files (Used As-Is):

- ✅ `backend/src/beautyai_inference/api/endpoints/webrtc_voice.py`
- ✅ `backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py`
- ✅ `backend/src/beautyai_inference/services/voice/webrtc_audio_processor.py`
- ✅ `backend/src/beautyai_inference/services/voice/webrtc_voice_service_adapter.py`
- ✅ `backend/src/beautyai_inference/services/voice/service_manager.py`
- ✅ `backend/src/beautyai_inference/services/voice/simple_voice_service.py`
- ✅ `tests/webrtc/webrtc_audio_test.py`
- ✅ `tests/webrtc/laser_hair.wav`

---

## 🎯 Test Scenarios

### Scenario 1: Dual VAD (Default)
```bash
./tests/webrtc/run_vad_test.sh dual_vad
```
- Uses both WebRTC VAD (fast) + Silero VAD (accurate)
- Requires Silero confirmation for speech detection
- Best for production use

### Scenario 2: Silero Only
```bash
./tests/webrtc/run_vad_test.sh silero_only
```
- Uses only Silero VAD (ML-based)
- More accurate but slightly slower
- Good for quality-focused applications

### Scenario 3: WebRTC Only
```bash
./tests/webrtc/run_vad_test.sh webrtc_only
```
- Uses only WebRTC VAD (rule-based)
- Fastest but less accurate
- Good for understanding baseline behavior

---

## 📊 Expected Output

After running the test, you'll see:

```
=================================================
   BeautyAI WebRTC + VAD Test Runner
=================================================
Scenario: dual_vad
...
Starting VAD test server...
Server PID: 12345
Waiting for server to start... OK
...
Running WebRTC audio test with VAD...
...
Test status: PASSED ✓

Generated artifacts:
  ✓ VAD debug audio: 3 file(s) in logs/webrtc/vad_debug
  ✓ Captured audio: 1 file(s) in reports/webRTC-VAD
  ✓ Metrics: logs/vad_test_metrics.json

Logs:
  Server log: logs/vad_test_server.log
  Test log: logs/vad_test_run.log

VAD Statistics:
  WebRTC detections: 45
  Silero detections: 38
  VAD state transitions: 12
  Speech segments: 2
```

---

## 🔍 Analyzing Results

### 1. Check VAD Debug Audio
```bash
# List detected audio segments
ls -lh logs/webrtc/vad_debug/*.wav

# Play Silero-detected segments
play logs/webrtc/vad_debug/*_silero.wav
```

### 2. Review Test Report
```bash
# Open latest test report
cat reports/webRTC-VAD/vad_test_report_*.md
```

### 3. Analyze Server Logs
```bash
# View VAD decisions
grep "VAD" logs/vad_test_server.log | tail -50

# Check state transitions
grep "state.*->" logs/vad_test_server.log

# Monitor WebRTC VAD
grep "WEBRTC-VAD" logs/vad_test_server.log

# Monitor Silero VAD
grep "SILERO-VAD" logs/vad_test_server.log
```

---

## 🐛 Troubleshooting

### Test Fails with "Server timeout"

**Check:**
```bash
# Verify Python environment
source .venv/bin/activate
python --version  # Should be 3.12+

# Check dependencies
pip list | grep -E "aiortc|fastapi|webrtcvad"
```

**Fix:**
```bash
# Reinstall dependencies
pip install aiortc fastapi uvicorn webrtcvad torch
```

### No VAD Detections

**Check audio file:**
```bash
# Verify laser_hair.wav exists and is valid
file tests/webrtc/laser_hair.wav
soxi tests/webrtc/laser_hair.wav
```

**Adjust VAD sensitivity** in `config/config.vad_test.yaml`:
```yaml
webrtc:
  vad_silero_sensitivity: 0.2  # Lower = more sensitive (default: 0.3)
  vad_webrtc_sensitivity: 1    # Lower = more sensitive (default: 2)
```

### Server Won't Start

**Check port availability:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process if needed
kill $(lsof -t -i :8000)
```

---

## 📚 Documentation

- **Full Testing Guide**: `tests/webrtc/README_VAD_TESTING.md`
- **VAD Service Code**: `backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py`
- **Test Configuration**: `config/config.vad_test.yaml`
- **Mock Models**: `backend/src/beautyai_inference/services/voice/mock_models.py`

---

## 🔄 Next Steps

### After Successful Test:

1. **Analyze VAD Behavior**
   - Review debug audio files
   - Check false positive/negative rates
   - Tune sensitivity thresholds

2. **Test with Live Microphone** (Future)
   - Modify test to use browser microphone
   - Test real-time VAD performance
   - Validate latency metrics

3. **Enable Real Models** (Optional)
   - Remove `VAD_TEST_MODE=1`
   - Test with actual Whisper/LLM/TTS
   - Validate full voice-to-voice pipeline

4. **Merge to Main** (When Ready)
   ```bash
   # Switch to local-development
   git checkout local-development
   
   # Merge VAD test infrastructure
   git merge WebRTC-VAD
   ```

---

## ⚠️ Important Notes

- ✅ **All changes are on `WebRTC-VAD` branch** - main codebase is safe
- ✅ **No GPU required** - uses mock models and CPU-only VAD
- ✅ **No model downloads** - lightweight testing (<100MB)
- ✅ **Existing infrastructure unchanged** - uses your WebRTC files as-is
- ✅ **Test with your audio file** - `tests/webrtc/laser_hair.wav`

---

## 🎯 Test Goals

This setup helps you:

1. ✅ **Debug VAD behavior** without model overhead
2. ✅ **Test WebRTC audio streaming** end-to-end
3. ✅ **Validate dual VAD strategy** (WebRTC + Silero)
4. ✅ **Identify detection issues** (false positives/negatives)
5. ✅ **Measure VAD performance** (latency, accuracy)
6. ✅ **Generate debug audio** for analysis

---

## 🚀 Ready to Test?

```bash
# Simple one-command test
./tests/webrtc/run_vad_test.sh

# Or manual testing
cd backend
VAD_TEST_MODE=1 python run_vad_test_server.py
```

**Questions?** Check `tests/webrtc/README_VAD_TESTING.md` for detailed docs!

---

**Branch**: WebRTC-VAD  
**Created**: October 29, 2025  
**Commit**: 4fd2f24 - feat(vad): Add WebRTC + VAD isolated testing infrastructure
