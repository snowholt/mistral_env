# WebRTC + VAD Testing Guide

This directory contains tools for testing WebRTC Voice Activity Detection (VAD) without loading heavy ML models (Whisper, LLM, TTS).

## Quick Start

### 1. Run VAD Test

```bash
# From repository root
./tests/webrtc/run_vad_test.sh
```

This will:
- Start test server with mock models
- Stream `laser_hair.wav` via WebRTC
- Monitor VAD detection (WebRTC + Silero)
- Generate test report and debug audio

### 2. Test Different VAD Scenarios

```bash
# Dual VAD (WebRTC + Silero confirmation)
./tests/webrtc/run_vad_test.sh dual_vad

# Silero-only mode (ML-based)
./tests/webrtc/run_vad_test.sh silero_only

# WebRTC-only mode (rule-based)
./tests/webrtc/run_vad_test.sh webrtc_only
```

### 3. Manual Server Testing

```bash
# Terminal 1: Start server
cd backend
VAD_TEST_MODE=1 python run_vad_test_server.py

# Terminal 2: Run test
cd tests/webrtc
pytest webrtc_audio_test.py -v -s
```

## Test Configuration

### Environment Variables

- `VAD_TEST_MODE=1` - Enable mock models
- `BEAUTYAI_VAD_DEBUG=1` - Enable VAD debug logging
- `VAD_DUAL_MODE=true/false` - Enable/disable WebRTC VAD
- `VAD_REQUIRE_CONFIRMATION=true/false` - Require Silero confirmation

### Configuration File

`config/config.vad_test.yaml` - Main test configuration:
- WebRTC enabled, models disabled
- VAD sensitivity settings
- Warmup filter configuration
- Debug logging enabled

## Test Artifacts

After running tests, check:

```
logs/
├── vad_test_server.log       # Server logs
├── vad_test_run.log          # Test execution logs
├── vad_test_metrics.json     # Performance metrics
└── webrtc/
    └── vad_debug/            # VAD-detected audio segments

reports/webRTC-VAD/
├── captured_webrtc_segment_*.wav  # Server-captured audio
└── vad_test_report_*.md           # Test reports
```

## Understanding VAD Output

### VAD States

1. **INACTIVE** - No voice detected
2. **VOICE_START** - Initial voice detected
3. **VOICE_ACTIVE** - Confirmed voice activity
4. **VOICE_END_PENDING** - Silence detected, waiting
5. **VOICE_END** - Confirmed end of speech

### Log Patterns

```log
[WEBRTC-VAD] detected=True   # WebRTC VAD triggered
[SILERO-VAD] prob=0.85       # Silero confidence score
[SUSTAINED] 3/3 frames       # Sustained speech confirmed
[WARMUP] Complete            # Warmup period finished
```

### VAD Metrics

Check server logs for:
- **WebRTC detections**: Fast initial detection count
- **Silero confirmations**: ML-confirmed speech count
- **False positives**: WebRTC detected but Silero rejected
- **Speech segments**: Number of complete utterances

## Debugging VAD Issues

### 1. Check Audio Pipeline

```bash
# Verify audio file is valid
file tests/webrtc/laser_hair.wav
soxi tests/webrtc/laser_hair.wav

# Check transmitted audio
ls -lh reports/webRTC-VAD/captured_webrtc_segment_*.wav
```

### 2. Analyze VAD Debug Audio

VAD saves detected speech segments to `logs/webrtc/vad_debug/`:

```bash
# List debug audio
ls -lh logs/webrtc/vad_debug/*.wav

# Play detected segments
play logs/webrtc/vad_debug/*_silero.wav
```

### 3. Review Server Logs

```bash
# Watch server logs in real-time
tail -f logs/vad_test_server.log

# Search for VAD decisions
grep "VAD" logs/vad_test_server.log

# Find state transitions
grep "state.*->" logs/vad_test_server.log
```

### 4. Common Issues

**No VAD Detection**:
- Check warmup filter (250ms initial silence required)
- Verify audio is not silent: `soxi captured_audio.wav`
- Lower Silero threshold in config

**False Positives**:
- Increase Silero threshold (0.3 → 0.5)
- Enable `require_silero_confirmation: true`
- Adjust WebRTC sensitivity (2 → 3 less aggressive)

**Delayed Detection**:
- Reduce `min_sustained_speech_frames` (3 → 2)
- Lower `min_speech_duration_ms` (30 → 20)
- Check warmup filter timing

## Test Audio File

**laser_hair.wav**:
- Duration: ~5 seconds
- Content: English narration ("How does laser hair removal work?")
- Sample Rate: 48kHz (resampled from 16kHz)
- Format: WAV PCM 16-bit mono
- Purpose: Tests VAD with clear speech after initial silence

## Architecture

```
┌─────────────────┐
│   Test Client   │
│  (Python Test)  │
└────────┬────────┘
         │ WebRTC
         ▼
┌─────────────────┐
│  VAD Test Server│
│   (FastAPI)     │
├─────────────────┤
│ Mock Whisper    │  ← Returns dummy transcription
│ Mock LLM        │  ← Returns dummy response
│ Mock TTS        │  ← Returns dummy audio
├─────────────────┤
│ WebRTC VAD      │  ← Fast detection (webrtcvad)
│ Silero VAD      │  ← ML confirmation (torch)
└─────────────────┘
```

## Mock Models

Mock models are used when `VAD_TEST_MODE=1`:

- **MockWhisperModel**: Returns `"[MOCK EN] Mock transcription #N"`
- **MockLLMModel**: Returns `"[MOCK LLM RESPONSE #N] ..."`
- **MockTTSModel**: Returns 1s of quiet sine wave audio

Benefits:
- No GPU required
- No model downloads
- Fast startup (<1s)
- Focus on VAD behavior

## CI/CD Integration

Add to your pipeline:

```yaml
- name: Test WebRTC VAD
  run: |
    source .venv/bin/activate
    ./tests/webrtc/run_vad_test.sh
```

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
lsof -i :8000

# Check Python environment
source .venv/bin/activate
python -c "import aiortc; print('aiortc OK')"
```

### Audio Not Streaming

```bash
# Check WebRTC dependencies
pip list | grep aiortc
pip list | grep webrtcvad

# Verify audio file
python -c "import soundfile as sf; print(sf.info('tests/webrtc/laser_hair.wav'))"
```

### No VAD Debug Output

Ensure these are set:
```bash
export BEAUTYAI_VAD_DEBUG=1
export VAD_TEST_MODE=1
```

Check config:
```yaml
webrtc:
  debug_logging: true
  log_vad_decisions: true

vad_test:
  vad_debug_enabled: true
  vad_save_debug_audio: true
```

## Next Steps

After successful VAD testing:

1. **Enable Real Models**: Remove `VAD_TEST_MODE=1`
2. **Test with Whisper**: Use actual STT transcription
3. **Add LLM**: Test full voice-to-voice pipeline
4. **Production Config**: Tune VAD thresholds for your use case

## References

- **VAD Service**: `backend/src/beautyai_inference/services/voice/vad/webrtc_vad_service.py`
- **Test Server**: `backend/run_vad_test_server.py`
- **Mock Models**: `backend/src/beautyai_inference/services/voice/mock_models.py`
- **Test Config**: `config/config.vad_test.yaml`

---

**Created**: October 29, 2025  
**Branch**: WebRTC-VAD  
**Purpose**: Debug WebRTC + VAD without heavy model dependencies
