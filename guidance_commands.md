# Audio Testing Quick Commands

## 1. Play Audio Files (Command Line)

```bash
# Play a single audio file
ffplay -nodisp -autoexit reports/debug/webrtc/layer1_48000hz_raw.wav

# Play Layer 1.5 (transient suppressed)
ffplay -nodisp -autoexit reports/debug/webrtc/layer15_transient_48000hz.wav

# Play Layer 3 (baseline 16kHz)
ffplay -nodisp -autoexit reports/debug/webrtc/layer3_16khz.wav

# Play Layer 3.1b (percentile gate)
ffplay -nodisp -autoexit reports/debug/webrtc/layer31b_percentile_16khz.wav

# Play Layer 3.6 (comb filter Q=2.0)
ffplay -nodisp -autoexit reports/debug/webrtc/layer36_comb_16khz.wav

# Alternative: Use aplay (simpler)
aplay reports/debug/webrtc/layer3_16khz.wav
```

## 2. Analyze Audio Files

```bash
# Full analysis with visualizations
python tools/analyze_audio_noise.py --compare --visualize

# Analyze specific layers only (8 layers: 1, 15, 2, 3, 31, 31b, 32, 36)
python tools/analyze_audio_noise.py --compare

# Run diagnostic tool for periodicity measurement
python tools/diagnose_periodic_noise.py --file reports/debug/webrtc/layer3_16khz.wav

# Check comb filter effectiveness
python tools/diagnose_periodic_noise.py --file reports/debug/webrtc/layer36_comb_16khz.wav
```

## 3. Quick Audio Info

```bash
# Get audio file details
ffprobe reports/debug/webrtc/layer3_16khz.wav

# Check all captured layers
ls -lh reports/debug/webrtc/*.wav

# Count crackles/transients
python tools/diagnose_periodic_noise.py --file reports/debug/webrtc/layer1_48000hz_raw.wav
```
