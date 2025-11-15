#!/bin/bash
# Quick demo: Capture 30 seconds of audio from HT813

echo "🎤 HT813 Audio Capture Demo"
echo "==========================="
echo
echo "This will capture RTP audio for 30 seconds."
echo "Make sure someone is making a call through the HT813!"
echo
echo "Press Enter to start, or Ctrl+C to cancel..."
read

echo "🎯 Starting capture for 30 seconds..."
echo

cd /home/lumi/beautyai/pabx

# Run capture
sudo venv/bin/python3 ht813_audio_capture.py -d 30

echo
echo "🔄 Converting captured audio to WAV..."
echo

# Convert to WAV
venv/bin/python3 convert_rtp_to_wav.py captures/ --all

echo
echo "✅ Done! Check the captures/ directory for results:"
echo
ls -lh captures/*/audio.wav 2>/dev/null || echo "   No audio files found (no calls were captured)"

echo
echo "To play the audio:"
echo "  aplay captures/session_*/audio.wav"
echo
