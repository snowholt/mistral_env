#!/bin/bash
"""
Quick Voice-to-Voice API Test Runner
Tests the voice-to-voice endpoint with real Arabic and English audio files.
"""

echo "🎤 BeautyAI Voice-to-Voice API Test Runner"
echo "=========================================="
echo ""
echo "This script will test the voice-to-voice endpoint with:"
echo "  🇸🇦 Arabic audio files: greeting_ar.wav, botox_ar.wav"
echo "  🇺🇸 English audio files: greeting.wav, botox.wav"
echo ""

# Check if Python script exists
if [ ! -f "/home/lumi/beautyai/tests/test_voice_to_voice_api.py" ]; then
    echo "❌ Test script not found!"
    exit 1
fi

# Check if audio files exist
echo "📁 Checking audio files..."
AUDIO_DIR="/home/lumi/beautyai/voice_tests/input_test_questions"
missing_files=0

for file in "greeting_ar.wav" "greeting.wav" "botox_ar.wav" "botox.wav"; do
    if [ ! -f "$AUDIO_DIR/$file" ]; then
        echo "❌ Missing: $file"
        missing_files=$((missing_files + 1))
    else
        echo "✅ Found: $file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ Missing $missing_files audio files. Please ensure all test files are present."
    exit 1
fi

echo ""
echo "🚀 Starting API tests..."
echo ""

# Run the Python test script
cd /home/lumi/beautyai
python tests/test_voice_to_voice_api.py

echo ""
echo "🏁 Test completed! Check the output above for results."
