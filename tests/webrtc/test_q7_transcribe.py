#!/usr/bin/env python3
"""Test direct Whisper transcription with q7 audio files to verify Whisper accuracy"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/home/lumi/beautyai/backend/src')

from beautyai_inference.core.persistent_model_manager import get_persistent_model_manager

# Test files
test_files = [
    ("/home/lumi/beautyai/tests/webrtc/q7.wav", "wav", "tests/webrtc/q7.wav (24kHz)"),
    ("/home/lumi/beautyai/tests/webrtc/q7.pcm", "pcm", "tests/webrtc/q7.pcm (16kHz)"),
    ("/home/lumi/beautyai/voice_tests/input_test_questions/pcm/q7.pcm", "pcm", "voice_tests/q7.pcm (16kHz)"),
]

print("=" * 80)
print("Direct Whisper Transcription Test")
print("=" * 80)

# Get persistent model manager
print("\n1. Loading persistent Whisper model...")
mgr = get_persistent_model_manager()
whisper = mgr.get_whisper_model()
print(f"   ✓ Whisper model loaded: {whisper.__class__.__name__}")

# Test each audio file
for file_path, audio_format, description in test_files:
    print(f"\n{'=' * 80}")
    print(f"Testing: {description}")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ⚠️  File not found, skipping...")
        continue
    
    file_size = os.path.getsize(file_path)
    
    # Calculate duration for PCM (16kHz, mono, 16-bit = 32000 bytes/sec)
    if audio_format == "pcm":
        duration = file_size / 32000
        print(f"   Size: {file_size:,} bytes ({duration:.2f}s @ 16kHz mono)")
    else:
        print(f"   Size: {file_size:,} bytes")
    
    # Read and transcribe
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    
    print(f"\n2. Transcribing with language='ar'...")
    try:
        result = whisper.transcribe_audio_bytes(audio_bytes, audio_format=audio_format, language='ar')
        print(f"   ✓ Transcription successful")
        print(f"\n   Result: {result.get('text', 'NO TEXT')}")
        
        if 'segments' in result:
            print(f"   Segments: {len(result['segments'])}")
            for i, seg in enumerate(result['segments'][:3]):  # Show first 3 segments
                print(f"      [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
    
    except Exception as e:
        print(f"   ✗ Transcription FAILED: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 80}")
print("Test complete!")
print("=" * 80)
