#!/usr/bin/env python3
"""Quick test to transcribe q7.wav using persistent Whisper engine"""

import sys
sys.path.insert(0, '/home/lumi/beautyai/backend/src')

from beautyai_inference.core.persistent_model_manager import get_persistent_model_manager

# Get persistent model manager
mgr = get_persistent_model_manager()

# Get Whisper model
whisper = mgr.get_whisper_model()

# Transcribe the WAV file
with open('/home/lumi/beautyai/tests/webrtc/q7.wav', 'rb') as f:
    audio_bytes = f.read()

print("Transcribing q7.wav...")
result = whisper.transcribe_audio_bytes(audio_bytes, audio_format='wav', language='ar')
print(f"\nTranscription result:\n{result}")
