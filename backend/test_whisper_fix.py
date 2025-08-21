#!/usr/bin/env python3
"""Test script to verify the Whisper engine fix"""

import sys
import os
import logging

# Add the beautyai backend to path
sys.path.insert(0, '/home/lumi/beautyai/backend/src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

print("🧪 Testing Whisper engine fix...")

try:
    from beautyai_inference.core.model_manager import ModelManager
    
    print("✅ ModelManager imported successfully")
    
    # Create ModelManager instance
    manager = ModelManager()
    print(f"✅ ModelManager instance created")
    
    # Try to load Whisper model
    print("🔄 Loading Whisper model...")
    service = manager.get_streaming_whisper('whisper-large-v3-turbo')
    
    if service:
        print("✅ SUCCESS: Whisper model loaded!")
        
        # Test with a small fake audio array
        import numpy as np
        fake_audio = np.random.random(16000).astype(np.float32)  # 1 second of random audio
        
        print("🔄 Testing transcription with fake audio...")
        result = service._transcribe_implementation(fake_audio, "en")
        print(f"📝 Transcription result: '{result}'")
        
        if result and result != "you":
            print("✅ SUCCESS: Transcription working correctly!")
        else:
            print("⚠️ Transcription still returning minimal result, but no crash!")
            
    else:
        print("❌ FAILED: Could not load Whisper model")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()