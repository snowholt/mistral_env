#!/usr/bin/env python3
"""
Test PCMU codec encoding directly
"""

import sys
sys.path.insert(0, '/home/lumi/beautyai/pabx')

import numpy as np
from src.modules.audio.codecs import PCMUCodec
from src.modules.audio.loader import AudioLoader

def test_encoding():
    print("=" * 60)
    print("Testing PCMU Codec Encoding")
    print("=" * 60)
    
    # Test 1: Simple array
    print("\n1. Testing with simple numpy array...")
    codec = PCMUCodec()
    test_data = np.array([0, 100, -100, 200, -200, 500, -500], dtype=np.int16)
    print(f"   Input: {test_data}")
    
    try:
        encoded = codec.encode(test_data)
        print(f"   Output: {len(encoded)} bytes")
        print(f"   First 10 bytes: {list(encoded[:10])}")
        print("   ✅ Simple encoding works!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Real audio file
    print("\n2. Testing with real audio file...")
    audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav"
    
    try:
        audio_data, sample_rate = AudioLoader.load(audio_file, target_sample_rate=8000, target_channels=1)
        print(f"   Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
        print(f"   Data type: {audio_data.dtype}, shape: {audio_data.shape}")
        print(f"   Min: {audio_data.min()}, Max: {audio_data.max()}, Mean: {audio_data.mean():.2f}")
        
        print(f"   Encoding...")
        encoded = codec.encode(audio_data)
        print(f"   Output: {len(encoded)} bytes")
        print(f"   First 20 bytes: {list(encoded[:20])}")
        print(f"   Expected duration: {len(audio_data) / sample_rate:.2f} seconds")
        print("   ✅ Real audio encoding works!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_encoding()
