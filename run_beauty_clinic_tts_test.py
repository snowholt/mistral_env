#!/usr/bin/env python3
"""
Quick test script for Arabic beauty clinic TTS scenarios.
Run this to test OuteTTS with 5 beauty clinic scenarios in Arabic only.
"""

import sys
import os

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

def main():
    """Run the Arabic beauty clinic TTS test."""
    print("🎙️ Starting Arabic Beauty Clinic TTS Test")
    print("="*60)
    
    try:
        # Import and run the main test
        import test_oute_tts
        test_oute_tts.main()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Arabic Beauty Clinic TTS test completed!")
        print("🎵 Check /home/lumi/beautyai/voice_tests/ for audio files")
    else:
        print("\n❌ Test failed - check errors above")
