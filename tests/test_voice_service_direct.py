#!/usr/bin/env python3
"""
Test Voice-to-Voice Service Directly
"""
import sys
import os
sys.path.append('/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService
import logging

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_voice_service_directly():
    """Test the voice service directly to isolate issues."""
    
    print("🧪 Testing Voice-to-Voice Service Directly")
    print("=" * 50)
    
    try:
        # Initialize service
        print("📱 Initializing VoiceToVoiceService...")
        service = VoiceToVoiceService()
        
        # Check model status
        models_loaded = service._validate_models_loaded()
        print(f"🔍 Models loaded status: {models_loaded}")
        
        if not models_loaded:
            print("🔧 Initializing models...")
            init_result = service.initialize_models(
                stt_model="whisper-large-v3-turbo-arabic",
                tts_model="coqui-tts-arabic", 
                chat_model="qwen3-unsloth-q4ks",
                language="ar"
            )
            print(f"🔍 Initialization result: {init_result}")
            
            if not all(init_result.values()):
                print(f"❌ Failed to initialize some models: {init_result}")
                return False
        
        # Test audio file
        audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav"
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"🎵 Using audio file: {audio_file}")
        
        # Read audio bytes
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        print(f"📏 Audio size: {len(audio_bytes)} bytes")
        
        # Test voice_to_voice_bytes method
        print("🚀 Calling voice_to_voice_bytes...")
        
        result = service.voice_to_voice_bytes(
            audio_bytes=audio_bytes,
            audio_format="wav",
            session_id="direct_test",
            input_language="auto",
            output_language="auto",
            speaker_voice="female",
            enable_content_filter=True,
            content_filter_strictness="balanced",
            thinking_mode=False,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "max_new_tokens": 128
            }
        )
        
        print("✅ Method completed!")
        print(f"🔍 Result success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"🎤 Transcription: {result.get('transcription', 'N/A')}")
            print(f"🤖 Response: {result.get('response_text', 'N/A')}")
            print(f"🎵 Audio output: {result.get('audio_output_path', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_service_directly()
    print("\n" + "=" * 50)
    if success:
        print("🎉 Direct test PASSED!")
    else:
        print("❌ Direct test FAILED!")
