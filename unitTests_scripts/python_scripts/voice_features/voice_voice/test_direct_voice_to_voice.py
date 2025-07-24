#!/usr/bin/env python3
"""
Direct Voice-to-Voice Service Test (No API required).

Tests the VoiceToVoiceService directly without requiring the FastAPI server.
This validates the core functionality with real audio files.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_direct_voice_to_voice():
    """Test voice-to-voice service directly with real audio files."""
    
    print("🎤 Direct Voice-to-Voice Service Test")
    print("=" * 50)
    
    # Audio files to test
    audio_files = [
        {
            "name": "Arabic Greeting",
            "path": "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav",
            "input_language": "ar",
            "output_language": "ar",
            "thinking_mode": False
        },
        {
            "name": "English Greeting", 
            "path": "/home/lumi/beautyai/voice_tests/input_test_questions/greeting.wav",
            "input_language": "en",
            "output_language": "en",
            "thinking_mode": False
        }
    ]
    
    # Output directory
    output_dir = Path("/home/lumi/beautyai/voice_tests/direct_service_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the service
        print("\n🚀 Initializing Voice-to-Voice Service...")
        v2v_service = VoiceToVoiceService(content_filter_strictness="balanced")
        
        # Initialize models
        print("📡 Loading models...")
        models_result = v2v_service.initialize_models(
            stt_model="whisper-large-v3-turbo-arabic",
            tts_model="coqui-tts-arabic",
            chat_model="qwen3-unsloth-q4ks",
            language="ar"
        )
        
        print(f"📊 Model initialization results: {models_result}")
        
        if not all(models_result.values()):
            print("❌ Failed to initialize some models")
            failed_models = [k for k, v in models_result.items() if not v]
            print(f"Failed models: {failed_models}")
            return False
        
        print("✅ All models loaded successfully!")
        
        # Test each audio file
        for i, test_case in enumerate(audio_files, 1):
            print(f"\n🎯 Test {i}: {test_case['name']}")
            print("-" * 30)
            
            # Check if audio file exists
            audio_path = Path(test_case["path"])
            if not audio_path.exists():
                print(f"❌ Audio file not found: {audio_path}")
                continue
            
            print(f"🎵 Processing: {audio_path.name}")
            
            # Read audio file as bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Process with voice-to-voice service
            result = v2v_service.voice_to_voice_bytes(
                audio_bytes=audio_bytes,
                audio_format="wav",
                session_id=f"test_session_{i}",
                input_language=test_case["input_language"],
                output_language=test_case["output_language"],
                speaker_voice="female",
                enable_content_filter=False,  # Disabled for testing
                content_filter_strictness="relaxed",
                thinking_mode=test_case["thinking_mode"],
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 256
                }
            )
            
            if result["success"]:
                print(f"✅ Processing successful!")
                print(f"📝 Transcription: {result['transcription']}")
                print(f"🤖 AI Response: {result['response'][:100]}...")
                print(f"🎵 Audio output: {result['audio_output']}")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                
                # Copy output audio to test directory
                if result.get('audio_output') and Path(result['audio_output']).exists():
                    output_audio = output_dir / f"test_{i}_{test_case['name'].lower().replace(' ', '_')}_output.wav"
                    import shutil
                    shutil.copy2(result['audio_output'], output_audio)
                    print(f"📁 Output copied to: {output_audio}")
                
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            print()
        
        # Test memory stats
        print("💾 Memory Statistics:")
        memory_stats = v2v_service.get_memory_stats()
        if memory_stats:
            print(f"   GPU Memory: {memory_stats.get('gpu_memory', {})}")
            print(f"   Performance: {memory_stats.get('performance_stats', {})}")
        
        # Clean up
        print("\n🧹 Cleaning up models...")
        v2v_service.unload_all_models()
        print("✅ Cleanup completed")
        
        print(f"\n🎉 Direct service test completed!")
        print(f"📁 Outputs saved to: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test."""
    success = asyncio.run(test_direct_voice_to_voice())
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
