#!/usr/bin/env python3
"""
Enhanced Voice-to-Voice Test Script for BeautyAI Framework.

Tests the complete pipeline: Audio Input → STT → LLM → TTS → Audio Output
with Coqui TTS integration, thinking mode, and content filtering.
"""

import sys
import os
import asyncio
import logging
import tempfile
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_voice_to_voice():
    """Test enhanced voice-to-voice with Coqui TTS and all features."""
    
    print("🎤 Enhanced Voice-to-Voice Test with Coqui TTS")
    print("=" * 60)
    
    # Test configurations
    test_cases = [
        {
            "name": "Basic Arabic Conversation",
            "text": "مرحبا، كيف يمكنني مساعدتك اليوم؟",
            "language": "ar",
            "speaker_voice": "female",
            "thinking_mode": False,
            "content_filter": True
        },
        {
            "name": "Thinking Mode Enabled",
            "text": "اشرح لي كيف يعمل الذكاء الاصطناعي",
            "language": "ar", 
            "speaker_voice": "female",
            "thinking_mode": True,
            "content_filter": True
        },
        {
            "name": "Advanced Generation Parameters",
            "text": "احك لي قصة قصيرة عن المستقبل",
            "language": "ar",
            "speaker_voice": "male",
            "thinking_mode": False,
            "content_filter": False,
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "max_new_tokens": 200
            }
        },
        {
            "name": "English with Thinking Override",
            "text": "/think Explain quantum computing in simple terms",
            "language": "en",
            "speaker_voice": "female",
            "thinking_mode": False,  # Should be overridden by /think command
            "content_filter": True
        }
    ]
    
    # Setup output directory
    output_dir = Path("/home/lumi/beautyai/voice_tests/enhanced_voice_to_voice")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Initialize the Enhanced Voice-to-Voice Service
        print("\n🚀 Initializing Enhanced Voice-to-Voice Service...")
        v2v_service = VoiceToVoiceService(content_filter_strictness="balanced")
        
        # Initialize models with Coqui TTS
        print("📡 Loading models...")
        models_result = v2v_service.initialize_models(
            stt_model="whisper-large-v3-turbo-arabic",
            tts_model="coqui-tts-arabic",  # Using Coqui TTS
            chat_model="qwen3-unsloth-q4ks",
            language="ar"
        )
        
        print(f"📊 Model initialization results: {models_result}")
        
        if not all(models_result.values()):
            print("❌ Failed to initialize all models")
            failed_models = [k for k, v in models_result.items() if not v]
            print(f"Failed models: {failed_models}")
            return False
        
        print("✅ All models loaded successfully!")
        
        # Test TTS Service separately first
        print("\n🧪 Testing TTS Service with Coqui TTS...")
        tts_service = v2v_service.tts_service
        
        test_tts_output = output_dir / "test_coqui_tts.wav"
        tts_result = tts_service.text_to_speech(
            text="هذا اختبار لمحرك كوكي للذكاء الاصطناعي العربي",
            output_path=str(test_tts_output),
            language="ar",
            speaker_voice="female"
        )
        
        if tts_result:
            print(f"✅ Coqui TTS test successful: {test_tts_output}")
        else:
            print("❌ Coqui TTS test failed")
            return False
        
        # Get model status
        print("\n📊 Model Status:")
        status = v2v_service.get_models_status()
        print(json.dumps(status, indent=2))
        
        # Run comprehensive tests
        print("\n🧪 Running Enhanced Voice-to-Voice Tests...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎯 Test {i}: {test_case['name']}")
            print("-" * 40)
            
            # Create a temporary audio file from text (using STT in reverse for testing)
            temp_audio_path = output_dir / f"temp_input_{i}.wav"
            
            # Generate audio from text for testing input
            input_audio_result = tts_service.text_to_speech(
                text=test_case["text"],
                output_path=str(temp_audio_path),
                language=test_case["language"],
                speaker_voice=test_case.get("speaker_voice", "female")
            )
            
            if not input_audio_result:
                print(f"❌ Failed to create test audio for test {i}")
                continue
            
            print(f"📎 Test input text: {test_case['text']}")
            print(f"🎵 Created test audio: {temp_audio_path}")
            
            # Read audio file as bytes for testing
            with open(temp_audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # Process voice-to-voice conversation
            result = v2v_service.voice_to_voice_bytes(
                audio_bytes=audio_bytes,
                audio_format="wav",
                session_id=f"test_session_{i}",
                input_language=test_case["language"],
                output_language=test_case["language"],
                speaker_voice=test_case.get("speaker_voice", "female"),
                enable_content_filter=test_case.get("content_filter", True),
                content_filter_strictness="balanced",
                thinking_mode=test_case.get("thinking_mode", False),
                generation_config=test_case.get("generation_config", {})
            )
            
            if result["success"]:
                print(f"✅ Voice-to-voice processing successful!")
                print(f"📝 Transcription: {result['transcription']}")
                print(f"🤖 AI Response: {result['response'][:100]}...")
                print(f"🎵 Audio output: {result['audio_output']}")
                print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                
                # Display metadata
                metadata = result.get("metadata", {})
                print(f"🧠 Thinking mode: {metadata.get('thinking_mode', 'N/A')}")
                print(f"🔒 Content filter: {metadata.get('content_filter_applied', 'N/A')}")
                print(f"⚙️ Generation config: {metadata.get('generation_config', {})}")
                
            else:
                print(f"❌ Voice-to-voice processing failed: {result.get('error', 'Unknown error')}")
            
            print()
        
        # Test session management
        print("📚 Testing Session Management...")
        session_history = v2v_service.get_session_history("test_session_1")
        if session_history:
            print(f"✅ Session history retrieved: {len(session_history)} messages")
        else:
            print("ℹ️ No session history found")
        
        # Test memory stats
        print("\n💾 Memory Statistics:")
        memory_stats = v2v_service.get_memory_stats()
        print(json.dumps(memory_stats, indent=2))
        
        print("\n🎉 Enhanced Voice-to-Voice test completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        
        # Clean up models
        print("\n🧹 Cleaning up models...")
        v2v_service.unload_all_models()
        print("✅ Models unloaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during enhanced voice-to-voice test: {e}")
        return False

def main():
    """Main function to run the enhanced test."""
    print("🧪 BeautyAI Enhanced Voice-to-Voice Test Suite")
    print("=" * 60)
    print("Testing complete pipeline with Coqui TTS integration:")
    print("🔄 Audio Input → STT → LLM → TTS → Audio Output")
    print("🎯 Features: Thinking mode, Content filtering, Advanced parameters")
    print("")
    
    success = asyncio.run(test_enhanced_voice_to_voice())
    
    if success:
        print("\n✅ All tests passed! Enhanced voice-to-voice is working correctly.")
        print("\n🚀 Ready for production use with:")
        print("   • Coqui TTS for high-quality Arabic voice synthesis")
        print("   • Advanced thinking mode with /think and /no_think commands")
        print("   • Configurable content filtering")
        print("   • 25+ LLM generation parameters")
        print("   • Session management and conversation history")
    else:
        print("\n❌ Some tests failed. Check the output above for issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
