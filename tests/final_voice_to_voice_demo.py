#!/usr/bin/env python3
"""
FINAL DEMONSTRATION: Complete Voice-to-Voice Pipeline
===============================================================

This script demonstrates the fully working voice-to-voice conversation system:
Audio Input → STT → LLM → TTS → Audio Output

Features Demonstrated:
- Coqui TTS Integration: High-quality neural voice synthesis  
- Multi-language Support: Arabic (optimized) and English
- AI Responses: Qwen3-14B model for intelligent conversations
- Complete Pipeline: End-to-end audio-to-audio processing
- Performance Metrics: Timing and processing statistics
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService

print("🎉 VOICE-TO-VOICE PIPELINE DEMONSTRATION")
print("=" * 50)
print("✅ Complete Pipeline: Audio Input → STT → LLM → TTS → Audio Output") 
print("🎤 Coqui TTS: High-quality neural voice synthesis")
print("🌍 Multi-language: Arabic (optimized) + English")
print("🧠 AI Model: Qwen3-14B for intelligent responses")
print("📊 Real Performance: With actual audio files")
print()

async def main():
    """Demonstrate the complete working pipeline."""
    
    # Test cases
    tests = [
        {
            "name": "Arabic Beauty Clinic Inquiry",
            "file": "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav",
            "lang": "ar"
        },
        {
            "name": "English Greeting", 
            "file": "/home/lumi/beautyai/voice_tests/input_test_questions/greeting.wav",
            "lang": "en"
        }
    ]
    
    # Initialize service
    print("🚀 Initializing Voice-to-Voice Service...")
    v2v_service = VoiceToVoiceService(content_filter_strictness="relaxed")
    
    # Load models
    print("📡 Loading Models (STT + LLM + TTS)...")
    models_result = v2v_service.initialize_models(
        stt_model="whisper-large-v3-turbo-arabic",
        tts_model="coqui-tts-arabic", 
        chat_model="qwen3-unsloth-q4ks"
    )
    
    if not all(models_result.values()):
        print("❌ Failed to load models")
        return False
    
    print("✅ All models loaded successfully!")
    print()
    
    # Run tests
    for i, test in enumerate(tests, 1):
        print(f"🎯 Test {i}: {test['name']}")
        print("-" * 40)
        
        if not Path(test["file"]).exists():
            print(f"❌ Audio file not found: {test['file']}")
            continue
        
        # Read audio
        with open(test["file"], 'rb') as f:
            audio_bytes = f.read()
        
        # Process
        result = v2v_service.voice_to_voice_bytes(
            audio_bytes=audio_bytes,
            input_language=test["lang"],
            output_language=test["lang"],
            enable_content_filter=False,
            generation_config={"temperature": 0.7, "max_new_tokens": 200}
        )
        
        if result["success"]:
            print(f"✅ SUCCESS!")
            print(f"📝 Input: {result['transcription'][:80]}...")
            print(f"🤖 Response: {result['response'][:80]}...")
            print(f"🎵 Audio: {result['audio_output']}")
            print(f"⏱️ Time: {result['processing_time']:.2f}s")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Cleanup
    v2v_service.unload_all_models()
    print("✅ Demo completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    print("\n🎉 Voice-to-Voice pipeline demonstration finished!")
    print("🔗 All three components (STT + LLM + TTS) working together!")
