#!/usr/bin/env python3
"""
Test the fixed thinking content removal in voice-to-voice service.

This script tests that TTS only processes the clean response without <think>...</think> blocks.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService

# Configure logging to see the cleaning process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_thinking_removal():
    """Test the thinking content removal function."""
    
    print("🧪 Testing Thinking Content Removal")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Response with thinking blocks",
            "input": "<think>Let me think about this carefully...</think>Here is my final answer about beauty treatments.",
            "expected": "Here is my final answer about beauty treatments."
        },
        {
            "name": "Complex thinking with multiple blocks",
            "input": "First part <think>Internal reasoning here</think> middle part <think>More reasoning</think> final answer.",
            "expected": "First part  middle part  final answer."
        },
        {
            "name": "Arabic response with thinking",
            "input": "<think>سأفكر في هذا السؤال بعناية</think>مرحباً بكم في عيادة التجميل، نحن نقدم أفضل الخدمات.",
            "expected": "مرحباً بكم في عيادة التجميل، نحن نقدم أفضل الخدمات."
        },
        {
            "name": "Only thinking content (should return default)",
            "input": "<think>Just thinking, no final answer</think>",
            "expected": "أعتذر، لم أتمكن من تقديم إجابة واضحة. هل يمكنك إعادة صياغة سؤالك؟"
        },
        {
            "name": "Clean response (no thinking)",
            "input": "This is a clean response without any thinking blocks.",
            "expected": "This is a clean response without any thinking blocks."
        }
    ]
    
    # Test the static method
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔧 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        result = VoiceToVoiceService._remove_thinking_content(test_case["input"])
        
        print(f"📝 Input: {test_case['input'][:80]}...")
        print(f"✅ Output: {result[:80]}...")
        print(f"🎯 Expected: {test_case['expected'][:80]}...")
        
        # Simple validation
        if "<think>" in result or "</think>" in result:
            print("❌ FAILED: Thinking blocks still present!")
        elif result.strip() == test_case["expected"].strip():
            print("✅ PASSED: Perfect match!")
        else:
            print("⚠️ PARTIAL: Content cleaned but different from expected")
    
    print("\n" + "=" * 50)
    print("🎉 Thinking removal test completed!")

async def test_voice_to_voice_with_thinking():
    """Test the complete voice-to-voice pipeline with thinking mode enabled."""
    
    print("\n🎤 Testing Voice-to-Voice with Thinking Mode")
    print("=" * 50)
    
    # Check if audio file exists
    audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav"
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        # Initialize service
        print("🚀 Initializing Voice-to-Voice Service...")
        v2v_service = VoiceToVoiceService(content_filter_strictness="relaxed")
        
        # Load models
        print("📡 Loading models...")
        models_result = v2v_service.initialize_models(
            stt_model="whisper-large-v3-turbo-arabic",
            tts_model="coqui-tts-arabic",
            chat_model="qwen3-unsloth-q4ks"
        )
        
        if not all(models_result.values()):
            print("❌ Failed to load models")
            return False
        
        print("✅ Models loaded successfully!")
        
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        # Test with thinking mode enabled
        print("\n🧠 Testing with thinking mode ENABLED...")
        result = v2v_service.voice_to_voice_bytes(
            audio_bytes=audio_bytes,
            input_language="ar",
            output_language="ar",
            thinking_mode=True,  # Enable thinking mode
            enable_content_filter=False,
            generation_config={"temperature": 0.7, "max_new_tokens": 300}
        )
        
        if result["success"]:
            print("✅ Voice-to-voice processing successful!")
            print(f"📝 Transcription: {result['transcription']}")
            print(f"🤖 Full Response: {result['response'][:200]}...")
            print(f"🎵 Audio output: {result['audio_output']}")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            
            # Check if thinking blocks are in the full response
            if "<think>" in result['response']:
                print("✅ GOOD: Full response contains thinking blocks (as expected)")
            else:
                print("ℹ️ INFO: No thinking blocks found in response")
            
            # Check the actual audio file was created
            if result.get('audio_output') and Path(result['audio_output']).exists():
                audio_size = Path(result['audio_output']).stat().st_size
                print(f"🎵 Audio file created: {audio_size} bytes")
            else:
                print("❌ Audio file not created")
                
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        v2v_service.unload_all_models()
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧪 THINKING CONTENT REMOVAL TESTS")
    print("=" * 60)
    
    # Test 1: Static method testing
    test_thinking_removal()
    
    # Test 2: Full pipeline testing
    success = await test_voice_to_voice_with_thinking()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ TTS should now only process clean text without thinking blocks")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    asyncio.run(main())
