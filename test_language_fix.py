#!/usr/bin/env python3
"""
Test script to verify the language detection fix for Simple Voice Service.
Tests both Arabic and English language handling.
"""
import asyncio
import logging
import sys
import os

# Add the beautyai_inference module to the path
sys.path.insert(0, '/home/lumi/beautyai')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_language_detection():
    """Test language detection with SimpleVoiceService."""
    try:
        from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
        
        # Initialize the service
        service = SimpleVoiceService()
        await service.initialize()
        
        # Test cases
        test_cases = [
            {
                "text": "Hello, how are you?",
                "expected_language": "en",
                "fallback_language": "en"
            },
            {
                "text": "مرحبا، كيف حالك؟",
                "expected_language": "ar",
                "fallback_language": "ar"
            },
            {
                "text": "unclear audio",
                "expected_language": "en",
                "fallback_language": "en"
            },
            {
                "text": "صوت غير واضح",
                "expected_language": "ar",
                "fallback_language": "ar"
            }
        ]
        
        print("🧪 Testing Language Detection")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected_language"]
            fallback = test_case["fallback_language"]
            
            # Test language detection
            detected = service._detect_language(text, fallback_language=fallback)
            
            status = "✅ PASS" if detected == expected else "❌ FAIL"
            print(f"Test {i}: {status}")
            print(f"  Input: '{text}'")
            print(f"  Expected: {expected}")
            print(f"  Detected: {detected}")
            print(f"  Fallback: {fallback}")
            print()
        
        print("🔧 Testing Voice Selection")
        print("=" * 50)
        
        # Test voice selection
        voice_tests = [
            {"lang": "en", "gender": "female", "expected_contains": "en-US"},
            {"lang": "en", "gender": "male", "expected_contains": "en-US"},
            {"lang": "ar", "gender": "female", "expected_contains": "ar-SA"},
            {"lang": "ar", "gender": "male", "expected_contains": "ar-SA"},
        ]
        
        for i, test in enumerate(voice_tests, 1):
            voice = service._select_voice(test["lang"], test["gender"])
            status = "✅ PASS" if test["expected_contains"] in voice else "❌ FAIL"
            print(f"Voice Test {i}: {status}")
            print(f"  Language: {test['lang']}, Gender: {test['gender']}")
            print(f"  Selected Voice: {voice}")
            print()
        
        await service.cleanup()
        print("🎉 Language detection tests completed!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

async def test_chat_service_language():
    """Test chat service language handling."""
    try:
        from beautyai_inference.services.inference.chat_service import ChatService
        
        chat_service = ChatService()
        success = chat_service.load_model("qwen3-unsloth-q4ks")
        
        if not success:
            print("❌ Failed to load chat model for testing")
            return
        
        print("🧪 Testing Chat Service Language Handling")
        print("=" * 50)
        
        test_cases = [
            {
                "message": "Hello, how are you?",
                "language": "en",
                "expected_language": "en"
            },
            {
                "message": "مرحبا، كيف حالك؟",
                "language": "ar", 
                "expected_language": "ar"
            },
            {
                "message": "unclear audio",
                "language": "en",
                "expected_language": "en"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            result = chat_service.chat(
                message=test_case["message"],
                language=test_case["language"],
                max_length=50
            )
            
            if result.get("success"):
                response = result.get("response", "")
                detected_lang = result.get("detected_language", "unknown")
                
                status = "✅ PASS" if detected_lang == test_case["expected_language"] else "❌ FAIL"
                print(f"Chat Test {i}: {status}")
                print(f"  Input: '{test_case['message']}'")
                print(f"  Target Language: {test_case['language']}")
                print(f"  Detected Language: {detected_lang}")
                print(f"  Response: '{response[:100]}...'")
                print()
            else:
                print(f"Chat Test {i}: ❌ FAIL - {result.get('error')}")
                print()
        
    except Exception as e:
        logger.error(f"Error testing chat service: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    print("🚀 Starting Language Fix Verification Tests")
    print("=" * 60)
    
    await test_language_detection()
    print("\n" + "=" * 60 + "\n")
    await test_chat_service_language()
    
    print("🏁 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
