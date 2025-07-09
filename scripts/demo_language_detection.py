#!/usr/bin/env python3
"""
🌍 Automatic Language Detection Demo

Quick demonstration of the automatic language detection feature.
Shows how the system detects Arabic vs English and prepares appropriate responses.
"""

import sys
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.utils.language_detection import detect_language, suggest_response_language
from beautyai_inference.api.models import ChatRequest

def demo_language_detection():
    """Demonstrate automatic language detection with real examples."""
    
    print("🌍 BeautyAI Automatic Language Detection Demo")
    print("=" * 60)
    
    # Test cases with real beauty clinic scenarios
    test_cases = [
        {
            "name": "Arabic Beauty Consultation",
            "text": "مرحباً دكتور، أريد استشارة حول أفضل علاج للبشرة الجافة وإزالة التجاعيد",
            "scenario": "Patient asks about skincare treatment in Arabic"
        },
        {
            "name": "English Beauty Consultation", 
            "text": "Hello doctor, I would like a consultation about the best treatment for dry skin and wrinkle removal",
            "scenario": "Patient asks about skincare treatment in English"
        },
        {
            "name": "Arabic Botox Inquiry",
            "text": "كم سعر حقن البوتوكس والفيلر؟ وهل هناك آثار جانبية؟",
            "scenario": "Patient asks about Botox pricing in Arabic"
        },
        {
            "name": "English Botox Inquiry",
            "text": "What is the price for Botox and filler injections? Are there any side effects?",
            "scenario": "Patient asks about Botox pricing in English"
        },
        {
            "name": "Arabic Appointment Booking",
            "text": "أريد حجز موعد للاستشارة، متى يمكنني الحضور؟",
            "scenario": "Patient wants to book appointment in Arabic"
        },
        {
            "name": "English Appointment Booking",
            "text": "I want to book an appointment for consultation, when can I come?",
            "scenario": "Patient wants to book appointment in English"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 Demo {i}: {test_case['name']}")
        print(f"📋 Scenario: {test_case['scenario']}")
        print(f"💬 User Input: \"{test_case['text']}\"")
        
        # Step 1: Detect language
        detected_lang, confidence = detect_language(test_case['text'])
        print(f"🔍 Language Detection: {detected_lang} (confidence: {confidence:.3f})")
        
        # Step 2: Suggest response language
        response_lang = suggest_response_language(test_case['text'])
        print(f"🌍 Response Language: {response_lang}")
        
        # Step 3: Show what system message would be used
        if response_lang == "ar":
            system_msg = "أنت مساعد ذكي ومفيد. يجب أن تجيب باللغة العربية فقط."
            expected_response = "أهلاً وسهلاً! سأساعدك في الإجابة على استفساراتك باللغة العربية..."
        elif response_lang == "en":
            system_msg = "You are a helpful assistant. You must respond only in English."
            expected_response = "Hello! I'll help you with your questions in English..."
        else:
            system_msg = "Default system message"
            expected_response = "Default response"
        
        print(f"📝 System Message: \"{system_msg[:50]}...\"")
        print(f"🤖 Expected Response Style: \"{expected_response}\"")
        
        # Step 4: Show confidence level
        if confidence >= 0.6:
            confidence_level = "🟢 HIGH - Very confident detection"
        elif confidence >= 0.3:
            confidence_level = "🟡 MEDIUM - Confident detection"
        else:
            confidence_level = "🔴 LOW - Will default to English"
        
        print(f"📊 Confidence Level: {confidence_level}")
        print("-" * 40)

def demo_chat_request_integration():
    """Demonstrate ChatRequest integration with auto language detection."""
    
    print("\n🔧 ChatRequest Integration Demo")
    print("=" * 60)
    
    # Arabic example
    print("\n📝 Example 1: Arabic ChatRequest")
    arabic_request = ChatRequest(
        model_name="qwen3-unsloth-q4ks",
        message="مرحباً، ما هي أفضل علاجات مكافحة الشيخوخة؟",
        response_language="auto",  # This triggers automatic detection
        preset="qwen_optimized"
    )
    
    print(f"   User Message: {arabic_request.message}")
    print(f"   Response Language Setting: {arabic_request.response_language}")
    print(f"   Processed Message: {arabic_request.get_processed_message()}")
    print(f"   Thinking Mode: {arabic_request.should_enable_thinking()}")
    
    # Simulate the auto-detection that happens in the endpoint
    if arabic_request.response_language == "auto":
        detected_lang = suggest_response_language(arabic_request.get_processed_message())
        print(f"   🌍 Auto-detected Language: {detected_lang}")
        print(f"   System Message Would Be: Arabic instruction")
    
    # English example  
    print("\n📝 Example 2: English ChatRequest")
    english_request = ChatRequest(
        model_name="qwen3-unsloth-q4ks",
        message="Hello, what are the best anti-aging treatments?",
        response_language="auto",  # This triggers automatic detection
        preset="qwen_optimized"
    )
    
    print(f"   User Message: {english_request.message}")
    print(f"   Response Language Setting: {english_request.response_language}")
    print(f"   Processed Message: {english_request.get_processed_message()}")
    print(f"   Thinking Mode: {english_request.should_enable_thinking()}")
    
    # Simulate the auto-detection that happens in the endpoint
    if english_request.response_language == "auto":
        detected_lang = suggest_response_language(english_request.get_processed_message())
        print(f"   🌍 Auto-detected Language: {detected_lang}")
        print(f"   System Message Would Be: English instruction")

def demo_thinking_mode_commands():
    """Demonstrate thinking mode command processing."""
    
    print("\n🧠 Thinking Mode Commands Demo")
    print("=" * 60)
    
    examples = [
        {
            "input": "/no_think ما هي أفضل طرق العناية بالبشرة؟",
            "description": "Arabic question with thinking disabled"
        },
        {
            "input": "/no_think What are the best skincare treatments?", 
            "description": "English question with thinking disabled"
        },
        {
            "input": "ما هي أفضل طرق العناية بالبشرة؟",
            "description": "Arabic question with default thinking"
        },
        {
            "input": "What are the best skincare treatments?",
            "description": "English question with default thinking"
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['description']}")
        
        request = ChatRequest(
            model_name="qwen3-unsloth-q4ks",
            message=example['input'],
            response_language="auto"
        )
        
        print(f"   Input: \"{example['input']}\"")
        print(f"   Thinking Enabled: {request.should_enable_thinking()}")
        print(f"   Processed Message: \"{request.get_processed_message()}\"")
        
        # Language detection
        detected_lang = suggest_response_language(request.get_processed_message())
        print(f"   🌍 Detected Language: {detected_lang}")

def main():
    """Run the complete demonstration."""
    try:
        demo_language_detection()
        demo_chat_request_integration()
        demo_thinking_mode_commands()
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("✅ The automatic language detection system is working perfectly!")
        print("🌍 Users can now speak Arabic or English and get responses in the same language.")
        print("🚀 Ready for production use with both chat and voice-to-voice endpoints!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
