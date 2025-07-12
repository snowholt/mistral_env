#!/usr/bin/env python3
"""
Test thinking content removal in voice-to-voice service.
"""
import sys
import re

sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService

def test_thinking_removal_comprehensive():
    """Test comprehensive thinking content removal scenarios."""
    
    print("🧪 COMPREHENSIVE THINKING CONTENT REMOVAL TEST")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Standard thinking with response",
            "input": "<think>Let me think about Botox carefully...</think>Botox is a neurotoxin used for cosmetic purposes.",
            "should_contain": "Botox is a neurotoxin used for cosmetic purposes",
            "should_not_contain": ["<think>", "</think>", "Let me think"]
        },
        {
            "name": "Multiple thinking blocks",
            "input": "<think>First thought</think>Main response here.<think>Second thought</think>More response.",
            "should_contain": "Main response here",
            "should_not_contain": ["<think>", "</think>", "First thought", "Second thought"]
        },
        {
            "name": "Thinking with newlines",
            "input": "<think>\nLet me analyze this:\n1. First point\n2. Second point\n</think>\n\nBotox injections are minimally invasive.",
            "should_contain": "Botox injections are minimally invasive",
            "should_not_contain": ["<think>", "</think>", "First point", "analyze"]
        },
        {
            "name": "Real Botox response example",
            "input": "<think>\nOkay, the user is asking about Botox, specifically the recovery period and possible side effects.\nLet me start by recalling what I know about Botox.\n</think>\n\nOkay, the user is asking about Botox, specifically the recovery period and possible side effects.\n\nFirst, Botox is a neurotoxin derived from Clostridium botulinum.",
            "should_contain": "Okay, the user is asking about Botox",
            "should_not_contain": ["<think>", "</think>", "Let me start by recalling"]
        },
        {
            "name": "Arabic with thinking",
            "input": "<think>دعني أفكر في هذا السؤال</think>البوتوكس هو علاج تجميلي آمن وفعال.",
            "should_contain": "البوتوكس هو علاج تجميلي آمن وفعال",
            "should_not_contain": ["<think>", "</think>", "دعني أفكر"]
        },
        {
            "name": "Only thinking (should get default)",
            "input": "<think>Just internal reasoning without any user response</think>",
            "should_contain": "أعتذر، لم أتمكن من تقديم إجابة واضحة",
            "should_not_contain": ["<think>", "</think>", "internal reasoning"]
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔧 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        result = VoiceToVoiceService._remove_thinking_content(test_case["input"])
        
        print(f"📝 Input: {test_case['input'][:80]}...")
        print(f"✅ Output: {result[:80]}...")
        
        # Check if required content is present
        if test_case["should_contain"] in result:
            print(f"✅ Contains required content: '{test_case['should_contain'][:50]}...'")
            content_check = True
        else:
            print(f"❌ Missing required content: '{test_case['should_contain'][:50]}...'")
            content_check = False
        
        # Check if forbidden content is absent
        forbidden_found = []
        for forbidden in test_case["should_not_contain"]:
            if forbidden in result:
                forbidden_found.append(forbidden)
        
        if forbidden_found:
            print(f"❌ Found forbidden content: {forbidden_found}")
            forbidden_check = False
        else:
            print("✅ No forbidden content found")
            forbidden_check = True
        
        # Overall test result
        if content_check and forbidden_check:
            print("🎉 PASSED")
            passed += 1
        else:
            print("❌ FAILED")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Thinking content removal is working correctly.")
    else:
        print("❌ Some tests failed. There may be issues with thinking content removal.")
    
    return failed == 0

if __name__ == "__main__":
    test_thinking_removal_comprehensive()
