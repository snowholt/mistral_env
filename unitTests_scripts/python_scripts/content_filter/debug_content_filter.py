#!/usr/bin/env python3
"""
Debug script to understand content filtering behavior.
"""

from beautyai_inference.services.inference.content_filter_service import ContentFilterService

def test_filter():
    """Test the content filter with the transcribed text."""
    
    # Initialize content filter
    filter_service = ContentFilterService(strictness_level="balanced")
    
    # The text that was transcribed from the audio
    transcribed_text = "مرحباً! كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصّة بكم"
    
    print(f"🔍 Testing content filter")
    print(f"📝 Text: {transcribed_text}")
    print(f"📊 Filter strictness: {filter_service.get_strictness_level()}")
    print()
    
    # Test the filtering
    result = filter_service.filter_content(transcribed_text, language='ar')
    
    print(f"✅ Filter result:")
    print(f"   - Allowed: {result.is_allowed}")
    print(f"   - Confidence: {result.confidence_score}")
    print(f"   - Reason: {result.filter_reason}")
    print(f"   - Matched patterns: {result.matched_patterns}")
    print()
    
    # Test individual parts
    parts = [
        "مرحباً! كيف حالك اليوم؟",
        "أتصل لأستفسر عن الخدمات المتوفرة", 
        "في عيادة التجميل الخاصّة بكم",
        "عيادة التجميل",
        "خدمات التجميل",
        "استفسار عن العيادة"
    ]
    
    print(f"🧪 Testing individual parts:")
    for part in parts:
        part_result = filter_service.filter_content(part, language='ar')
        print(f"   '{part}' -> Allowed: {part_result.is_allowed} (reason: {part_result.filter_reason})")
    
    print()
    
    # Check what medical keywords are detected
    words = transcribed_text.lower().split()
    medical_words_found = []
    for word in words:
        if word in filter_service.allowed_medical_keywords:
            medical_words_found.append(word)
    
    print(f"🏥 Medical keywords found: {medical_words_found}")
    print(f"📈 Medical word ratio: {len(medical_words_found)}/{len(words)} = {len(medical_words_found)/len(words):.2%}")
    
    # Test with different strictness levels
    print(f"\n🎯 Testing different strictness levels:")
    for level in ["relaxed", "balanced", "strict"]:
        filter_service.set_strictness_level(level)
        level_result = filter_service.filter_content(transcribed_text, language='ar')
        print(f"   {level}: Allowed = {level_result.is_allowed}")

if __name__ == "__main__":
    test_filter()
