#!/usr/bin/env python3
"""
Debug script to check specific medical keyword detection.
"""

from beautyai_inference.services.inference.content_filter_service import ContentFilterService
import re

def debug_keywords():
    """Debug keyword detection."""
    
    filter_service = ContentFilterService(strictness_level="relaxed")
    
    # Test text
    text = "مرحباً! كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصّة بكم"
    
    print(f"🔍 Text: {text}")
    print()
    
    # Extract words using the same regex as the filter
    words = re.findall(r'[\u0600-\u06FF\w]+', text.lower())
    print(f"📝 Extracted words: {words}")
    print()
    
    # Check each word
    print(f"🏥 Checking each word against medical keywords:")
    for word in words:
        is_medical = word in filter_service.allowed_medical_keywords
        print(f"   '{word}' -> Medical: {is_medical}")
    
    print()
    
    # Check if 'تجميل' and related words are in allowed keywords
    beauty_terms = ['تجميل', 'التجميل', 'عيادة', 'الجمال', 'جمال', 'خدمات']
    print(f"🎯 Checking specific beauty terms:")
    for term in beauty_terms:
        is_allowed = term in filter_service.allowed_medical_keywords
        print(f"   '{term}' -> Allowed: {is_allowed}")
    
    print()
    
    # Print a sample of allowed medical keywords in Arabic
    arabic_keywords = [kw for kw in filter_service.allowed_medical_keywords if any('\u0600' <= c <= '\u06FF' for c in kw)]
    print(f"📋 Sample of allowed Arabic medical keywords (first 20):")
    for i, kw in enumerate(arabic_keywords[:20]):
        print(f"   {i+1}. '{kw}'")

if __name__ == "__main__":
    debug_keywords()
