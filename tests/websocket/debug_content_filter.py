#!/usr/bin/env python3
"""
Debug script to understand content filtering issues with voice-to-voice tests.
This script will help identify why legitimate medical/beauty content is being blocked.
"""

import sys
import os
import asyncio
import whisper
from pathlib import Path

# Add the beautyai_inference module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from beautyai_inference.services.inference.content_filter_service import ContentFilterService
from beautyai_inference.services.voice.transcription.audio_transcription_service import AudioTranscriptionService

async def debug_content_filter():
    """Debug content filtering for voice-to-voice tests."""
    
    print("🔍 BEAUTYAI CONTENT FILTER DEBUG")
    print("=" * 60)
    
    # Initialize services
    print("🚀 Initializing services...")
    content_filter = ContentFilterService(strictness_level="balanced")
    stt_service = AudioTranscriptionService()
    
    # Test audio files
    audio_files = [
        "/home/lumi/beautyai/voice_tests/botox.wav",
        "/home/lumi/beautyai/voice_tests/botox_ar.wav"
    ]
    
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            continue
            
        print(f"\n🎵 TESTING: {Path(audio_file).name}")
        print("-" * 40)
        
        # Transcribe audio
        print("📝 Transcribing audio...")
        try:
            # Read audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Transcribe using AudioTranscription service
            transcription_result = await stt_service.transcribe_audio(
                audio_data=audio_data,
                language="auto"
            )
            
            transcription = transcription_result.get("text", "")
            detected_language = transcription_result.get("language", "unknown")
            
            print(f"🗣️  Transcription: '{transcription}'")
            print(f"🌐 Detected Language: {detected_language}")
            
            if not transcription.strip():
                print("⚠️  Empty transcription - skipping filter test")
                continue
            
            # Test content filter
            print(f"\n🛡️  Testing content filter...")
            filter_result = content_filter.filter_content(transcription, detected_language)
            
            print(f"✅ Filter Result:")
            print(f"   - Allowed: {filter_result.is_allowed}")
            print(f"   - Reason: {filter_result.filter_reason}")
            print(f"   - Confidence: {filter_result.confidence_score}")
            
            # Debug filter details
            print(f"\n🔍 Filter Debug:")
            print(f"   - Strictness Level: {content_filter.get_strictness_level()}")
            
            # Test with different strictness levels
            for strictness in ["relaxed", "balanced", "strict"]:
                print(f"\n📊 Testing with strictness: {strictness}")
                content_filter.set_strictness_level(strictness)
                result = content_filter.filter_content(transcription, detected_language)
                print(f"   - {strictness}: {'✅ ALLOWED' if result.is_allowed else '❌ BLOCKED'} - {result.filter_reason}")
            
            # Test manual medical content detection
            print(f"\n🔬 Manual Medical Content Check:")
            is_medical = content_filter._is_medical_beauty_related(transcription.lower())
            print(f"   - Is Medical/Beauty Related: {is_medical}")
            
            # Check forbidden content
            forbidden_matches = content_filter._check_forbidden_content(transcription.lower())
            print(f"   - Forbidden Content Matches: {forbidden_matches}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test some manual text inputs
    print(f"\n🧪 MANUAL TEXT TESTS")
    print("=" * 60)
    
    test_texts = [
        ("ما هو البوتوكس؟", "ar", "Basic botox question in Arabic"),
        ("What is botox?", "en", "Basic botox question in English"),
        ("أريد معلومات عن حقن البوتوكس", "ar", "Botox injection info in Arabic"),
        ("Tell me about botox treatments", "en", "Botox treatments in English"),
        ("كم سعر البوتوكس؟", "ar", "Botox price question in Arabic"),
        ("How much does botox cost?", "en", "Botox cost question in English"),
        ("Hello, how are you?", "en", "Non-medical greeting"),
        ("ما هو الطقس اليوم؟", "ar", "Weather question in Arabic"),
    ]
    
    content_filter.set_strictness_level("balanced")
    
    for text, lang, description in test_texts:
        print(f"\n📝 Testing: {description}")
        print(f"   Text: '{text}'")
        print(f"   Language: {lang}")
        
        result = content_filter.filter_content(text, lang)
        print(f"   Result: {'✅ ALLOWED' if result.is_allowed else '❌ BLOCKED'} - {result.filter_reason}")
        
        # Check medical content detection
        is_medical = content_filter._is_medical_beauty_related(text.lower())
        print(f"   Medical Related: {is_medical}")

if __name__ == "__main__":
    asyncio.run(debug_content_filter())
