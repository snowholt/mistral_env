#!/usr/bin/env python3
"""
Comprehensive Voice-to-Voice Test with Format Comparison.
Tests both WAV and WebM formats to ensure consistent behavior.
"""
import requests
import json
from pathlib import Path

def test_format_comparison():
    """Test voice-to-voice with both WebM and WAV formats."""
    
    print("🎤 Voice-to-Voice Format Comparison Test")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8000/inference/voice-to-voice"
    
    # Test cases: same content in different formats
    test_cases = [
        {
            "name": "WebM Format Test",
            "file": "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.webm",
            "format": "webm"
        },
        {
            "name": "WAV Format Test", 
            "file": "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav",
            "format": "wav"
        }
    ]
    
    # Test parameters
    test_data = {
        "input_language": "ar",
        "output_language": "ar",
        "stt_model_name": "whisper-large-v3-turbo-arabic",
        "tts_model_name": "coqui-tts-arabic",
        "chat_model_name": "qwen3-unsloth-q4ks",
        "speaker_voice": "female",
        "preset": "balanced",
        "disable_content_filter": "true",
        "thinking_mode": "false"
    }
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔧 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        if not Path(test_case['file']).exists():
            print(f"❌ File not found: {test_case['file']}")
            continue
            
        try:
            with open(test_case['file'], "rb") as audio_file:
                files = {"audio_file": audio_file}
                
                print(f"🚀 Testing {test_case['format'].upper()} format...")
                response = requests.post(url, files=files, data=test_data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for thinking content
                    response_text = result.get('response_text', '')
                    has_thinking = '<think>' in response_text or '</think>' in response_text
                    
                    test_result = {
                        "format": test_case['format'],
                        "success": True,
                        "transcription": result.get('transcription', 'N/A'),
                        "response_clean": not has_thinking,
                        "response_length": len(response_text),
                        "processing_time": result.get('total_processing_time_ms', 0),
                        "audio_generated": result.get('data', {}).get('audio_output_available', False)
                    }
                    
                    print(f"✅ SUCCESS! Format: {test_case['format'].upper()}")
                    print(f"📝 Transcription: {result.get('transcription', 'N/A')}")
                    print(f"💬 Response Length: {len(response_text)} chars")
                    print(f"🧹 Clean Response: {'✅ Yes' if not has_thinking else '❌ Contains thinking'}")
                    print(f"⏱️ Processing Time: {result.get('total_processing_time_ms', 0):.1f}ms")
                    print(f"🎵 Audio Generated: {'✅ Yes' if test_result['audio_generated'] else '❌ No'}")
                    
                else:
                    print(f"❌ FAILED! Status: {response.status_code}")
                    try:
                        error = response.json()
                        print(f"Error: {error.get('detail', 'Unknown error')}")
                    except:
                        print(f"Raw error: {response.text}")
                    
                    test_result = {
                        "format": test_case['format'],
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                
                results.append(test_result)
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                "format": test_case['format'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if r.get('success', False)]
    
    if successful_tests:
        print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
        
        # Check consistency
        if len(successful_tests) >= 2:
            transcriptions = [r['transcription'] for r in successful_tests]
            if len(set(transcriptions)) == 1:
                print("✅ Transcription consistency: PASS (same across formats)")
            else:
                print("⚠️ Transcription consistency: Different results across formats")
            
            clean_responses = [r['response_clean'] for r in successful_tests]
            if all(clean_responses):
                print("✅ Thinking content removal: PASS (all responses clean)")
            else:
                print("❌ Thinking content removal: FAIL (some responses contain thinking)")
                
            audio_generated = [r['audio_generated'] for r in successful_tests]
            if all(audio_generated):
                print("✅ Audio generation: PASS (all formats generated audio)")
            else:
                print("❌ Audio generation: FAIL (some formats failed to generate audio)")
        
        print(f"\n📈 Average processing time: {sum(r['processing_time'] for r in successful_tests) / len(successful_tests):.1f}ms")
        
    else:
        print("❌ No successful tests")
    
    print("\n🏁 Voice-to-Voice format comparison completed!")

if __name__ == "__main__":
    test_format_comparison()
