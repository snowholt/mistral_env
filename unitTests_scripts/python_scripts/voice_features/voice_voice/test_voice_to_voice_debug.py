#!/usr/bin/env python3
"""
Debug script to test the voice-to-voice endpoint with a simple audio file.
"""
import requests
import json
from pathlib import Path

def test_voice_to_voice_endpoint():
    """Test the voice-to-voice endpoint with debug information."""
    
    print("🎤 Voice-to-Voice Endpoint Debug Test")
    print("=" * 50)
    
    # API endpoint
    url = "http://localhost:8000/inference/voice-to-voice"
    
    # Check if we have any test audio files
    test_audio_paths = [
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.webm",  # Test WebM first
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.webm",
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting.webm",
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
    ]
    
    available_audio = None
    for audio_path in test_audio_paths:
        if Path(audio_path).exists():
            available_audio = audio_path
            print(f"✅ Found test audio: {audio_path}")
            break
    
    if not available_audio:
        print("❌ No test audio files found!")
        print("Available paths checked:")
        for path in test_audio_paths:
            print(f"  - {path}")
        return
    
    # Test parameters
    test_data = {
        "input_language": "ar",
        "output_language": "ar",
        "stt_model_name": "whisper-large-v3-turbo-arabic",
        "tts_model_name": "coqui-tts-arabic",
        "chat_model_name": "qwen3-unsloth-q4ks",
        "speaker_voice": "female",
        "preset": "high_quality",
        "disable_content_filter": "true",
        "thinking_mode": "false"
    }
    
    print(f"📁 Using audio file: {available_audio}")
    print(f"⚙️ Test parameters: {json.dumps(test_data, indent=2)}")
    print()
    
    try:
        # Prepare the multipart form data
        with open(available_audio, "rb") as audio_file:
            files = {"audio_file": audio_file}
            
            print("🚀 Sending request to voice-to-voice endpoint...")
            response = requests.post(
                url,
                files=files,
                data=test_data,
                timeout=120  # 2 minutes timeout
            )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            print("✅ SUCCESS! Voice-to-voice completed successfully")
            try:
                result = response.json()
                print(f"📝 Transcription: {result.get('transcription', 'N/A')}")
                response_text = result.get('response_text', 'N/A')
                print(f"💬 Response: {response_text[:100]}...")
                
                # Check for thinking content in response
                if '<think>' in response_text or '</think>' in response_text:
                    print("⚠️  WARNING: Response contains thinking content!")
                    print(f"🔍 Full response: {response_text}")
                else:
                    print("✅ Response is clean (no thinking content)")
                
                print(f"⏱️ Processing Time: {result.get('total_processing_time_ms', 'N/A')}ms")
                print(f"🎵 Audio Available: {result.get('data', {}).get('audio_output_available', False)}")
                print(f"🎵 Audio Path: {result.get('data', {}).get('audio_output_path', 'N/A')}")
            except json.JSONDecodeError:
                print("Response is not JSON (might be audio bytes)")
                print(f"Content type: {response.headers.get('content-type')}")
                print(f"Content length: {len(response.content)} bytes")
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 2 minutes")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_voice_to_voice_endpoint()
