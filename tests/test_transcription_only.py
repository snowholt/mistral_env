#!/usr/bin/env python3
"""
Test only the transcription part of the audio chat endpoint.
"""

import requests
import io
import json
import os


def test_transcription_only(audio_file_path, audio_format, language="ar", port=8000):
    """Test only the transcription part by using a simple model."""
    print(f"🎙️ Testing transcription with {audio_format.upper()} file...")
    print(f"📁 File: {audio_file_path}")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(audio_file_path)
    print(f"📊 File size: {file_size} bytes")
    
    # Prepare the request
    url = f"http://localhost:{port}/inference/audio-chat"
    
    # Read the audio file
    with open(audio_file_path, 'rb') as f:
        audio_data = f.read()
    
    # Determine MIME type based on format
    mime_types = {
        'mp3': 'audio/mpeg',
        'webm': 'audio/webm',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'flac': 'audio/flac',
        'm4a': 'audio/mp4'
    }
    
    mime_type = mime_types.get(audio_format.lower(), 'audio/mpeg')
    filename = f"test_audio.{audio_format}"
    
    # Prepare form data - using the default model and minimal params
    files = {
        'audio_file': (filename, io.BytesIO(audio_data), mime_type)
    }
    
    data = {
        'model_name': 'qwen3-unsloth-q4ks',  # This is the default model
        'whisper_model_name': 'whisper-large-v3-turbo-arabic',
        'audio_language': language,
        'temperature': '0.1',  # Low temperature
        'max_new_tokens': '50',  # Short response
        'disable_content_filter': 'true',
        'session_id': f'transcription_test_{audio_format}'
    }
    
    try:
        print("📤 Sending transcription request...")
        response = requests.post(url, files=files, data=data, timeout=120)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request completed!")
            
            # Focus on transcription results
            transcription = result.get('transcription', '')
            print(f"🎙️ Transcription Success: {bool(transcription)}")
            print(f"🎙️ Transcription Length: {len(transcription)} chars")
            
            if transcription:
                print(f"🎙️ Transcribed Text: '{transcription}'")
                print(f"✅ TRANSCRIPTION SUCCESSFUL for {audio_format.upper()}")
            else:
                print(f"❌ TRANSCRIPTION FAILED for {audio_format.upper()}")
                if result.get('transcription_error'):
                    print(f"🔍 Transcription Error: {result.get('transcription_error')}")
            
            # Check if there's a whisper model issue
            whisper_model = result.get('whisper_model_used')
            if whisper_model:
                print(f"🎙️ Whisper Model Used: {whisper_model}")
            
            # Check for any errors
            if result.get('error'):
                print(f"⚠️ General Error: {result.get('error')}")
                
            return bool(transcription)
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_result = response.json()
                print(f"❌ Error details: {json.dumps(error_result, indent=2)}")
            except:
                print(f"❌ Raw response: {response.text}")
            return False
                
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def check_whisper_model_status(port=8000):
    """Check if Whisper model is loaded."""
    try:
        # Try to check model status - this might not exist but worth trying
        url = f"http://localhost:{port}/models/"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            whisper_models = [m for m in models if 'whisper' in m.get('name', '').lower()]
            print(f"🎙️ Whisper models available: {len(whisper_models)}")
            for model in whisper_models:
                print(f"   - {model.get('name', 'Unknown')}: {model.get('description', 'No description')}")
            return len(whisper_models) > 0
    except:
        pass
    return True  # Assume it's fine


def main():
    print("🎙️ BeautyAI Transcription Test")
    print("Focus: Testing MP3 and WebM transcription capabilities")
    print("="*80)
    
    # Check whisper model
    print("🔍 Checking Whisper model availability...")
    check_whisper_model_status()
    
    # Test files
    test_files = [
        ("/home/lumi/beautyai/q1.mp3", "mp3", "ar"),
        ("/home/lumi/beautyai/q1.webm", "webm", "ar"),
    ]
    
    results = {}
    
    for file_path, format_name, language in test_files:
        print(f"\n{'='*60}")
        print(f"Testing {format_name.upper()} transcription")
        print(f"{'='*60}")
        
        if os.path.exists(file_path):
            results[format_name] = test_transcription_only(file_path, format_name, language)
        else:
            print(f"❌ File not found: {file_path}")
            results[format_name] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TRANSCRIPTION TEST RESULTS")
    print(f"{'='*80}")
    
    for format_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{format_name.upper()} transcription: {status}")
    
    successful_transcriptions = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    print(f"\nTranscription Success Rate: {successful_transcriptions}/{total_tests}")
    
    if successful_transcriptions == total_tests:
        print("🎉 All transcriptions working!")
    elif successful_transcriptions > 0:
        print("⚠️ Some transcriptions working, some failing")
    else:
        print("❌ All transcriptions failed")
    
    # Recommendations
    print(f"\n💡 FINDINGS:")
    if results.get('mp3'):
        print("✅ MP3 files can be transcribed successfully")
    else:
        print("❌ MP3 transcription is failing")
        
    if results.get('webm'):
        print("✅ WebM files can be transcribed successfully") 
    else:
        print("❌ WebM transcription is failing - this might be a format support issue")
        print("   Consider converting WebM to MP3/WAV for better compatibility")


if __name__ == "__main__":
    main()
