#!/usr/bin/env python3
"""
Test the audio chat endpoint on the actual BeautyAI API.
"""

import requests
import io
import json
import tempfile
import wave
import numpy as np
import os

# Create a simple test audio file (1 second of 440Hz tone)
def create_test_audio():
    """Create a simple WAV audio file for testing."""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # 440 Hz (A note)
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Read the file back as bytes
        with open(temp_file.name, 'rb') as f:
            return f.read()

def test_audio_file_endpoint(audio_file_path, audio_format, language="ar", port=8000):
    """Test the audio chat endpoint with a specific audio file."""
    print(f"🧪 Testing audio chat endpoint with {audio_format.upper()} file...")
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
    
    # Prepare form data
    files = {
        'audio_file': (filename, io.BytesIO(audio_data), mime_type)
    }
    
    data = {
        'model_name': 'qwen3-unsloth-q4ks',
        'whisper_model_name': 'whisper-large-v3-turbo-arabic',
        'audio_language': language,
        'preset': 'qwen_optimized',
        'temperature': '0.3',
        'top_p': '0.95',
        'top_k': '20',
        'max_new_tokens': '512',
        'disable_content_filter': 'true',
        'session_id': f'test_session_{audio_format}'
    }
    
    try:
        print("📤 Sending request...")
        response = requests.post(url, files=files, data=data, timeout=60)
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"✅ Transcription: '{result.get('transcription', '')}' ({len(result.get('transcription', ''))} chars)")
            print(f"✅ Whisper Model: {result.get('whisper_model_used', 'N/A')}")
            print(f"✅ Audio Language Detected: {result.get('audio_language_detected', 'N/A')}")
            
            if result.get('response'):
                response_text = result.get('response', '')
                print(f"✅ Chat Response: '{response_text[:200]}{'...' if len(response_text) > 200 else ''}'")
            
            if result.get('metrics'):
                metrics = result.get('metrics')
                print(f"⏱️  Transcription Time: {metrics.get('transcription_time', 'N/A')}s")
                print(f"⏱️  Generation Time: {metrics.get('generation_time', 'N/A')}s")
                print(f"📊 Total Tokens: {metrics.get('total_tokens', 'N/A')}")
            
            if result.get('error'):
                print(f"⚠️  Error: {result.get('error')}")
            if result.get('transcription_error'):
                print(f"⚠️  Transcription Error: {result.get('transcription_error')}")
                
            return result.get('success', False)
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


def test_mp3_file(port=8000):
    """Test with the MP3 file."""
    print("\n" + "="*60)
    print("🎵 Testing MP3 File (q1.mp3)")
    print("="*60)
    
    # Try both locations
    mp3_paths = [
        "/home/lumi/beautyai/q1.mp3",
        "/home/lumi/beautyai/tests/q1.mp3"
    ]
    
    for mp3_path in mp3_paths:
        if os.path.exists(mp3_path):
            return test_audio_file_endpoint(mp3_path, "mp3", language="ar", port=port)
    
    print("❌ No MP3 file found in expected locations")
    return False


def test_webm_file(port=8000):
    """Test with the WebM file."""
    print("\n" + "="*60)
    print("🎬 Testing WebM File (q1.webm)")
    print("="*60)
    
    webm_path = "/home/lumi/beautyai/q1.webm"
    if os.path.exists(webm_path):
        return test_audio_file_endpoint(webm_path, "webm", language="ar", port=port)
    else:
        print("❌ WebM file not found")
        return False


def test_both_formats(port=8000):
    """Test both MP3 and WebM formats."""
    print("🚀 Starting comprehensive audio format testing...")
    
    results = {}
    
    # Test MP3
    print("\n" + "🎵" * 20)
    results['mp3'] = test_mp3_file(port)
    
    # Test WebM  
    print("\n" + "🎬" * 20)
    results['webm'] = test_webm_file(port)
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    for format_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{format_name.upper()}: {status}")
    
    total_passed = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️  Some tests FAILED - check the logs above for details")
    
    return results



def test_port_8000():
    """Test the service on port 8000 with real audio files."""
    print("\n" + "="*50)
    print("🧪 Testing port 8000 (Main BeautyAI API)")
    print("Testing with real MP3 and WebM audio files")
    print("="*50)
    test_both_formats(8000)


def check_server_health(port=8000):
    """Check if the API server is running."""
    try:
        health_url = f"http://localhost:{port}/health"
        response = requests.get(health_url, timeout=5)
        print(f"🏥 Server health check: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"❌ Server not reachable on port {port}")
        return False
    except Exception as e:
        print(f"⚠️  Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 BeautyAI Audio Endpoint Tester")
    print("Testing MP3 and WebM file transcription")
    print("="*80)
    
    # Check server health first
    if not check_server_health():
        print("\n💡 Make sure the BeautyAI API server is running:")
        print("   cd /home/lumi/beautyai")
        print("   python -m beautyai_inference.api.main")
        exit(1)
    
    # Run the tests
    test_port_8000()
