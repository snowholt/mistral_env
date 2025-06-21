import requests
import os
import json

def check_server_health():
    """Check if the API server is running and healthy."""
    try:
        # Try to hit a basic endpoint first
        health_url = "http://localhost:8000/health"
        response = requests.get(health_url, timeout=5)
        print(f"🏥 Health check: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server not reachable - Is the API server running on localhost:8000?")
        return False
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
        return True  # Continue anyway

def test_audio_chat_endpoint():
    url = "http://localhost:8000/inference/audio-chat"
    model_name = "qwen3-unsloth-q4ks"
    audio_file_path = "/home/lumi/beautyai/tests/q1.mp3"

    print(f"Testing audio chat endpoint: {url}")
    print(f"Using model: {model_name}")
    print(f"Audio file: {audio_file_path}")

    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False

    # Check file size
    file_size = os.path.getsize(audio_file_path)
    print(f"Audio file size: {file_size} bytes")

    with open(audio_file_path, "rb") as audio_file:
        files = {"audio_file": audio_file}
        payload = {
            "model_name": model_name,
            "whisper_model_name": "whisper-large-v3-turbo-arabic",
            "audio_language": "ar",
            "temperature": 0.7,
            "max_new_tokens": 512,
            "disable_content_filter": True
        }

        try:
            print("Sending request...")
            print(f"Payload: {payload}")
            response = requests.post(url, files=files, data=payload, timeout=60)
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result.get('success')}")
                print(f"📝 Transcription: {result.get('transcription', 'No transcription')}")
                print(f"🤖 Response: {result.get('response', 'No response')}")
                print(f"🔢 Tokens generated: {result.get('tokens_generated')}")
                print(f"⏱️ Generation time: {result.get('generation_time_ms')}ms")
                if 'transcription_time_ms' in result:
                    print(f"🎙️ Transcription time: {result.get('transcription_time_ms')}ms")
                return True
            else:
                print(f"❌ Error response:")
                print(f"Status: {response.status_code}")
                print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
                try:
                    error_json = response.json()
                    print(f"Error JSON: {json.dumps(error_json, indent=2)}")
                except:
                    print(f"Raw error text: {response.text}")
                
                # Additional debugging for 500 errors
                if response.status_code == 500:
                    print("\n🔍 Debugging suggestions for 500 error:")
                    print("1. Check server logs for detailed error messages")
                    print("2. Verify the Whisper model is properly loaded")
                    print("3. Ensure the chat model is available")
                    print("4. Check audio file format compatibility (supported: WAV, MP3, OGG, FLAC, M4A, WMA, WebM)")
                
                return False

        except requests.exceptions.ConnectionError:
            print("❌ Connection error - Is the API server running on localhost:8000?")
            return False
        except requests.exceptions.Timeout:
            print("❌ Request timed out after 60 seconds")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_simple_audio_request():
    """Test with minimal parameters to isolate issues."""
    url = "http://localhost:8000/inference/audio-chat"
    audio_file_path = "/home/lumi/beautyai/tests/q1.mp3"
    
    print(f"\n🔧 Testing simplified request...")
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False

    with open(audio_file_path, "rb") as audio_file:
        files = {"audio_file": audio_file}
        payload = {"model_name": "qwen3-unsloth-q4ks"}

        try:
            response = requests.post(url, files=files, data=payload, timeout=30)
            print(f"Simple test status: {response.status_code}")
            if response.status_code != 500:
                print("✅ Simple test shows different result!")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Response: {result.get('response', 'No response')}")
                    return True
            return False
        except Exception as e:
            print(f"Simple test error: {e}")
            return False

if __name__ == "__main__":
    print("🚀 Starting audio chat endpoint test...")
    
    # Check server health first
    if not check_server_health():
        exit(1)
    
    # Run main test
    success = test_audio_chat_endpoint()
    
    # If main test fails with 500, try simplified test
    if not success:
        print("\n" + "="*50)
        success = test_simple_audio_request()
    
    print(f"\n🏁 Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if not success:
        print("\n💡 Next steps:")
        print("1. Check API server logs for detailed error information")
        print("2. Verify models are properly loaded and available")
        print("3. Test with the regular /chat endpoint first")
        print("4. Ensure audio transcription service is configured")
