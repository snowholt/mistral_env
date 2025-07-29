#!/usr/bin/env python3
"""
Test WebSocket Voice Conversation with Audio
"""
import asyncio
import json
import logging
import websockets
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_voice_audio():
    """Test WebSocket voice conversation with actual audio file."""
    
    # Test audio files (use available ones)
    test_audio_paths = [
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav",
        "/home/lumi/beautyai/voice_tests/edge_tts_ar_female_7.wav",
        "/home/lumi/beautyai/tests/test_audio_ar.wav",
        "/home/lumi/beautyai/tests/arabic_test.wav"
    ]
    
    audio_file = None
    for path in test_audio_paths:
        if Path(path).exists():
            audio_file = path
            break
    
    if not audio_file:
        print("❌ No test audio files found!")
        print("Available paths checked:")
        for path in test_audio_paths:
            print(f"  - {path}")
        return False
    
    print(f"🎵 Using audio file: {audio_file}")
    print(f"📏 Audio file size: {Path(audio_file).stat().st_size} bytes")
    
    # WebSocket URL with simplified parameters
    ws_url = "wss://api.gmai.sa/ws/voice-conversation?preset=qwen_optimized&session_id=test_audio_session"
    
    print(f"🔗 Connecting to: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection established!")
            
            # Wait for connection confirmation
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Connection confirmed: {data.get('type')}")
            print(f"🆔 Session ID: {data.get('session_id')}")
            
            # Read and send audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            print(f"📤 Sending audio data: {len(audio_data)} bytes")
            await websocket.send(audio_data)
            
            # Wait for processing messages
            start_time = time.time()
            timeout = 60  # 1 minute timeout
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    response = json.loads(message)
                    
                    print(f"📥 Received: {response.get('type')}")
                    
                    if response.get('type') == 'processing_started':
                        print("⏳ Processing started...")
                        
                    elif response.get('type') == 'voice_response':
                        processing_time = time.time() - start_time
                        print(f"🎉 Voice response received! (took {processing_time:.2f}s)")
                        
                        success = response.get('success', False)
                        print(f"✅ Success: {success}")
                        
                        if response.get('transcription'):
                            print(f"🎤 Transcription: {response['transcription']}")
                        
                        if response.get('response_text'):
                            print(f"🤖 Response: {response['response_text']}")
                        
                        if response.get('audio_base64'):
                            audio_size = response.get('audio_size_bytes', 0)
                            print(f"🎵 Audio received: {audio_size} bytes")
                            print(f"🎵 Audio format: {response.get('audio_format', 'unknown')}")
                        else:
                            print("❌ No audio data in response")
                        
                        if response.get('error'):
                            print(f"❌ Error: {response['error']}")
                        
                        if response.get('warning'):
                            print(f"⚠️ Warning: {response['warning']}")
                        
                        return success
                        
                    elif response.get('type') == 'pong':
                        continue  # Ignore pong messages
                        
                    else:
                        print(f"📨 Other message: {response}")
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        print("⏰ Timeout waiting for response")
                        return False
                        
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for message")
                    return False
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse response: {e}")
                    return False
                    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False

async def main():
    """Main test function."""
    print("🎤 WebSocket Voice Audio Test")
    print("=" * 50)
    
    success = await test_websocket_voice_audio()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test PASSED! WebSocket voice conversation is working!")
    else:
        print("❌ Test FAILED! Issues found in voice processing.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
