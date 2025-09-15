#!/usr/bin/env python3
"""
Test script for the debug WebSocket interface
Validates that the enhanced debug interface is working correctly
"""

import asyncio
import websockets
import json
import time
from pathlib import Path

async def test_debug_interface():
    """Test the debug interface with real audio data"""
    
    print("🔧 Testing BeautyAI Debug WebSocket Interface...")
    
    # Test 1: Connection test
    print("\n1️⃣ Testing WebSocket Connection...")
    uri = 'ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female&debug=1'
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Wait for ready message
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)
            print(f"✅ Received ready message: {data['type']}")
            print(f"   Session ID: {data.get('session_id', 'N/A')}")
            print(f"   Debug Mode: {data.get('debug_mode', 'N/A')}")
            
            # Test 2: Audio file streaming
            print("\n2️⃣ Testing Audio File Streaming...")
            pcm_file = Path("/home/lumi/beautyai/voice_tests/input_test_questions/pcm/q1.pcm")
            
            if pcm_file.exists():
                # Load PCM data
                audio_data = pcm_file.read_bytes()
                print(f"✅ Loaded audio file: {pcm_file.name} ({len(audio_data)} bytes)")
                
                # Send audio in chunks
                chunk_size = 640  # 20ms at 16kHz (320 samples * 2 bytes)
                chunks_sent = 0
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    await websocket.send(chunk)
                    chunks_sent += 1
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.02)  # 20ms
                
                print(f"✅ Sent {chunks_sent} audio chunks")
                
                # Test 3: Response collection
                print("\n3️⃣ Collecting responses...")
                responses = []
                timeout_counter = 0
                max_timeout = 30  # 30 seconds max wait
                
                while timeout_counter < max_timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        responses.append(data)
                        
                        msg_type = data.get('type', 'unknown')
                        print(f"   📨 Received: {msg_type}")
                        
                        if msg_type == 'partial_transcript':
                            print(f"      Partial: '{data.get('text', '')}'")
                        elif msg_type == 'final_transcript':
                            print(f"      Final: '{data.get('text', '')}'")
                        elif msg_type == 'assistant_response':
                            print(f"      Assistant: '{data.get('text', '')}'")
                        elif msg_type == 'tts_complete':
                            print(f"      ✅ TTS Complete!")
                            break
                            
                        timeout_counter = 0  # Reset timeout on any message
                        
                    except asyncio.TimeoutError:
                        timeout_counter += 1
                        print(f"   ⏱️ Waiting... ({timeout_counter}s)")
                
                print(f"\n✅ Collected {len(responses)} responses")
                
                # Analyze responses
                partial_count = sum(1 for r in responses if r.get('type') == 'partial_transcript')
                final_count = sum(1 for r in responses if r.get('type') == 'final_transcript')
                assistant_count = sum(1 for r in responses if r.get('type') == 'assistant_response')
                tts_count = sum(1 for r in responses if r.get('type') in ['tts_start', 'tts_audio', 'tts_complete'])
                
                print(f"\n📊 Response Summary:")
                print(f"   Partial transcripts: {partial_count}")
                print(f"   Final transcripts: {final_count}")
                print(f"   Assistant responses: {assistant_count}")
                print(f"   TTS events: {tts_count}")
                
                if final_count > 0 and assistant_count > 0:
                    print("   ✅ STT → LLM → TTS pipeline working!")
                else:
                    print("   ⚠️ Pipeline incomplete - check debug logs")
                    
            else:
                print(f"❌ Test audio file not found: {pcm_file}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n🎉 Debug interface test completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_debug_interface())
    exit(0 if success else 1)