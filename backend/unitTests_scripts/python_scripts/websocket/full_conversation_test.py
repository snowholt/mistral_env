#!/usr/bin/env python3
"""Enhanced WebSocket replay script that captures full conversation including model responses."""

import asyncio
import json
import sys
import time
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets", file=sys.stderr)
    sys.exit(1)


async def full_conversation_test(pcm_file: str, language: str, show_model_response=True):
    """Test full conversation flow including model response."""
    
    # Read PCM file
    pcm_path = Path(pcm_file)
    if not pcm_path.exists():
        print(f"PCM file not found: {pcm_file}", file=sys.stderr)
        return
    
    pcm_data = pcm_path.read_bytes()
    
    # WebSocket URL
    url = f"ws://localhost:8000/api/v1/ws/streaming-voice?language={language}"
    
    print(f"🔗 Connecting to: {url}")
    print(f"🎤 Testing: {pcm_file} ({language})")
    print(f"📊 PCM size: {len(pcm_data)} bytes")
    print("=" * 60)
    
    model_responses = []
    transcription_results = []
    
    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            
            # Send PCM data in chunks
            chunk_size = 640  # 20ms at 16kHz
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.01)  # 10ms between chunks for faster testing
            
            # Listen for events for up to 15 seconds
            timeout_time = time.time() + 15
            
            while time.time() < timeout_time:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    
                    try:
                        event = json.loads(message)
                        event_type = event.get("type", event.get("event"))
                        
                        # Print all events for debugging
                        print(f"📨 {event_type}: {json.dumps(event, ensure_ascii=False, indent=2)}")
                        
                        # Capture important events
                        if event_type == "final_transcript":
                            transcription_results.append(event.get("text", ""))
                            
                        elif event_type == "assistant_response" or event_type == "model_response":
                            model_responses.append(event.get("text", event.get("content", "")))
                            
                        elif event_type == "tts_start":
                            # This indicates TTS is starting, so we should have the model response
                            print(f"🔊 TTS Starting for utterance {event.get('utterance_index')}")
                            
                        elif "error" in event_type.lower():
                            print(f"❌ Error: {event}")
                            
                    except json.JSONDecodeError:
                        print(f"📜 Non-JSON message: {message}")
                        
                except asyncio.TimeoutError:
                    # Check if we should continue waiting
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connection closed")
                    break
                    
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    print(f"🗣️  Language: {language}")
    
    if transcription_results:
        print(f"📝 Transcription: {transcription_results[-1]}")
    else:
        print("📝 Transcription: ❌ No transcription received")
    
    if model_responses:
        print(f"🤖 Model Response: {model_responses[-1]}")
    else:
        print("🤖 Model Response: ❌ No model response captured")
        print("💡 Note: Model response might be in TTS pipeline or not fully implemented")
    
    print("=" * 60)


async def main():
    if len(sys.argv) < 3:
        print("Usage: python full_conversation_test.py <pcm_file> <language>")
        print("Example: python full_conversation_test.py voice_tests/input_test_questions/pcm/botox.pcm en")
        return
    
    pcm_file = sys.argv[1]
    language = sys.argv[2]
    
    await full_conversation_test(pcm_file, language)


if __name__ == "__main__":
    asyncio.run(main())