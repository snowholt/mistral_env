#!/usr/bin/env python3
"""
Simple test to verify buffer reset fixes are working.
Uploads one WebM file and checks for log messages.
"""

import asyncio
import websockets
import json
import time
from pathlib import Path

async def test_single_file():
    print("Starting single file test to verify buffer reset fixes...")
    
    # WebM file to test with
    webm_file = Path("/home/lumi/beautyai/voice_tests/input_test_questions/webm/greeting.webm")
    
    if not webm_file.exists():
        print(f"Error: Test file not found: {webm_file}")
        return
    
    uri = "ws://127.0.0.1:8000/api/v1/ws/streaming-voice?language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            print(f"Ready: {ready_data['type']}")
            
            # Upload the WebM file
            print(f"Uploading {webm_file.name} ({webm_file.stat().st_size} bytes)")
            with open(webm_file, 'rb') as f:
                webm_data = f.read()
            
            await websocket.send(webm_data)
            print("WebM data sent")
            
            # Listen for messages for 10 seconds
            print("Listening for messages...")
            start_time = time.time()
            
            while time.time() - start_time < 10:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'final_transcript':
                        print(f"Final transcript: '{data.get('text', '')}'")
                    elif msg_type == 'partial_transcript':
                        print(f"Partial transcript: '{data.get('text', '')}'")
                    elif msg_type == 'endpoint_event':
                        event = data.get('event', '')
                        if event == 'start':
                            print(f"Endpoint START event: utterance_index={data.get('utterance_index')}")
                        elif event == 'final':
                            print(f"Endpoint FINAL event: utterance_index={data.get('utterance_index')}")
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_single_file())