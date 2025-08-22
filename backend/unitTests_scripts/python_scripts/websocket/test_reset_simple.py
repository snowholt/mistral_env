#!/usr/bin/env python3
"""
Simple test to verify conversation reset is working.
"""

import asyncio
import json
import websockets
import time

async def test_reset():
    """Test conversation reset functionality."""
    uri = "ws://localhost:8000/api/v1/ws/streaming-voice?language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            print(f"📦 Ready: {ready_msg}")
            
            # Send a beauty question first
            print(f"\n🗣️  Sending initial question...")
            question_msg = {
                "type": "final_transcript",
                "text": "What is the best skincare routine for dry skin?",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(question_msg))
            
            # Process messages for a bit
            print(f"📦 Waiting for responses...")
            for i in range(10):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    print(f"📦 Message {i+1}: {response_data.get('type', 'unknown')} - {response_data.get('message', response_data.get('text', ''))[:50]}...")
                except asyncio.TimeoutError:
                    break
            
            # Send reset message
            print(f"\n🔄 Sending conversation reset...")
            reset_msg = {
                "type": "reset_conversation"
            }
            await websocket.send(json.dumps(reset_msg))
            
            # Wait for reset confirmation
            print(f"📦 Waiting for reset confirmation...")
            for i in range(10):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    print(f"📦 Reset Response {i+1}: {response_data}")
                    
                    if response_data.get("type") == "conversation_reset":
                        print(f"✅ Conversation reset confirmed!")
                        break
                except asyncio.TimeoutError:
                    print(f"⏰ Timeout waiting for reset response")
                    break
            
            print("\n✅ Test completed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_reset())