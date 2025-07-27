#!/usr/bin/env python3
"""
Debug test to check WebSocket connection
"""

import asyncio
import json
import websockets

async def test_connection():
    try:
        print("Attempting to connect to WebSocket...")
        uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"📝 Welcome message: {welcome_msg}")
            
            # Try sending a simple message
            print("📤 Sending test message...")
            await websocket.send(b"hello")
            
            # Wait for response
            async for message in websocket:
                print(f"📥 Received: {message}")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
