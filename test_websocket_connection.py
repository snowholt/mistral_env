#!/usr/bin/env python3
"""Test WebSocket connection to the simple voice chat endpoint."""

import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """Test WebSocket connection with proper parameters."""
    
    # Test URLs - test different debug parameter values
    test_urls = [
        "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=en&voice_type=female&debug_mode=true",
        "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=en&voice_type=female&debug_mode=1", 
        "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=en&voice_type=female&debug_mode=True",
    ]
    
    for url in test_urls:
        print(f"\n🔗 Testing connection to: {url}")
        try:
            # Add proper headers for WebSocket connection
            headers = [
                ('Origin', 'https://dev.gmai.sa' if url.startswith('wss://') else 'http://localhost:8000'),
                ('User-Agent', 'WebSocket Test Client')
            ]
            
            async with websockets.connect(url, additional_headers=headers) as websocket:
                print("✅ Connection successful!")
                
                # Listen for initial messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(message)
                    print(f"📨 Received message type: {data.get('type')}")
                    print(f"    Language: {data.get('config', {}).get('language', 'N/A')}")
                    print(f"    Voice Type: {data.get('config', {}).get('voice_type', 'N/A')}")
                    print(f"    Debug Mode: {data.get('config', {}).get('debug_enabled', 'N/A')}")
                    
                    if data.get('type') == 'connection_established':
                        print(f"🎉 Session ready! Session ID: {data.get('session_id')}")
                        
                    # Try to close gracefully
                    await websocket.close()
                        
                except asyncio.TimeoutError:
                    print("⏰ No initial message received within 5 seconds")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse message as JSON: {e}")
                    print(f"Raw message: {message}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Connection closed: {e}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing WebSocket connections...")
    asyncio.run(test_websocket_connection())