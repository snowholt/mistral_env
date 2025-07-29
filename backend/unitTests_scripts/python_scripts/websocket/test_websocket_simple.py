#!/usr/bin/env python3
"""
Simple WebSocket connection test for BeautyAI voice conversation endpoint.
Tests if the WebSocket endpoint is responding and accepting connections.
"""
import asyncio
import json
import logging
import time
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test basic WebSocket connection to voice conversation endpoint."""
    
    # WebSocket URL
    ws_url = "ws://dev.gmai.sa:8000/ws/voice-conversation"
    
    print("🔍 Testing WebSocket Voice Conversation Endpoint")
    print("=" * 60)
    print(f"📡 URL: {ws_url}")
    print()
    
    try:
        print("🚀 Attempting to connect to WebSocket...")
        
        # Try to connect with a short timeout
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection established successfully!")
            print(f"📊 Connection state: {websocket.state}")
            
            # Wait for the connection confirmation message
            try:
                print("⏳ Waiting for connection confirmation message...")
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                
                try:
                    data = json.loads(message)
                    print("📨 Received connection message:")
                    print(f"   Type: {data.get('type', 'unknown')}")
                    print(f"   Message: {data.get('message', 'N/A')}")
                    print(f"   Connection ID: {data.get('connection_id', 'N/A')}")
                    print(f"   Session ID: {data.get('session_id', 'N/A')}")
                except json.JSONDecodeError:
                    print(f"📨 Received raw message: {message}")
                
                # Send a ping to test bidirectional communication
                print("\n🏓 Testing ping/pong...")
                ping_msg = json.dumps({
                    "type": "ping",
                    "timestamp": time.time(),
                    "test": True
                })
                await websocket.send(ping_msg)
                print("📤 Ping sent")
                
                # Wait for pong response
                try:
                    pong_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    pong_data = json.loads(pong_response)
                    print("📥 Pong received:")
                    print(f"   Type: {pong_data.get('type', 'unknown')}")
                    print(f"   Timestamp: {pong_data.get('timestamp', 'N/A')}")
                    print("✅ Bidirectional communication working!")
                    
                except asyncio.TimeoutError:
                    print("⚠️  No pong response received (timeout)")
                except json.JSONDecodeError:
                    print(f"📥 Raw pong response: {pong_response}")
                
            except asyncio.TimeoutError:
                print("⚠️  No initial message received (timeout)")
            
            print("\n🎯 WebSocket endpoint is working and responding!")
            return True
            
    except ConnectionClosed as e:
        print(f"❌ WebSocket connection closed: {e}")
        return False
        
    except InvalidURI as e:
        print(f"❌ Invalid WebSocket URI: {e}")
        return False
        
    except OSError as e:
        print(f"❌ Network error: {e}")
        print("   Possible causes:")
        print("   - Server is not running")
        print("   - Wrong hostname/port")
        print("   - Firewall blocking connection")
        return False
        
    except asyncio.TimeoutError:
        print("❌ Connection timeout")
        print("   Possible causes:")
        print("   - Server is not responding")
        print("   - Network latency issues")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_websocket_with_parameters():
    """Test WebSocket connection with query parameters."""
    
    # Test with some parameters
    params = {
        "input_language": "ar",
        "output_language": "ar", 
        "speaker_voice": "female",
        "preset": "qwen_optimized",
        "session_id": f"test_session_{int(time.time())}"
    }
    
    # Build URL with parameters
    param_string = "&".join([f"{k}={v}" for k, v in params.items()])
    ws_url = f"ws://dev.gmai.sa:8000/ws/voice-conversation?{param_string}"
    
    print("\n" + "=" * 60)
    print("🔧 Testing WebSocket with Parameters")
    print("=" * 60)
    print(f"📡 URL: {ws_url}")
    print(f"📋 Parameters: {params}")
    print()
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection with parameters successful!")
            
            # Wait for connection message
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(message)
                print("📨 Connection confirmation:")
                print(f"   Session ID: {data.get('session_id', 'N/A')}")
                print(f"   Type: {data.get('type', 'unknown')}")
                return True
                
            except asyncio.TimeoutError:
                print("⚠️  No response received")
                return False
                
    except Exception as e:
        print(f"❌ Error testing with parameters: {e}")
        return False

async def main():
    """Run all WebSocket tests."""
    
    print("🎤 BeautyAI WebSocket Voice Conversation Test")
    print("=" * 60)
    print(f"🕒 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Basic connection
    result1 = await test_websocket_connection()
    
    # Test 2: Connection with parameters
    result2 = await test_websocket_with_parameters()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Basic Connection: {'PASS' if result1 else 'FAIL'}")
    print(f"✅ Parametrized Connection: {'PASS' if result2 else 'FAIL'}")
    
    if result1 and result2:
        print("\n🎉 WebSocket endpoint is fully functional!")
        print("🚀 Ready for voice conversation testing!")
    else:
        print("\n❌ WebSocket endpoint has issues")
        print("🔧 Check server status and configuration")
    
    print(f"🕒 Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
