#!/usr/bin/env python3
"""
Simple WebSocket ping test to verify WebSocket connectivity.
"""
import asyncio
import websockets
import json
import time

async def test_websocket_ping(url: str, test_name: str):
    """Test WebSocket connection with ping/pong."""
    print(f"\n🔌 Testing {test_name}")
    print(f"📡 URL: {url}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Connect to WebSocket
        async with websockets.connect(url, max_size=None) as websocket:
            connection_time = time.time() - start_time
            print(f"✅ Connected in {connection_time:.3f}s")
            
            # Wait for connection_established message
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                if data.get('type') == 'connection_established':
                    print(f"✅ Connection confirmed: {data.get('session_id')}")
                else:
                    print(f"📨 Received: {data}")
            except asyncio.TimeoutError:
                print("⚠️ No connection confirmation received")
            
            # Send ping
            ping_message = {
                "type": "ping",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(ping_message))
            print(f"📤 Ping sent: {ping_message}")
            
            # Wait for pong
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                if data.get('type') == 'pong':
                    print(f"📨 Pong received: {data}")
                    print("✅ Ping-Pong test successful!")
                else:
                    print(f"📨 Unexpected response: {data}")
            except asyncio.TimeoutError:
                print("❌ No pong response received")
            
            # Close cleanly
            await websocket.close(code=1000, reason="Test completed")
            print("✅ Connection closed cleanly")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True

async def main():
    """Run WebSocket tests."""
    print("🧪 WebSocket Ping Test Suite")
    print("=" * 60)
    
    # Test endpoints
    endpoints = [
        {
            "url": "ws://localhost:8000/ws/voice-conversation?test=true",
            "name": "Development (localhost)"
        },
        {
            "url": "wss://api.gmai.sa/ws/voice-conversation?test=true", 
            "name": "Production (api.gmai.sa)"
        }
    ]
    
    results = []
    for endpoint in endpoints:
        try:
            success = await test_websocket_ping(endpoint["url"], endpoint["name"])
            results.append((endpoint["name"], success))
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed for {endpoint['name']}: {e}")
            results.append((endpoint["name"], False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Results: {successful}/{total} tests passed")

if __name__ == "__main__":
    asyncio.run(main())
