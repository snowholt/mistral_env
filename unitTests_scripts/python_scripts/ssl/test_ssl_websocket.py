#!/usr/bin/env python3
"""
Test SSL WebSocket connections with proper SSL handling
"""

import asyncio
import json
import websockets
import ssl
import time

async def test_localhost_direct():
    """Test direct localhost connection first"""
    url = "ws://localhost:8000/ws/voice-conversation"
    print("🔍 Testing localhost direct connection...")
    print(f"📡 URL: {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            print("✅ Connected to localhost!")
            
            # Wait for connection message
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(msg)
                print(f"📨 Received: {data.get('type', 'unknown')}")
                return True
            except asyncio.TimeoutError:
                print("⏰ No initial message (but connection succeeded)")
                return True
                
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_api_ssl_bypass():
    """Test API through nginx with SSL certificate verification disabled"""
    url = "wss://api.gmai.sa/ws/voice-conversation"
    print("\n🔍 Testing api.gmai.sa with SSL bypass...")
    print(f"📡 URL: {url}")
    
    try:
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with websockets.connect(url, ssl=ssl_context) as websocket:
            print("✅ Connected to api.gmai.sa (SSL bypass)!")
            
            # Wait for connection message
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(msg)
                print(f"📨 Received: {data.get('type', 'unknown')}")
                
                # Send a test ping
                ping = {"type": "ping", "timestamp": time.time()}
                await websocket.send(json.dumps(ping))
                print("📤 Sent ping")
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                resp_data = json.loads(response)
                print(f"📥 Response: {resp_data.get('type', 'unknown')}")
                
                return True
                
            except asyncio.TimeoutError:
                print("⏰ No initial message (but connection succeeded)")
                return True
                
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_api_ssl_proper():
    """Test API through nginx with proper SSL certificate verification"""
    url = "wss://api.gmai.sa/ws/voice-conversation"
    print("\n🔍 Testing api.gmai.sa with proper SSL...")
    print(f"📡 URL: {url}")
    
    try:
        # Use default SSL context (proper verification)
        async with websockets.connect(url) as websocket:
            print("✅ Connected to api.gmai.sa (proper SSL)!")
            
            # Wait for connection message
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(msg)
                print(f"📨 Received: {data.get('type', 'unknown')}")
                return True
            except asyncio.TimeoutError:
                print("⏰ No initial message (but connection succeeded)")
                return True
                
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("   This indicates SSL certificate issues")
        return False


async def main():
    print("🚀 SSL WebSocket Test for api.gmai.sa")
    print("=" * 50)
    
    # Test 1: Localhost direct (baseline)
    local_success = await test_localhost_direct()
    
    # Test 2: SSL bypass to verify proxy works
    bypass_success = await test_api_ssl_bypass()
    
    # Test 3: Proper SSL verification
    ssl_success = await test_api_ssl_proper()
    
    print("\n📊 Results Summary:")
    print(f"   Localhost Direct: {'✅' if local_success else '❌'}")
    print(f"   SSL Bypass:       {'✅' if bypass_success else '❌'}")
    print(f"   Proper SSL:       {'✅' if ssl_success else '❌'}")
    
    if bypass_success and not ssl_success:
        print("\n🔧 Issue Identified: SSL Certificate Problems")
        print("   The nginx proxy works, but SSL certificates need fixing")
        print("   Solution: Check Let's Encrypt certificate renewal")
    elif not bypass_success:
        print("\n🔧 Issue Identified: Nginx Proxy Problems")
        print("   The nginx configuration may not be working correctly")
    elif ssl_success:
        print("\n✅ All tests passed! SSL WebSocket is working correctly")


if __name__ == "__main__":
    asyncio.run(main())
