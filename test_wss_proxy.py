#!/usr/bin/env python3
"""
Test script to verify WSS connections through nginx proxy.
Tests both the direct port 8000 connection and the nginx proxy route.
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

async def test_direct_websocket():
    """Test direct WebSocket connection to port 8000."""
    ws_url = "ws://dev.gmai.sa:8000/ws/voice-conversation"
    
    print("🔍 Testing Direct WebSocket Connection (Port 8000)")
    print("=" * 60)
    print(f"📡 URL: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ Direct WebSocket connection successful!")
            
            # Test basic communication
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Received: {data.get('type', 'unknown')}")
            return True
            
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False

async def test_proxy_websocket():
    """Test WebSocket connection through nginx proxy."""
    ws_url = "wss://dev.gmai.sa/ws/voice-conversation"
    
    print("\n🔍 Testing Proxied WebSocket Connection (HTTPS/WSS)")
    print("=" * 60)
    print(f"📡 URL: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ Proxied WebSocket connection successful!")
            
            # Test basic communication
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Received: {data.get('type', 'unknown')}")
            
            # Test ping/pong through proxy
            ping_msg = json.dumps({
                "type": "ping",
                "timestamp": time.time(),
                "test": "proxy_test"
            })
            await websocket.send(ping_msg)
            
            response = await websocket.recv()
            pong_data = json.loads(response)
            print(f"📥 Pong through proxy: {pong_data.get('type', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Proxied connection failed: {e}")
        print("   This is expected until nginx is reconfigured")
        return False

async def test_fallback_mechanism():
    """Test the automatic fallback mechanism."""
    print("\n🔍 Testing Automatic Fallback Mechanism")
    print("=" * 60)
    
    # Test URLs in order of preference
    test_urls = [
        "wss://dev.gmai.sa/ws/voice-conversation",  # Preferred (through proxy)
        "ws://dev.gmai.sa:8000/ws/voice-conversation"  # Fallback (direct)
    ]
    
    for i, url in enumerate(test_urls, 1):
        protocol = "WSS" if url.startswith("wss") else "WS"
        print(f"\n🚀 Attempt {i}: {protocol} Connection")
        print(f"📡 URL: {url}")
        
        try:
            async with websockets.connect(url) as websocket:
                print(f"✅ {protocol} connection successful!")
                
                message = await websocket.recv()
                data = json.loads(message)
                print(f"📨 Connection confirmed: {data.get('type', 'unknown')}")
                
                return True, protocol
                
        except Exception as e:
            print(f"❌ {protocol} failed: {e}")
            continue
    
    return False, None

async def main():
    """Run all WebSocket tests."""
    print("🌐 BeautyAI WebSocket SSL/Proxy Configuration Test")
    print("=" * 70)
    print(f"🕒 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Direct connection (current working method)
    direct_result = await test_direct_websocket()
    
    # Test 2: Proxy connection (what we want to achieve)
    proxy_result = await test_proxy_websocket()
    
    # Test 3: Fallback mechanism
    fallback_result, fallback_protocol = await test_fallback_mechanism()
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Direct WS (port 8000): {'PASS' if direct_result else 'FAIL'}")
    print(f"✅ Proxied WSS (nginx): {'PASS' if proxy_result else 'FAIL'}")
    print(f"✅ Fallback mechanism: {'PASS' if fallback_result else 'FAIL'}")
    
    if fallback_result:
        print(f"🎯 Working protocol: {fallback_protocol}")
    
    print("\n🔧 NGINX CONFIGURATION INSTRUCTIONS:")
    print("=" * 70)
    print("1. Copy the updated nginx config to: /etc/nginx/sites-available/")
    print("2. Test nginx config: sudo nginx -t")
    print("3. Reload nginx: sudo systemctl reload nginx")
    print("4. Verify BeautyAI API is running: sudo systemctl status beautyai-api")
    print("5. Test the proxy connection again")
    
    if proxy_result:
        print("\n🎉 WSS through nginx proxy is working!")
        print("🚀 Your voice conversation will work over HTTPS!")
    else:
        print("\n⚠️  WSS through nginx needs configuration")
        print("🔄 But fallback to WS will work for testing")
    
    print(f"\n🕒 Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
