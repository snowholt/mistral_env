#!/usr/bin/env python3
"""
Test script to verify the WebSocket fix is working correctly.

This script tests:
1. Direct backend WebSocket connection
2. Nginx-proxied WebSocket connection 
3. Frontend integration verification
"""

import asyncio
import ssl
import json
import time
import subprocess
import sys

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

async def test_direct_backend():
    """Test direct connection to backend WebSocket"""
    print("\n🔍 Testing direct backend WebSocket connection...")
    
    try:
        uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to: {uri}")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"✅ Welcome message: {welcome_data.get('type', 'unknown')}")
            print(f"   └─ Connection ID: {welcome_data.get('connection_id', 'N/A')}")
            
            # Send a ping
            ping_msg = json.dumps({"type": "ping", "timestamp": time.time()})
            await websocket.send(ping_msg)
            
            # Wait for pong
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"✅ Pong received: {pong_data.get('type', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Direct backend test failed: {e}")
        return False

async def test_nginx_proxy():
    """Test WebSocket connection through nginx proxy"""
    print("\n🔍 Testing nginx-proxied WebSocket connection...")
    
    try:
        # Create SSL context that ignores certificate verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        uri = "wss://dev.gmai.sa/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print(f"✅ Connected to: {uri}")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"✅ Welcome message: {welcome_data.get('type', 'unknown')}")
            print(f"   └─ Connection ID: {welcome_data.get('connection_id', 'N/A')}")
            
            # Send a ping
            ping_msg = json.dumps({"type": "ping", "timestamp": time.time()})
            await websocket.send(ping_msg)
            
            # Wait for pong
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"✅ Pong received: {pong_data.get('type', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Nginx proxy test failed: {e}")
        return False

def test_frontend_config():
    """Test frontend configuration"""
    print("\n🔍 Testing frontend configuration...")
    
    try:
        # Check frontend config file
        with open('/home/lumi/beautyai/.env.production', 'r') as f:
            config_content = f.read()
            
        if 'ENVIRONMENT=production' in config_content:
            print("✅ Frontend environment: production")
        else:
            print("❌ Frontend environment not set to production")
            return False
            
        if 'WSS_PROTOCOL=wss' in config_content:
            print("✅ WebSocket protocol: WSS")
        else:
            print("❌ WebSocket protocol not set to WSS")
            return False
            
        if 'WSS_HOST=dev.gmai.sa' in config_content:
            print("✅ WebSocket host: dev.gmai.sa")
        else:
            print("❌ WebSocket host not set to dev.gmai.sa")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Frontend config test failed: {e}")
        return False

def test_nginx_config():
    """Test nginx configuration"""
    print("\n🔍 Testing nginx configuration...")
    
    try:
        # Test nginx syntax
        result = subprocess.run(['sudo', 'nginx', '-t'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Nginx configuration syntax: valid")
        else:
            print(f"❌ Nginx configuration syntax: invalid\n{result.stderr}")
            return False
            
        # Check if nginx is running
        result = subprocess.run(['sudo', 'systemctl', 'is-active', 'nginx'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip() == 'active':
            print("✅ Nginx service: running")
        else:
            print("❌ Nginx service: not running")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Nginx config test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 BeautyAI WebSocket Fix Verification")
    print("=" * 50)
    
    # Test results
    results = {}
    
    # Test nginx configuration
    results['nginx_config'] = test_nginx_config()
    
    # Test frontend configuration  
    results['frontend_config'] = test_frontend_config()
    
    # Test direct backend connection
    results['direct_backend'] = await test_direct_backend()
    
    # Test nginx proxy connection
    results['nginx_proxy'] = await test_nginx_proxy()
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! WebSocket fix is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Visit https://dev.gmai.sa/voice to test the voice interface")
        print("   2. Click the Connect button to establish WebSocket connection")
        print("   3. Test voice input/output functionality")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)