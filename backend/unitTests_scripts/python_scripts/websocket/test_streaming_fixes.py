#!/usr/bin/env python3
"""
Test script to verify the streaming voice websocket fixes.
This tests the language parameter passing and basic WebSocket functionality.
"""

import asyncio
import websockets
import json
import sys

async def test_language_parameter(language="en"):
    """Test WebSocket connection with language parameter"""
    
    # Test the language parameter passing fix
    url = f"ws://localhost:8000/api/v1/ws/streaming-voice?language={language}"
    
    print(f"🧪 Testing WebSocket connection with language={language}")
    print(f"📡 Connecting to: {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Wait for ready message
            response = await websocket.recv()
            message = json.loads(response)
            print(f"📨 Received: {message}")
            
            if message.get("type") == "ready":
                print("✅ Backend is ready to receive audio")
                return True
            elif message.get("type") == "error":
                print(f"❌ Backend error: {message.get('message')}")
                return False
            else:
                print(f"⚠️ Unexpected message type: {message.get('type')}")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def main():
    """Test both English and Arabic language parameters"""
    print("🚀 Testing Streaming Voice WebSocket Fixes")
    print("=" * 50)
    
    # Test English
    print("\n1️⃣ Testing English language parameter...")
    en_success = await test_language_parameter("en")
    
    # Test Arabic  
    print("\n2️⃣ Testing Arabic language parameter...")
    ar_success = await test_language_parameter("ar")
    
    # Test auto-detect
    print("\n3️⃣ Testing auto-detect language parameter...")
    auto_success = await test_language_parameter("auto")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   English (en): {'✅ PASS' if en_success else '❌ FAIL'}")
    print(f"   Arabic (ar):  {'✅ PASS' if ar_success else '❌ FAIL'}")
    print(f"   Auto-detect:  {'✅ PASS' if auto_success else '❌ FAIL'}")
    
    all_passed = en_success and ar_success and auto_success
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✨ The language parameter fix is working correctly!")
        print("   You can now use the frontend debug interface with confidence.")
    else:
        print("\n🔧 There are still issues that need to be resolved.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)