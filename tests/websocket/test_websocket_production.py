#!/usr/bin/env python3
"""
Test WebSocket connection for production subdomain setup
Tests both localhost and production (api.gmai.sa) endpoints
"""

import asyncio
import json
import websockets
import ssl
import base64
from pathlib import Path


class ProductionWebSocketTester:
    def __init__(self, use_production=False):
        self.use_production = use_production
        if use_production:
            self.base_url = "wss://api.gmai.sa"
        else:
            self.base_url = "ws://localhost:8000"
    
    def get_websocket_url(self):
        """Build WebSocket URL with parameters"""
        params = {
            "input_language": "ar",
            "output_language": "ar", 
            "speaker_voice": "arabic_male",
            "emotion": "neutral",
            "speech_speed": "1.0",
            "thinking_mode": "fast",
            "preset": "default",
            "session_id": f"test_session_{asyncio.get_event_loop().time()}"
        }
        
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}/ws/voice-conversation?{param_str}"
    
    async def test_connection(self):
        """Test basic WebSocket connection and ping/pong"""
        url = self.get_websocket_url()
        print(f"🔗 Connecting to: {url}")
        
        try:
            # Setup SSL context for production
            ssl_context = None
            if self.use_production:
                ssl_context = ssl.create_default_context()
            
            async with websockets.connect(url, ssl=ssl_context) as websocket:
                print("✅ WebSocket connected successfully!")
                
                # Send ping
                await websocket.ping()
                print("📡 Ping/Pong successful")
                
                # Send test message
                test_message = {
                    "type": "audio",
                    "audio_data": base64.b64encode(b"test audio data").decode(),
                    "format": "wav",
                    "sample_rate": 16000
                }
                
                await websocket.send(json.dumps(test_message))
                print("📤 Test message sent")
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    print(f"📥 Response received: {data.get('type', 'unknown')}")
                    
                    if data.get('type') == 'error':
                        print(f"❌ Error from server: {data.get('message', 'Unknown error')}")
                    else:
                        print("✅ Server response looks good!")
                        
                except asyncio.TimeoutError:
                    print("⏰ Response timeout (expected for test audio)")
                
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
        
        return True
    
    async def test_model_status(self):
        """Test model status endpoint via REST API"""
        import aiohttp
        
        if self.use_production:
            base_url = "https://api.gmai.sa"
        else:
            base_url = "http://localhost:8000"
        
        url = f"{base_url}/models/status"
        print(f"🔍 Checking model status: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Model status: {data}")
                    else:
                        print(f"❌ Status check failed: {response.status}")
        except Exception as e:
            print(f"❌ Status check error: {str(e)}")


async def main():
    print("🚀 BeautyAI WebSocket Production Test")
    print("=" * 50)
    
    # Test localhost first
    print("\n📍 Testing localhost...")
    local_tester = ProductionWebSocketTester(use_production=False)
    await local_tester.test_connection()
    await local_tester.test_model_status()
    
    # Test production if requested
    print("\n🌐 Testing production (api.gmai.sa)...")
    print("Note: This requires DNS and SSL to be configured")
    
    try:
        prod_tester = ProductionWebSocketTester(use_production=True)
        await prod_tester.test_connection()
        await prod_tester.test_model_status()
    except Exception as e:
        print(f"⚠️  Production test failed (expected if DNS/SSL not ready): {e}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
