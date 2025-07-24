#!/usr/bin/env python3
"""
WebSocket Voice-to-Voice Test Script for BeautyAI Framework.

Tests the real-time WebSocket voice conversation endpoint with real audio files.
This script simulates a browser client connecting via WebSocket.
"""
import asyncio
import websockets
import json
import base64
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketVoiceClient:
    """WebSocket client for testing voice conversations."""
    
    def __init__(self, base_url: str = "ws://dev.gmai.sa:8000"):
        self.base_url = base_url
        self.websocket = None
        self.session_id = f"test_session_{int(time.time())}"
        self.message_count = 0
        
    async def connect(
        self,
        input_language: str = "ar",
        output_language: str = "ar",
        speaker_voice: str = "female",
        preset: str = "qwen_optimized",
        thinking_mode: bool = False
    ) -> bool:
        """Connect to the WebSocket voice endpoint."""
        try:
            # Build connection URL with parameters
            params = {
                "session_id": self.session_id,
                "input_language": input_language,
                "output_language": output_language,
                "speaker_voice": speaker_voice,
                "preset": preset,
                "thinking_mode": str(thinking_mode).lower(),
                "emotion": "neutral",
                "speech_speed": "1.0"
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{self.base_url}/ws/voice-conversation?{query_string}"
            
            logger.info(f"Connecting to: {url}")
            
            self.websocket = await websockets.connect(url)
            logger.info("✅ WebSocket connection established")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Disconnected from WebSocket")
    
    async def send_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """Send an audio file via WebSocket and wait for response."""
        if not self.websocket:
            return {"success": False, "error": "Not connected"}
        
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            return {"success": False, "error": f"Audio file not found: {audio_file_path}"}
        
        try:
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"📤 Sending audio file: {audio_path.name} ({len(audio_data)} bytes)")
            
            # Send audio data as binary message
            await self.websocket.send(audio_data)
            self.message_count += 1
            
            # Wait for response
            start_time = time.time()
            response = await self.wait_for_response(timeout=120)  # 2 minutes timeout
            processing_time = time.time() - start_time
            
            if response:
                response["client_processing_time"] = processing_time
                logger.info(f"✅ Received response in {processing_time:.2f}s")
                return response
            else:
                return {"success": False, "error": "No response received"}
        
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")
            return {"success": False, "error": str(e)}
    
    async def wait_for_response(self, timeout: float = 60) -> Optional[Dict[str, Any]]:
        """Wait for a voice response message."""
        try:
            # Wait for messages until we get a voice_response
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "connection_established":
                        logger.info(f"🔗 Connection confirmed: {data.get('session_id')}")
                        continue
                    
                    elif data.get("type") == "processing_started":
                        logger.info("⚡ Processing started...")
                        continue
                    
                    elif data.get("type") == "voice_response":
                        logger.info("🎤 Voice response received")
                        return data
                    
                    else:
                        logger.info(f"📨 Received message: {data.get('type', 'unknown')}")
                        continue
                
                except json.JSONDecodeError:
                    logger.warning("⚠️ Received non-JSON message")
                    continue
        
        except asyncio.TimeoutError:
            logger.error("⏰ Timeout waiting for response")
            return None
        except Exception as e:
            logger.error(f"❌ Error waiting for response: {e}")
            return None
    
    async def send_ping(self):
        """Send a ping message to test connection."""
        if self.websocket:
            ping_msg = json.dumps({"type": "ping", "timestamp": time.time()})
            await self.websocket.send(ping_msg)
            logger.info("📡 Ping sent")


async def test_websocket_voice_conversation():
    """Test the WebSocket voice conversation with real audio files."""
    
    print("🎤 WebSocket Voice-to-Voice Test")
    print("=" * 60)
    
    # Test audio files
    test_audio_files = [
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.webm",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.webm",
        "/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
    ]
    
    # Find available test file
    available_file = None
    for file_path in test_audio_files:
        if Path(file_path).exists():
            available_file = file_path
            break
    
    if not available_file:
        print("❌ No test audio files found!")
        print("Checked paths:")
        for path in test_audio_files:
            print(f"  - {path}")
        return False
    
    print(f"📁 Using test file: {available_file}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Arabic Conversation - Qwen Optimized",
            "config": {
                "input_language": "ar",
                "output_language": "ar",
                "speaker_voice": "female",
                "preset": "qwen_optimized",
                "thinking_mode": False
            }
        },
        {
            "name": "Arabic with Thinking Mode",
            "config": {
                "input_language": "ar",
                "output_language": "ar",
                "speaker_voice": "female",
                "preset": "high_quality",
                "thinking_mode": True
            }
        }
    ]
    
    # Output directory for results
    output_dir = Path("/home/lumi/beautyai/voice_tests/websocket_test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_tests = 0
    
    for i, test_case in enumerate(test_configs, 1):
        print(f"\n🎯 Test {i}: {test_case['name']}")
        print("-" * 50)
        
        client = WebSocketVoiceClient()
        
        try:
            # Connect to WebSocket
            connected = await client.connect(**test_case["config"])
            
            if not connected:
                print(f"❌ Failed to connect for test {i}")
                continue
            
            # Send audio and get response
            response = await client.send_audio_file(available_file)
            
            if response.get("success", False):
                print("✅ Test successful!")
                print(f"📝 Transcription: {response.get('transcription', 'N/A')}")
                print(f"🤖 AI Response: {response.get('response_text', 'N/A')[:100]}...")
                print(f"⏱️ Processing Time: {response.get('processing_time_ms', 0):.0f}ms")
                print(f"🔄 Client Total Time: {response.get('client_processing_time', 0):.2f}s")
                
                # Save audio response if available
                if response.get("audio_base64"):
                    audio_data = base64.b64decode(response["audio_base64"])
                    audio_format = response.get("audio_format", "wav")
                    output_file = output_dir / f"test_{i}_response.{audio_format}"
                    
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    
                    print(f"💾 Audio saved: {output_file}")
                    print(f"🎵 Audio size: {len(audio_data)} bytes")
                
                # Show models used
                models_used = response.get("models_used", {})
                if models_used:
                    print(f"🧠 Models used:")
                    for model_type, model_name in models_used.items():
                        print(f"   {model_type}: {model_name}")
                
                successful_tests += 1
            
            else:
                print(f"❌ Test failed: {response.get('error', 'Unknown error')}")
        
        except Exception as e:
            import traceback
            print(f"❌ Test exception: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
        
        finally:
            # Disconnect
            await client.disconnect()
            
            # Wait a bit between tests
            if i < len(test_configs):
                print("⏳ Waiting 3 seconds before next test...")
                await asyncio.sleep(3)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful tests: {successful_tests}/{len(test_configs)}")
    print(f"❌ Failed tests: {len(test_configs) - successful_tests}/{len(test_configs)}")
    
    if successful_tests > 0:
        print(f"📁 Output files saved to: {output_dir}")
    
    return successful_tests == len(test_configs)


async def test_websocket_connection_only():
    """Simple connection test without audio."""
    print("🔗 Testing WebSocket Connection Only")
    print("-" * 40)
    
    client = WebSocketVoiceClient()
    
    try:
        connected = await client.connect()
        
        if connected:
            print("✅ Connection successful!")
            
            # Send a ping
            await client.send_ping()
            
            # Wait a moment
            await asyncio.sleep(2)
            
            return True
        else:
            print("❌ Connection failed")
            return False
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    finally:
        await client.disconnect()


async def main():
    """Main test function."""
    print("🚀 BeautyAI WebSocket Voice-to-Voice Test Suite")
    print("=" * 60)
    
    # Test 1: Basic connection
    print("\n🔌 Test 1: WebSocket Connection")
    connection_ok = await test_websocket_connection_only()
    
    if not connection_ok:
        print("❌ Basic connection failed. Ensure the API server is running on localhost:8000")
        return
    
    # Test 2: Full voice conversation
    print("\n🎤 Test 2: Full Voice Conversation")
    conversation_ok = await test_websocket_voice_conversation()
    
    # Final result
    print("\n🎉 All tests completed!")
    if conversation_ok:
        print("✅ WebSocket voice conversation is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
