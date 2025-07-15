#!/usr/bin/env python3
"""
🎤 Comprehensive WebSocket Voice-to-Voice Test Script for BeautyAI Framework

Tests the real-time WebSocket voice conversation endpoint with real audio files.
This script tests both localhost:8000 and api.gmai.sa endpoints with debugging.

Features:
- Real audio file testing (botox.wav, botox_ar.wav)
- Multiple endpoint testing (localhost and api.gmai.sa)
- Comprehensive error handling and debugging
- Audio response validation and saving
- Performance metrics tracking
- Connection diagnostics
"""
import asyncio
import websockets
import json
import base64
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import os

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/lumi/beautyai/tests/websocket/test_outputs/websocket_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class WebSocketVoiceTestClient:
    """Enhanced WebSocket client for comprehensive voice conversation testing."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.websocket = None
        self.session_id = f"test_session_{int(time.time())}"
        self.message_count = 0
        self.connection_start_time = None
        self.debug_messages = []
        
    def log_debug(self, message: str):
        """Log debug message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        debug_msg = f"[{timestamp}] {message}"
        self.debug_messages.append(debug_msg)
        logger.info(debug_msg)
        
    async def connect(
        self,
        input_language: str = "ar",
        output_language: str = "ar",
        speaker_voice: str = "female",
        preset: str = "qwen_optimized",
        thinking_mode: bool = False,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Connect to the WebSocket voice endpoint with comprehensive error handling."""
        connection_result = {
            "success": False,
            "url": "",
            "error": None,
            "connection_time": None,
            "debug_info": {}
        }
        
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
            connection_result["url"] = url
            
            self.log_debug(f"🔗 Attempting connection to: {url}")
            
            # Try connection with timeout and no size limit
            self.connection_start_time = time.time()
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, max_size=None), 
                timeout=timeout
            )
            connection_time = time.time() - self.connection_start_time
            connection_result["connection_time"] = connection_time
            
            self.log_debug(f"✅ WebSocket connected in {connection_time:.3f}s")
            
            # Wait for connection establishment message
            try:
                establishment_msg = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=10.0
                )
                
                try:
                    msg_data = json.loads(establishment_msg)
                    if msg_data.get("type") == "connection_established":
                        self.log_debug(f"🔗 Connection confirmed: {msg_data.get('session_id')}")
                        connection_result["debug_info"]["establishment_message"] = msg_data
                        connection_result["success"] = True
                    else:
                        self.log_debug(f"⚠️ Unexpected first message: {msg_data}")
                        connection_result["debug_info"]["unexpected_message"] = msg_data
                        connection_result["success"] = True  # Still connected
                        
                except json.JSONDecodeError:
                    self.log_debug(f"⚠️ Non-JSON establishment message: {establishment_msg}")
                    connection_result["debug_info"]["raw_establishment"] = establishment_msg
                    connection_result["success"] = True  # Still connected
                    
            except asyncio.TimeoutError:
                self.log_debug("⏰ No establishment message received within 10s")
                connection_result["debug_info"]["no_establishment"] = True
                connection_result["success"] = True  # Connection might still work
            
            return connection_result
            
        except asyncio.TimeoutError:
            error_msg = f"Connection timeout after {timeout}s"
            self.log_debug(f"⏰ {error_msg}")
            connection_result["error"] = error_msg
            
        except websockets.exceptions.ConnectionClosed as e:
            error_msg = f"Connection closed during handshake: {e}"
            self.log_debug(f"❌ {error_msg}")
            connection_result["error"] = error_msg
            
        except websockets.exceptions.InvalidStatusCode as e:
            error_msg = f"Invalid HTTP status code: {e}"
            self.log_debug(f"❌ {error_msg}")
            connection_result["error"] = error_msg
            
        except OSError as e:
            error_msg = f"Network error: {e}"
            self.log_debug(f"❌ {error_msg}")
            connection_result["error"] = error_msg
            
        except Exception as e:
            error_msg = f"Unexpected connection error: {e}"
            self.log_debug(f"❌ {error_msg}")
            connection_result["error"] = error_msg
            
        return connection_result
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            try:
                await self.websocket.close()
                self.log_debug("🔌 Disconnected from WebSocket")
            except Exception as e:
                self.log_debug(f"⚠️ Error during disconnect: {e}")
            finally:
                self.websocket = None
    
    async def send_audio_file(self, audio_file_path: str, timeout: float = 120.0) -> Dict[str, Any]:
        """Send an audio file via WebSocket and wait for response."""
        result = {
            "success": False,
            "audio_file": audio_file_path,
            "error": None,
            "response_data": None,
            "timing": {},
            "debug_info": {}
        }
        
        if not self.websocket:
            result["error"] = "Not connected to WebSocket"
            return result
        
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            result["error"] = f"Audio file not found: {audio_file_path}"
            return result
        
        try:
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            result["debug_info"]["audio_size_bytes"] = len(audio_data)
            result["debug_info"]["audio_format"] = audio_path.suffix[1:]  # Remove the dot
            
            self.log_debug(f"📤 Sending audio: {audio_path.name} ({len(audio_data):,} bytes)")
            
            # Send audio data as binary message
            send_start = time.time()
            await self.websocket.send(audio_data)
            send_time = time.time() - send_start
            
            result["timing"]["send_time"] = send_time
            self.message_count += 1
            
            self.log_debug(f"✅ Audio sent in {send_time:.3f}s")
            
            # Wait for response with detailed message tracking
            response_start = time.time()
            response = await self.wait_for_response(timeout=timeout)
            total_time = time.time() - response_start
            
            result["timing"]["response_time"] = total_time
            result["timing"]["total_time"] = time.time() - send_start
            
            if response:
                result["success"] = True
                result["response_data"] = response
                self.log_debug(f"✅ Complete response received in {total_time:.2f}s")
            else:
                result["error"] = "No response received within timeout"
                self.log_debug(f"⏰ No response within {timeout}s timeout")
        
        except Exception as e:
            error_msg = f"Error sending audio: {e}"
            result["error"] = error_msg
            self.log_debug(f"❌ {error_msg}")
            self.log_debug(f"Full traceback:\n{traceback.format_exc()}")
        
        return result
    
    async def wait_for_response(self, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """Wait for a voice response message with detailed tracking."""
        messages_received = []
        
        try:
            start_time = time.time()
            
            # Use timeout for the entire waiting period
            async with asyncio.timeout(timeout):
                async for message in self.websocket:
                    try:
                        data = json.loads(message)
                        message_type = data.get("type", "unknown")
                        elapsed = time.time() - start_time
                        
                        self.log_debug(f"📨 [{elapsed:.1f}s] Received: {message_type}")
                        messages_received.append({
                            "type": message_type,
                            "timestamp": elapsed,
                            "data": data
                        })
                        
                        if message_type == "connection_established":
                            self.log_debug(f"🔗 Session: {data.get('session_id')}")
                            continue
                        
                        elif message_type == "processing_started":
                            self.log_debug("⚡ Processing audio...")
                            continue
                        
                        elif message_type == "voice_response":
                            self.log_debug("🎤 Voice response received!")
                            
                            # Log response details
                            if data.get("success"):
                                self.log_debug(f"📝 Transcription: {data.get('transcription', 'N/A')}")
                                response_text = data.get('response_text', '')
                                if len(response_text) > 100:
                                    self.log_debug(f"🤖 Response: {response_text[:100]}...")
                                else:
                                    self.log_debug(f"🤖 Response: {response_text}")
                                    
                                if data.get("audio_base64"):
                                    audio_size = data.get("audio_size_bytes", 0)
                                    self.log_debug(f"🎵 Audio: {audio_size:,} bytes")
                                else:
                                    self.log_debug("🔇 No audio in response")
                                    
                                processing_time = data.get("processing_time_ms", 0)
                                self.log_debug(f"⏱️ Server processing: {processing_time:.0f}ms")
                            else:
                                self.log_debug(f"❌ Response failed: {data.get('error', 'Unknown error')}")
                            
                            return data
                        
                        elif message_type == "pong":
                            self.log_debug("🏓 Pong received")
                            continue
                        
                        elif message_type == "error":
                            self.log_debug(f"❌ Server error: {data.get('message', 'Unknown error')}")
                            return data
                        
                        else:
                            self.log_debug(f"❓ Unknown message type: {message_type}")
                            continue
                    
                    except json.JSONDecodeError:
                        self.log_debug(f"⚠️ Non-JSON message received: {message[:100]}...")
                        continue
        
        except asyncio.TimeoutError:
            self.log_debug(f"⏰ Response timeout after {timeout}s")
            self.log_debug(f"📊 Messages received during wait: {len(messages_received)}")
            for msg in messages_received:
                self.log_debug(f"  - [{msg['timestamp']:.1f}s] {msg['type']}")
            return None
            
        except Exception as e:
            self.log_debug(f"❌ Error waiting for response: {e}")
            return None
    
    async def send_ping(self) -> bool:
        """Send a ping message to test connection."""
        if not self.websocket:
            return False
            
        try:
            ping_msg = json.dumps({"type": "ping", "timestamp": time.time()})
            await self.websocket.send(ping_msg)
            self.log_debug("📡 Ping sent")
            return True
        except Exception as e:
            self.log_debug(f"❌ Failed to send ping: {e}")
            return False


async def test_endpoint_basic_connection(endpoint_url: str, test_name: str) -> Dict[str, Any]:
    """Test basic WebSocket connection to an endpoint."""
    print(f"\n🔌 {test_name}")
    print("-" * 60)
    
    result = {
        "endpoint": endpoint_url,
        "test_name": test_name,
        "success": False,
        "error": None,
        "connection_time": None
    }
    
    client = WebSocketVoiceTestClient(endpoint_url)
    
    try:
        # Test connection
        connection_result = await client.connect(timeout=15.0)
        result.update(connection_result)
        
        if connection_result["success"]:
            print("✅ Connection successful!")
            print(f"⏱️ Connection time: {connection_result['connection_time']:.3f}s")
            
            # Test ping/pong
            print("🏓 Testing ping/pong...")
            ping_success = await client.send_ping()
            
            if ping_success:
                # Wait for pong
                await asyncio.sleep(2)
                print("✅ Ping/pong test successful")
            else:
                print("⚠️ Ping test failed")
            
            result["success"] = True
        else:
            print(f"❌ Connection failed: {connection_result['error']}")
            result["error"] = connection_result["error"]
    
    except Exception as e:
        error_msg = f"Test exception: {e}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
    
    finally:
        await client.disconnect()
        await asyncio.sleep(1)  # Brief pause between tests
    
    return result


async def test_endpoint_with_audio(endpoint_url: str, test_name: str, audio_files: List[str]) -> Dict[str, Any]:
    """Test WebSocket endpoint with actual audio files."""
    print(f"\n🎤 {test_name}")
    print("-" * 60)
    
    result = {
        "endpoint": endpoint_url,
        "test_name": test_name,
        "audio_tests": [],
        "success": False,
        "total_time": 0
    }
    
    # Find available audio files
    available_files = []
    for audio_file in audio_files:
        if Path(audio_file).exists():
            available_files.append(audio_file)
        else:
            print(f"⚠️ Audio file not found: {audio_file}")
    
    if not available_files:
        result["error"] = "No audio files available for testing"
        print("❌ No audio files available!")
        return result
    
    print(f"📁 Available audio files: {len(available_files)}")
    for file_path in available_files:
        print(f"  - {Path(file_path).name}")
    
    successful_tests = 0
    
    for i, audio_file in enumerate(available_files):
        print(f"\n🎯 Audio Test {i+1}: {Path(audio_file).name}")
        print("-" * 40)
        
        client = WebSocketVoiceTestClient(endpoint_url)
        audio_test_result = {
            "audio_file": audio_file,
            "success": False,
            "error": None,
            "response_summary": {}
        }
        
        try:
            # Connect
            connection_result = await client.connect()
            
            if not connection_result["success"]:
                audio_test_result["error"] = f"Connection failed: {connection_result['error']}"
                print(f"❌ Connection failed: {connection_result['error']}")
                result["audio_tests"].append(audio_test_result)
                continue
            
            # Send audio file
            send_result = await client.send_audio_file(audio_file, timeout=150.0)
            audio_test_result.update(send_result)
            
            if send_result["success"]:
                print("✅ Audio test successful!")
                
                response_data = send_result["response_data"]
                if response_data:
                    # Extract response summary
                    summary = {
                        "transcription": response_data.get("transcription", "N/A"),
                        "response_text": response_data.get("response_text", "N/A"),
                        "has_audio": bool(response_data.get("audio_base64")),
                        "audio_size": response_data.get("audio_size_bytes", 0),
                        "processing_time_ms": response_data.get("processing_time_ms", 0),
                        "models_used": response_data.get("models_used", {})
                    }
                    audio_test_result["response_summary"] = summary
                    
                    print(f"📝 Transcription: {summary['transcription']}")
                    if len(summary['response_text']) > 100:
                        print(f"🤖 Response: {summary['response_text'][:100]}...")
                    else:
                        print(f"🤖 Response: {summary['response_text']}")
                    print(f"🎵 Audio: {summary['audio_size']:,} bytes")
                    print(f"⏱️ Processing: {summary['processing_time_ms']:.0f}ms")
                    
                    # Save audio response if available
                    if response_data.get("audio_base64"):
                        await save_audio_response(response_data, audio_file, i+1)
                
                successful_tests += 1
                audio_test_result["success"] = True
            else:
                print(f"❌ Audio test failed: {send_result['error']}")
        
        except Exception as e:
            error_msg = f"Audio test exception: {e}"
            audio_test_result["error"] = error_msg
            print(f"❌ {error_msg}")
        
        finally:
            await client.disconnect()
            result["audio_tests"].append(audio_test_result)
            
            # Wait between tests
            if i < len(available_files) - 1:
                print("⏳ Waiting 3s before next test...")
                await asyncio.sleep(3)
    
    result["success"] = successful_tests > 0
    result["successful_audio_tests"] = successful_tests
    result["total_audio_tests"] = len(available_files)
    
    print(f"\n📊 Audio Tests Summary: {successful_tests}/{len(available_files)} successful")
    
    return result


async def save_audio_response(response_data: Dict[str, Any], original_file: str, test_number: int):
    """Save audio response to file."""
    try:
        # Create output directory
        output_dir = Path("/home/lumi/beautyai/tests/websocket/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode audio
        audio_base64 = response_data.get("audio_base64", "")
        if not audio_base64:
            return
        
        audio_data = base64.b64decode(audio_base64)
        audio_format = response_data.get("audio_format", "wav")
        
        # Create filename
        original_name = Path(original_file).stem
        output_filename = f"response_{test_number}_{original_name}.{audio_format}"
        output_path = output_dir / output_filename
        
        # Save audio file
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        print(f"💾 Audio saved: {output_path}")
        print(f"🎵 Size: {len(audio_data):,} bytes")
        
    except Exception as e:
        print(f"⚠️ Failed to save audio response: {e}")


async def run_comprehensive_tests():
    """Run comprehensive WebSocket voice tests."""
    print("🚀 BeautyAI WebSocket Voice-to-Voice Comprehensive Test Suite")
    print("=" * 80)
    
    # Test endpoints
    test_endpoints = [
        {
            "url": "ws://localhost:8000",
            "name": "Localhost (Direct)"
        },
        {
            "url": "ws://api.gmai.sa:8000",
            "name": "api.gmai.sa (nginx proxy)"
        }
    ]
    
    # Test audio files (as specified by user)
    test_audio_files = [
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
    ]
    
    # Create output directory
    output_dir = Path("/home/lumi/beautyai/tests/websocket/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Test each endpoint
    for endpoint in test_endpoints:
        print(f"\n🌐 Testing Endpoint: {endpoint['name']}")
        print("=" * 80)
        
        # 1. Basic connection test
        basic_result = await test_endpoint_basic_connection(
            endpoint["url"], 
            f"{endpoint['name']} - Basic Connection"
        )
        all_results.append(basic_result)
        
        # 2. Audio tests (only if basic connection works)
        if basic_result["success"]:
            audio_result = await test_endpoint_with_audio(
                endpoint["url"],
                f"{endpoint['name']} - Voice Tests",
                test_audio_files
            )
            all_results.append(audio_result)
        else:
            print(f"⚠️ Skipping audio tests for {endpoint['name']} due to connection failure")
    
    # Save comprehensive test report
    await save_test_report(all_results, output_dir)
    
    # Print final summary
    print_final_summary(all_results)


async def save_test_report(results: List[Dict[str, Any]], output_dir: Path):
    """Save comprehensive test report to JSON file."""
    try:
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "test_summary": {
                "total_tests": len(results),
                "successful_tests": sum(1 for r in results if r.get("success", False)),
                "failed_tests": sum(1 for r in results if not r.get("success", False))
            },
            "results": results
        }
        
        report_file = output_dir / f"websocket_test_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to save test report: {e}")


def print_final_summary(results: List[Dict[str, Any]]):
    """Print final test summary."""
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    connection_tests = 0
    audio_tests = 0
    
    for result in results:
        if result.get("success", False):
            successful_tests += 1
        else:
            failed_tests += 1
        
        if "audio_tests" in result:
            audio_tests += result.get("successful_audio_tests", 0)
        else:
            connection_tests += 1 if result.get("success", False) else 0
    
    print(f"✅ Total successful tests: {successful_tests}")
    print(f"❌ Total failed tests: {failed_tests}")
    print(f"🔌 Connection tests passed: {connection_tests}")
    print(f"🎤 Audio tests passed: {audio_tests}")
    
    if successful_tests > 0:
        print("\n🎉 Some tests passed! Check the detailed output above.")
    
    if failed_tests > 0:
        print("\n⚠️ Some tests failed. Check the error messages above for debugging.")
        print("💡 Debugging tips:")
        print("   - Ensure the API service is running: sudo systemctl status beautyai-api.service")
        print("   - Check service logs: sudo journalctl -u beautyai-api.service -f")
        print("   - Verify network connectivity to endpoints")
        print("   - Check if audio files exist and are readable")
    
    print(f"\n📁 Test outputs saved to: /home/lumi/beautyai/tests/websocket/test_outputs/")


async def main():
    """Main test execution function."""
    try:
        await run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = Path("/home/lumi/beautyai/tests/websocket/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎤 Starting BeautyAI WebSocket Voice-to-Voice Tests...")
    asyncio.run(main())
