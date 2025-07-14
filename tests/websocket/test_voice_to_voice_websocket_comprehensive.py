#!/usr/bin/env python3
"""
Comprehensive WebSocket Voice-to-Voice Test Script for BeautyAI Framework.

This script tests the WebSocket voice-to-voice functionality using real audio files
and both localhost:8000 and api.gmai.sa (nginx proxy) endpoints.

Usage:
    python test_voice_to_voice_websocket_comprehensive.py
    
Test Files:
    - /home/lumi/beautyai/voice_tests/input_test_questions/botox.wav
    - /home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav
    
Endpoints Tested:
    - ws://localhost:8000/ws/voice-conversation
    - wss://api.gmai.sa/ws/voice-conversation
"""

import asyncio
import json
import logging
import time
import base64
import ssl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/lumi/beautyai/tests/websocket/websocket_test.log')
    ]
)
logger = logging.getLogger(__name__)


class VoiceWebSocketTester:
    """Comprehensive WebSocket voice-to-voice tester."""
    
    def __init__(self):
        self.test_results = []
        self.test_audio_files = [
            "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav",
            "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
        ]
        
        # Test endpoints
        self.endpoints = {
            "localhost": "ws://localhost:8000/ws/voice-conversation",
            "nginx_proxy": "wss://api.gmai.sa/ws/voice-conversation"
        }
        
        # Test configurations
        self.test_configs = [
            {
                "name": "basic_arabic",
                "params": {
                    "input_language": "ar",
                    "output_language": "ar", 
                    "speaker_voice": "female",
                    "preset": "qwen_optimized",
                    "thinking_mode": "false"
                }
            },
            {
                "name": "basic_english",
                "params": {
                    "input_language": "en",
                    "output_language": "en",
                    "speaker_voice": "female", 
                    "preset": "qwen_optimized",
                    "thinking_mode": "false"
                }
            },
            {
                "name": "auto_detection",
                "params": {
                    "input_language": "auto",
                    "output_language": "auto",
                    "speaker_voice": "female",
                    "preset": "qwen_optimized",
                    "thinking_mode": "false"
                }
            },
            {
                "name": "speed_optimized",
                "params": {
                    "input_language": "auto",
                    "output_language": "auto", 
                    "speaker_voice": "female",
                    "preset": "speed_optimized",
                    "thinking_mode": "false",
                    "temperature": "0.1",
                    "max_new_tokens": "64"
                }
            }
        ]
    
    def validate_audio_files(self) -> bool:
        """Validate that test audio files exist."""
        logger.info("🔍 Validating test audio files...")
        
        for audio_file in self.test_audio_files:
            path = Path(audio_file)
            if not path.exists():
                logger.error(f"❌ Audio file not found: {audio_file}")
                return False
            
            file_size = path.stat().st_size
            logger.info(f"✅ Found: {path.name} ({file_size} bytes)")
        
        return True
    
    def build_websocket_url(self, base_url: str, config: Dict[str, Any]) -> str:
        """Build WebSocket URL with query parameters."""
        params = []
        for key, value in config["params"].items():
            params.append(f"{key}={value}")
        
        session_id = f"test_{config['name']}_{int(time.time())}"
        params.append(f"session_id={session_id}")
        
        url = f"{base_url}?{'&'.join(params)}"
        return url
    
    async def test_websocket_connection(
        self, 
        endpoint_name: str, 
        base_url: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test WebSocket connection and basic functionality."""
        
        test_name = f"{endpoint_name}_{config['name']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING: {test_name}")
        logger.info(f"{'='*60}")
        
        url = self.build_websocket_url(base_url, config)
        logger.info(f"📡 URL: {url}")
        
        test_result = {
            "test_name": test_name,
            "endpoint": endpoint_name,
            "config": config["name"],
            "url": url,
            "success": False,
            "connection_time": 0.0,
            "total_time": 0.0,
            "error": None,
            "transcription": None,
            "response_text": None,
            "audio_received": False,
            "audio_size": 0,
            "messages_received": []
        }
        
        try:
            # Setup SSL context for wss:// connections
            ssl_context = None
            if base_url.startswith("wss://"):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.info("🔒 Using SSL context with certificate verification disabled")
            
            start_time = time.time()
            
            # Connect to WebSocket
            logger.info("🚀 Attempting WebSocket connection...")
            async with websockets.connect(url, ssl=ssl_context) as websocket:
                connection_time = time.time() - start_time
                test_result["connection_time"] = connection_time
                logger.info(f"✅ WebSocket connected in {connection_time:.2f}s")
                
                # Wait for connection confirmation
                logger.info("⏳ Waiting for connection confirmation...")
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    test_result["messages_received"].append(data)
                    
                    if data.get("type") == "connection_established":
                        logger.info(f"🔗 Connection confirmed: {data.get('session_id')}")
                        logger.info(f"📊 Connection ID: {data.get('connection_id')}")
                    else:
                        logger.warning(f"⚠️ Unexpected first message: {data.get('type')}")
                        
                except asyncio.TimeoutError:
                    logger.warning("⏰ No connection confirmation received")
                
                # Test with both audio files
                for audio_file in self.test_audio_files:
                    if not Path(audio_file).exists():
                        continue
                        
                    logger.info(f"\n🎵 Testing with audio file: {Path(audio_file).name}")
                    audio_result = await self.send_audio_and_wait_response(
                        websocket, audio_file, test_result
                    )
                    
                    if audio_result["success"]:
                        test_result["success"] = True
                        test_result["transcription"] = audio_result.get("transcription")
                        test_result["response_text"] = audio_result.get("response_text") 
                        test_result["audio_received"] = audio_result.get("audio_received", False)
                        test_result["audio_size"] = audio_result.get("audio_size", 0)
                        logger.info("✅ Audio test completed successfully")
                        break
                    else:
                        logger.warning(f"⚠️ Audio test failed: {audio_result.get('error')}")
                
                test_result["total_time"] = time.time() - start_time
                
        except ConnectionClosed as e:
            error_msg = f"WebSocket connection closed: {e}"
            logger.error(f"❌ {error_msg}")
            test_result["error"] = error_msg
            
        except InvalidURI as e:
            error_msg = f"Invalid WebSocket URI: {e}"
            logger.error(f"❌ {error_msg}")
            test_result["error"] = error_msg
            
        except OSError as e:
            error_msg = f"Network error: {e}"
            logger.error(f"❌ {error_msg}")
            test_result["error"] = error_msg
            
        except asyncio.TimeoutError:
            error_msg = "Connection timeout"
            logger.error(f"❌ {error_msg}")
            test_result["error"] = error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            test_result["error"] = error_msg
        
        # Log test summary
        status = "✅ PASSED" if test_result["success"] else "❌ FAILED"
        logger.info(f"\n📊 TEST RESULT: {status}")
        logger.info(f"   Connection Time: {test_result['connection_time']:.2f}s")
        logger.info(f"   Total Time: {test_result['total_time']:.2f}s")
        if test_result["transcription"]:
            logger.info(f"   Transcription: {test_result['transcription'][:100]}...")
        if test_result["response_text"]:
            logger.info(f"   Response: {test_result['response_text'][:100]}...")
        if test_result["audio_received"]:
            logger.info(f"   Audio Size: {test_result['audio_size']} bytes")
        if test_result["error"]:
            logger.info(f"   Error: {test_result['error']}")
        
        return test_result
    
    async def send_audio_and_wait_response(
        self, 
        websocket, 
        audio_file: str, 
        test_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send audio file and wait for voice response."""
        
        result = {
            "success": False,
            "transcription": None,
            "response_text": None,
            "audio_received": False,
            "audio_size": 0,
            "error": None
        }
        
        try:
            # Read audio file
            audio_path = Path(audio_file)
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"📤 Sending audio: {audio_path.name} ({len(audio_data)} bytes)")
            
            # Send audio as binary message
            await websocket.send(audio_data)
            
            # Wait for response with timeout
            timeout = 120  # 2 minutes
            start_time = time.time()
            
            logger.info(f"⏳ Waiting for voice response (timeout: {timeout}s)...")
            
            while (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(message)
                        test_result["messages_received"].append(data)
                        
                        message_type = data.get("type", "unknown")
                        logger.info(f"📨 Received: {message_type}")
                        
                        if message_type == "processing_started":
                            logger.info("⚡ Processing started...")
                            continue
                            
                        elif message_type == "voice_response":
                            success = data.get("success", False)
                            
                            if success:
                                result["success"] = True
                                result["transcription"] = data.get("transcription", "")
                                result["response_text"] = data.get("response_text", "")
                                
                                # Check for audio data
                                audio_base64 = data.get("audio_base64")
                                if audio_base64:
                                    result["audio_received"] = True
                                    result["audio_size"] = data.get("audio_size_bytes", 0)
                                    logger.info(f"🎵 Audio received: {result['audio_size']} bytes")
                                    
                                    # Save audio file for verification
                                    audio_output_dir = Path("/home/lumi/beautyai/tests/websocket/test_outputs")
                                    audio_output_dir.mkdir(exist_ok=True)
                                    
                                    output_file = audio_output_dir / f"response_{int(time.time())}.wav"
                                    audio_bytes = base64.b64decode(audio_base64)
                                    with open(output_file, "wb") as f:
                                        f.write(audio_bytes)
                                    logger.info(f"💾 Audio saved to: {output_file}")
                                else:
                                    logger.warning("⚠️ No audio data in response")
                                
                                logger.info(f"📝 Transcription: {result['transcription']}")
                                logger.info(f"💬 Response: {result['response_text']}")
                                
                                return result
                            else:
                                error_msg = data.get("error", "Voice response failed")
                                result["error"] = error_msg
                                logger.error(f"❌ Voice response failed: {error_msg}")
                                return result
                                
                        elif message_type == "error":
                            error_msg = data.get("error", "Unknown error")
                            result["error"] = error_msg
                            logger.error(f"❌ Server error: {error_msg}")
                            return result
                            
                        else:
                            logger.info(f"📨 Other message: {message_type}")
                            continue
                            
                    except json.JSONDecodeError:
                        logger.warning("⚠️ Received non-JSON message")
                        continue
                        
                except asyncio.TimeoutError:
                    logger.info("⏳ Still waiting for response...")
                    continue
            
            # Timeout reached
            result["error"] = f"Timeout waiting for response after {timeout}s"
            logger.error(f"⏰ {result['error']}")
            
        except Exception as e:
            result["error"] = f"Error sending audio: {str(e)}"
            logger.error(f"❌ {result['error']}")
        
        return result
    
    async def test_ping_pong(self, websocket) -> bool:
        """Test WebSocket ping/pong functionality."""
        try:
            logger.info("🏓 Testing ping/pong...")
            
            ping_message = {
                "type": "ping",
                "timestamp": time.time(),
                "test": "ping_pong_test"
            }
            
            await websocket.send(json.dumps(ping_message))
            logger.info("📤 Ping sent")
            
            # Wait for pong
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            
            if data.get("type") == "pong":
                logger.info("✅ Pong received")
                return True
            else:
                logger.warning(f"⚠️ Expected pong, got: {data.get('type')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ping/pong failed: {e}")
            return False
    
    async def run_comprehensive_tests(self) -> List[Dict[str, Any]]:
        """Run comprehensive WebSocket tests."""
        logger.info("🚀 Starting Comprehensive WebSocket Voice-to-Voice Tests")
        logger.info("="*80)
        
        # Validate audio files first
        if not self.validate_audio_files():
            logger.error("❌ Audio file validation failed")
            return []
        
        all_results = []
        
        # Test each endpoint with each configuration
        for endpoint_name, base_url in self.endpoints.items():
            logger.info(f"\n🌐 TESTING ENDPOINT: {endpoint_name}")
            logger.info(f"📡 URL: {base_url}")
            
            for config in self.test_configs:
                result = await self.test_websocket_connection(
                    endpoint_name, base_url, config
                )
                all_results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(2)
        
        self.test_results = all_results
        return all_results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("="*80)
        report.append("🧪 BEAUTYAI WEBSOCKET VOICE-TO-VOICE TEST REPORT")
        report.append("="*80)
        report.append(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Total Tests: {len(self.test_results)}")
        
        # Summary statistics
        passed_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        report.append(f"✅ Passed: {len(passed_tests)}")
        report.append(f"❌ Failed: {len(failed_tests)}")
        report.append(f"📈 Success Rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        report.append("")
        
        # Endpoint summary
        endpoints = set(r["endpoint"] for r in self.test_results)
        for endpoint in endpoints:
            endpoint_results = [r for r in self.test_results if r["endpoint"] == endpoint]
            endpoint_passed = [r for r in endpoint_results if r["success"]]
            
            report.append(f"🌐 {endpoint.upper()}:")
            report.append(f"   Tests: {len(endpoint_results)}")
            report.append(f"   Passed: {len(endpoint_passed)}")
            report.append(f"   Success Rate: {len(endpoint_passed)/len(endpoint_results)*100:.1f}%")
            report.append("")
        
        # Detailed results
        report.append("📋 DETAILED TEST RESULTS:")
        report.append("-"*60)
        
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report.append(f"{status} {result['test_name']}")
            report.append(f"   Connection: {result['connection_time']:.2f}s")
            report.append(f"   Total Time: {result['total_time']:.2f}s")
            
            if result["transcription"]:
                report.append(f"   Transcription: {result['transcription'][:100]}...")
            if result["response_text"]:
                report.append(f"   Response: {result['response_text'][:100]}...")
            if result["audio_received"]:
                report.append(f"   Audio: {result['audio_size']} bytes")
            if result["error"]:
                report.append(f"   Error: {result['error']}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("-"*30)
        
        if failed_tests:
            common_errors = {}
            for test in failed_tests:
                error = test.get("error", "Unknown error")
                common_errors[error] = common_errors.get(error, 0) + 1
            
            report.append("🔍 Most common errors:")
            for error, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   {count}x: {error}")
            report.append("")
        
        if passed_tests:
            avg_time = sum(r["total_time"] for r in passed_tests) / len(passed_tests)
            report.append(f"⚡ Average processing time: {avg_time:.2f}s")
            
            audio_success = [r for r in passed_tests if r["audio_received"]]
            report.append(f"🎵 Audio generation rate: {len(audio_success)}/{len(passed_tests)} ({len(audio_success)/len(passed_tests)*100:.1f}%)")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_test_report(self) -> str:
        """Save test report to file."""
        report = self.generate_test_report()
        
        # Save to file
        report_file = Path("/home/lumi/beautyai/tests/websocket/websocket_test_report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        
        # Also save JSON results
        json_file = Path("/home/lumi/beautyai/tests/websocket/websocket_test_results.json")
        with open(json_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        logger.info(f"📊 JSON results saved to: {json_file}")
        
        return str(report_file)


async def main():
    """Main test function."""
    tester = VoiceWebSocketTester()
    
    try:
        # Run all tests
        results = await tester.run_comprehensive_tests()
        
        # Generate and display report
        report = tester.generate_test_report()
        print(report)
        
        # Save report
        report_file = tester.save_test_report()
        
        # Summary
        passed = len([r for r in results if r["success"]])
        total = len(results)
        
        logger.info(f"\n🎯 TEST SUMMARY:")
        logger.info(f"   Passed: {passed}/{total}")
        logger.info(f"   Success Rate: {passed/total*100:.1f}%")
        logger.info(f"   Report: {report_file}")
        
        return passed == total  # Return True if all tests passed
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    # Ensure output directory exists
    Path("/home/lumi/beautyai/tests/websocket/test_outputs").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)
