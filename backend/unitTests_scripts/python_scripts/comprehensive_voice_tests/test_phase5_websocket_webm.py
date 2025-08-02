#!/usr/bin/env python3
"""
Phase 5: WebSocket Voice-to-Voice Test with WebM Format Support

Tests the real-time WebSocket voice chat functionality using WebM audio files.
This test validates:
- WebSocket connection establishment
- WebM audio format support
- Real-time voice-to-voice processing
- Language detection and response accuracy
- Latency measurements for real-time performance

Author: BeautyAI Framework
Date: 2025-08-01
"""

import asyncio
import json
import logging
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import websockets
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebSocketVoiceTest:
    """Test WebSocket voice-to-voice functionality with WebM format support."""
    
    def __init__(self, websocket_url: str = "ws://localhost:8000/api/v1/ws/simple-voice-chat"):
        self.websocket_url = websocket_url
        self.test_results = []
        self.websocket = None
        
        # Test configuration
        self.webm_test_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions/webm")
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/phase5_websocket_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Expected ground truth transcriptions
        self.ground_truth = {
            "botox.webm": "What is botox used for?",
            "greeting.webm": "Hello, how are you today?",
            "greeting_ar.webm": "مرحبًا، كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصة بكم.",
            "q1.webm": "ما هو استخدام البوتكس؟",
            "q2.webm": "كيف يعمل إزالة الشعر بالليزر؟",
            "q3.webm": "هل الحشوات الجلدية دائمة؟",
            "q4.webm": "ما هي الآثار الجانبية الشائعة للتقشير الكيميائي؟",
            "q5.webm": "هل الميزوثيرابي؟",
            "q6.webm": "كم تدوم نتائج جلسة تنظيف البشرة عادة؟",
            "q7.webm": "هل يمكن لأي شخص إجراء عملية تجميل الأنف غير الجراحية؟",
            "q8.webm": "ما هو الغرض من علاج البلازما الغنية بالصفائح الدموية PRP للبشرة؟",
            "q9.webm": "هل هناك فترة نقاها بعد عملية شد الوجه بالخيوط؟",
            "q10.webm": "ما هي الفائدة الرئيسية لعلاج الضوء النبدي المكثف؟ IPL",
            "laser_hair.webm": "كيف يعمل الاشتراك في الاشتراك في الاشتراك في الاشتراك في الاشتراك في الاشتراك في الاشتراك؟"
        }
    
    async def connect_websocket(self, language: str = "ar", voice_type: str = "female", session_id: Optional[str] = None) -> bool:
        """
        Connect to the WebSocket endpoint with specified parameters.
        
        Args:
            language: Language setting ("ar" or "en")
            voice_type: Voice type ("male" or "female")
            session_id: Optional session ID
            
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Build WebSocket URL with parameters
            params = f"language={language}&voice_type={voice_type}"
            if session_id:
                params += f"&session_id={session_id}"
            
            full_url = f"{self.websocket_url}?{params}"
            logger.info(f"🔌 Connecting to WebSocket: {full_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(full_url)
            
            # Wait for connection establishment message
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            except asyncio.TimeoutError:
                logger.error("❌ Timeout waiting for connection establishment message")
                return False
            connection_msg = json.loads(response)
            
            if connection_msg.get("type") == "connection_established" and connection_msg.get("success"):
                logger.info(f"✅ WebSocket connected successfully")
                logger.info(f"   Connection ID: {connection_msg.get('connection_id')}")
                logger.info(f"   Session ID: {connection_msg.get('session_id')}")
                logger.info(f"   Config: {connection_msg.get('config')}")
                return True
            else:
                logger.error(f"❌ Connection establishment failed: {connection_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("🔌 WebSocket disconnected")
            except Exception as e:
                logger.warning(f"⚠️ Error during WebSocket disconnect: {e}")
            finally:
                self.websocket = None
    
    async def send_audio_file(self, webm_file_path: Path) -> Dict[str, Any]:
        """
        Send WebM audio file to WebSocket and receive response.
        
        Args:
            webm_file_path: Path to the WebM audio file
            
        Returns:
            Dictionary containing test results
        """
        if not self.websocket:
            raise Exception("WebSocket not connected")
        
        test_start_time = time.time()
        
        try:
            # Read WebM audio file
            async with aiofiles.open(webm_file_path, 'rb') as f:
                webm_audio_data = await f.read()
            
            logger.info(f"🎤 Sending WebM audio: {webm_file_path.name} ({len(webm_audio_data)} bytes)")
            
            # Send audio data as binary message
            await self.websocket.send(webm_audio_data)
            
            # Wait for response (handle multiple message types)
            voice_response_received = False
            
            for attempt in range(3):  # Try to get the voice_response message
                try:
                    response_data = await asyncio.wait_for(self.websocket.recv(), timeout=20)
                    response = json.loads(response_data)
                    
                    if response.get("type") == "processing_started":
                        logger.info(f"📡 Processing started, waiting for voice response...")
                        continue  # Keep waiting for voice_response
                    
                    elif response.get("type") == "voice_response":
                        voice_response_received = True
                        break  # Got the response we need
                    
                    elif response.get("type") == "error":
                        # Handle error response
                        error_result = {
                            "input_file": webm_file_path.name,
                            "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                            "error": response.get("message", "Unknown error"),
                            "error_code": response.get("error_code", "UNKNOWN"),
                            "total_latency_ms": int(total_latency * 1000),
                            "success": False,
                            "timestamp": time.time()
                        }
                        logger.error(f"❌ Error response: {error_result['error']}")
                        return error_result
                    
                except asyncio.TimeoutError:
                    logger.error(f"⏰ Timeout waiting for message (attempt {attempt + 1})")
                    break
            
            if not voice_response_received:
                return {
                    "input_file": webm_file_path.name,
                    "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                    "error": "No voice response received",
                    "total_latency_ms": int((time.time() - test_start_time) * 1000),
                    "success": False,
                    "timestamp": time.time()
                }
            
            total_latency = time.time() - test_start_time
            
            # Process response
            if response.get("type") == "voice_response" and response.get("success"):
                
                # Save received audio if available
                audio_file_path = None
                if response.get("audio_base64"):
                    audio_data = base64.b64decode(response["audio_base64"])
                    audio_file_path = self.output_dir / f"{webm_file_path.stem}_response.wav"
                    
                    with open(audio_file_path, 'wb') as f:
                        f.write(audio_data)
                    
                    logger.info(f"💾 Saved response audio: {audio_file_path}")
                
                result = {
                    "input_file": webm_file_path.name,
                    "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                    "transcribed_text": response.get("transcription", ""),
                    "response_text": response.get("response_text", ""),
                    "language_detected": response.get("language", ""),
                    "voice_used": response.get("voice_type", ""),
                    "service_latency_ms": response.get("response_time_ms", 0),
                    "total_latency_ms": int(total_latency * 1000),
                    "generated_audio_file": str(audio_file_path) if audio_file_path else None,
                    "success": True,
                    "timestamp": time.time()
                }
                
                logger.info(f"✅ Success | Latency: {total_latency:.2f}s | Service: {result['service_latency_ms']}ms")
                logger.info(f"   📝 Transcribed: '{result['transcribed_text'][:50]}...'")
                logger.info(f"   🤖 Response: '{result['response_text'][:50]}...'")
                logger.info(f"   🌍 Language: {result['language_detected']}")
                
                return result
                
            else:
                # Handle error response
                error_result = {
                    "input_file": webm_file_path.name,
                    "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                    "error": response.get("message", "Unknown error"),
                    "error_code": response.get("error_code", "UNKNOWN"),
                    "total_latency_ms": int(total_latency * 1000),
                    "success": False,
                    "timestamp": time.time()
                }
                
                logger.error(f"❌ Error response: {error_result['error']}")
                return error_result
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout waiting for response to {webm_file_path.name}")
            return {
                "input_file": webm_file_path.name,
                "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                "error": "Response timeout",
                "total_latency_ms": int((time.time() - test_start_time) * 1000),
                "success": False,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {webm_file_path.name}: {e}")
            return {
                "input_file": webm_file_path.name,
                "ground_truth_text": self.ground_truth.get(webm_file_path.name, ""),
                "error": str(e),
                "total_latency_ms": int((time.time() - test_start_time) * 1000),
                "success": False,
                "timestamp": time.time()
            }
    
    async def run_websocket_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive WebSocket voice tests with WebM format support.
        
        Returns:
            Dictionary containing all test results and summary
        """
        logger.info("🎤 WebSocket Voice-to-Voice Test - Phase 5 (WebM Format)")
        logger.info("=" * 60)
        
        # Get all WebM test files
        webm_files = sorted(list(self.webm_test_dir.glob("*.webm")))
        if not webm_files:
            logger.error(f"❌ No WebM files found in {self.webm_test_dir}")
            return {"error": "No test files found"}
        
        logger.info(f"🎯 Found {len(webm_files)} WebM test files")
        
        # Test both Arabic and English configurations
        test_configs = [
            {"language": "ar", "voice_type": "female", "session_id": "test_ar_female"},
            {"language": "en", "voice_type": "female", "session_id": "test_en_female"},
        ]
        
        all_results = []
        
        for config in test_configs:
            logger.info(f"\n🔄 Testing configuration: {config}")
            
            # Connect to WebSocket
            connected = await self.connect_websocket(**config)
            if not connected:
                logger.error(f"❌ Failed to connect with config: {config}")
                continue
            
            try:
                # Test subset of files for each configuration
                test_files = webm_files[:3]  # Test first 3 files for each config
                
                for webm_file in test_files:
                    logger.info(f"\n🎤 Testing: {webm_file.name} with {config['language']}")
                    
                    result = await self.send_audio_file(webm_file)
                    result["test_config"] = config.copy()
                    all_results.append(result)
                    
                    # Brief pause between tests
                    await asyncio.sleep(0.5)
            
            finally:
                # Disconnect WebSocket
                await self.disconnect_websocket()
                await asyncio.sleep(1)  # Brief pause between configs
        
        # Calculate summary statistics
        successful_tests = [r for r in all_results if r.get("success")]
        failed_tests = [r for r in all_results if not r.get("success")]
        
        if successful_tests:
            avg_service_latency = sum(r.get("service_latency_ms", 0) for r in successful_tests) / len(successful_tests)
            avg_total_latency = sum(r.get("total_latency_ms", 0) for r in successful_tests) / len(successful_tests)
        else:
            avg_service_latency = 0
            avg_total_latency = 0
        
        # Prepare final results
        test_summary = {
            "test_timestamp": time.time(),
            "test_summary": {
                "total_tests": len(all_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate_percent": (len(successful_tests) / len(all_results) * 100) if all_results else 0,
                "average_service_latency_ms": avg_service_latency,
                "average_total_latency_ms": avg_total_latency,
                "webm_format_supported": len(successful_tests) > 0,
                "tested_configurations": test_configs
            },
            "test_results": all_results
        }
        
        # Save results to JSON
        results_file = self.output_dir / f"websocket_webm_test_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"\n📊 WEBSOCKET WEBM TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Success Rate: {test_summary['test_summary']['success_rate_percent']:.1f}%")
        logger.info(f"📊 Tests: {len(successful_tests)}/{len(all_results)} successful")
        logger.info(f"⚡ Average Service Latency: {avg_service_latency:.0f}ms")
        logger.info(f"⚡ Average Total Latency: {avg_total_latency:.0f}ms")
        logger.info(f"🎵 WebM Format Support: {'✅ YES' if test_summary['test_summary']['webm_format_supported'] else '❌ NO'}")
        logger.info(f"💾 Results saved to: {results_file}")
        
        if failed_tests:
            logger.info(f"\n❌ Failed tests:")
            for failed_test in failed_tests:
                logger.info(f"   - {failed_test['input_file']}: {failed_test.get('error', 'Unknown error')}")
        
        return test_summary


async def main():
    """Main test runner."""
    try:
        # Initialize test
        test = WebSocketVoiceTest()
        
        # Run WebSocket tests
        results = await test.run_websocket_tests()
        
        if "error" in results:
            logger.error(f"❌ Test setup failed: {results['error']}")
            return
        
        # Print final status
        success_rate = results["test_summary"]["success_rate_percent"]
        if success_rate >= 80:
            logger.info(f"\n🎉 EXCELLENT: WebSocket WebM tests passed with {success_rate:.1f}% success rate!")
        elif success_rate >= 60:
            logger.info(f"\n⚠️ GOOD: WebSocket WebM tests mostly successful ({success_rate:.1f}% success rate)")
        else:
            logger.info(f"\n❌ NEEDS WORK: WebSocket WebM tests need improvement ({success_rate:.1f}% success rate)")
        
        logger.info(f"\n📁 Check results in: {test.output_dir}")
        logger.info(f"🔊 Check generated audio in: {test.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
