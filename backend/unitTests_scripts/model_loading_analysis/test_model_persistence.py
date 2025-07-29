#!/usr/bin/env python3
"""
Comprehensive Model Loading Analysis Test Suite

This script tests and verifies model loading behavior across different API endpoints:
1. Regular Chat API (text-based)
2. Simple Voice WebSocket 
3. Advanced Voice WebSocket
4. Model Manager singleton behavior

Author: BeautyAI Framework
Date: 2025-07-27
"""

import asyncio
import aiohttp
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import websockets
import base64
import wave
import tempfile

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/model_persistence_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelLoadingAnalyzer:
    """Comprehensive model loading behavior analyzer."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.session = None
        self.results = {
            "test_timestamp": time.time(),
            "base_url": base_url,
            "tests": {}
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_api_health(self) -> bool:
        """Check if the API is running and accessible."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ API Health Check: {data}")
                    return True
                else:
                    logger.error(f"❌ API Health Check Failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ API Health Check Error: {e}")
            return False
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model loading status from API."""
        try:
            async with self.session.get(f"{self.base_url}/models/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"📊 Model Status: {json.dumps(data, indent=2)}")
                    return data
                else:
                    logger.error(f"❌ Failed to get model status: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"❌ Error getting model status: {e}")
            return {}
    
    async def test_chat_api_model_persistence(self) -> Dict[str, Any]:
        """
        Test if models stay loaded between regular chat API requests.
        
        This tests the hypothesis that text chat API keeps models loaded
        while WebSocket services reload them each time.
        """
        logger.info("\n" + "="*60)
        logger.info("🧪 TESTING CHAT API MODEL PERSISTENCE")
        logger.info("="*60)
        
        test_results = {
            "test_name": "chat_api_model_persistence",
            "requests": [],
            "model_status_snapshots": [],
            "analysis": {}
        }
        
        # Initial model status check
        initial_status = await self.get_model_status()
        test_results["model_status_snapshots"].append({
            "timestamp": time.time(),
            "checkpoint": "initial",
            "status": initial_status
        })
        
        # Prepare chat requests
        chat_requests = [
            {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "مرحبا، كيف حالك؟",
                "max_new_tokens": 50,
                "temperature": 0.7
            },
            {
                "model_name": "qwen3-unsloth-q4ks", 
                "message": "ما هو الذكاء الاصطناعي؟",
                "max_new_tokens": 50,
                "temperature": 0.7
            },
            {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "Tell me about machine learning in English",
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        ]
        
        # Execute chat requests and monitor model loading
        for i, request_data in enumerate(chat_requests):
            logger.info(f"\n📤 Chat Request {i+1}: {request_data['message'][:50]}...")
            
            start_time = time.time()
            
            try:
                async with self.session.post(
                    f"{self.base_url}/inference/chat",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info(f"✅ Chat Request {i+1} Success: {response_time:.2f}s")
                        logger.info(f"   Response: {response_data.get('response', '')[:100]}...")
                        
                        test_results["requests"].append({
                            "request_number": i+1,
                            "request_data": request_data,
                            "response_time": response_time,
                            "success": True,
                            "response_length": len(response_data.get('response', '')),
                            "tokens_generated": response_data.get('tokens_generated', 0),
                            "generation_time_ms": response_data.get('generation_time_ms', 0)
                        })
                        
                    else:
                        logger.error(f"❌ Chat Request {i+1} Failed: {response.status}")
                        error_text = await response.text()
                        test_results["requests"].append({
                            "request_number": i+1,
                            "request_data": request_data,
                            "response_time": response_time,
                            "success": False,
                            "error": error_text
                        })
            
            except Exception as e:
                logger.error(f"❌ Chat Request {i+1} Exception: {e}")
                test_results["requests"].append({
                    "request_number": i+1,
                    "request_data": request_data,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                })
            
            # Check model status after each request
            status_after = await self.get_model_status()
            test_results["model_status_snapshots"].append({
                "timestamp": time.time(),
                "checkpoint": f"after_request_{i+1}",
                "status": status_after
            })
            
            # Small delay between requests
            await asyncio.sleep(2)
        
        # Final model status check
        final_status = await self.get_model_status()
        test_results["model_status_snapshots"].append({
            "timestamp": time.time(),
            "checkpoint": "final",
            "status": final_status
        })
        
        # Analyze results
        successful_requests = [r for r in test_results["requests"] if r["success"]]
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            first_request_time = successful_requests[0]["response_time"]
            subsequent_times = [r["response_time"] for r in successful_requests[1:]]
            avg_subsequent_time = sum(subsequent_times) / len(subsequent_times) if subsequent_times else 0
            
            test_results["analysis"] = {
                "total_requests": len(chat_requests),
                "successful_requests": len(successful_requests),
                "success_rate": len(successful_requests) / len(chat_requests) * 100,
                "first_request_time": first_request_time,
                "avg_subsequent_time": avg_subsequent_time,
                "avg_response_time": avg_response_time,
                "performance_improvement": first_request_time - avg_subsequent_time if avg_subsequent_time > 0 else 0,
                "model_persistency_evidence": avg_subsequent_time < first_request_time * 0.8 if avg_subsequent_time > 0 else False
            }
            
            logger.info(f"\n📊 CHAT API ANALYSIS:")
            logger.info(f"   First Request Time: {first_request_time:.2f}s")
            logger.info(f"   Avg Subsequent Time: {avg_subsequent_time:.2f}s")
            logger.info(f"   Performance Improvement: {test_results['analysis']['performance_improvement']:.2f}s")
            logger.info(f"   Model Persistence Evidence: {test_results['analysis']['model_persistency_evidence']}")
        
        return test_results
    
    async def create_test_audio(self) -> bytes:
        """Create a simple test audio file in memory."""
        # Create a simple 1-second sine wave audio
        import numpy as np
        
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Read the file back as bytes
            temp_file.seek(0)
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return audio_bytes
    
    async def test_simple_voice_websocket_loading(self) -> Dict[str, Any]:
        """
        Test model loading behavior with Simple Voice WebSocket.
        
        This tests if each WebSocket connection triggers fresh model loading.
        """
        logger.info("\n" + "="*60)
        logger.info("🧪 TESTING SIMPLE VOICE WEBSOCKET MODEL LOADING")
        logger.info("="*60)
        
        test_results = {
            "test_name": "simple_voice_websocket_loading",
            "connections": [],
            "model_status_snapshots": [],
            "analysis": {}
        }
        
        # Initial model status
        initial_status = await self.get_model_status()
        test_results["model_status_snapshots"].append({
            "timestamp": time.time(),
            "checkpoint": "initial",
            "status": initial_status
        })
        
        # Create test audio
        test_audio = await self.create_test_audio()
        
        # Test multiple WebSocket connections
        for connection_num in range(1, 4):  # Test 3 connections
            logger.info(f"\n🔌 Simple Voice WebSocket Connection {connection_num}")
            
            connection_start = time.time()
            
            try:
                # WebSocket URL for simple voice
                ws_url = f"{self.ws_url}/ws/simple-voice-chat?language=ar&voice_type=female"
                
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"✅ WebSocket connected: {ws_url}")
                    
                    # Wait for connection establishment message
                    try:
                        initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        logger.info(f"📨 Initial message: {initial_message}")
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ No initial message received")
                    
                    # Send test audio
                    audio_send_time = time.time()
                    await websocket.send(test_audio)
                    logger.info(f"📤 Sent test audio ({len(test_audio)} bytes)")
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=60.0)  # 60s timeout for processing
                        response_time = time.time() - audio_send_time
                        
                        try:
                            response_data = json.loads(response)
                            logger.info(f"✅ Response received in {response_time:.2f}s")
                            logger.info(f"   Type: {response_data.get('type', 'unknown')}")
                            logger.info(f"   Success: {response_data.get('success', False)}")
                            
                            connection_result = {
                                "connection_number": connection_num,
                                "connection_time": time.time() - connection_start,
                                "response_time": response_time,
                                "success": True,
                                "response_type": response_data.get('type'),
                                "has_audio": 'audio_base64' in response_data,
                                "transcription": response_data.get('transcription', ''),
                                "response_text": response_data.get('response_text', '')
                            }
                            
                        except json.JSONDecodeError:
                            logger.info(f"✅ Binary response received in {response_time:.2f}s ({len(response)} bytes)")
                            connection_result = {
                                "connection_number": connection_num,
                                "connection_time": time.time() - connection_start,
                                "response_time": response_time,
                                "success": True,
                                "response_type": "binary",
                                "response_size": len(response)
                            }
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Response timeout for connection {connection_num}")
                        connection_result = {
                            "connection_number": connection_num,
                            "connection_time": time.time() - connection_start,
                            "response_time": None,
                            "success": False,
                            "error": "timeout"
                        }
                    
                    test_results["connections"].append(connection_result)
            
            except Exception as e:
                logger.error(f"❌ WebSocket connection {connection_num} failed: {e}")
                test_results["connections"].append({
                    "connection_number": connection_num,
                    "connection_time": time.time() - connection_start,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                })
            
            # Check model status after each connection
            status_after = await self.get_model_status()
            test_results["model_status_snapshots"].append({
                "timestamp": time.time(),
                "checkpoint": f"after_connection_{connection_num}",
                "status": status_after
            })
            
            # Delay between connections
            await asyncio.sleep(5)
        
        # Analyze results
        successful_connections = [c for c in test_results["connections"] if c["success"] and c.get("response_time")]
        if successful_connections:
            response_times = [c["response_time"] for c in successful_connections]
            avg_response_time = sum(response_times) / len(response_times)
            first_response_time = response_times[0]
            
            test_results["analysis"] = {
                "total_connections": len(test_results["connections"]),
                "successful_connections": len(successful_connections),
                "success_rate": len(successful_connections) / len(test_results["connections"]) * 100,
                "response_times": response_times,
                "first_response_time": first_response_time,
                "avg_response_time": avg_response_time,
                "consistent_slow_times": all(t > 30 for t in response_times),  # All responses >30s indicate model loading
                "model_loading_evidence": avg_response_time > 30  # >30s suggests model loading each time
            }
            
            logger.info(f"\n📊 SIMPLE VOICE WEBSOCKET ANALYSIS:")
            logger.info(f"   Response Times: {response_times}")
            logger.info(f"   Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"   Consistent Slow Times: {test_results['analysis']['consistent_slow_times']}")
            logger.info(f"   Model Loading Evidence: {test_results['analysis']['model_loading_evidence']}")
        
        return test_results
    
    async def test_advanced_voice_websocket_loading(self) -> Dict[str, Any]:
        """
        Test model loading behavior with Advanced Voice WebSocket.
        """
        logger.info("\n" + "="*60)
        logger.info("🧪 TESTING ADVANCED VOICE WEBSOCKET MODEL LOADING")
        logger.info("="*60)
        
        test_results = {
            "test_name": "advanced_voice_websocket_loading",
            "connections": [],
            "model_status_snapshots": [],
            "analysis": {}
        }
        
        # Initial model status
        initial_status = await self.get_model_status()
        test_results["model_status_snapshots"].append({
            "timestamp": time.time(),
            "checkpoint": "initial",
            "status": initial_status
        })
        
        # Create test audio
        test_audio = await self.create_test_audio()
        
        # Test multiple WebSocket connections
        for connection_num in range(1, 4):  # Test 3 connections
            logger.info(f"\n🔌 Advanced Voice WebSocket Connection {connection_num}")
            
            connection_start = time.time()
            
            try:
                # WebSocket URL for advanced voice
                ws_url = f"{self.ws_url}/ws/voice-conversation?preset=qwen_optimized&session_id=test_{connection_num}"
                
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"✅ WebSocket connected: {ws_url}")
                    
                    # Wait for connection establishment message
                    try:
                        initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        logger.info(f"📨 Initial message: {initial_message}")
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ No initial message received")
                    
                    # Send test audio
                    audio_send_time = time.time()
                    await websocket.send(test_audio)
                    logger.info(f"📤 Sent test audio ({len(test_audio)} bytes)")
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=60.0)  # 60s timeout for processing
                        response_time = time.time() - audio_send_time
                        
                        try:
                            response_data = json.loads(response)
                            logger.info(f"✅ Response received in {response_time:.2f}s")
                            logger.info(f"   Type: {response_data.get('type', 'unknown')}")
                            logger.info(f"   Success: {response_data.get('success', False)}")
                            
                            connection_result = {
                                "connection_number": connection_num,
                                "connection_time": time.time() - connection_start,
                                "response_time": response_time,
                                "success": True,
                                "response_type": response_data.get('type'),
                                "has_audio": 'audio_base64' in response_data,
                                "transcription": response_data.get('transcription', ''),
                                "response_text": response_data.get('response_text', ''),
                                "models_used": response_data.get('models_used', {})
                            }
                            
                        except json.JSONDecodeError:
                            logger.info(f"✅ Binary response received in {response_time:.2f}s ({len(response)} bytes)")
                            connection_result = {
                                "connection_number": connection_num,
                                "connection_time": time.time() - connection_start,
                                "response_time": response_time,
                                "success": True,
                                "response_type": "binary",
                                "response_size": len(response)
                            }
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Response timeout for connection {connection_num}")
                        connection_result = {
                            "connection_number": connection_num,
                            "connection_time": time.time() - connection_start,
                            "response_time": None,
                            "success": False,
                            "error": "timeout"
                        }
                    
                    test_results["connections"].append(connection_result)
            
            except Exception as e:
                logger.error(f"❌ WebSocket connection {connection_num} failed: {e}")
                test_results["connections"].append({
                    "connection_number": connection_num,
                    "connection_time": time.time() - connection_start,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                })
            
            # Check model status after each connection
            status_after = await self.get_model_status()
            test_results["model_status_snapshots"].append({
                "timestamp": time.time(),
                "checkpoint": f"after_connection_{connection_num}",
                "status": status_after
            })
            
            # Delay between connections
            await asyncio.sleep(5)
        
        # Analyze results
        successful_connections = [c for c in test_results["connections"] if c["success"] and c.get("response_time")]
        if successful_connections:
            response_times = [c["response_time"] for c in successful_connections]
            avg_response_time = sum(response_times) / len(response_times)
            first_response_time = response_times[0]
            
            test_results["analysis"] = {
                "total_connections": len(test_results["connections"]),
                "successful_connections": len(successful_connections),
                "success_rate": len(successful_connections) / len(test_results["connections"]) * 100,
                "response_times": response_times,
                "first_response_time": first_response_time,
                "avg_response_time": avg_response_time,
                "consistent_slow_times": all(t > 30 for t in response_times),  # All responses >30s indicate model loading
                "model_loading_evidence": avg_response_time > 30  # >30s suggests model loading each time
            }
            
            logger.info(f"\n📊 ADVANCED VOICE WEBSOCKET ANALYSIS:")
            logger.info(f"   Response Times: {response_times}")
            logger.info(f"   Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"   Consistent Slow Times: {test_results['analysis']['consistent_slow_times']}")
            logger.info(f"   Model Loading Evidence: {test_results['analysis']['model_loading_evidence']}")
        
        return test_results
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive analysis."""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING COMPREHENSIVE MODEL LOADING ANALYSIS")
        logger.info("="*80)
        
        # Check API health first
        if not await self.check_api_health():
            logger.error("❌ API is not accessible. Please start the BeautyAI API service.")
            return {"error": "API not accessible"}
        
        # Run all tests
        chat_results = await self.test_chat_api_model_persistence()
        self.results["tests"]["chat_api"] = chat_results
        
        simple_ws_results = await self.test_simple_voice_websocket_loading()
        self.results["tests"]["simple_voice_websocket"] = simple_ws_results
        
        advanced_ws_results = await self.test_advanced_voice_websocket_loading()
        self.results["tests"]["advanced_voice_websocket"] = advanced_ws_results
        
        # Generate comprehensive analysis
        await self.generate_final_analysis()
        
        return self.results
    
    async def generate_final_analysis(self):
        """Generate final comprehensive analysis."""
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE ANALYSIS RESULTS")
        logger.info("="*80)
        
        analysis = {
            "summary": {},
            "evidence": {},
            "recommendations": []
        }
        
        # Analyze chat API results
        chat_test = self.results["tests"].get("chat_api", {})
        chat_analysis = chat_test.get("analysis", {})
        
        # Analyze WebSocket results
        simple_ws_test = self.results["tests"].get("simple_voice_websocket", {})
        simple_ws_analysis = simple_ws_test.get("analysis", {})
        
        advanced_ws_test = self.results["tests"].get("advanced_voice_websocket", {})
        advanced_ws_analysis = advanced_ws_test.get("analysis", {})
        
        # Summary
        analysis["summary"] = {
            "chat_api_model_persistence": chat_analysis.get("model_persistency_evidence", False),
            "chat_api_avg_response_time": chat_analysis.get("avg_response_time", 0),
            "simple_ws_model_loading_each_time": simple_ws_analysis.get("model_loading_evidence", False),
            "simple_ws_avg_response_time": simple_ws_analysis.get("avg_response_time", 0),
            "advanced_ws_model_loading_each_time": advanced_ws_analysis.get("model_loading_evidence", False),
            "advanced_ws_avg_response_time": advanced_ws_analysis.get("avg_response_time", 0)
        }
        
        # Evidence
        analysis["evidence"] = {
            "hypothesis_confirmed": (
                simple_ws_analysis.get("model_loading_evidence", False) or 
                advanced_ws_analysis.get("model_loading_evidence", False)
            ) and not chat_analysis.get("model_persistency_evidence", True),
            "chat_api_fast": chat_analysis.get("avg_response_time", 999) < 10,
            "websockets_slow": (
                simple_ws_analysis.get("avg_response_time", 0) > 30 or
                advanced_ws_analysis.get("avg_response_time", 0) > 30
            )
        }
        
        # Recommendations
        if analysis["evidence"]["hypothesis_confirmed"]:
            analysis["recommendations"] = [
                "🎯 **CRITICAL ISSUE CONFIRMED**: WebSocket services load models on each connection",
                "⚡ **SOLUTION**: Pre-load required models at API startup",
                "🏗️ **IMPLEMENTATION**: Modify WebSocket services to use ModelManager singleton properly",
                "📊 **MONITORING**: Add model loading status endpoints for real-time monitoring",
                "🔧 **OPTIMIZATION**: Consider model warm-up strategies for faster initialization"
            ]
        else:
            analysis["recommendations"] = [
                "✅ **GOOD NEWS**: No major model loading issues detected",
                "🔍 **CONTINUE MONITORING**: Keep tracking response times for performance regression",
                "🏗️ **OPTIMIZATION**: Consider further performance improvements"
            ]
        
        self.results["final_analysis"] = analysis
        
        # Print summary
        logger.info(f"\n🎯 **FINAL VERDICT**:")
        logger.info(f"   Hypothesis Confirmed: {analysis['evidence']['hypothesis_confirmed']}")
        logger.info(f"   Chat API Fast: {analysis['evidence']['chat_api_fast']}")
        logger.info(f"   WebSockets Slow: {analysis['evidence']['websockets_slow']}")
        
        logger.info(f"\n📋 **RECOMMENDATIONS**:")
        for rec in analysis["recommendations"]:
            logger.info(f"   {rec}")
    
    def save_results(self, output_file: str = None):
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_file}")
        return output_file


async def main():
    """Main test execution function."""
    try:
        async with ModelLoadingAnalyzer() as analyzer:
            results = await analyzer.run_comprehensive_analysis()
            output_file = analyzer.save_results()
            
            logger.info(f"\n✅ **ANALYSIS COMPLETE**")
            logger.info(f"📄 Full results saved to: {output_file}")
            logger.info(f"📊 Check the log file for detailed analysis")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import websockets
        import numpy as np
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("📦 Please install required packages:")
        logger.info("   pip install websockets numpy")
        sys.exit(1)
    
    asyncio.run(main())
