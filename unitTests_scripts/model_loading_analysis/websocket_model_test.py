#!/usr/bin/env python3
"""
🔍 WEBSOCKET MODEL LOADING ANALYSIS SCRIPT
===========================================

This script specifically tests the WebSocket endpoints to verify:
1. Simple Voice WebSocket model loading behavior
2. Advanced Voice WebSocket model loading behavior
3. Model persistence between WebSocket connections
4. Comparison with Chat API performance

Focus: Identify if WebSocket connections load models on-demand vs using persistent models
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import requests
import websockets
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/websocket_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WebSocketModelAnalyzer:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_base_url = "ws://localhost:8000"
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "initial_model_status": None,
            "simple_voice_tests": [],
            "advanced_voice_tests": [],
            "model_loading_timeline": [],
            "performance_comparison": {},
            "websocket_endpoints_status": {}
        }
        
    def log_model_state(self, stage: str) -> Dict[str, Any]:
        """Log current model loading state"""
        try:
            response = requests.get(f"{self.base_url}/models/loaded", timeout=5)
            if response.status_code == 200:
                data = response.json()
                model_state = {
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                    "total_loaded": data.get("data", {}).get("total_loaded", 0),
                    "loaded_models": data.get("data", {}).get("models", [])
                }
                self.results["model_loading_timeline"].append(model_state)
                logger.info(f"📊 {stage}: {model_state['total_loaded']} models loaded - {[m['name'] for m in model_state['loaded_models']]}")
                return model_state
            else:
                logger.error(f"❌ Failed to get model status at {stage}: {response.status_code}")
                return {"stage": stage, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error checking model status at {stage}: {e}")
            return {"stage": stage, "error": str(e)}

    def check_websocket_status_endpoints(self):
        """Check WebSocket service status endpoints"""
        logger.info("🔍 CHECKING WEBSOCKET SERVICE STATUS")
        logger.info("=" * 60)
        
        endpoints = [
            "/api/v1/ws/simple-voice-chat/status",
            "/api/v1/voice/endpoints",
            "/api/v1/health/voice"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                logger.info(f"✅ {endpoint}: HTTP {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    self.results["websocket_endpoints_status"][endpoint] = data
                    if "models" in str(data).lower():
                        logger.info(f"   📦 Model info found in response")
                else:
                    logger.warning(f"   ⚠️  Non-200 response: {response.text[:100]}")
            except Exception as e:
                logger.error(f"❌ {endpoint}: {e}")
                self.results["websocket_endpoints_status"][endpoint] = {"error": str(e)}

    async def test_simple_voice_websocket(self):
        """Test Simple Voice WebSocket model loading"""
        logger.info("\n🧪 TESTING SIMPLE VOICE WEBSOCKET")
        logger.info("=" * 60)
        
        ws_url = f"{self.ws_base_url}/ws/simple-voice-chat"
        
        for attempt in range(1, 4):
            logger.info(f"\n🔌 Simple Voice WebSocket Attempt {attempt}")
            
            # Check model state before connection
            before_state = self.log_model_state(f"Before Simple Voice WS {attempt}")
            
            start_time = time.time()
            try:
                # Try different connection approaches
                async with websockets.connect(
                    ws_url, 
                    timeout=10,
                    extra_headers={
                        "Origin": "http://localhost:8000",
                        "User-Agent": "BeautyAI-Test-Client/1.0"
                    }
                ) as websocket:
                    connection_time = time.time() - start_time
                    logger.info(f"✅ WebSocket connected in {connection_time:.2f}s")
                    
                    # Send a test audio message (dummy base64 audio)
                    test_message = {
                        "type": "audio",
                        "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",  # Minimal WAV header
                        "format": "wav"
                    }
                    
                    # Send message and time response
                    message_start = time.time()
                    await websocket.send(json.dumps(test_message))
                    
                    # Wait for response with timeout
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=60)
                        response_time = time.time() - message_start
                        
                        logger.info(f"✅ Response received in {response_time:.2f}s")
                        logger.info(f"   Response preview: {response[:200]}...")
                        
                        # Check model state after response
                        after_state = self.log_model_state(f"After Simple Voice WS {attempt}")
                        
                        test_result = {
                            "attempt": attempt,
                            "connection_time": connection_time,
                            "response_time": response_time,
                            "total_time": response_time + connection_time,
                            "success": True,
                            "models_before": before_state.get("total_loaded", 0),
                            "models_after": after_state.get("total_loaded", 0),
                            "models_loaded_before": [m["name"] for m in before_state.get("loaded_models", [])],
                            "models_loaded_after": [m["name"] for m in after_state.get("loaded_models", [])],
                            "error": None
                        }
                        
                        self.results["simple_voice_tests"].append(test_result)
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ WebSocket response timeout after 60s")
                        after_state = self.log_model_state(f"After Simple Voice WS {attempt} (timeout)")
                        
                        test_result = {
                            "attempt": attempt,
                            "connection_time": connection_time,
                            "response_time": None,
                            "total_time": None,
                            "success": False,
                            "models_before": before_state.get("total_loaded", 0),
                            "models_after": after_state.get("total_loaded", 0),
                            "error": "Response timeout after 60s"
                        }
                        self.results["simple_voice_tests"].append(test_result)
                    
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"❌ WebSocket connection closed: {e}")
                test_result = {
                    "attempt": attempt,
                    "connection_time": None,
                    "response_time": None,
                    "success": False,
                    "error": f"Connection closed: {e}"
                }
                self.results["simple_voice_tests"].append(test_result)
                
            except Exception as e:
                logger.error(f"❌ WebSocket connection failed: {e}")
                test_result = {
                    "attempt": attempt,
                    "connection_time": None,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                }
                self.results["simple_voice_tests"].append(test_result)
            
            # Wait between attempts
            await asyncio.sleep(5)

    async def test_advanced_voice_websocket(self):
        """Test Advanced Voice WebSocket model loading"""
        logger.info("\n🧪 TESTING ADVANCED VOICE WEBSOCKET")
        logger.info("=" * 60)
        
        ws_url = f"{self.ws_base_url}/ws/voice-conversation"
        
        for attempt in range(1, 4):
            logger.info(f"\n🔌 Advanced Voice WebSocket Attempt {attempt}")
            
            # Check model state before connection
            before_state = self.log_model_state(f"Before Advanced Voice WS {attempt}")
            
            start_time = time.time()
            try:
                async with websockets.connect(
                    ws_url,
                    timeout=10,
                    extra_headers={
                        "Origin": "http://localhost:8000",
                        "User-Agent": "BeautyAI-Test-Client/1.0"
                    }
                ) as websocket:
                    connection_time = time.time() - start_time
                    logger.info(f"✅ WebSocket connected in {connection_time:.2f}s")
                    
                    # Send a test audio message
                    test_message = {
                        "type": "audio",
                        "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
                        "format": "wav",
                        "language": "ar"
                    }
                    
                    message_start = time.time()
                    await websocket.send(json.dumps(test_message))
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=60)
                        response_time = time.time() - message_start
                        
                        logger.info(f"✅ Response received in {response_time:.2f}s")
                        logger.info(f"   Response preview: {response[:200]}...")
                        
                        after_state = self.log_model_state(f"After Advanced Voice WS {attempt}")
                        
                        test_result = {
                            "attempt": attempt,
                            "connection_time": connection_time,
                            "response_time": response_time,
                            "total_time": response_time + connection_time,
                            "success": True,
                            "models_before": before_state.get("total_loaded", 0),
                            "models_after": after_state.get("total_loaded", 0),
                            "models_loaded_before": [m["name"] for m in before_state.get("loaded_models", [])],
                            "models_loaded_after": [m["name"] for m in after_state.get("loaded_models", [])],
                            "error": None
                        }
                        
                        self.results["advanced_voice_tests"].append(test_result)
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ WebSocket response timeout after 60s")
                        after_state = self.log_model_state(f"After Advanced Voice WS {attempt} (timeout)")
                        
                        test_result = {
                            "attempt": attempt,
                            "connection_time": connection_time,
                            "response_time": None,
                            "total_time": None,
                            "success": False,
                            "models_before": before_state.get("total_loaded", 0),
                            "models_after": after_state.get("total_loaded", 0),
                            "error": "Response timeout after 60s"
                        }
                        self.results["advanced_voice_tests"].append(test_result)
                    
            except Exception as e:
                logger.error(f"❌ WebSocket connection failed: {e}")
                test_result = {
                    "attempt": attempt,
                    "connection_time": None,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                }
                self.results["advanced_voice_tests"].append(test_result)
            
            await asyncio.sleep(5)

    def analyze_websocket_results(self):
        """Analyze WebSocket test results"""
        logger.info("\n📊 ANALYZING WEBSOCKET RESULTS")
        logger.info("=" * 60)
        
        # Analyze Simple Voice results
        simple_successful = [test for test in self.results["simple_voice_tests"] if test.get("success")]
        simple_failed = [test for test in self.results["simple_voice_tests"] if not test.get("success")]
        
        logger.info(f"🎵 Simple Voice: {len(simple_successful)} successful, {len(simple_failed)} failed")
        
        if simple_successful:
            avg_response_time = sum(test["response_time"] for test in simple_successful if test["response_time"]) / len([test for test in simple_successful if test["response_time"]])
            logger.info(f"   ⏱️  Average Response Time: {avg_response_time:.2f}s")
            
            # Check for model loading patterns
            for test in simple_successful:
                models_added = test["models_after"] - test["models_before"]
                if models_added > 0:
                    logger.info(f"   📦 Models loaded during attempt {test['attempt']}: +{models_added}")
        
        # Analyze Advanced Voice results
        advanced_successful = [test for test in self.results["advanced_voice_tests"] if test.get("success")]
        advanced_failed = [test for test in self.results["advanced_voice_tests"] if not test.get("success")]
        
        logger.info(f"🎭 Advanced Voice: {len(advanced_successful)} successful, {len(advanced_failed)} failed")
        
        if advanced_successful:
            avg_response_time = sum(test["response_time"] for test in advanced_successful if test["response_time"]) / len([test for test in advanced_successful if test["response_time"]])
            logger.info(f"   ⏱️  Average Response Time: {avg_response_time:.2f}s")
            
            for test in advanced_successful:
                models_added = test["models_after"] - test["models_before"]
                if models_added > 0:
                    logger.info(f"   📦 Models loaded during attempt {test['attempt']}: +{models_added}")

    def save_results(self):
        """Save results to JSON file"""
        timestamp = int(time.time())
        filename = f"/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/websocket_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 WebSocket test results saved to: {filename}")
        return filename

    async def run_comprehensive_websocket_analysis(self):
        """Run the complete WebSocket analysis"""
        logger.info("🚀 STARTING WEBSOCKET MODEL LOADING ANALYSIS")
        logger.info("=" * 80)
        
        # Initial state
        self.results["initial_model_status"] = self.log_model_state("Initial WebSocket test state")
        
        # Check WebSocket endpoints status
        self.check_websocket_status_endpoints()
        
        # Test Simple Voice WebSocket
        await self.test_simple_voice_websocket()
        
        # Test Advanced Voice WebSocket  
        await self.test_advanced_voice_websocket()
        
        # Analyze results
        self.analyze_websocket_results() 
        
        # Save results
        results_file = self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ WEBSOCKET ANALYSIS COMPLETE")
        logger.info(f"📄 Results saved to: {results_file}")
        logger.info("=" * 80)

if __name__ == "__main__":
    analyzer = WebSocketModelAnalyzer()
    asyncio.run(analyzer.run_comprehensive_websocket_analysis())
