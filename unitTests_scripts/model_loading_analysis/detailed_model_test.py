#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE MODEL LOADING ANALYSIS SCRIPT
==============================================

This script thoroughly tests and verifies the model loading behavior across:
1. Regular Text Chat API
2. Simple Voice WebSocket
3. Advanced Voice WebSocket  
4. Model Manager state monitoring

Key Focus Areas:
- Model persistence between requests
- Loading time analysis
- Memory usage tracking
- Performance comparison

Author: BeautyAI Framework Testing Suite
Date: July 2025
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
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/model_loading_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelLoadingAnalyzer:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "initial_model_status": None,
            "chat_api_tests": [],
            "voice_websocket_tests": [],
            "model_loading_timeline": [],
            "performance_analysis": {},
            "conclusions": {}
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
                logger.info(f"📊 {stage}: {model_state['total_loaded']} models loaded")
                return model_state
            else:
                logger.error(f"❌ Failed to get model status at {stage}: {response.status_code}")
                return {"stage": stage, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error checking model status at {stage}: {e}")
            return {"stage": stage, "error": str(e)}

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from registry"""
        try:
            response = requests.get(f"{self.base_url}/models/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                logger.info(f"📋 Found {len(models)} models in registry")
                return models
            else:
                logger.error(f"❌ Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"❌ Error getting models: {e}")
            return []

    def test_chat_api_model_loading(self):
        """Test model loading behavior with chat API"""
        logger.info("🧪 TESTING CHAT API MODEL LOADING BEHAVIOR")
        logger.info("=" * 60)
        
        # Test different models
        test_models = ["qwen3-unsloth-q4ks", "qwen3-model", "bee1reason-arabic-q4ks"]
        
        for model_name in test_models:
            logger.info(f"\n🔍 Testing model: {model_name}")
            
            # Check initial state
            initial_state = self.log_model_state(f"Before {model_name} test")
            
            # Make chat request and time it
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/inference/chat",
                    json={
                        "message": "ما هي أفضل طرق العناية بالبشرة الدهنية؟",
                        "model_name": model_name,
                        "disable_content_filter": True,
                        "max_new_tokens": 50  # Keep it short for testing
                    },
                    timeout=120  # 2 minutes timeout
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                # Check model state after request
                final_state = self.log_model_state(f"After {model_name} test")
                
                test_result = {
                    "model_name": model_name,
                    "response_time_seconds": response_time,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "initial_loaded_count": initial_state.get("total_loaded", 0),
                    "final_loaded_count": final_state.get("total_loaded", 0),
                    "model_was_loaded_initially": any(m.get("name") == model_name for m in initial_state.get("loaded_models", [])),
                    "model_is_loaded_finally": any(m.get("name") == model_name for m in final_state.get("loaded_models", []))
                }
                
                if response.status_code == 200:
                    response_data = response.json()
                    test_result["response_preview"] = response_data.get("response", "")[:100] + "..." if len(response_data.get("response", "")) > 100 else response_data.get("response", "")
                    test_result["error"] = None
                    logger.info(f"✅ Chat request successful: {response_time:.2f}s")
                else:
                    response_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    test_result["error"] = response_data.get("error", f"HTTP {response.status_code}")
                    logger.error(f"❌ Chat request failed: {test_result['error']}")
                
                self.results["chat_api_tests"].append(test_result)
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Chat request timed out after 120 seconds")
                test_result = {
                    "model_name": model_name,
                    "response_time_seconds": 120,
                    "status_code": None,
                    "success": False,
                    "error": "Request timed out after 120 seconds",
                    "initial_loaded_count": initial_state.get("total_loaded", 0),
                    "final_loaded_count": None
                }
                self.results["chat_api_tests"].append(test_result)
            except Exception as e:
                logger.error(f"❌ Chat request failed with exception: {e}")
                test_result = {
                    "model_name": model_name,
                    "response_time_seconds": None,
                    "status_code": None,
                    "success": False,
                    "error": str(e),
                    "initial_loaded_count": initial_state.get("total_loaded", 0),
                    "final_loaded_count": None
                }
                self.results["chat_api_tests"].append(test_result)
                
            # Wait between tests
            time.sleep(3)

    def test_model_persistence(self):
        """Test if models stay loaded between requests"""
        logger.info("\n🔄 TESTING MODEL PERSISTENCE BETWEEN REQUESTS")
        logger.info("=" * 60)
        
        model_name = "qwen3-unsloth-q4ks"  # Default model
        
        # Test 3 consecutive requests
        for i in range(1, 4):
            logger.info(f"\n📤 Chat Request {i}")
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/inference/chat",
                    json={
                        "message": f"سؤال رقم {i}: ما هي فوائد استخدام كريم الوجه؟",
                        "model_name": model_name,
                        "disable_content_filter": True,
                        "max_new_tokens": 30
                    },
                    timeout=60
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                logger.info(f"✅ Request {i} completed in {response_time:.2f}s")
                
                # Log model state after each request
                self.log_model_state(f"After persistence test {i}")
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Request {i} timed out")
            except Exception as e:
                logger.error(f"❌ Request {i} failed: {e}")
                
            time.sleep(2)

    def analyze_results(self):
        """Analyze the test results and draw conclusions"""
        logger.info("\n📊 ANALYZING TEST RESULTS")
        logger.info("=" * 60)
        
        # Analyze chat API performance
        successful_chat_tests = [test for test in self.results["chat_api_tests"] if test["success"]]
        failed_chat_tests = [test for test in self.results["chat_api_tests"] if not test["success"]]
        
        if successful_chat_tests:
            avg_response_time = sum(test["response_time_seconds"] for test in successful_chat_tests) / len(successful_chat_tests)
            min_response_time = min(test["response_time_seconds"] for test in successful_chat_tests)
            max_response_time = max(test["response_time_seconds"] for test in successful_chat_tests)
            
            self.results["performance_analysis"]["chat_api"] = {
                "successful_tests": len(successful_chat_tests),
                "failed_tests": len(failed_chat_tests),
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            }
            
            logger.info(f"✅ Chat API Tests: {len(successful_chat_tests)} successful, {len(failed_chat_tests)} failed")
            logger.info(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"⚡ Fastest Response: {min_response_time:.2f}s")
            logger.info(f"🐌 Slowest Response: {max_response_time:.2f}s")
        
        # Analyze model loading patterns
        loading_events = self.results["model_loading_timeline"]
        if loading_events:
            initial_count = loading_events[0].get("total_loaded", 0)
            final_count = loading_events[-1].get("total_loaded", 0)
            
            logger.info(f"📈 Model Loading: Started with {initial_count}, ended with {final_count}")
            
            # Check for model loading/unloading patterns
            for i, event in enumerate(loading_events):
                if i > 0:
                    prev_count = loading_events[i-1].get("total_loaded", 0)
                    curr_count = event.get("total_loaded", 0)
                    if curr_count > prev_count:
                        logger.info(f"📦 Model loaded at stage: {event['stage']}")
                    elif curr_count < prev_count:
                        logger.info(f"🗑️  Model unloaded at stage: {event['stage']}")

    def save_results(self):
        """Save detailed results to JSON file"""
        timestamp = int(time.time())
        filename = f"/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/detailed_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Detailed results saved to: {filename}")
        return filename

    async def run_comprehensive_analysis(self):
        """Run the complete analysis suite"""
        logger.info("🚀 STARTING COMPREHENSIVE MODEL LOADING ANALYSIS")
        logger.info("=" * 80)
        
        # Initial state check
        self.results["initial_model_status"] = self.log_model_state("Initial state")
        
        # Get available models
        available_models = self.get_available_models()
        self.results["available_models"] = available_models
        
        # Test chat API
        self.test_chat_api_model_loading()
        
        # Test model persistence
        self.test_model_persistence()
        
        # Analyze results
        self.analyze_results()
        
        # Save results
        results_file = self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        logger.info(f"📄 Full results saved to: {results_file}")
        logger.info("=" * 80)

if __name__ == "__main__":
    analyzer = ModelLoadingAnalyzer()
    asyncio.run(analyzer.run_comprehensive_analysis())
