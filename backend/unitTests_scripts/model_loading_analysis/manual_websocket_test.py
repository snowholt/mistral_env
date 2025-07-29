#!/usr/bin/env python3
"""
🔍 MANUAL WEBSOCKET CONNECTION TEST
==================================

This script uses a simpler approach to test WebSocket connections
and monitor model loading behavior during voice chat operations.
"""

import json
import logging
import time
import sys
import requests
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/manual_websocket_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ManualWebSocketTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        
    def log_model_state(self, stage: str):
        """Log current model loading state"""
        try:
            response = requests.get(f"{self.base_url}/models/loaded", timeout=5)
            if response.status_code == 200:
                data = response.json()
                total_loaded = data.get("data", {}).get("total_loaded", 0)
                loaded_models = [m["name"] for m in data.get("data", {}).get("models", [])]
                logger.info(f"📊 {stage}: {total_loaded} models loaded - {loaded_models}")
                return {"total": total_loaded, "models": loaded_models}
            else:
                logger.error(f"❌ Failed to get model status: {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error checking model status: {e}")
            return {"error": str(e)}

    def test_voice_endpoints_status(self):
        """Test voice endpoint status endpoints to understand model requirements"""
        logger.info("🔍 TESTING VOICE ENDPOINT STATUS")
        logger.info("=" * 60)
        
        endpoints = [
            "/api/v1/ws/simple-voice-chat/status",
            "/api/v1/health/voice", 
            "/api/v1/voice/endpoints"
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"\n📡 Testing: {endpoint}")
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                logger.info(f"   Status: HTTP {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   Response: {json.dumps(data, indent=2)[:500]}...")
                    
                    # Look for model information in the response
                    response_str = json.dumps(data).lower()
                    if "model" in response_str:
                        logger.info("   🎯 CONTAINS MODEL INFO!")
                    if "whisper" in response_str:
                        logger.info("   🎙️  References Whisper (STT)")
                    if "qwen" in response_str:
                        logger.info("   🧠 References Qwen (Chat)")
                    if "tts" in response_str:
                        logger.info("   🔊 References TTS")
                        
                else:
                    logger.warning(f"   Response: {response.text[:200]}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")

    def test_model_loading_before_websocket_simulation(self):
        """Simulate what happens when WebSocket endpoints try to use models"""
        logger.info("\n🧪 SIMULATING WEBSOCKET MODEL REQUIREMENTS")
        logger.info("=" * 60)
        
        # Models typically needed for voice chat
        voice_models = [
            "whisper-large-v3-turbo-arabic",  # For STT
            "qwen3-unsloth-q4ks",             # For Chat (already loaded)
            "coqui-tts-arabic",               # For TTS (Advanced Voice)
            "edge-tts"                        # For TTS (Simple Voice)
        ]
        
        logger.info("📋 Models typically required for voice chat:")
        for model in voice_models:
            logger.info(f"   - {model}")
        
        # Check current state
        initial_state = self.log_model_state("Before WebSocket simulation")
        
        # Test loading each model individually to measure time
        for model_name in voice_models:
            if model_name in initial_state.get("models", []):
                logger.info(f"✅ {model_name} - Already loaded")
                continue
                
            logger.info(f"\n🔄 Testing load time for: {model_name}")
            start_time = time.time()
            
            try:
                response = requests.post(f"{self.base_url}/models/{model_name}/load", timeout=120)
                load_time = time.time() - start_time
                
                if response.status_code == 200:
                    logger.info(f"✅ {model_name} loaded in {load_time:.2f}s")
                    
                    # Check if model is actually loaded
                    after_state = self.log_model_state(f"After loading {model_name}")
                    
                    if model_name not in after_state.get("models", []):
                        logger.warning(f"⚠️  {model_name} not found in loaded models list!")
                        
                else:
                    logger.error(f"❌ {model_name} failed to load: HTTP {response.status_code}")
                    logger.error(f"   Response: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                load_time = time.time() - start_time
                logger.error(f"❌ {model_name} load timeout after {load_time:.2f}s")
            except Exception as e:
                load_time = time.time() - start_time 
                logger.error(f"❌ {model_name} load failed after {load_time:.2f}s: {e}")
            
            # Wait between model loads
            time.sleep(2)

    def test_websocket_using_curl(self):
        """Use curl to test WebSocket connections"""
        logger.info("\n🔧 TESTING WEBSOCKET WITH CURL")
        logger.info("=" * 60)
        
        websocket_urls = [
            "ws://localhost:8000/ws/simple-voice-chat",
            "ws://localhost:8000/ws/voice-conversation"
        ]
        
        for ws_url in websocket_urls:
            logger.info(f"\n📡 Testing WebSocket: {ws_url}")
            
            # Check model state before
            before_state = self.log_model_state(f"Before {ws_url}")
            
            try:
                # Use curl to test WebSocket upgrade
                result = subprocess.run([
                    'curl', '-i', '-N', '-H', 'Connection: Upgrade', 
                    '-H', 'Upgrade: websocket', 
                    '-H', 'Sec-WebSocket-Version: 13',
                    '-H', 'Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==',
                    '-H', 'Origin: http://localhost:8000',
                    ws_url.replace('ws://', 'http://')
                ], capture_output=True, text=True, timeout=10)
                
                logger.info(f"   Return code: {result.returncode}")
                if result.stdout:
                    logger.info(f"   STDOUT: {result.stdout[:300]}")
                if result.stderr:
                    logger.info(f"   STDERR: {result.stderr[:300]}")
                    
                # Check model state after
                after_state = self.log_model_state(f"After {ws_url}")
                
                models_change = after_state.get("total", 0) - before_state.get("total", 0)
                if models_change > 0:
                    logger.info(f"   📦 Models loaded during connection attempt: +{models_change}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"   ❌ Curl timeout after 10s")
            except Exception as e:
                logger.error(f"   ❌ Curl error: {e}")

    def comprehensive_model_analysis(self):
        """Run comprehensive analysis"""
        logger.info("🚀 STARTING MANUAL WEBSOCKET MODEL ANALYSIS")
        logger.info("=" * 80)
        
        # Initial state
        self.log_model_state("Initial state")
        
        # Test voice endpoint status
        self.test_voice_endpoints_status()
        
        # Test model loading simulation
        self.test_model_loading_before_websocket_simulation()
        
        # Test WebSocket with curl
        self.test_websocket_using_curl()
        
        # Final state
        self.log_model_state("Final state")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ MANUAL WEBSOCKET ANALYSIS COMPLETE")
        logger.info("=" * 80)

if __name__ == "__main__":
    tester = ManualWebSocketTester()
    tester.comprehensive_model_analysis()
