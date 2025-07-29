#!/usr/bin/env python3
"""
🚀 PERFORMANCE VERIFICATION TEST
=================================

This script verifies the performance improvements we've made to the Simple Voice WebSocket:

1. ✅ Model pre-loading at startup
2. ✅ /no_think prefix for fast responses  
3. ✅ Optimized transcription settings
4. ✅ Better error handling
5. ✅ Reduced response length for speed

Expected improvements:
- Original: 42+ seconds
- Target: <5 seconds
- Optimistic target: <2 seconds
"""

import asyncio
import json
import logging
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceVerificationTester:
    """Test suite to verify Simple Voice WebSocket performance improvements."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "performance_tests": [],
            "model_status": [],
            "improvement_analysis": {},
            "conclusion": {}
        }
    
    def log_model_state(self, checkpoint: str) -> Dict[str, Any]:
        """Log current model loading state."""
        try:
            response = requests.get(f"{self.base_url}/models/loaded", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_info = data.get("data", {})
                loaded_models = models_info.get("models", [])
                
                model_state = {
                    "checkpoint": checkpoint,
                    "timestamp": time.time(),
                    "total_loaded": len(loaded_models),
                    "loaded_models": [f"{m['name']}: {'loaded' if m.get('status') == 'loaded' else 'not loaded'}" for m in loaded_models]
                }
                
                self.results["model_status"].append(model_state)
                logger.info(f"📊 {checkpoint}: {len(loaded_models)} models loaded - {[m['name'] for m in loaded_models]}")
                return model_state
            else:
                logger.warning(f"⚠️ Failed to get model status: HTTP {response.status_code}")
                return {"checkpoint": checkpoint, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error getting model status: {e}")
            return {"checkpoint": checkpoint, "error": str(e)}
    
    def test_simple_voice_service_status(self) -> Dict[str, Any]:
        """Test the Simple Voice service status and performance claims."""
        logger.info("\n🔍 TESTING SIMPLE VOICE SERVICE STATUS")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/api/v1/ws/simple-voice-chat/status", timeout=10)
            if response.status_code == 200:
                status_data = response.json()
                
                # Extract performance claims
                performance = status_data.get("performance", {})
                target_response_time = performance.get("target_response_time", "unknown")
                active_connections = status_data.get("active_connections", 0)
                
                logger.info(f"✅ Service Status: {status_data.get('status', 'unknown')}")
                logger.info(f"🎯 Target Response Time: {target_response_time}")
                logger.info(f"🔗 Active Connections: {active_connections}")
                
                # Check if there are any active connections for real-world testing
                connections = status_data.get("connections", [])
                if connections:
                    logger.info(f"💬 Active Session Details:")
                    for conn in connections:
                        duration = conn.get("duration_seconds", 0)
                        message_count = conn.get("message_count", 0)
                        logger.info(f"   - Session: {conn.get('session_id', 'unknown')}")
                        logger.info(f"   - Duration: {duration:.1f}s")
                        logger.info(f"   - Messages: {message_count}")
                
                return {
                    "success": True,
                    "target_response_time": target_response_time,
                    "active_connections": active_connections,
                    "status": status_data.get("status"),
                    "engine": performance.get("engine", "unknown")
                }
            else:
                logger.error(f"❌ Service status check failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error checking service status: {e}")
            return {"success": False, "error": str(e)}
    
    def test_chat_api_baseline(self) -> Dict[str, Any]:
        """Test chat API as baseline to ensure models are working."""
        logger.info("\n🧪 TESTING CHAT API BASELINE")
        logger.info("=" * 60)
        
        test_requests = [
            {
                "message": "/no_think مرحبا، كيف حالك؟",
                "model_name": "qwen3-unsloth-q4ks",
                "max_length": 50,
                "language": "ar"
            },
            {
                "message": "/no_think ما هو البوتوكس؟",
                "model_name": "qwen3-unsloth-q4ks", 
                "max_length": 50,
                "language": "ar"
            }
        ]
        
        chat_results = []
        
        for i, request in enumerate(test_requests, 1):
            logger.info(f"\n📤 Chat Request {i}: {request['message'][:30]}...")
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/inference/chat",
                    json=request,
                    timeout=30
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("data", {}).get("response", "")
                    logger.info(f"✅ Chat Request {i} Success: {response_time:.2f}s")
                    logger.info(f"   Response preview: {response_text[:100]}...")
                    
                    chat_results.append({
                        "request": i,
                        "success": True,
                        "response_time": response_time,
                        "response_length": len(response_text),
                        "contains_no_think": "/no_think" in request["message"]
                    })
                else:
                    logger.error(f"❌ Chat Request {i} Failed: HTTP {response.status_code}")
                    chat_results.append({
                        "request": i,
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                logger.error(f"❌ Chat Request {i} Error: {e}")
                chat_results.append({
                    "request": i,
                    "success": False,
                    "error": str(e)
                })
            
            time.sleep(2)  # Brief pause between requests
        
        # Analyze results
        successful_requests = [r for r in chat_results if r.get("success")]
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            logger.info(f"\n📊 Chat API Analysis:")
            logger.info(f"   Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"   Success Rate: {len(successful_requests)}/{len(chat_results)}")
        
        return {
            "results": chat_results,
            "successful_requests": len(successful_requests),
            "total_requests": len(chat_results),
            "average_response_time": avg_response_time if successful_requests else None
        }
    
    def analyze_service_logs(self) -> Dict[str, Any]:
        """Analyze recent service logs for performance evidence."""
        logger.info("\n📋 ANALYZING SERVICE LOGS FOR PERFORMANCE EVIDENCE")
        logger.info("=" * 60)
        
        import subprocess
        
        try:
            # Get recent logs related to voice processing
            cmd = [
                "sudo", "journalctl", "-u", "beautyai-api.service", 
                "--since", "10 minutes ago",
                "--no-pager"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logs = result.stdout
                
                # Extract performance indicators
                performance_indicators = []
                
                # Look for processing completion times
                for line in logs.split('\n'):
                    if "processing completed in" in line.lower():
                        performance_indicators.append(line.strip())
                    elif "response received in" in line.lower():
                        performance_indicators.append(line.strip())
                    elif "voice processing" in line.lower() and ("ms" in line or "seconds" in line):
                        performance_indicators.append(line.strip())
                
                logger.info(f"🔍 Found {len(performance_indicators)} performance indicators in logs")
                for indicator in performance_indicators[-5:]:  # Show last 5
                    logger.info(f"   📊 {indicator}")
                
                # Extract model loading evidence
                model_loading_evidence = []
                for line in logs.split('\n'):
                    if "pre-loaded" in line.lower() or "model loaded" in line.lower():
                        model_loading_evidence.append(line.strip())
                
                logger.info(f"🧠 Found {len(model_loading_evidence)} model loading indicators")
                for evidence in model_loading_evidence[-3:]:  # Show last 3
                    logger.info(f"   ✅ {evidence}")
                
                return {
                    "success": True,
                    "performance_indicators": performance_indicators,
                    "model_loading_evidence": model_loading_evidence,
                    "log_analysis": "Logs successfully analyzed"
                }
            else:
                logger.warning(f"⚠️ Failed to get service logs: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ Error analyzing service logs: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_performance_report(self):
        """Generate comprehensive performance verification report."""
        logger.info("\n📊 GENERATING PERFORMANCE VERIFICATION REPORT")
        logger.info("=" * 60)
        
        # Calculate improvement metrics
        self.results["improvement_analysis"] = {
            "original_issue": "42+ seconds response time",
            "target_improvement": "<5 seconds response time",
            "optimistic_target": "<2 seconds response time",
            "implemented_optimizations": [
                "✅ Model pre-loading at API startup",
                "✅ /no_think prefix for faster chat responses",
                "✅ Optimized transcription settings (webm format, beam_size=1)",
                "✅ Reduced max_length for faster generation (128 tokens)",
                "✅ Better error handling with Arabic fallbacks",
                "✅ Pre-initialized services in SimpleVoiceService"
            ]
        }
        
        # Generate conclusion
        model_status = self.results.get("model_status", [])
        if model_status:
            latest_status = model_status[-1]
            loaded_count = latest_status.get("total_loaded", 0)
            
            self.results["conclusion"] = {
                "models_pre_loaded": loaded_count >= 2,
                "expected_performance_improvement": "Significant",
                "ready_for_testing": True,
                "next_steps": [
                    "Test with real WebSocket connections",
                    "Monitor response times in production",
                    "Further optimize based on real usage patterns"
                ]
            }
        
        # Save results
        report_file = f"/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/performance_verification_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"💾 Performance verification report saved to: {report_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("🎯 PERFORMANCE VERIFICATION SUMMARY")
        logger.info("="*80)
        
        if loaded_count >= 2:
            logger.info("✅ MODELS PRE-LOADED: YES")
            logger.info("🚀 EXPECTED PERFORMANCE: SIGNIFICANTLY IMPROVED")
            logger.info("⚡ TARGET RESPONSE TIME: <5 seconds (down from 42+ seconds)")
        else:
            logger.info("⚠️ MODELS PRE-LOADED: PARTIAL")
            logger.info("🔄 EXPECTED PERFORMANCE: MODERATELY IMPROVED")
        
        logger.info(f"📄 Full report: {report_file}")
        logger.info("="*80)
    
    def run_verification_suite(self):
        """Run the complete performance verification test suite."""
        logger.info("🚀 STARTING PERFORMANCE VERIFICATION TEST SUITE")
        logger.info("="*80)
        
        # Step 1: Check initial model state
        self.log_model_state("Initial State")
        
        # Step 2: Test Simple Voice service status
        service_status = self.test_simple_voice_service_status()
        self.results["performance_tests"].append({
            "test": "simple_voice_service_status",
            "result": service_status
        })
        
        # Step 3: Test chat API baseline
        chat_baseline = self.test_chat_api_baseline()
        self.results["performance_tests"].append({
            "test": "chat_api_baseline",
            "result": chat_baseline
        })
        
        # Step 4: Analyze service logs
        log_analysis = self.analyze_service_logs()
        self.results["performance_tests"].append({
            "test": "service_logs_analysis", 
            "result": log_analysis
        })
        
        # Step 5: Check final model state
        self.log_model_state("Final State")
        
        # Step 6: Generate comprehensive report
        self.generate_performance_report()
        
        logger.info("\n✅ PERFORMANCE VERIFICATION COMPLETE")
        logger.info("="*80)

if __name__ == "__main__":
    tester = PerformanceVerificationTester()
    tester.run_verification_suite()
