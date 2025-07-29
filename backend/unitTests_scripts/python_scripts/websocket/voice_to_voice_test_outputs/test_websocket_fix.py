#!/usr/bin/env python3
"""
Quick test script to verify WebSocket endpoint paths after nginx fix.

This script tests the WebSocket endpoint paths to ensure they're correctly 
configured after the nginx proxy_pass fix.
"""

import asyncio
import json
import logging
from typing import Dict, Any
import websockets
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketEndpointTester:
    """Test WebSocket endpoints for proper routing."""
    
    def __init__(self, base_url: str = "https://api.gmai.sa"):
        self.base_url = base_url
        self.ws_base_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        
    async def test_http_status_endpoints(self) -> Dict[str, Any]:
        """Test HTTP status endpoints first."""
        results = {}
        
        endpoints_to_test = [
            "/api/v1/ws/simple-voice-chat/status",
            "/api/v1/health/voice",
            "/api/v1/voice/endpoints"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Testing HTTP endpoint: {url}")
                
                response = requests.get(url, timeout=10)
                results[endpoint] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_size": len(response.text),
                    "content_type": response.headers.get("content-type", "unknown")
                }
                
                if response.status_code == 200:
                    logger.info(f"✅ {endpoint}: SUCCESS")
                else:
                    logger.warning(f"❌ {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ {endpoint}: ERROR - {e}")
                results[endpoint] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def test_websocket_connection(self, endpoint: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        """Test a WebSocket connection."""
        result = {
            "endpoint": endpoint,
            "success": False,
            "error": None,
            "connection_time": None,
            "close_code": None
        }
        
        # Build URL with parameters
        url = f"{self.ws_base_url}{endpoint}"
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{param_str}"
        
        logger.info(f"Testing WebSocket: {url}")
        
        try:
            import time
            start_time = time.time()
            
            # Test connection
            async with websockets.connect(
                url,
                ping_interval=None,  # Disable ping for quick test
                timeout=10
            ) as websocket:
                connection_time = time.time() - start_time
                logger.info(f"✅ WebSocket connected in {connection_time:.2f}s")
                
                # Wait for welcome message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    welcome_data = json.loads(message)
                    logger.info(f"📨 Received welcome: {welcome_data.get('type', 'unknown')}")
                    
                    result.update({
                        "success": True,
                        "connection_time": connection_time,
                        "welcome_message": welcome_data
                    })
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ No welcome message received within 5 seconds")
                    result.update({
                        "success": True,
                        "connection_time": connection_time,
                        "warning": "No welcome message received"
                    })
                
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"❌ WebSocket connection failed with status: {e.status_code}")
            result.update({
                "error": f"HTTP {e.status_code}",
                "status_code": e.status_code
            })
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            result.update({
                "error": str(e)
            })
        
        return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive WebSocket endpoint tests."""
        logger.info("🧪 Starting comprehensive WebSocket endpoint tests...")
        
        results = {
            "test_time": asyncio.get_event_loop().time(),
            "base_url": self.base_url,
            "ws_base_url": self.ws_base_url,
            "http_endpoints": {},
            "websocket_endpoints": {}
        }
        
        # Test HTTP endpoints first
        logger.info("🔍 Testing HTTP status endpoints...")
        results["http_endpoints"] = await self.test_http_status_endpoints()
        
        # Test WebSocket endpoints
        logger.info("🔌 Testing WebSocket connections...")
        
        websocket_tests = [
            {
                "name": "simple_voice_ar_female",
                "endpoint": "/api/v1/ws/simple-voice-chat",
                "params": {"language": "ar", "voice_type": "female"}
            },
            {
                "name": "simple_voice_en_male", 
                "endpoint": "/api/v1/ws/simple-voice-chat",
                "params": {"language": "en", "voice_type": "male"}
            },
            {
                "name": "legacy_voice_conversation",
                "endpoint": "/ws/voice-conversation",
                "params": {"preset": "qwen_optimized"}
            }
        ]
        
        for test in websocket_tests:
            test_result = await self.test_websocket_connection(
                test["endpoint"], 
                test.get("params")
            )
            results["websocket_endpoints"][test["name"]] = test_result
        
        # Summary
        http_success = sum(1 for r in results["http_endpoints"].values() if r.get("success", False))
        ws_success = sum(1 for r in results["websocket_endpoints"].values() if r.get("success", False))
        
        results["summary"] = {
            "http_endpoints_tested": len(results["http_endpoints"]),
            "http_endpoints_successful": http_success,
            "websocket_endpoints_tested": len(results["websocket_endpoints"]),
            "websocket_endpoints_successful": ws_success,
            "overall_success": http_success > 0 and ws_success > 0
        }
        
        logger.info(f"📊 Test Summary:")
        logger.info(f"   HTTP Endpoints: {http_success}/{len(results['http_endpoints'])} successful")
        logger.info(f"   WebSocket Endpoints: {ws_success}/{len(results['websocket_endpoints'])} successful")
        
        return results


async def main():
    """Main test execution."""
    print("🚀 BeautyAI WebSocket Endpoint Tester")
    print("=====================================")
    
    # Test remote API
    tester = WebSocketEndpointTester("https://api.gmai.sa")
    results = await tester.run_comprehensive_test()
    
    # Save results
    results_file = Path(__file__).parent / "websocket_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Print key findings
    print("\n🔍 Key Findings:")
    if results["summary"]["overall_success"]:
        print("✅ WebSocket endpoints are properly configured and accessible!")
    else:
        print("❌ Some endpoints are not working. Check the detailed results.")
        
        # Show specific errors
        for name, result in results["websocket_endpoints"].items():
            if not result.get("success"):
                print(f"   • {name}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
