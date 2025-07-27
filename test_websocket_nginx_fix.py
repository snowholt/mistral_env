#!/usr/bin/env python3
"""
WebSocket Nginx Configuration Test Script

This script tests the fixed nginx configuration to ensure WebSocket routing
works correctly for the SimpleVoiceService endpoints.

Author: BeautyAI Framework
Date: 2025-01-27
"""

import asyncio
import websockets
import json
import sys
from pathlib import Path
import time

class WebSocketNginxTester:
    """Test WebSocket connections through nginx proxy."""
    
    def __init__(self, base_url: str = "api.gmai.sa"):
        self.base_url = base_url
        self.results = []
    
    async def test_websocket_endpoint(
        self, 
        endpoint_path: str, 
        description: str,
        expected_status: str = "success"
    ) -> dict:
        """Test a specific WebSocket endpoint."""
        
        # Construct full WebSocket URL
        ws_url = f"wss://{self.base_url}{endpoint_path}"
        
        print(f"\n🔍 Testing: {description}")
        print(f"   URL: {ws_url}")
        
        test_result = {
            "endpoint": endpoint_path,
            "description": description,
            "url": ws_url,
            "status": "failed",
            "error": None,
            "response_time": None
        }
        
        start_time = time.time()
        
        try:
            # Try to connect 
            async with websockets.connect(ws_url) as websocket:
                
                # Send a ping message to test basic connectivity
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(ping_message))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    
                    test_result["status"] = "success"
                    test_result["response"] = response_data
                    test_result["response_time"] = time.time() - start_time
                    
                    print(f"   ✅ SUCCESS: Connected and received response")
                    print(f"   📈 Response time: {test_result['response_time']:.2f}s")
                    
                except asyncio.TimeoutError:
                    test_result["status"] = "timeout"
                    test_result["error"] = "No response received within 5 seconds"
                    print(f"   ⚠️  TIMEOUT: Connected but no response")
                
        except websockets.exceptions.ConnectionClosedError as e:
            test_result["error"] = f"Connection closed: {str(e)}"
            print(f"   ❌ FAILED: {test_result['error']}")
            
        except OSError as e:
            test_result["error"] = f"Network error: {str(e)}"
            print(f"   ❌ FAILED: {test_result['error']}")
            
        except websockets.exceptions.InvalidURI as e:
            test_result["error"] = f"Invalid URI: {str(e)}"
            print(f"   ❌ FAILED: {test_result['error']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"   ❌ FAILED: {test_result['error']}")
        
        self.results.append(test_result)
        return test_result
    
    async def test_http_endpoint(self, endpoint_path: str, description: str) -> dict:
        """Test HTTP endpoint accessibility."""
        
        import aiohttp
        
        url = f"https://{self.base_url}{endpoint_path}"
        
        print(f"\n🌐 Testing HTTP: {description}")
        print(f"   URL: {url}")
        
        test_result = {
            "endpoint": endpoint_path,
            "description": description,
            "url": url,
            "status": "failed",
            "error": None,
            "response_time": None
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    test_result["status_code"] = response.status
                    test_result["response_time"] = time.time() - start_time
                    
                    if response.status == 200:
                        test_result["status"] = "success"
                        response_text = await response.text()
                        test_result["response"] = response_text[:200]  # First 200 chars
                        print(f"   ✅ SUCCESS: HTTP {response.status}")
                    else:
                        test_result["status"] = "http_error"
                        test_result["error"] = f"HTTP {response.status}"
                        print(f"   ⚠️  HTTP ERROR: {response.status}")
                        
        except Exception as e:
            test_result["error"] = str(e)
            print(f"   ❌ FAILED: {test_result['error']}")
        
        return test_result
    
    async def run_comprehensive_test(self):
        """Run comprehensive WebSocket and HTTP tests."""
        
        print("🚀 Starting Nginx WebSocket Configuration Tests")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            # Main SimpleVoice WebSocket endpoint
            {
                "path": "/api/v1/ws/simple-voice-chat?language=ar&voice_type=female",
                "description": "SimpleVoice WebSocket (Primary Path)",
                "type": "websocket"
            },
            
            # Legacy WebSocket endpoint  
            {
                "path": "/ws/simple-voice-chat?language=ar&voice_type=female",
                "description": "SimpleVoice WebSocket (Legacy Path)",
                "type": "websocket"
            },
            
            # HTTP Status endpoints
            {
                "path": "/api/v1/ws/simple-voice-chat/status",
                "description": "SimpleVoice Status Endpoint",
                "type": "http"
            }
        ]
        
        # Run all tests
        for scenario in test_scenarios:
            if scenario["type"] == "websocket":
                await self.test_websocket_endpoint(
                    scenario["path"], 
                    scenario["description"]
                )
            elif scenario["type"] == "http":
                await self.test_http_endpoint(
                    scenario["path"],
                    scenario["description"]
                )
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        success_count = len([r for r in self.results if r["status"] == "success"])
        total_count = len(self.results)
        
        print(f"✅ Successful: {success_count}/{total_count}")
        print(f"❌ Failed: {total_count - success_count}/{total_count}")
        
        # Detailed results
        for result in self.results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"\n{status_icon} {result['description']}")
            print(f"   Path: {result['endpoint']}")
            print(f"   Status: {result['status']}")
            
            if result.get("response_time"):
                print(f"   Response Time: {result['response_time']:.2f}s")
            
            if result.get("error"):
                print(f"   Error: {result['error']}")
        
        # Recommendations
        print("\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)
        
        if success_count == total_count:
            print("🎉 All tests passed! Nginx configuration is working correctly.")
        else:
            print("🔧 Some tests failed. Check the following:")
            
            failed_results = [r for r in self.results if r["status"] != "success"]
            for result in failed_results:
                if "404" in str(result.get("error", "")):
                    print(f"   • Path not found: {result['endpoint']}")
                    print("     → Check FastAPI router registration")
                elif "timeout" in result["status"]:
                    print(f"   • Timeout on: {result['endpoint']}")
                    print("     → Check backend service status")
                elif "connection" in str(result.get("error", "")).lower():
                    print(f"   • Connection issue: {result['endpoint']}")
                    print("     → Check nginx WebSocket proxy configuration")


async def main():
    """Main test function."""
    
    print("🧪 WebSocket Nginx Configuration Tester")
    print("Testing the fixed nginx configuration for BeautyAI WebSocket endpoints")
    
    # Initialize tester
    tester = WebSocketNginxTester()
    
    # Add aiohttp for HTTP testing
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp for HTTP testing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp"], check=True)
        import aiohttp
    
    # Run comprehensive tests
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())
