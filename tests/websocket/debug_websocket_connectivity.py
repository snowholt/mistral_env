#!/usr/bin/env python3
"""
WebSocket Connectivity Debug Tool
Tests multiple WebSocket endpoints to identify connectivity issues
"""

import asyncio
import websockets
import ssl
import socket
import requests
import json
import time
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketConnectivityDebugger:
    """Debug WebSocket connectivity issues across different endpoints."""
    
    def __init__(self):
        self.results = {}
        
    async def test_basic_connection(self, url: str, name: str, timeout: int = 10) -> Dict[str, Any]:
        """Test basic WebSocket connection."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Testing {name}: {url}")
        logger.info(f"{'='*60}")
        
        result = {
            "name": name,
            "url": url,
            "success": False,
            "error": None,
            "response_time_ms": None,
            "connection_established": False,
            "can_send_message": False,
            "ssl_info": None
        }
        
        try:
            start_time = time.time()
            
            # Create SSL context for secure connections
            ssl_context = None
            if url.startswith("wss://"):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                result["ssl_info"] = "SSL context created (no verification)"
            
            # Test connection
            logger.info(f"📡 Attempting connection...")
            
            async with websockets.connect(
                url, 
                ssl=ssl_context,
                timeout=timeout,
                extra_headers={
                    "User-Agent": "BeautyAI-WebSocket-Debug/1.0",
                    "Origin": "https://dev.gmai.sa"
                }
            ) as websocket:
                response_time = (time.time() - start_time) * 1000
                result["response_time_ms"] = response_time
                result["connection_established"] = True
                
                logger.info(f"✅ Connection established in {response_time:.2f}ms")
                
                # Test sending a message
                test_message = json.dumps({
                    "type": "ping",
                    "timestamp": time.time(),
                    "test": True
                })
                
                logger.info(f"📤 Sending test message...")
                await websocket.send(test_message)
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    logger.info(f"📥 Received response: {response[:100]}...")
                    result["can_send_message"] = True
                    result["success"] = True
                    
                except asyncio.TimeoutError:
                    logger.warning("⏰ No response received within 5 seconds")
                    result["can_send_message"] = False
                    result["success"] = True  # Connection worked, just no response
                    
        except websockets.exceptions.InvalidStatusCode as e:
            result["error"] = f"Invalid status code: {e.status_code}"
            logger.error(f"❌ Invalid status code: {e.status_code}")
            
        except websockets.exceptions.InvalidURI as e:
            result["error"] = f"Invalid URI: {e}"
            logger.error(f"❌ Invalid URI: {e}")
            
        except ssl.SSLError as e:
            result["error"] = f"SSL Error: {e}"
            logger.error(f"❌ SSL Error: {e}")
            
        except socket.gaierror as e:
            result["error"] = f"DNS Resolution Error: {e}"
            logger.error(f"❌ DNS Resolution Error: {e}")
            
        except ConnectionRefusedError as e:
            result["error"] = f"Connection Refused: {e}"
            logger.error(f"❌ Connection Refused: {e}")
            
        except asyncio.TimeoutError:
            result["error"] = "Connection timeout"
            logger.error(f"❌ Connection timeout after {timeout} seconds")
            
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            logger.error(f"❌ Unexpected error: {e}")
        
        return result
    
    def test_http_endpoint(self, url: str, name: str) -> Dict[str, Any]:
        """Test if the HTTP endpoint is accessible."""
        logger.info(f"\n🌐 Testing HTTP endpoint for {name}")
        
        result = {
            "name": f"{name} (HTTP)",
            "url": url,
            "success": False,
            "status_code": None,
            "error": None,
            "response_time_ms": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10, verify=False)
            response_time = (time.time() - start_time) * 1000
            
            result["status_code"] = response.status_code
            result["response_time_ms"] = response_time
            result["success"] = response.status_code < 500
            
            logger.info(f"📊 HTTP Status: {response.status_code} in {response_time:.2f}ms")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ HTTP test failed: {e}")
        
        return result
    
    async def test_websocket_upgrade(self, base_url: str, name: str) -> Dict[str, Any]:
        """Test WebSocket upgrade process manually."""
        logger.info(f"\n🔄 Testing WebSocket upgrade for {name}")
        
        # This will help us see if the upgrade is happening correctly
        result = {
            "name": f"{name} (Upgrade Test)",
            "success": False,
            "error": None
        }
        
        try:
            import aiohttp
            
            url = base_url.replace("ws://", "http://").replace("wss://", "https://")
            url += "/ws/voice-conversation"
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    url,
                    headers={
                        "Origin": "https://dev.gmai.sa",
                        "User-Agent": "BeautyAI-Debug/1.0"
                    }
                ) as ws:
                    logger.info("✅ WebSocket upgrade successful with aiohttp")
                    result["success"] = True
                    
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ WebSocket upgrade failed: {e}")
        
        return result
    
    def check_dns_resolution(self, hostname: str) -> Dict[str, Any]:
        """Check DNS resolution for the hostname."""
        logger.info(f"\n🔍 Testing DNS resolution for {hostname}")
        
        result = {
            "hostname": hostname,
            "resolved_ips": [],
            "success": False,
            "error": None
        }
        
        try:
            import socket
            ips = socket.gethostbyname_ex(hostname)[2]
            result["resolved_ips"] = ips
            result["success"] = len(ips) > 0
            
            logger.info(f"✅ DNS resolved to: {', '.join(ips)}")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ DNS resolution failed: {e}")
        
        return result
    
    async def run_comprehensive_test(self):
        """Run all connectivity tests."""
        logger.info("🚀 Starting Comprehensive WebSocket Connectivity Test")
        logger.info("="*80)
        
        # Test endpoints
        endpoints = [
            ("ws://localhost:8000/ws/voice-conversation", "Localhost WebSocket"),
            ("ws://api.gmai.sa/ws/voice-conversation", "API.GMAI.SA WebSocket (WS)"),
            ("wss://api.gmai.sa/ws/voice-conversation", "API.GMAI.SA WebSocket (WSS)"),
            ("ws://127.0.0.1:8000/ws/voice-conversation", "Local IP WebSocket"),
        ]
        
        # Test DNS resolution
        dns_results = []
        for hostname in ["api.gmai.sa", "dev.gmai.sa", "localhost"]:
            dns_result = self.check_dns_resolution(hostname)
            dns_results.append(dns_result)
        
        # Test HTTP endpoints first
        http_results = []
        http_endpoints = [
            ("http://localhost:8000/docs", "Localhost API Docs"),
            ("https://api.gmai.sa/docs", "API.GMAI.SA Docs"),
            ("https://api.gmai.sa/", "API.GMAI.SA Root"),
        ]
        
        for url, name in http_endpoints:
            result = self.test_http_endpoint(url, name)
            http_results.append(result)
        
        # Test WebSocket connections
        websocket_results = []
        for url, name in endpoints:
            result = await self.test_basic_connection(url, name, timeout=15)
            websocket_results.append(result)
        
        # Test WebSocket upgrades
        upgrade_results = []
        upgrade_endpoints = [
            ("ws://localhost:8000", "Localhost"),
            ("wss://api.gmai.sa", "API.GMAI.SA"),
        ]
        
        for url, name in upgrade_endpoints:
            try:
                result = await self.test_websocket_upgrade(url, name)
                upgrade_results.append(result)
            except ImportError:
                logger.warning("⚠️ aiohttp not available for upgrade tests")
        
        # Generate report
        self.generate_report(dns_results, http_results, websocket_results, upgrade_results)
    
    def generate_report(self, dns_results, http_results, websocket_results, upgrade_results):
        """Generate a comprehensive connectivity report."""
        logger.info("\n" + "="*80)
        logger.info("📊 CONNECTIVITY TEST REPORT")
        logger.info("="*80)
        
        # DNS Results
        logger.info("\n🔍 DNS RESOLUTION RESULTS:")
        for result in dns_results:
            status = "✅" if result["success"] else "❌"
            logger.info(f"{status} {result['hostname']}: {result.get('resolved_ips', result.get('error'))}")
        
        # HTTP Results
        logger.info("\n🌐 HTTP ENDPOINT RESULTS:")
        for result in http_results:
            status = "✅" if result["success"] else "❌"
            details = f"Status: {result['status_code']}" if result["status_code"] else f"Error: {result['error']}"
            logger.info(f"{status} {result['name']}: {details}")
        
        # WebSocket Results
        logger.info("\n📡 WEBSOCKET CONNECTION RESULTS:")
        for result in websocket_results:
            status = "✅" if result["success"] else "❌"
            if result["success"]:
                details = f"Connected in {result['response_time_ms']:.2f}ms"
                if result["can_send_message"]:
                    details += " (can send messages)"
                else:
                    details += " (no response to messages)"
            else:
                details = f"Failed: {result['error']}"
            
            logger.info(f"{status} {result['name']}: {details}")
        
        # Upgrade Results
        if upgrade_results:
            logger.info("\n🔄 WEBSOCKET UPGRADE RESULTS:")
            for result in upgrade_results:
                status = "✅" if result["success"] else "❌"
                details = "Upgrade successful" if result["success"] else f"Failed: {result['error']}"
                logger.info(f"{status} {result['name']}: {details}")
        
        # Analysis
        logger.info("\n🎯 ANALYSIS:")
        
        # Check if localhost works
        localhost_works = any(r["success"] and "localhost" in r["url"].lower() for r in websocket_results)
        
        # Check if external domain works
        external_works = any(r["success"] and "api.gmai.sa" in r["url"] for r in websocket_results)
        
        if localhost_works and not external_works:
            logger.info("🔍 LIKELY ISSUE: Nginx proxy configuration or firewall blocking WebSocket")
            logger.info("   Recommendations:")
            logger.info("   1. Check nginx error logs: sudo tail -f /var/log/nginx/error.log")
            logger.info("   2. Verify WebSocket headers are properly forwarded")
            logger.info("   3. Check if router/firewall blocks WebSocket traffic")
            logger.info("   4. Test direct connection bypassing nginx")
            
        elif not localhost_works:
            logger.info("🔍 LIKELY ISSUE: BeautyAI service or WebSocket endpoint problem")
            logger.info("   Recommendations:")
            logger.info("   1. Check BeautyAI service status")
            logger.info("   2. Verify WebSocket endpoint is properly registered")
            logger.info("   3. Check service logs for errors")
            
        elif localhost_works and external_works:
            logger.info("✅ ALL CONNECTIONS WORKING: WebSocket setup appears correct")
            
        else:
            logger.info("🔍 MIXED RESULTS: Further investigation needed")
        
        logger.info("="*80)

async def main():
    """Run the connectivity debugger."""
    debugger = WebSocketConnectivityDebugger()
    await debugger.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
