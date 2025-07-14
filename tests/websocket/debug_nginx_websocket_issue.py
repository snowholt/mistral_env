#!/usr/bin/env python3
"""
Nginx WebSocket Configuration Debug Tool

This script specifically tests the nginx WebSocket proxy configuration
to identify why WebSocket connections fail through api.gmai.sa but work locally.

Tests:
1. Direct localhost WebSocket connection
2. WebSocket via nginx proxy (api.gmai.sa)
3. HTTP API calls via nginx proxy (for comparison)
4. WebSocket upgrade headers
5. SSL/TLS WebSocket handling
6. Nginx error log analysis
"""

import asyncio
import json
import logging
import time
import ssl
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import websockets
import aiohttp
from websockets.exceptions import ConnectionClosed, InvalidURI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/lumi/beautyai/tests/websocket/nginx_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class NginxWebSocketDebugger:
    """Debug nginx WebSocket proxy configuration issues."""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_localhost_websocket(self) -> Dict[str, Any]:
        """Test direct localhost WebSocket connection."""
        logger.info("🔍 TESTING LOCALHOST WEBSOCKET")
        logger.info("="*50)
        
        result = {
            "test": "localhost_websocket",
            "success": False,
            "connection_time": 0.0,
            "error": None,
            "messages_received": []
        }
        
        url = "ws://localhost:8000/ws/voice-conversation?session_id=debug_localhost"
        
        try:
            start_time = time.time()
            async with websockets.connect(url) as websocket:
                connection_time = time.time() - start_time
                result["connection_time"] = connection_time
                logger.info(f"✅ Localhost WebSocket connected in {connection_time:.3f}s")
                
                # Wait for connection message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(message)
                    result["messages_received"].append(data)
                    logger.info(f"📨 Received: {data.get('type', 'unknown')}")
                    result["success"] = True
                except asyncio.TimeoutError:
                    logger.warning("⏰ No connection message received")
                    result["success"] = True  # Connection worked, just no message
                    
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Localhost WebSocket failed: {e}")
            
        return result
    
    async def test_nginx_websocket(self) -> Dict[str, Any]:
        """Test WebSocket through nginx proxy."""
        logger.info("\n🔍 TESTING NGINX WEBSOCKET PROXY")
        logger.info("="*50)
        
        result = {
            "test": "nginx_websocket",
            "success": False,
            "connection_time": 0.0,
            "error": None,
            "ssl_error": None,
            "messages_received": []
        }
        
        url = "wss://api.gmai.sa/ws/voice-conversation?session_id=debug_nginx"
        
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.info("🔒 Using SSL bypass for self-signed certificate")
            
            start_time = time.time()
            async with websockets.connect(url, ssl=ssl_context) as websocket:
                connection_time = time.time() - start_time
                result["connection_time"] = connection_time
                logger.info(f"✅ Nginx WebSocket connected in {connection_time:.3f}s")
                
                # Wait for connection message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    result["messages_received"].append(data)
                    logger.info(f"📨 Received: {data.get('type', 'unknown')}")
                    result["success"] = True
                except asyncio.TimeoutError:
                    logger.warning("⏰ No connection message received")
                    result["success"] = True  # Connection worked, just no message
                    
        except ssl.SSLError as e:
            result["ssl_error"] = str(e)
            logger.error(f"🔒 SSL Error: {e}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Nginx WebSocket failed: {e}")
            
        return result
    
    async def test_http_api_comparison(self) -> Dict[str, Any]:
        """Test HTTP API through nginx proxy for comparison."""
        logger.info("\n🔍 TESTING HTTP API VIA NGINX (COMPARISON)")
        logger.info("="*50)
        
        result = {
            "test": "nginx_http_api",
            "success": False,
            "response_time": 0.0,
            "status_code": None,
            "error": None
        }
        
        url = "https://api.gmai.sa/health"
        
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            start_time = time.time()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    result["response_time"] = response_time
                    result["status_code"] = response.status
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ HTTP API works: {response.status} in {response_time:.3f}s")
                        logger.info(f"📊 Response: {data.get('status', 'unknown')}")
                        result["success"] = True
                    else:
                        logger.error(f"❌ HTTP API error: {response.status}")
                        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ HTTP API failed: {e}")
            
        return result
    
    async def test_websocket_with_curl(self) -> Dict[str, Any]:
        """Test WebSocket upgrade with curl."""
        logger.info("\n🔍 TESTING WEBSOCKET UPGRADE WITH CURL")
        logger.info("="*50)
        
        result = {
            "test": "curl_websocket_upgrade",
            "success": False,
            "output": "",
            "error": None
        }
        
        try:
            # Test WebSocket upgrade headers
            cmd = [
                "curl", "-v", "-k",  # -k for insecure SSL
                "-H", "Connection: Upgrade",
                "-H", "Upgrade: websocket", 
                "-H", "Sec-WebSocket-Version: 13",
                "-H", "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
                "https://api.gmai.sa/ws/voice-conversation"
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            result["output"] = process.stderr  # curl outputs headers to stderr
            
            if "101 Switching Protocols" in result["output"]:
                logger.info("✅ WebSocket upgrade successful")
                result["success"] = True
            elif "Connection: upgrade" in result["output"]:
                logger.info("✅ WebSocket upgrade headers accepted")
                result["success"] = True
            else:
                logger.warning("⚠️ WebSocket upgrade may have failed")
                logger.info(f"Curl output: {result['output'][:500]}...")
                
        except subprocess.TimeoutExpired:
            result["error"] = "Curl command timeout"
            logger.error("⏰ Curl command timed out")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Curl test failed: {e}")
            
        return result
    
    def check_nginx_logs(self) -> Dict[str, Any]:
        """Check nginx error logs for WebSocket-related issues."""
        logger.info("\n🔍 CHECKING NGINX ERROR LOGS")
        logger.info("="*50)
        
        result = {
            "test": "nginx_logs",
            "success": False,
            "recent_errors": [],
            "websocket_errors": [],
            "error": None
        }
        
        try:
            # Check nginx error log
            cmd = ["sudo", "tail", "-n", "50", "/var/log/nginx/error.log"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                error_lines = process.stdout.strip().split('\n')
                recent_errors = [line for line in error_lines if line.strip()]
                result["recent_errors"] = recent_errors[-10:]  # Last 10 errors
                
                # Look for WebSocket-specific errors
                websocket_keywords = ["websocket", "upgrade", "connection", "proxy"]
                websocket_errors = []
                
                for line in recent_errors:
                    if any(keyword.lower() in line.lower() for keyword in websocket_keywords):
                        websocket_errors.append(line)
                
                result["websocket_errors"] = websocket_errors
                
                if websocket_errors:
                    logger.warning(f"⚠️ Found {len(websocket_errors)} WebSocket-related errors")
                    for error in websocket_errors[-3:]:  # Show last 3
                        logger.warning(f"   {error}")
                else:
                    logger.info("✅ No WebSocket-related errors in recent logs")
                
                result["success"] = True
                
            else:
                result["error"] = f"Failed to read nginx logs: {process.stderr}"
                logger.error(f"❌ {result['error']}")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Log check failed: {e}")
            
        return result
    
    def check_nginx_config(self) -> Dict[str, Any]:
        """Validate nginx configuration for WebSocket support."""
        logger.info("\n🔍 CHECKING NGINX WEBSOCKET CONFIGURATION")
        logger.info("="*50)
        
        result = {
            "test": "nginx_config",
            "success": False,
            "config_issues": [],
            "recommendations": [],
            "error": None
        }
        
        try:
            # Read the nginx config
            config_path = "/etc/nginx/sites-enabled/gmai.sa"
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            issues = []
            recommendations = []
            
            # Check for required WebSocket directives
            required_directives = [
                "proxy_http_version 1.1",
                "proxy_set_header Upgrade $http_upgrade",
                "proxy_set_header Connection \"upgrade\""
            ]
            
            for directive in required_directives:
                if directive not in config_content:
                    issues.append(f"Missing directive: {directive}")
                    
            # Check for WebSocket-specific timeouts
            if "proxy_read_timeout" not in config_content:
                recommendations.append("Consider adding proxy_read_timeout for long WebSocket connections")
                
            if "proxy_send_timeout" not in config_content:
                recommendations.append("Consider adding proxy_send_timeout for WebSocket responses")
                
            # Check for buffering settings
            if "proxy_buffering off" not in config_content:
                recommendations.append("Consider adding 'proxy_buffering off' for real-time WebSocket data")
                
            result["config_issues"] = issues
            result["recommendations"] = recommendations
            
            if issues:
                logger.error(f"❌ Found {len(issues)} configuration issues:")
                for issue in issues:
                    logger.error(f"   {issue}")
            else:
                logger.info("✅ Required WebSocket directives found")
                
            if recommendations:
                logger.info(f"💡 {len(recommendations)} recommendations:")
                for rec in recommendations:
                    logger.info(f"   {rec}")
                    
            result["success"] = len(issues) == 0
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Config check failed: {e}")
            
        return result
    
    async def test_websocket_headers(self) -> Dict[str, Any]:
        """Test WebSocket connection with detailed header analysis."""
        logger.info("\n🔍 TESTING WEBSOCKET HEADERS AND HANDSHAKE")
        logger.info("="*50)
        
        result = {
            "test": "websocket_headers",
            "success": False,
            "handshake_details": {},
            "error": None
        }
        
        try:
            # Create a WebSocket connection with detailed logging
            url = "wss://api.gmai.sa/ws/voice-conversation"
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # This will show us the handshake details
            async with websockets.connect(url, ssl=ssl_context) as websocket:
                logger.info("✅ WebSocket handshake successful")
                logger.info(f"📊 WebSocket state: {websocket.state}")
                
                result["handshake_details"] = {
                    "state": str(websocket.state),
                    "local_address": str(websocket.local_address),
                    "remote_address": str(websocket.remote_address)
                }
                result["success"] = True
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ WebSocket handshake failed: {e}")
            
        return result
    
    async def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run all debug tests."""
        logger.info("🚀 Starting Comprehensive Nginx WebSocket Debug")
        logger.info("="*80)
        
        # Run all tests
        tests = [
            ("localhost_websocket", self.test_localhost_websocket()),
            ("nginx_websocket", self.test_nginx_websocket()),
            ("nginx_http_api", self.test_http_api_comparison()),
            ("curl_websocket", self.test_websocket_with_curl()),
            ("nginx_logs", self.check_nginx_logs()),
            ("nginx_config", self.check_nginx_config()),
            ("websocket_headers", self.test_websocket_headers()),
        ]
        
        results = {}
        
        for test_name, test_coro in tests:
            if asyncio.iscoroutine(test_coro):
                results[test_name] = await test_coro
            else:
                results[test_name] = test_coro
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        self.test_results = results
        return results
    
    def generate_diagnosis_report(self) -> str:
        """Generate diagnosis report with specific recommendations."""
        if not self.test_results:
            return "No test results available"
            
        report = []
        report.append("="*80)
        report.append("🔧 NGINX WEBSOCKET DIAGNOSIS REPORT")
        report.append("="*80)
        report.append(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Test summary
        localhost_works = self.test_results.get("localhost_websocket", {}).get("success", False)
        nginx_works = self.test_results.get("nginx_websocket", {}).get("success", False)
        http_works = self.test_results.get("nginx_http_api", {}).get("success", False)
        
        report.append("📊 TEST SUMMARY:")
        report.append(f"   Localhost WebSocket: {'✅ WORKS' if localhost_works else '❌ FAILS'}")
        report.append(f"   Nginx WebSocket: {'✅ WORKS' if nginx_works else '❌ FAILS'}")
        report.append(f"   Nginx HTTP API: {'✅ WORKS' if http_works else '❌ FAILS'}")
        report.append("")
        
        # Diagnosis
        report.append("🔍 DIAGNOSIS:")
        
        if localhost_works and not nginx_works and http_works:
            report.append("   🎯 ISSUE: Nginx WebSocket proxy configuration problem")
            report.append("   📌 HTTP API works but WebSocket doesn't through nginx")
            report.append("   🔧 This is a classic nginx WebSocket configuration issue")
            
        elif localhost_works and not nginx_works and not http_works:
            report.append("   🎯 ISSUE: General nginx proxy problem")
            report.append("   📌 Both HTTP and WebSocket fail through nginx")
            
        elif not localhost_works:
            report.append("   🎯 ISSUE: Local WebSocket service problem")
            report.append("   📌 WebSocket doesn't work locally - check BeautyAI service")
            
        elif localhost_works and nginx_works:
            report.append("   ✅ NO ISSUE: Both localhost and nginx WebSocket work")
            report.append("   📌 The problem may be intermittent or client-side")
            
        report.append("")
        
        # Configuration issues
        config_result = self.test_results.get("nginx_config", {})
        if config_result.get("config_issues"):
            report.append("⚠️ CONFIGURATION ISSUES:")
            for issue in config_result["config_issues"]:
                report.append(f"   - {issue}")
            report.append("")
            
        # Log errors
        log_result = self.test_results.get("nginx_logs", {})
        if log_result.get("websocket_errors"):
            report.append("📋 NGINX LOG ERRORS:")
            for error in log_result["websocket_errors"][-3:]:
                report.append(f"   - {error}")
            report.append("")
            
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        
        if not nginx_works and localhost_works:
            report.append("   1. Check nginx WebSocket proxy configuration")
            report.append("   2. Verify WebSocket upgrade headers are properly set")
            report.append("   3. Check nginx error logs for specific errors")
            report.append("   4. Test with nginx configuration reload")
            report.append("   5. Consider firewall/router WebSocket support")
            
        if config_result.get("recommendations"):
            report.append("   Configuration improvements:")
            for rec in config_result["recommendations"]:
                report.append(f"   - {rec}")
                
        # Detailed results
        report.append("")
        report.append("📋 DETAILED TEST RESULTS:")
        report.append("-"*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            report.append(f"{status} {test_name}")
            
            if result.get("error"):
                report.append(f"   Error: {result['error']}")
            if result.get("connection_time"):
                report.append(f"   Time: {result['connection_time']:.3f}s")
            if result.get("response_time"):
                report.append(f"   Time: {result['response_time']:.3f}s")
                
            report.append("")
            
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_diagnosis_report(self) -> str:
        """Save diagnosis report to file."""
        report = self.generate_diagnosis_report()
        
        report_file = Path("/home/lumi/beautyai/tests/websocket/nginx_websocket_diagnosis.txt")
        with open(report_file, "w") as f:
            f.write(report)
            
        # Also save JSON results
        json_file = Path("/home/lumi/beautyai/tests/websocket/nginx_websocket_results.json")
        with open(json_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
            
        logger.info(f"📄 Diagnosis report saved to: {report_file}")
        logger.info(f"📊 JSON results saved to: {json_file}")
        
        return str(report_file)


async def main():
    """Main debug function."""
    debugger = NginxWebSocketDebugger()
    
    try:
        # Run comprehensive debug
        results = await debugger.run_comprehensive_debug()
        
        # Generate and display report
        report = debugger.generate_diagnosis_report()
        print(report)
        
        # Save report
        report_file = debugger.save_diagnosis_report()
        
        # Summary
        localhost_works = results.get("localhost_websocket", {}).get("success", False)
        nginx_works = results.get("nginx_websocket", {}).get("success", False)
        
        logger.info(f"\n🎯 DIAGNOSIS SUMMARY:")
        logger.info(f"   Localhost WebSocket: {'✅ WORKS' if localhost_works else '❌ FAILS'}")
        logger.info(f"   Nginx WebSocket: {'✅ WORKS' if nginx_works else '❌ FAILS'}")
        logger.info(f"   Report: {report_file}")
        
        if localhost_works and not nginx_works:
            logger.info(f"\n🔧 LIKELY CAUSE: Nginx WebSocket proxy configuration issue")
            logger.info(f"   Check the diagnosis report for specific recommendations")
        
        return nginx_works
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Debug interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Debug execution failed: {e}")
        return False


if __name__ == "__main__":
    # Run debug
    success = asyncio.run(main())
    exit(0 if success else 1)
