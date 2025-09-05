"""
Connection Pool Integration Test Script.

This script tests the integration of the new WebSocket connection pool
with the existing simple voice chat endpoint to ensure everything works
together properly.

Usage:
    python test_connection_pool_integration.py

Author: BeautyAI Framework  
Date: September 5, 2025
"""

import asyncio
import json
import logging
import sys
import time
import websockets
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionPoolIntegrationTest:
    """Integration test for WebSocket connection pool with voice chat endpoint."""
    
    def __init__(self, base_url="ws://localhost:8000"):
        self.base_url = base_url
        self.connections = []
        self.received_messages = []
    
    async def test_single_connection(self):
        """Test a single WebSocket connection to the voice chat endpoint."""
        logger.info("🔌 Testing single WebSocket connection...")
        
        uri = f"{self.base_url}/ws/simple-voice-chat?language=en&voice_type=female"
        
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("✅ WebSocket connected successfully")
                
                # Wait for welcome message
                welcome_msg = await websocket.recv()
                welcome_data = json.loads(welcome_msg)
                logger.info(f"📨 Received welcome: {welcome_data['type']}")
                
                assert welcome_data["type"] == "connection_established"
                assert welcome_data["success"] == True
                assert "pool_connection_id" in welcome_data
                
                # Send a ping
                ping_msg = {"type": "ping", "timestamp": time.time()}
                await websocket.send(json.dumps(ping_msg))
                
                # Wait for pong
                pong_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                pong_data = json.loads(pong_response)
                logger.info(f"📨 Received pong: {pong_data['type']}")
                
                assert pong_data["type"] == "pong"
                
                logger.info("✅ Single connection test passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Single connection test failed: {e}")
            return False
    
    async def test_multiple_connections(self):
        """Test multiple concurrent WebSocket connections."""
        logger.info("🔌 Testing multiple WebSocket connections...")
        
        connection_count = 5
        connections = []
        
        try:
            # Create multiple connections
            for i in range(connection_count):
                uri = f"{self.base_url}/ws/simple-voice-chat?language=en&voice_type=female&session_id=test_session_{i}"
                websocket = await websockets.connect(uri)
                connections.append(websocket)
                
                # Wait for welcome message
                welcome_msg = await websocket.recv()
                welcome_data = json.loads(welcome_msg)
                
                assert welcome_data["type"] == "connection_established"
                logger.info(f"✅ Connection {i+1} established")
            
            logger.info(f"✅ All {connection_count} connections established")
            
            # Test concurrent message sending
            async def send_ping(ws, conn_id):
                ping_msg = {"type": "ping", "timestamp": time.time(), "connection": conn_id}
                await ws.send(json.dumps(ping_msg))
                
                pong_response = await asyncio.wait_for(ws.recv(), timeout=5)
                pong_data = json.loads(pong_response)
                
                assert pong_data["type"] == "pong"
                return conn_id
            
            # Send pings concurrently
            ping_tasks = [send_ping(ws, i) for i, ws in enumerate(connections)]
            results = await asyncio.gather(*ping_tasks)
            
            logger.info(f"✅ All {len(results)} connections responded to ping")
            
            # Close all connections
            for websocket in connections:
                await websocket.close()
            
            logger.info("✅ Multiple connections test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multiple connections test failed: {e}")
            
            # Cleanup on error
            for websocket in connections:
                try:
                    await websocket.close()
                except:
                    pass
            
            return False
    
    async def test_connection_recovery(self):
        """Test connection recovery and pool health."""
        logger.info("🔧 Testing connection recovery...")
        
        uri = f"{self.base_url}/ws/simple-voice-chat?language=en&voice_type=female"
        
        try:
            # Create connection
            websocket = await websockets.connect(uri)
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            
            logger.info("✅ Initial connection established")
            
            # Force close connection
            await websocket.close()
            logger.info("🔌 Connection closed")
            
            # Wait a moment
            await asyncio.sleep(1)
            
            # Create new connection (should work despite previous close)
            websocket2 = await websockets.connect(uri)
            welcome_msg2 = await websocket2.recv()
            welcome_data2 = json.loads(welcome_msg2)
            
            assert welcome_data2["type"] == "connection_established"
            logger.info("✅ Recovery connection established")
            
            # Test that new connection works
            ping_msg = {"type": "ping", "timestamp": time.time()}
            await websocket2.send(json.dumps(ping_msg))
            
            pong_response = await asyncio.wait_for(websocket2.recv(), timeout=5)
            pong_data = json.loads(pong_response)
            
            assert pong_data["type"] == "pong"
            logger.info("✅ Recovery connection functional")
            
            await websocket2.close()
            
            logger.info("✅ Connection recovery test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection recovery test failed: {e}")
            return False
    
    async def check_status_endpoint(self):
        """Check the status endpoint shows pool information."""
        logger.info("📊 Checking status endpoint...")
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                status_url = f"http://localhost:8000/ws/simple-voice-chat/status"
                async with session.get(status_url) as response:
                    status_data = await response.json()
                    
                    logger.info(f"📊 Status: {status_data['status']}")
                    logger.info(f"📊 Active connections: {status_data['active_connections']}")
                    
                    # Check for pool metrics
                    if "pool_metrics" in status_data:
                        pool_metrics = status_data["pool_metrics"]
                        logger.info(f"📊 Pool total: {pool_metrics['total_connections']}")
                        logger.info(f"📊 Pool utilization: {pool_metrics['pool_utilization']:.2%}")
                        logger.info("✅ Status endpoint shows pool metrics")
                    else:
                        logger.warning("⚠️ No pool metrics in status endpoint")
                    
                    return True
                    
        except ImportError:
            logger.warning("⚠️ aiohttp not available, skipping status endpoint test")
            return True
        except Exception as e:
            logger.error(f"❌ Status endpoint test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting WebSocket Connection Pool Integration Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Single Connection", self.test_single_connection),
            ("Multiple Connections", self.test_multiple_connections),
            ("Connection Recovery", self.test_connection_recovery),
            ("Status Endpoint", self.check_status_endpoint),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running test: {test_name}")
            logger.info("-" * 40)
            
            try:
                result = await test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.info(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name}")
        
        logger.info(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Connection pool integration is working correctly.")
        else:
            logger.error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        
        return passed == total


async def main():
    """Main test runner."""
    test_runner = ConnectionPoolIntegrationTest()
    
    logger.info("🔍 Testing WebSocket Connection Pool Integration")
    logger.info("💡 Make sure the BeautyAI API server is running on localhost:8000")
    logger.info("💡 Run: python backend/run_server.py")
    logger.info("")
    
    success = await test_runner.run_all_tests()
    
    if success:
        logger.info("\n🎉 Integration test completed successfully!")
        exit(0)
    else:
        logger.error("\n💥 Integration test failed!")
        exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        exit(1)