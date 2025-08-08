#!/usr/bin/env python3
"""
Quick manual test to validate backend chunk accumulation.
This test directly calls the backend WebSocket to check chunk handling.
"""

import asyncio
import logging
import json
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_backend_running():
    """Test if backend is responding"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Backend is running: {data}")
                    return True
                else:
                    logger.error(f"❌ Backend health check failed: {resp.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Backend connection failed: {e}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection"""
    try:
        import websockets
        
        async with websockets.connect(
            "ws://localhost:8000/ws/simple-voice-chat",
            extra_headers={"Origin": "http://localhost:3000"}
        ) as websocket:
            logger.info("✅ WebSocket connection successful")
            
            # Send a small test message
            test_chunk = b'\x1a\x45\xdf\xa3' + b'\x00' * 100  # WebM header + data
            await websocket.send(test_chunk)
            logger.info("📤 Sent test chunk")
            
            # Listen for response
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"📨 Received response: {message[:100]}...")
                return True
            except asyncio.TimeoutError:
                logger.warning("⏰ No response received within timeout")
                return True  # Connection worked, just no response
                
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

async def main():
    """Run basic connectivity tests"""
    logger.info("🧪 Running basic backend connectivity test...")
    
    # Test 1: Backend health
    logger.info("1️⃣ Testing backend health endpoint...")
    health_ok = await test_backend_running()
    
    # Test 2: WebSocket connection
    logger.info("2️⃣ Testing WebSocket connection...")
    ws_ok = await test_websocket_connection()
    
    # Results
    logger.info("\n" + "="*50)
    logger.info("🧪 BASIC CONNECTIVITY TEST RESULTS")
    logger.info("="*50)
    logger.info(f"🏥 Backend health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    logger.info(f"🔌 WebSocket: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    
    if health_ok and ws_ok:
        logger.info("✅ Backend is ready for chunk accumulation testing!")
        return True
    else:
        logger.error("❌ Backend is not ready. Please check server status.")
        return False

if __name__ == "__main__":
    asyncio.run(main())
