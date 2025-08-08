#!/usr/bin/env python3
"""
Fixed connectivity test for WebSocket VAD endpoint.
Tests basic WebSocket connectivity with proper websocket library usage.
"""

import asyncio
import websockets
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
            async with session.get('http://localhost:8000/health/') as resp:
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
    """Test WebSocket connection with proper syntax"""
    try:
        # Try the exact endpoint from the frontend logs
        uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ WebSocket connection successful")
            
            # Listen for initial connection message
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"📨 Received initial message: {message[:100]}...")
                
                # Try to parse as JSON
                try:
                    data = json.loads(message)
                    if data.get("type") == "connection_established":
                        logger.info("✅ Connection establishment confirmed")
                        return True
                except json.JSONDecodeError:
                    logger.warning("⚠️ Non-JSON initial message")
                
                return True
                
            except asyncio.TimeoutError:
                logger.warning("⏰ No initial response received")
                return True  # Connection worked, just no response
                
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

async def test_real_webm_file():
    """Test with actual WebM file from voice_tests"""
    try:
        webm_file = "/home/lumi/beautyai/voice_tests/input_test_questions/webm/q1.webm"
        
        # Read the WebM file
        with open(webm_file, 'rb') as f:
            webm_data = f.read()
        
        logger.info(f"📁 Loaded WebM file: {len(webm_data)} bytes")
        
        # Connect and send the file
        uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
        
        async with websockets.connect(uri) as websocket:
            # Wait for connection established
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                logger.info("📨 Connection message received")
            except asyncio.TimeoutError:
                pass
            
            # Send the WebM file as chunks (simulating MediaRecorder)
            chunk_size = 1024
            chunks_sent = 0
            
            for i in range(0, len(webm_data), chunk_size):
                chunk = webm_data[i:i+chunk_size]
                await websocket.send(chunk)
                chunks_sent += 1
                logger.info(f"📤 Sent chunk {chunks_sent}: {len(chunk)} bytes")
                await asyncio.sleep(0.1)  # Simulate real-time recording
            
            logger.info(f"✅ Sent {chunks_sent} chunks total")
            
            # Listen for responses
            responses = []
            start_time = time.time()
            
            while time.time() - start_time < 10.0:  # Listen for 10 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    responses.append(message)
                    
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        logger.info(f"📨 Response type: {msg_type}")
                        
                        if msg_type == "voice_response":
                            logger.info("🎤 Voice response received!")
                            break
                            
                    except json.JSONDecodeError:
                        logger.info(f"📨 Non-JSON response: {len(message)} bytes")
                        
                except asyncio.TimeoutError:
                    continue
            
            logger.info(f"📊 Total responses received: {len(responses)}")
            return len(responses) > 0
            
    except Exception as e:
        logger.error(f"❌ WebM file test failed: {e}")
        return False

async def main():
    """Run comprehensive connectivity tests"""
    logger.info("🧪 Running comprehensive connectivity test...")
    
    # Test 1: Backend health
    logger.info("1️⃣ Testing backend health endpoint...")
    health_ok = await test_backend_running()
    
    # Test 2: WebSocket connection
    logger.info("2️⃣ Testing WebSocket connection...")
    ws_ok = await test_websocket_connection()
    
    # Test 3: Real WebM file processing
    logger.info("3️⃣ Testing with real WebM file...")
    webm_ok = await test_real_webm_file()
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("🧪 COMPREHENSIVE CONNECTIVITY TEST RESULTS")
    logger.info("="*60)
    logger.info(f"🏥 Backend health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    logger.info(f"🔌 WebSocket connect: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    logger.info(f"🎤 WebM processing: {'✅ PASS' if webm_ok else '❌ FAIL'}")
    
    if health_ok and ws_ok and webm_ok:
        logger.info("✅ All systems operational - ready for VAD testing!")
        return True
    elif health_ok and ws_ok:
        logger.info("⚠️ Basic connectivity OK, WebM processing needs investigation")
        return True
    else:
        logger.error("❌ Critical connectivity issues detected")
        return False

if __name__ == "__main__":
    asyncio.run(main())
