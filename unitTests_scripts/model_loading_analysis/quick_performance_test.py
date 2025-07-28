#!/usr/bin/env python3
"""
🎯 QUICK PERFORMANCE TEST
=======================

Simple test to verify WebSocket voice performance improvements.
"""

import asyncio
import websockets
import json
import base64
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_performance():
    """Test Simple Voice WebSocket performance."""
    
    websocket_url = "ws://localhost:8000/ws/simple-voice-chat"
    
    # Create a minimal test audio (silence)
    # This represents about 1 second of silence in webm format
    test_audio_data = b'\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x44\x89\x84webm\x42\x87\x81\x04\x42\x85\x81\x02'  # Minimal webm header
    test_audio_b64 = base64.b64encode(test_audio_data).decode('utf-8')
    
    test_message = {
        "type": "voice_message",
        "audio_data": test_audio_b64,
        "format": "webm",
        "language": "ar"
    }
    
    try:
        logger.info("🔗 Connecting to WebSocket...")
        async with websockets.connect(websocket_url) as websocket:
            logger.info("✅ Connected successfully!")
            
            # Send test message
            logger.info("📤 Sending voice message...")
            start_time = time.time()
            
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            logger.info("⏳ Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"✅ Response received in {response_time:.2f} seconds")
            
            # Parse response
            try:
                response_data = json.loads(response)
                logger.info(f"📝 Response type: {response_data.get('type', 'unknown')}")
                
                if response_data.get('type') == 'voice_response':
                    text_response = response_data.get('text_response', '')
                    logger.info(f"💬 Text response: {text_response[:100]}...")
                    
                    if response_data.get('audio_data'):
                        logger.info("🔊 Audio response included")
                    
                    # Performance assessment
                    if response_time < 2:
                        logger.info("🎯 EXCELLENT: Response time < 2 seconds!")
                    elif response_time < 5:
                        logger.info("✅ GOOD: Response time < 5 seconds (target met)")
                    elif response_time < 10:
                        logger.info("⚠️ ACCEPTABLE: Response time < 10 seconds (improved)")
                    else:
                        logger.info("❌ NEEDS IMPROVEMENT: Response time > 10 seconds")
                        
                elif response_data.get('type') == 'error':
                    error_msg = response_data.get('message', 'Unknown error')
                    logger.error(f"❌ Error response: {error_msg}")
                    
            except json.JSONDecodeError:
                logger.error("❌ Failed to parse response as JSON")
                logger.info(f"Raw response: {response[:200]}")
            
            return response_time
            
    except asyncio.TimeoutError:
        logger.error("❌ Request timed out after 30 seconds")
        return None
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

async def main():
    """Main test function."""
    logger.info("🚀 STARTING WEBSOCKET PERFORMANCE TEST")
    logger.info("=" * 60)
    logger.info(f"⏰ Test started at: {datetime.now().strftime('%H:%M:%S')}")
    
    response_time = await test_websocket_performance()
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE TEST SUMMARY")
    logger.info("=" * 60)
    
    if response_time:
        logger.info(f"⏱️ Response Time: {response_time:.2f} seconds")
        
        # Improvement analysis
        original_time = 42  # seconds
        improvement = ((original_time - response_time) / original_time) * 100
        
        logger.info(f"📈 Improvement vs. Original (42s): {improvement:.1f}%")
        
        if response_time < 2:
            logger.info("🏆 STATUS: OPTIMISTIC TARGET ACHIEVED (<2s)")
        elif response_time < 5:
            logger.info("✅ STATUS: PRIMARY TARGET ACHIEVED (<5s)")
        else:
            logger.info("⚠️ STATUS: PARTIAL IMPROVEMENT (still room for optimization)")
    else:
        logger.info("❌ STATUS: TEST FAILED")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
