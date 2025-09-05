#!/usr/bin/env python3
"""
Test script to verify the streaming voice fixes
"""

import asyncio
import websockets
import json
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_streaming_voice():
    """Test the fixed streaming voice endpoint"""
    
    uri = "ws://localhost:8000/api/v1/ws/streaming-voice?language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to streaming voice endpoint")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            logger.info(f"📡 Ready message: {ready_data.get('message', 'No message')}")
            
            if ready_data.get('type') != 'ready':
                logger.error(f"❌ Expected 'ready' message, got: {ready_data}")
                return
            
            # Send a test audio chunk (16-bit PCM, 320 samples = 640 bytes)
            sample_rate = 16000
            duration_ms = 20  # 20ms chunk
            samples_per_chunk = int(sample_rate * duration_ms / 1000)  # 320 samples
            
            # Create a simple sine wave at 440Hz (A4 note)
            frequency = 440
            t = np.linspace(0, duration_ms/1000, samples_per_chunk)
            audio_samples = (np.sin(2 * np.pi * frequency * t) * 16383).astype(np.int16)
            
            logger.info(f"🎵 Generated test audio: {len(audio_samples)} samples ({len(audio_samples) * 2} bytes)")
            
            # Send multiple chunks to simulate speech
            for i in range(10):  # Send 10 chunks (200ms total)
                audio_bytes = audio_samples.tobytes()
                await websocket.send(audio_bytes)
                logger.info(f"📤 Sent audio chunk {i+1}/10 ({len(audio_bytes)} bytes)")
                await asyncio.sleep(0.02)  # 20ms interval
                
            # Wait for responses
            logger.info("👂 Waiting for responses...")
            
            responses = []
            timeout_count = 0
            max_timeouts = 10
            
            while timeout_count < max_timeouts:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    event_type = data.get('type', 'unknown')
                    logger.info(f"📨 Received: {event_type}")
                    
                    if event_type == 'partial_transcript':
                        text = data.get('text', '')
                        logger.info(f"📝 Partial transcript: '{text}'")
                    elif event_type == 'final_transcript':
                        text = data.get('text', '')
                        logger.info(f"✅ Final transcript: '{text}'")
                    elif event_type == 'transcription_error':
                        error = data.get('error', 'Unknown error')
                        logger.error(f"❌ Transcription error: {error}")
                    elif event_type == 'decoder_error':
                        error = data.get('error', 'Unknown error')
                        logger.error(f"❌ Decoder error: {error}")
                    elif event_type == 'error':
                        error = data.get('message', 'Unknown error')
                        logger.error(f"❌ General error: {error}")
                    elif event_type == 'heartbeat':
                        bytes_received = data.get('bytes_received', 0)
                        logger.info(f"💓 Heartbeat: {bytes_received} bytes received")
                    
                    timeout_count = 0  # Reset timeout counter on successful receive
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    logger.info(f"⏰ Timeout {timeout_count}/{max_timeouts}")
                    continue
                    
            logger.info(f"📊 Test completed. Received {len(responses)} responses")
            
            # Send reset message
            reset_msg = {"type": "reset_conversation"}
            await websocket.send(json.dumps(reset_msg))
            logger.info("🔄 Sent reset message")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_streaming_voice())