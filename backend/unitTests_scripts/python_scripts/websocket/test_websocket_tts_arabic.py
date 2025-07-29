#!/usr/bin/env python3
"""
Quick WebSocket TTS Test
========================

Test the WebSocket specifically with Arabic audio to see if TTS bytes work.
Since we confirmed Arabic TTS bytes work, let's see if they make it through the WebSocket.
"""

import asyncio
import websockets
import json
import base64
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_websocket_tts():
    """Test WebSocket with Arabic audio to check TTS bytes."""
    
    # Test with Arabic audio file
    audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
    
    if not Path(audio_file).exists():
        logger.error(f"❌ Audio file not found: {audio_file}")
        return
    
    # Read audio file
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    
    logger.info(f"🎯 Testing WebSocket TTS with Arabic audio: {len(audio_data)} bytes")
    
    try:
        # Connect to WebSocket
        uri = "ws://localhost:8000/ws/voice-conversation?thinking_mode=false&preset=qwen_optimized"
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket")
            
            # Listen for connection message
            connection_msg = await websocket.recv()
            connection_data = json.loads(connection_msg)
            logger.info(f"📨 Connection: {connection_data['type']}")
            
            # Send audio
            logger.info("🎤 Sending Arabic audio...")
            await websocket.send(audio_data)
            
            # Wait for processing message
            processing_msg = await websocket.recv()
            processing_data = json.loads(processing_msg)
            logger.info(f"📨 Processing: {processing_data['type']}")
            
            # Wait for response
            logger.info("⏳ Waiting for voice response...")
            response_msg = await websocket.recv()
            response_data = json.loads(response_msg)
            
            logger.info(f"📨 Response type: {response_data['type']}")
            logger.info(f"✅ Success: {response_data.get('success', False)}")
            
            if response_data.get('success'):
                logger.info(f"📝 Transcription: {response_data.get('transcription', '')}")
                logger.info(f"💬 Response: {response_data.get('response_text', '')[:100]}...")
                
                # Check audio data
                audio_base64 = response_data.get('audio_base64')
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    logger.info(f"🎵 ✅ AUDIO RECEIVED: {len(audio_bytes)} bytes")
                    
                    # Save the audio for verification
                    output_file = Path("/home/lumi/beautyai/tests/websocket/websocket_tts_test.wav")
                    with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                    logger.info(f"💾 Audio saved to: {output_file}")
                    
                    return True
                else:
                    logger.error("❌ No audio data in response!")
                    logger.info(f"🔍 Response keys: {list(response_data.keys())}")
                    if 'warning' in response_data:
                        logger.warning(f"⚠️ Warning: {response_data['warning']}")
                    return False
            else:
                error_msg = response_data.get('error', 'Unknown error')
                logger.error(f"❌ Request failed: {error_msg}")
                return False
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Testing WebSocket TTS with Arabic Audio")
    logger.info("=" * 60)
    
    success = await test_websocket_tts()
    
    if success:
        logger.info("\n🎉 WebSocket TTS is working! Audio data received successfully.")
        logger.info("📋 The voice-to-voice WebSocket pipeline is functional.")
    else:
        logger.info("\n⚠️ WebSocket TTS has issues. Need to investigate further.")
        logger.info("📋 Check the voice service TTS integration.")

if __name__ == "__main__":
    asyncio.run(main())
