#!/usr/bin/env python3
"""
Test script to verify WebSocket voice conversation fix for None/float comparison error.
"""
import asyncio
import websockets
import websockets.exceptions
import json
import base64
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_voice():
    """Test WebSocket voice conversation with a simple audio message."""
    uri = "ws://localhost:8000/ws/voice-conversation"
    
    try:
        # Create a simple audio file for testing (silence)
        import wave
        import numpy as np
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Save to temporary WAV file
        temp_file = "/tmp/test_audio.wav"
        with wave.open(temp_file, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Read and encode audio file
        with open(temp_file, 'rb') as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info("🔊 Connecting to WebSocket voice conversation...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected successfully!")
            
            # Wait for connection established message first
            logger.info("⏳ Waiting for connection established...")
            
            connection_msg = await websocket.recv()
            connection_data = json.loads(connection_msg)
            logger.info(f"📥 Connection message: {connection_data.get('type', 'unknown')}")
            
            # Send voice message as binary data (not JSON)
            logger.info("📤 Sending voice message as binary audio...")
            send_time = time.time()
            await websocket.send(audio_bytes)  # Send raw audio bytes
            
            # Wait for voice response (may receive multiple messages)
            logger.info("⏳ Waiting for voice response...")
            
            timeout = 120  # 2 minutes timeout for model loading
            start_time = time.time()
            
            try:
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10)
                        response_data = json.loads(response)
                        
                        logger.info(f"📥 Received response: {response_data.get('type', 'unknown')}")
                        
                        if response_data.get('type') == 'processing_started':
                            logger.info("⚡ Processing started, waiting for result...")
                            continue
                        
                        elif response_data.get('type') == 'voice_response':
                            if response_data.get('success'):
                                logger.info("✅ Voice conversation test PASSED!")
                                logger.info(f"Response time: {time.time() - send_time:.2f}s")
                                if 'audio_base64' in response_data:
                                    logger.info(f"🎵 Audio response received (length: {len(response_data['audio_base64'])} chars)")
                                if 'response_text' in response_data:
                                    logger.info(f"💬 Text response: {response_data['response_text']}")
                                if 'transcription' in response_data:
                                    logger.info(f"🎤 Transcription: {response_data['transcription']}")
                                return True
                            else:
                                logger.error(f"❌ Voice conversation test FAILED: {response_data.get('error', 'Unknown error')}")
                                return False
                        
                        elif response_data.get('type') == 'error':
                            logger.error(f"❌ Server error: {response_data.get('message', 'Unknown error')}")
                            return False
                        
                        else:
                            logger.info(f"📨 Other message: {response_data.get('type')} - {response_data.get('message', '')}")
                            continue
                    
                    except asyncio.TimeoutError:
                        logger.info("⏰ Waiting for response... (10s timeout per message)")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("🔌 WebSocket connection closed by server")
                        break
                    except Exception as e:
                        logger.error(f"❌ Error receiving message: {e}")
                        break
                
                logger.error(f"❌ Timeout after {timeout}s - no voice response received")
                return False
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout after {timeout}s - no response received")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket connection error: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🧪 Testing WebSocket Voice Conversation Fix")
    logger.info("=" * 50)
    
    success = await test_websocket_voice()
    
    if success:
        print("\n🎉 Test COMPLETED SUCCESSFULLY!")
        print("The None/float comparison error has been FIXED!")
    else:
        print("\n💥 Test FAILED!")
        print("Check the server logs for more details.")

if __name__ == "__main__":
    asyncio.run(main())
