#!/usr/bin/env python3
"""
Simple WebSocket test to isolate specific issues found in comprehensive testing.

This script tests individual issues:
1. Content filtering problems
2. TTS generation issues  
3. WebSocket frame size limits
4. Different presets behavior
"""

import asyncio
import json
import logging
import time
import base64
import ssl
from pathlib import Path
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_content_filter_issue():
    """Test why content filter is rejecting Botox-related content."""
    
    logger.info("🔍 TESTING CONTENT FILTER ISSUE")
    logger.info("="*50)
    
    # Test with disabled content filter
    url = "ws://localhost:8000/ws/voice-conversation?disable_content_filter=true&preset=qwen_optimized&input_language=auto&output_language=auto"
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("✅ Connected to WebSocket")
            
            # Wait for connection confirmation
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"🔗 Connection: {data.get('type')}")
            
            # Test with botox.wav (the English file that was filtered)
            audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav"
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"📤 Sending audio: {len(audio_data)} bytes (content filter disabled)")
            await websocket.send(audio_data)
            
            # Wait for response
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                data = json.loads(message)
                
                if data.get("type") == "voice_response":
                    success = data.get("success", False)
                    logger.info(f"🎯 Voice response success: {success}")
                    
                    if success:
                        logger.info(f"📝 Transcription: {data.get('transcription', '')}")
                        logger.info(f"💬 Response: {data.get('response_text', '')}")
                        logger.info("✅ Content filter disabled test PASSED")
                    else:
                        logger.error(f"❌ Still failed: {data.get('error', 'Unknown error')}")
                    break
                elif data.get("type") == "processing_started":
                    logger.info("⚡ Processing started...")
                    continue
                    
    except Exception as e:
        logger.error(f"❌ Content filter test failed: {e}")


async def test_tts_issue():
    """Test TTS generation issues with English language."""
    
    logger.info("\n🔍 TESTING TTS GENERATION ISSUE")
    logger.info("="*50)
    
    # Test with different TTS settings for English
    url = "ws://localhost:8000/ws/voice-conversation?input_language=en&output_language=en&tts_model_name=coqui-tts-arabic&disable_content_filter=true&preset=speed_optimized"
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("✅ Connected to WebSocket")
            
            # Wait for connection confirmation
            message = await websocket.recv()
            logger.info("🔗 Connection confirmed")
            
            # Test with botox.wav
            audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav"
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"📤 Sending audio: {len(audio_data)} bytes (English TTS test)")
            await websocket.send(audio_data)
            
            # Wait for response
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                data = json.loads(message)
                
                if data.get("type") == "voice_response":
                    success = data.get("success", False)
                    logger.info(f"🎯 Voice response success: {success}")
                    
                    if success:
                        audio_received = data.get("audio_base64") is not None
                        logger.info(f"🎵 Audio received: {audio_received}")
                        if audio_received:
                            logger.info(f"📏 Audio size: {data.get('audio_size_bytes', 0)} bytes")
                            logger.info("✅ TTS test PASSED")
                        else:
                            logger.warning("⚠️ No audio in response")
                    else:
                        logger.error(f"❌ TTS failed: {data.get('error', 'Unknown error')}")
                    break
                elif data.get("type") == "processing_started":
                    logger.info("⚡ Processing started...")
                    continue
                    
    except Exception as e:
        logger.error(f"❌ TTS test failed: {e}")


async def test_frame_size_issue():
    """Test WebSocket frame size limit issue."""
    
    logger.info("\n🔍 TESTING WEBSOCKET FRAME SIZE ISSUE")
    logger.info("="*50)
    
    url = "ws://localhost:8000/ws/voice-conversation?preset=speed_optimized&disable_content_filter=true"
    
    try:
        async with websockets.connect(url, max_size=None) as websocket:  # Remove size limit
            logger.info("✅ Connected to WebSocket (no size limit)")
            
            # Wait for connection confirmation
            message = await websocket.recv()
            logger.info("🔗 Connection confirmed")
            
            # Test with larger botox_ar.wav file
            audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"📤 Sending large audio: {len(audio_data)} bytes")
            await websocket.send(audio_data)
            
            # Wait for response
            timeout = 120
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    
                    if data.get("type") == "voice_response":
                        success = data.get("success", False)
                        logger.info(f"🎯 Voice response success: {success}")
                        
                        if success:
                            audio_received = data.get("audio_base64") is not None
                            logger.info(f"🎵 Audio received: {audio_received}")
                            logger.info("✅ Large frame test PASSED")
                        else:
                            logger.error(f"❌ Large frame failed: {data.get('error', 'Unknown error')}")
                        break
                    elif data.get("type") == "processing_started":
                        logger.info("⚡ Processing started...")
                        continue
                        
                except asyncio.TimeoutError:
                    logger.info("⏳ Still waiting...")
                    continue
                    
    except Exception as e:
        logger.error(f"❌ Frame size test failed: {e}")


async def test_preset_comparison():
    """Test different presets to understand why some work and others don't."""
    
    logger.info("\n🔍 TESTING PRESET COMPARISON")
    logger.info("="*50)
    
    presets = ["qwen_optimized", "speed_optimized", "high_quality", None]
    
    for preset in presets:
        preset_name = preset or "default"
        logger.info(f"\n🧪 Testing preset: {preset_name}")
        
        # Build URL
        params = ["disable_content_filter=true", "input_language=auto", "output_language=auto"]
        if preset:
            params.append(f"preset={preset}")
        
        url = f"ws://localhost:8000/ws/voice-conversation?{'&'.join(params)}"
        
        try:
            async with websockets.connect(url) as websocket:
                logger.info(f"✅ Connected for {preset_name}")
                
                # Wait for connection confirmation
                message = await websocket.recv()
                
                # Test with botox.wav
                audio_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav"
                with open(audio_file, "rb") as f:
                    audio_data = f.read()
                
                logger.info(f"📤 Sending audio for {preset_name}")
                await websocket.send(audio_data)
                
                # Wait for response
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=60)
                    data = json.loads(message)
                    
                    if data.get("type") == "voice_response":
                        success = data.get("success", False)
                        audio_received = data.get("audio_base64") is not None
                        
                        logger.info(f"🎯 {preset_name}: Success={success}, Audio={audio_received}")
                        
                        if not success:
                            logger.info(f"   Error: {data.get('error', 'Unknown')}")
                        
                        break
                    elif data.get("type") == "processing_started":
                        continue
                        
        except Exception as e:
            logger.error(f"❌ {preset_name} failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)


async def test_service_logs():
    """Get current service logs to see what's happening."""
    
    logger.info("\n🔍 CHECKING SERVICE LOGS")
    logger.info("="*50)
    
    import subprocess
    
    try:
        # Get last 50 lines of service logs
        result = subprocess.run(
            ["sudo", "journalctl", "-u", "beautyai-api.service", "-n", "50", "--no-pager"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("📄 Recent service logs:")
            print(result.stdout)
        else:
            logger.error(f"❌ Failed to get logs: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Error getting logs: {e}")


async def main():
    """Run all diagnostic tests."""
    
    logger.info("🚀 Starting WebSocket Diagnostic Tests")
    logger.info("="*60)
    
    # Run tests in sequence
    await test_content_filter_issue()
    await test_tts_issue()
    await test_frame_size_issue()
    await test_preset_comparison()
    await test_service_logs()
    
    logger.info("\n🎯 Diagnostic tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
