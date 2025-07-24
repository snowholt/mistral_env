#!/usr/bin/env python3
"""
Test script for Simple Voice WebSocket functionality.

This script tests the simple voice WebSocket endpoint with real audio files
to ensure the voice-to-voice functionality works correctly.

Usage:
    python test_simple_websocket.py

Author: BeautyAI Framework
Date: 2025-07-24
"""

import asyncio
import json
import base64
import time
import websockets
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleWebSocketTester:
    """Test client for Simple Voice WebSocket endpoint."""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def test_audio_file(
        self, 
        audio_file_path: Path, 
        language: str = "ar", 
        voice_type: str = "female",
        test_name: str = None
    ) -> dict:
        """
        Test WebSocket with a specific audio file.
        
        Args:
            audio_file_path: Path to the audio file to test
            language: Language for the conversation ("ar" or "en")
            voice_type: Voice type ("male" or "female")
            test_name: Name for this test case
            
        Returns:
            Dictionary with test results
        """
        if not audio_file_path.exists():
            logger.error(f"Audio file not found: {audio_file_path}")
            return {"success": False, "error": f"File not found: {audio_file_path}"}
        
        test_name = test_name or f"{audio_file_path.stem}_{language}_{voice_type}"
        logger.info(f"🎤 Starting test: {test_name}")
        logger.info(f"   File: {audio_file_path}")
        logger.info(f"   Language: {language}, Voice: {voice_type}")
        
        # Read audio file
        try:
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            logger.info(f"   Audio size: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            return {"success": False, "error": f"Failed to read file: {e}"}
        
        # Build WebSocket URL
        ws_url = f"{self.base_url}/api/v1/ws/simple-voice-chat"
        ws_url += f"?language={language}&voice_type={voice_type}"
        
        start_time = time.time()
        result = {
            "test_name": test_name,
            "audio_file": str(audio_file_path),
            "language": language,
            "voice_type": voice_type,
            "success": False,
            "start_time": start_time,
            "messages_received": [],
            "errors": []
        }
        
        try:
            logger.info(f"   Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                connection_time = time.time()
                logger.info(f"   ✅ Connected in {connection_time - start_time:.2f}s")
                
                # Wait for connection established message
                try:
                    welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    welcome_data = json.loads(welcome_msg)
                    result["messages_received"].append(welcome_data)
                    
                    if welcome_data.get("type") == "connection_established":
                        logger.info(f"   ✅ Connection established: {welcome_data.get('message')}")
                        logger.info(f"   Session ID: {welcome_data.get('session_id')}")
                    else:
                        logger.warning(f"   Unexpected welcome message: {welcome_data}")
                        
                except asyncio.TimeoutError:
                    logger.error("   ❌ Timeout waiting for welcome message")
                    result["errors"].append("Timeout waiting for welcome message")
                    return result
                
                # Send audio data
                logger.info(f"   📤 Sending audio data...")
                await websocket.send(audio_data)
                send_time = time.time()
                
                # Wait for processing response
                logger.info(f"   ⏳ Waiting for voice response...")
                
                processing_started = False
                voice_response_received = False
                
                # Listen for messages with timeout
                async for message in websocket:
                    try:
                        message_data = json.loads(message)
                        result["messages_received"].append(message_data)
                        message_type = message_data.get("type")
                        
                        if message_type == "processing_started":
                            processing_started = True
                            processing_start_time = time.time()
                            logger.info(f"   🔄 Processing started ({processing_start_time - send_time:.2f}s after send)")
                            
                        elif message_type == "voice_response":
                            voice_response_received = True
                            response_time = time.time()
                            total_time = response_time - start_time
                            processing_time = message_data.get("response_time_ms", 0) / 1000
                            
                            logger.info(f"   ✅ Voice response received!")
                            logger.info(f"   Total time: {total_time:.2f}s")
                            logger.info(f"   Server processing: {processing_time:.2f}s") 
                            logger.info(f"   Transcription: {message_data.get('transcription', 'N/A')}")
                            logger.info(f"   Response text: {message_data.get('response_text', 'N/A')}")
                            
                            # Check if we got audio
                            audio_base64 = message_data.get("audio_base64")
                            if audio_base64:
                                # Decode and save audio response
                                try:
                                    audio_bytes = base64.b64decode(audio_base64)
                                    output_file = Path(f"voice_tests/output_{test_name}_{int(time.time())}.wav")
                                    output_file.parent.mkdir(exist_ok=True)
                                    
                                    with open(output_file, 'wb') as f:
                                        f.write(audio_bytes)
                                    
                                    logger.info(f"   💾 Audio saved to: {output_file}")
                                    logger.info(f"   Audio size: {len(audio_bytes)} bytes")
                                    
                                    result["output_audio_file"] = str(output_file)
                                    result["output_audio_size"] = len(audio_bytes)
                                    
                                except Exception as e:
                                    logger.error(f"   ❌ Failed to save audio: {e}")
                                    result["errors"].append(f"Failed to save audio: {e}")
                            else:
                                logger.warning("   ⚠️ No audio data in response")
                                result["errors"].append("No audio data in response")
                            
                            # Mark as successful
                            result["success"] = True
                            result["total_time"] = total_time
                            result["server_processing_time"] = processing_time
                            result["transcription"] = message_data.get("transcription")
                            result["response_text"] = message_data.get("response_text")
                            
                            break  # Exit message loop
                            
                        elif message_type == "error":
                            error_msg = message_data.get("message", "Unknown error")
                            logger.error(f"   ❌ Server error: {error_msg}")
                            result["errors"].append(f"Server error: {error_msg}")
                            break
                            
                        else:
                            logger.info(f"   📥 Message: {message_type}")
                    
                    except json.JSONDecodeError:
                        logger.warning(f"   ⚠️ Non-JSON message received: {message[:100]}...")
                        continue
                    
                    # Timeout for response
                    if time.time() - send_time > 30:  # 30 second timeout
                        logger.error("   ❌ Timeout waiting for voice response")
                        result["errors"].append("Timeout waiting for voice response")
                        break
                
                if not voice_response_received:
                    logger.error(f"   ❌ No voice response received for {test_name}")
                    if not result["errors"]:
                        result["errors"].append("No voice response received")
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"   ❌ WebSocket connection closed: {e}")
            result["errors"].append(f"Connection closed: {e}")
        
        except Exception as e:
            logger.error(f"   ❌ Test failed: {e}")
            result["errors"].append(f"Test exception: {e}")
        
        # Calculate final time
        result["end_time"] = time.time()
        result["total_duration"] = result["end_time"] - result["start_time"]
        
        # Log final result
        if result["success"]:
            logger.info(f"   ✅ Test {test_name} PASSED in {result.get('total_time', 0):.2f}s")
        else:
            logger.error(f"   ❌ Test {test_name} FAILED: {', '.join(result['errors'])}")
        
        self.results.append(result)
        return result
    
    async def run_test_suite(self, audio_files: list, language: str = "ar", voice_type: str = "female"):
        """Run tests on multiple audio files."""
        logger.info(f"🚀 Starting Simple WebSocket Test Suite")
        logger.info(f"   Language: {language}, Voice: {voice_type}")
        logger.info(f"   Files to test: {len(audio_files)}")
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}/{len(audio_files)}: {audio_file.name}")
            logger.info(f"{'='*60}")
            
            await self.test_audio_file(
                audio_file_path=audio_file,
                language=language,
                voice_type=voice_type,
                test_name=f"test_{i}_{audio_file.stem}"
            )
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        successful_tests = [r for r in self.results if r["success"]]
        failed_tests = [r for r in self.results if not r["success"]]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tests: {len(self.results)}")
        logger.info(f"✅ Successful: {len(successful_tests)}")
        logger.info(f"❌ Failed: {len(failed_tests)}")
        
        if successful_tests:
            avg_time = sum(r.get("total_time", 0) for r in successful_tests) / len(successful_tests)
            logger.info(f"⏱️  Average response time: {avg_time:.2f}s")
            
            logger.info(f"\n✅ SUCCESSFUL TESTS:")
            for result in successful_tests:
                logger.info(f"   - {result['test_name']}: {result.get('total_time', 0):.2f}s")
        
        if failed_tests:
            logger.info(f"\n❌ FAILED TESTS:")
            for result in failed_tests:
                errors = ', '.join(result['errors'][:2])  # Show first 2 errors
                logger.info(f"   - {result['test_name']}: {errors}")
        
        logger.info(f"{'='*80}")


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Simple Voice WebSocket")
    parser.add_argument("--url", default="ws://localhost:8000", help="WebSocket base URL")
    parser.add_argument("--language", default="ar", choices=["ar", "en"], help="Language to test")
    parser.add_argument("--voice", default="female", choices=["male", "female"], help="Voice type")
    parser.add_argument("--file", help="Test single audio file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = SimpleWebSocketTester(base_url=args.url)
    
    if args.file:
        # Test single file
        audio_file = Path(args.file)
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return
        
        await tester.test_audio_file(
            audio_file_path=audio_file,
            language=args.language,
            voice_type=args.voice
        )
    else:
        # Test the specified files
        base_path = Path("voice_tests/input_test_questions")
        
        # Test files as specified by user
        test_files = []
        
        # Add botox_ar.webm if it exists
        botox_ar_webm = base_path / "botox_ar.webm"
        if botox_ar_webm.exists():
            test_files.append(botox_ar_webm)
        
        # Add botox.wav if it exists
        botox_wav = base_path / "botox.wav"
        if botox_wav.exists():
            test_files.append(botox_wav)
        
        if not test_files:
            logger.error("No test files found! Expected:")
            logger.error(f"  - {botox_ar_webm}")
            logger.error(f"  - {botox_wav}")
            return
        
        # Run test suite
        await tester.run_test_suite(
            audio_files=test_files,
            language=args.language,
            voice_type=args.voice
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
