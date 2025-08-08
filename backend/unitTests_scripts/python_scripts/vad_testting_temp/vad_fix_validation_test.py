#!/usr/bin/env python3
"""
Test script to validate the VAD WebM chunk accumulation fix.

This tests the specific fix:
1. WebM format detection and VAD bypass  
2. Time-based chunk accumulation (30 chunks)
3. Complete WebM processing without individual chunk failures
4. No infinite loop from "unclear audio" responses
"""

import asyncio
import json
import time
import websockets
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VADFixValidationTest:
    def __init__(self):
        self.websocket = None
        self.messages_received = []
        self.connection_id = None
        self.voice_responses = []
        self.unclear_audio_count = 0
        self.webm_errors = 0
        self.vad_errors = 0
        
    async def connect(self):
        """Connect to the WebSocket endpoint"""
        try:
            uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
            self.websocket = await websockets.connect(uri)
            
            # Wait for connection established message
            message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(message)
            
            if data.get("type") == "connection_established":
                self.connection_id = data.get("connection_id")
                logger.info(f"✅ Connected to WebSocket: {self.connection_id}")
                logger.info(f"🔧 VAD enabled: {data.get('config', {}).get('vad_config') is not None}")
                return True
            else:
                logger.error(f"❌ Unexpected initial message: {data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def send_webm_chunks_slowly(self):
        """Send WebM file as chunks with realistic timing to test accumulation"""
        webm_file = "/home/lumi/beautyai/voice_tests/input_test_questions/webm/q1.webm"
        
        with open(webm_file, 'rb') as f:
            webm_data = f.read()
        
        logger.info(f"📦 Sending WebM file as slow chunks: {len(webm_data)} bytes")
        
        # Send exactly 30 chunks to trigger the fix's time-based processing
        chunk_size = len(webm_data) // 30
        chunks_sent = 0
        
        for i in range(30):
            start_idx = i * chunk_size
            if i == 29:  # Last chunk gets remainder
                chunk = webm_data[start_idx:]
            else:
                chunk = webm_data[start_idx:start_idx + chunk_size]
            
            await self.websocket.send(chunk)
            chunks_sent += 1
            logger.info(f"📤 Sent chunk {chunks_sent}/30: {len(chunk)} bytes")
            
            # Simulate realistic MediaRecorder timing
            await asyncio.sleep(0.1)  # 100ms between chunks
        
        logger.info(f"✅ Sent all 30 chunks - fix should trigger processing now")
        return chunks_sent
    
    async def listen_and_analyze(self, timeout=20):
        """Listen for responses and analyze for fix validation"""
        logger.info(f"👂 Listening for responses to validate fix (timeout: {timeout}s)...")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            self.messages_received.append(data)
                            
                            msg_type = data.get("type", "unknown")
                            logger.info(f"📨 Received: {msg_type}")
                            
                            # Track different message types
                            if msg_type == "voice_response":
                                self.voice_responses.append(data)
                                
                                # Check for "unclear audio" content
                                if "unclear audio" in str(data).lower():
                                    self.unclear_audio_count += 1
                                    logger.warning(f"⚠️ UNCLEAR AUDIO detected in response #{len(self.voice_responses)}")
                            
                            elif msg_type == "processing_started":
                                logger.info("🏭 Processing started - this is good!")
                            
                            elif msg_type == "error":
                                error_msg = str(data).lower()
                                if "webm" in error_msg or "ebml" in error_msg:
                                    self.webm_errors += 1
                                    logger.warning(f"⚠️ WebM error detected: {data.get('message', '')}")
                                if "input audio chunk is too short" in error_msg:
                                    self.vad_errors += 1
                                    logger.warning(f"⚠️ VAD chunk error detected")
                                    
                        except json.JSONDecodeError:
                            logger.info(f"📨 Non-JSON message: {len(message)} bytes")
                            self.messages_received.append({"raw_message": message[:100]})
                    
                    else:
                        # Binary message (audio response)
                        logger.info(f"🎵 Binary audio response: {len(message)} bytes")
                        self.messages_received.append({"audio_response": f"{len(message)} bytes"})
                
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error listening for responses: {e}")
    
    async def test_vad_fix(self):
        """Test the VAD fix with WebM chunk accumulation"""
        logger.info("🧪 Testing VAD fix with WebM chunk accumulation...")
        
        # Connect 
        if not await self.connect():
            return {"success": False, "error": "Connection failed"}
        
        # Start listening in background
        listen_task = asyncio.create_task(self.listen_and_analyze(25))
        
        # Wait for connection to stabilize
        await asyncio.sleep(1)
        
        # Send chunks slowly to test accumulation
        chunks_sent = await self.send_webm_chunks_slowly()
        
        # Wait for processing to complete
        logger.info("⏳ Waiting for backend processing (fix should prevent infinite loop)...")
        await asyncio.sleep(15)
        
        # Stop listening
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        
        await self.websocket.close()
        
        # Analyze results
        results = {
            "success": True,
            "connection_id": self.connection_id,
            "chunks_sent": chunks_sent,
            "total_messages": len(self.messages_received),
            "voice_responses": len(self.voice_responses),
            "unclear_audio_count": self.unclear_audio_count,
            "webm_errors": self.webm_errors,
            "vad_errors": self.vad_errors
        }
        
        # Determine if fix is working
        results["fix_working"] = (
            results["unclear_audio_count"] <= 1 and  # At most 1 unclear audio (initial processing)
            results["webm_errors"] == 0 and          # No WebM decoding errors
            results["vad_errors"] == 0 and           # No VAD chunk errors
            results["voice_responses"] <= 2           # At most 1-2 responses, not infinite loop
        )
        
        results["infinite_loop_prevented"] = results["unclear_audio_count"] <= 1
        
        return results
    
    def print_results(self, results):
        """Print detailed test results"""
        print("\n" + "="*80)
        print("🧪 VAD FIX VALIDATION TEST RESULTS")  
        print("="*80)
        
        if results.get("success"):
            print(f"🔌 Connection ID: {results.get('connection_id', 'unknown')}")
            print(f"📦 WebM chunks sent: {results['chunks_sent']}")
            print(f"📨 Total messages: {results['total_messages']}")
            print(f"🎤 Voice responses: {results['voice_responses']}")
            print(f"⚠️ Unclear audio responses: {results['unclear_audio_count']}")
            print(f"❌ WebM parsing errors: {results['webm_errors']}")
            print(f"❌ VAD chunk errors: {results['vad_errors']}")
            
            print("\n" + "-"*60)
            print("🎯 FIX VALIDATION:")
            
            if results["fix_working"]:
                print("✅ VAD FIX IS WORKING!")
                print("   - No infinite loop of unclear audio responses")
                print("   - No WebM chunk parsing errors")  
                print("   - No VAD processing errors on individual chunks")
                print("   - Reasonable number of voice responses")
            else:
                print("❌ VAD FIX IS NOT WORKING!")
                if results["unclear_audio_count"] > 1:
                    print(f"   - Too many unclear audio responses: {results['unclear_audio_count']}")
                if results["webm_errors"] > 0:
                    print(f"   - WebM parsing errors detected: {results['webm_errors']}")
                if results["vad_errors"] > 0:
                    print(f"   - VAD chunk errors still occurring: {results['vad_errors']}")
                if results["voice_responses"] > 2:
                    print(f"   - Too many responses (possible loop): {results['voice_responses']}")
                    
            print(f"\n🔄 Infinite loop prevented: {'✅ YES' if results['infinite_loop_prevented'] else '❌ NO'}")
            
        else:
            print(f"❌ Test failed: {results.get('error')}")
        
        print("="*80)

async def main():
    """Run the VAD fix validation test"""
    print("🚀 Starting VAD Fix Validation Test...")
    print("This test validates that the WebM chunk accumulation fix prevents")  
    print("the infinite loop caused by VAD processing individual chunks.\n")
    
    test = VADFixValidationTest()
    results = await test.test_vad_fix()
    test.print_results(results)
    
    # Save results
    results_file = Path("vad_fix_validation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Log results to debug file
    with open("/home/lumi/beautyai/vad_debug_log.txt", "a") as log:
        log.write(f"\n=== VAD FIX VALIDATION TEST ===\n")
        log.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Fix working: {results.get('fix_working', False)}\n")
        log.write(f"Infinite loop prevented: {results.get('infinite_loop_prevented', False)}\n")
        log.write(f"Voice responses: {results.get('voice_responses', 0)}\n")
        log.write(f"Unclear audio count: {results.get('unclear_audio_count', 0)}\n")
        log.write(f"WebM errors: {results.get('webm_errors', 0)}\n")
        log.write(f"VAD errors: {results.get('vad_errors', 0)}\n\n")
    
    return results.get("fix_working", False)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
