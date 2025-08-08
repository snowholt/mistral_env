#!/usr/bin/env python3
"""
UPDATED: Test script to validate backend chunk accumulation fix.

This script tests:
1. WebM chunk accumulation and buffering
2. Complete audio processing on VAD turn complete
3. Proper preventing of individual chunk decoding
4. Backend duplicate response elimination
"""

import asyncio
import json
import time
import websockets
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkAccumulationTest:
    def __init__(self):
        self.websocket = None
        self.test_results = []
        self.messages_received = []
        self.connection_id = None
        
    async def connect(self):
        """Connect to the WebSocket endpoint with proper parameters"""
        try:
            uri = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"
            self.websocket = await websockets.connect(uri)
            
            # Wait for connection established message
            message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(message)
            
            if data.get("type") == "connection_established":
                self.connection_id = data.get("connection_id")
                logger.info(f"✅ Connected to WebSocket: {self.connection_id}")
                return True
            else:
                logger.error(f"❌ Unexpected initial message: {data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def send_real_webm_as_chunks(self):
        """Send real WebM file as chunks to simulate MediaRecorder behavior"""
        webm_file = "/home/lumi/beautyai/voice_tests/input_test_questions/webm/q1.webm"
        
        with open(webm_file, 'rb') as f:
            webm_data = f.read()
        
        logger.info(f"📦 Sending real WebM file as chunks: {len(webm_data)} bytes")
        
        # Split into chunks like MediaRecorder would do
        chunk_size = 1024
        chunks_sent = 0
        
        for i in range(0, len(webm_data), chunk_size):
            chunk = webm_data[i:i+chunk_size]
            await self.websocket.send(chunk)
            chunks_sent += 1
            logger.info(f"📤 Sent chunk {chunks_sent}: {len(chunk)} bytes")
            
            # Delay to simulate real recording
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Sent {chunks_sent} chunks total from real WebM file")
        return chunks_sent
    
    async def listen_for_responses(self, timeout=15):
        """Listen for backend responses and analyze them"""
        logger.info(f"👂 Listening for responses (timeout: {timeout}s)...")
        
        start_time = time.time()
        duplicate_count = 0
        unique_responses = set()
        voice_responses = []
        vad_updates = []
        processing_messages = []
        error_messages = []
        
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
                            
                            # Categorize messages
                            if msg_type == "voice_response":
                                turn_id = data.get("turn_id", data.get("message_id", "unknown"))
                                voice_responses.append(turn_id)
                                
                                if turn_id in unique_responses:
                                    duplicate_count += 1
                                    logger.warning(f"⚠️ DUPLICATE RESPONSE detected: {turn_id}")
                                else:
                                    unique_responses.add(turn_id)
                                    logger.info(f"✅ New unique response: {turn_id}")
                            
                            elif msg_type == "vad_update":
                                vad_updates.append(data)
                                logger.info(f"🔊 VAD update: {data.get('state', 'unknown')}")
                            
                            elif msg_type == "processing_started":
                                processing_messages.append(data)
                                logger.info("🏭 Processing started")
                                
                            elif msg_type == "error":
                                error_messages.append(data)
                                logger.warning(f"⚠️ Error message: {data.get('message', 'unknown')}")
                            
                            else:
                                logger.info(f"📨 Received: {msg_type}")
                            
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Non-JSON message: {message[:100]}...")
                            self.messages_received.append({"raw_message": message[:200]})
                    
                    else:
                        # Binary message
                        logger.info(f"📨 Binary message: {len(message)} bytes")
                        self.messages_received.append({"binary_data": f"{len(message)} bytes"})
                
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error listening for responses: {e}")
        
        return {
            "total_messages": len(self.messages_received),
            "voice_responses": voice_responses,
            "unique_responses": len(unique_responses),
            "duplicate_count": duplicate_count,
            "vad_updates": len(vad_updates),
            "processing_messages": len(processing_messages),
            "error_messages": len(error_messages),
            "messages": self.messages_received
        }
    
    async def test_chunk_accumulation(self):
        """Test the chunk accumulation functionality with real WebM data"""
        logger.info("🧪 Testing chunk accumulation with real WebM file...")
        
        # Connect to WebSocket
        if not await self.connect():
            return {"success": False, "error": "Failed to connect"}
        
        # Start listening in background
        listen_task = asyncio.create_task(self.listen_for_responses(20))
        
        # Wait for connection to stabilize
        await asyncio.sleep(1)
        
        # Send real WebM file as chunks
        chunks_sent = await self.send_real_webm_as_chunks()
        
        # Wait for processing
        logger.info("⏳ Waiting for backend processing...")
        await asyncio.sleep(10)
        
        # Cancel listening and get results
        listen_task.cancel()
        try:
            response_data = await listen_task
        except asyncio.CancelledError:
            response_data = await self.listen_for_responses(0)  # Get current state
        
        await self.websocket.close()
        
        # Analyze results
        results = {
            "success": True,
            "chunks_sent": chunks_sent,
            "connection_id": self.connection_id,
            **response_data
        }
        
        # Evaluate if chunk accumulation is working
        results["chunk_accumulation_working"] = (
            results["duplicate_count"] == 0 and
            len(results["voice_responses"]) <= 2  # Should get 1-2 responses max
        )
        
        results["backend_processing_properly"] = (
            results["vad_updates"] > 0 or
            results["processing_messages"] > 0
        )
        
        # Check for the specific error patterns we identified
        webm_errors = 0
        vad_errors = 0
        unclear_audio = 0
        
        for msg in self.messages_received:
            if isinstance(msg, dict):
                msg_str = str(msg).lower()
                if "ebml" in msg_str or "webm" in msg_str:
                    webm_errors += 1
                if "input audio chunk is too short" in msg_str:
                    vad_errors += 1
                if "unclear audio" in msg_str:
                    unclear_audio += 1
        
        results["webm_parsing_errors"] = webm_errors
        results["vad_chunk_errors"] = vad_errors  
        results["unclear_audio_fallbacks"] = unclear_audio
        
        return results
    
    def print_results(self, results):
        """Print test results in a readable format"""
        print("\n" + "="*70)
        print("🧪 BACKEND CHUNK ACCUMULATION TEST RESULTS")
        print("="*70)
        
        if results.get("success"):
            print(f"🔌 Connection ID: {results.get('connection_id', 'unknown')}")
            print(f"📦 WebM chunks sent: {results['chunks_sent']}")
            print(f"📨 Total messages received: {results['total_messages']}")
            print(f"🎤 Voice responses received: {len(results['voice_responses'])}")
            print(f"🔄 Duplicate responses: {results['duplicate_count']}")
            print(f"📊 VAD updates: {results['vad_updates']}")
            print(f"🏭 Processing messages: {results['processing_messages']}")
            print(f"⚠️ Error messages: {results['error_messages']}")
            
            print(f"\n🔍 ERROR PATTERN ANALYSIS:")
            print(f"❌ WebM parsing errors: {results['webm_parsing_errors']}")
            print(f"❌ VAD chunk errors: {results['vad_chunk_errors']}")
            print(f"❌ Unclear audio fallbacks: {results['unclear_audio_fallbacks']}")
            
            print("\n" + "-"*50)
            print("🎯 EVALUATION:")
            
            if results["chunk_accumulation_working"]:
                print("✅ Chunk accumulation: WORKING")
                print("   - No duplicate responses detected")
                print("   - Reasonable number of voice responses")
            else:
                print("❌ Chunk accumulation: FAILED")
                print(f"   - {results['duplicate_count']} duplicate responses")
                print(f"   - {len(results['voice_responses'])} total responses (should be 1-2)")
            
            if results["backend_processing_properly"]:
                print("✅ Backend processing: WORKING")
                print("   - VAD updates or processing messages detected")
            else:
                print("⚠️ Backend processing: UNCLEAR")
                print("   - No clear indication of proper buffering")
            
            # Key diagnostic information
            if results['webm_parsing_errors'] > 0 or results['vad_chunk_errors'] > 0:
                print("🚨 CRITICAL ISSUES DETECTED:")
                if results['vad_chunk_errors'] > 0:
                    print("   - VAD is still processing individual chunks")
                if results['webm_parsing_errors'] > 0:
                    print("   - WebM chunk parsing still failing")
                    
                print("   ➡️ CHUNK ACCUMULATION IS NOT WORKING PROPERLY")
            else:
                print("✅ No critical chunk processing errors detected")
                
        else:
            print(f"❌ Test failed: {results.get('error')}")
        
        print("="*70)

async def main():
    """Run the chunk accumulation test with real WebM data"""
    print("🚀 Starting Real WebM Chunk Accumulation Test...")
    print("This tests if the backend properly accumulates WebM chunks")
    print("from a real recorded voice file instead of processing individual fragments.\n")
    
    test = ChunkAccumulationTest()
    results = await test.test_chunk_accumulation()
    test.print_results(results)
    
    # Save detailed results
    results_file = Path("real_webm_chunk_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Add to debug log
    with open("/home/lumi/beautyai/vad_debug_log.txt", "a") as log:
        log.write(f"\n=== CHUNK ACCUMULATION TEST RESULTS ===\n")
        log.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Chunks sent: {results.get('chunks_sent', 0)}\n")
        log.write(f"Voice responses: {len(results.get('voice_responses', []))}\n")
        log.write(f"Duplicates: {results.get('duplicate_count', 0)}\n")
        log.write(f"WebM errors: {results.get('webm_parsing_errors', 0)}\n")
        log.write(f"VAD errors: {results.get('vad_chunk_errors', 0)}\n")
        log.write(f"Chunk accumulation working: {results.get('chunk_accumulation_working', False)}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
