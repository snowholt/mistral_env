#!/usr/bin/env python3
"""
Test script to validate backend chunk accumulation fix.

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
        
    async def connect(self):
        """Connect to the WebSocket endpoint"""
        try:
            self.websocket = await websockets.connect(
                "ws://localhost:8000/ws/simple-voice-chat",
                extra_headers={
                    "Origin": "http://localhost:3000"
                }
            )
            logger.info("✅ Connected to WebSocket")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def send_chunk_sequence(self, chunk_count=10):
        """Send a sequence of mock WebM chunks to test accumulation"""
        logger.info(f"📦 Sending {chunk_count} mock WebM chunks...")
        
        # Create mock WebM header for first chunk
        webm_header = b'\x1a\x45\xdf\xa3'  # WebM signature
        mock_chunk_data = b'\x00' * 1024  # 1KB of mock audio data
        
        for i in range(chunk_count):
            chunk_data = webm_header + mock_chunk_data if i == 0 else mock_chunk_data
            
            # Send as binary message
            await self.websocket.send(chunk_data)
            logger.info(f"📤 Sent chunk {i+1}/{chunk_count} ({len(chunk_data)} bytes)")
            
            # Small delay between chunks to simulate real MediaRecorder
            await asyncio.sleep(0.1)
    
    async def listen_for_responses(self, timeout=30):
        """Listen for backend responses and analyze them"""
        logger.info(f"👂 Listening for responses (timeout: {timeout}s)...")
        
        start_time = time.time()
        duplicate_count = 0
        unique_responses = set()
        
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
                            
                            # Check for voice responses
                            if data.get("type") == "voice_response":
                                turn_id = data.get("turn_id", "unknown")
                                
                                if turn_id in unique_responses:
                                    duplicate_count += 1
                                    logger.warning(f"⚠️ DUPLICATE RESPONSE detected: {turn_id}")
                                else:
                                    unique_responses.add(turn_id)
                                    logger.info(f"✅ New unique response: {turn_id}")
                            
                            # Log all message types
                            msg_type = data.get("type", "unknown")
                            logger.info(f"📨 Received: {msg_type}")
                            
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Non-JSON message: {message[:100]}...")
                
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error listening for responses: {e}")
        
        return {
            "total_messages": len(self.messages_received),
            "unique_responses": len(unique_responses),
            "duplicate_count": duplicate_count,
            "messages": self.messages_received
        }
    
    async def test_chunk_accumulation(self):
        """Test the chunk accumulation functionality"""
        logger.info("🧪 Testing chunk accumulation...")
        
        # Connect to WebSocket
        if not await self.connect():
            return {"success": False, "error": "Failed to connect"}
        
        # Start listening in background
        listen_task = asyncio.create_task(self.listen_for_responses(15))
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(1)
        
        # Send chunk sequence
        await self.send_chunk_sequence(15)
        
        # Wait for processing and responses
        logger.info("⏳ Waiting for backend processing...")
        await asyncio.sleep(10)
        
        # Stop listening and get results
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        
        # Analyze results
        results = {
            "success": True,
            "chunks_sent": 15,
            "messages_received": len(self.messages_received),
            "duplicate_responses": 0,
            "chunk_accumulation_working": False,
            "backend_processing_properly": False
        }
        
        # Check for duplicate responses
        voice_responses = [msg for msg in self.messages_received if msg.get("type") == "voice_response"]
        turn_ids = [resp.get("turn_id") for resp in voice_responses]
        unique_turn_ids = set(turn_ids)
        
        results["duplicate_responses"] = len(turn_ids) - len(unique_turn_ids)
        results["total_voice_responses"] = len(voice_responses)
        results["unique_voice_responses"] = len(unique_turn_ids)
        
        # Check for chunk accumulation indicators
        vad_updates = [msg for msg in self.messages_received if msg.get("type") == "vad_update"]
        results["vad_updates_received"] = len(vad_updates)
        
        # Check if we got processing messages indicating buffering
        processing_msgs = [msg for msg in self.messages_received 
                          if "buffer" in str(msg).lower() or "chunk" in str(msg).lower()]
        results["buffering_messages"] = len(processing_msgs)
        
        # Evaluate success criteria
        results["chunk_accumulation_working"] = (
            results["duplicate_responses"] == 0 and
            results["total_voice_responses"] <= 2  # Should be 1-2 responses max, not 15
        )
        
        results["backend_processing_properly"] = (
            results["vad_updates_received"] > 0 or
            results["buffering_messages"] > 0
        )
        
        await self.websocket.close()
        return results
    
    def print_results(self, results):
        """Print test results in a readable format"""
        print("\n" + "="*60)
        print("🧪 BACKEND CHUNK ACCUMULATION TEST RESULTS")
        print("="*60)
        
        if results.get("success"):
            print(f"📦 Chunks sent: {results['chunks_sent']}")
            print(f"📨 Total messages received: {results['messages_received']}")
            print(f"🎤 Voice responses received: {results['total_voice_responses']}")
            print(f"🔄 Duplicate responses: {results['duplicate_responses']}")
            print(f"📊 VAD updates: {results['vad_updates_received']}")
            print(f"📋 Buffering messages: {results['buffering_messages']}")
            
            print("\n" + "-"*40)
            print("EVALUATION:")
            
            if results["chunk_accumulation_working"]:
                print("✅ Chunk accumulation: WORKING")
                print("   - No duplicate responses detected")
                print("   - Reasonable number of voice responses")
            else:
                print("❌ Chunk accumulation: FAILED")
                print(f"   - {results['duplicate_responses']} duplicate responses")
                print(f"   - {results['total_voice_responses']} total responses (should be 1-2)")
            
            if results["backend_processing_properly"]:
                print("✅ Backend processing: WORKING")
                print("   - VAD updates or buffering messages detected")
            else:
                print("⚠️ Backend processing: UNCLEAR")
                print("   - No clear indication of proper buffering")
                
        else:
            print(f"❌ Test failed: {results.get('error')}")
        
        print("="*60)

async def main():
    """Run the chunk accumulation test"""
    print("🚀 Starting Backend Chunk Accumulation Test...")
    print("This will test if the backend properly accumulates WebM chunks")
    print("instead of processing individual MediaRecorder fragments.\n")
    
    test = ChunkAccumulationTest()
    results = await test.test_chunk_accumulation()
    test.print_results(results)
    
    # Save detailed results
    results_file = Path("chunk_accumulation_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
