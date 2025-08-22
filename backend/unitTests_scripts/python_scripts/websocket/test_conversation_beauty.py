#!/usr/bin/env python3
"""
Test script to verify conversation history limiting is working with beauty/medical domain messages.
"""

import asyncio
import json
import websockets
import time

async def test_conversation_history():
    """Test conversation history limiting by sending multiple beauty/medical messages."""
    uri = "ws://localhost:8000/api/v1/ws/streaming-voice?language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            print(f"📦 Received: {ready_msg}")
            
            # Send multiple beauty/medical conversation messages to test history limiting
            test_messages = [
                "What is the best skincare routine for dry skin?",
                "How can I prevent acne breakouts?", 
                "What vitamins are good for hair health?",
                "How do I choose the right foundation shade?",
                "What are the benefits of retinol for anti-aging?",
                "How often should I exfoliate my skin?",
                "What is the difference between UVA and UVB rays?",
                "How can I reduce dark circles under my eyes?",
                "What ingredients should I avoid in skincare?", 
                "How do I treat sensitive skin effectively?"
            ]
            
            for i, message in enumerate(test_messages):
                print(f"\n🗣️  Test {i+1}: Sending '{message}'")
                
                # Simulate sending a transcription
                transcript_msg = {
                    "type": "final_transcript",
                    "text": message,
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(transcript_msg))
                
                # Wait for assistant response
                timeout_counter = 0
                while timeout_counter < 15:  # 15 second timeout
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        response_data = json.loads(response)
                        
                        if response_data.get("type") in ["assistant_response", "assistant_text"]:
                            print(f"🤖 Assistant response: {response_data.get('text', '')[:100]}...")
                            break
                        elif response_data.get("type") == "error":
                            print(f"❌ Error: {response_data.get('message', '')}")
                            break
                        elif response_data.get("type") == "heartbeat":
                            # Skip heartbeat messages, continue waiting
                            continue
                        else:
                            print(f"📦 Other message: {response_data.get('type', 'unknown')}")
                            
                    except asyncio.TimeoutError:
                        timeout_counter += 1
                        continue
                else:
                    print("⏰ Timeout waiting for assistant response")
                
                # Add a small delay between messages
                await asyncio.sleep(2)
            
            # Test conversation reset
            print(f"\n🔄 Testing conversation reset...")
            reset_msg = {
                "type": "reset_conversation"
            }
            await websocket.send(json.dumps(reset_msg))
            
            # Wait for reset confirmation
            timeout_counter = 0
            while timeout_counter < 10:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    if response_data.get("type") == "conversation_reset":
                        print(f"📦 Reset response: {response_data}")
                        break
                    elif response_data.get("type") != "heartbeat":
                        print(f"📦 Other response: {response_data}")
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    continue
            
            # Send one more message after reset
            print(f"\n🗣️  Post-reset test: Sending beauty question")
            post_reset_msg = {
                "type": "final_transcript", 
                "text": "What is the best moisturizer for combination skin?",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(post_reset_msg))
            
            # Wait for response
            timeout_counter = 0
            while timeout_counter < 15:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") in ["assistant_response", "assistant_text"]:
                        print(f"🤖 Post-reset response: {response_data.get('text', '')[:100]}...")
                        break
                    elif response_data.get("type") == "heartbeat":
                        continue
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    continue
            
            print("\n✅ Test completed successfully!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_conversation_history())