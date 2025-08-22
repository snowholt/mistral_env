#!/usr/bin/env python3
"""
Focused test to understand conversation bleeding root cause.
Uploads two different files with a long pause and examines exact transcripts.
"""

import asyncio
import websockets
import json
import time
from pathlib import Path

async def focused_bleeding_test():
    print("Starting focused conversation bleeding test...")
    
    # Two different WebM files
    file1 = Path("/home/lumi/beautyai/voice_tests/input_test_questions/webm/greeting.webm")
    file2 = Path("/home/lumi/beautyai/voice_tests/input_test_questions/webm/botox.webm")
    
    if not (file1.exists() and file2.exists()):
        print(f"Error: Test files not found")
        return
    
    uri = "ws://127.0.0.1:8000/api/v1/ws/streaming-voice?language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            print(f"Ready: {ready_data['type']}")
            session_id = ready_data.get('session_id', 'unknown')
            print(f"Session ID: {session_id}")
            
            # Upload first file
            print(f"\n=== UPLOADING FILE 1: {file1.name} ===")
            with open(file1, 'rb') as f:
                webm_data1 = f.read()
            await websocket.send(webm_data1)
            print(f"File 1 uploaded ({len(webm_data1)} bytes)")
            
            # Listen for first file transcription
            transcripts = []
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'final_transcript':
                        transcript = data.get('text', '')
                        utterance_idx = data.get('utterance_index', 'N/A')
                        print(f"📝 FINAL TRANSCRIPT 1: '{transcript}' (utterance {utterance_idx})")
                        transcripts.append(('file1', transcript, utterance_idx))
                        break
                    elif msg_type == 'partial_transcript':
                        partial = data.get('text', '')
                        print(f"📄 partial: '{partial}'")
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    return
            
            # Long pause between files
            print(f"\n=== PAUSING 5 SECONDS ===")
            await asyncio.sleep(5)
            
            # Upload second file  
            print(f"\n=== UPLOADING FILE 2: {file2.name} ===")
            with open(file2, 'rb') as f:
                webm_data2 = f.read()
            await websocket.send(webm_data2)
            print(f"File 2 uploaded ({len(webm_data2)} bytes)")
            
            # Listen for second file transcription
            start_time = time.time()
            while time.time() - start_time < 15:  # Wait longer
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'final_transcript':
                        transcript = data.get('text', '')
                        utterance_idx = data.get('utterance_index', 'N/A')
                        print(f"📝 FINAL TRANSCRIPT 2: '{transcript}' (utterance {utterance_idx})")
                        transcripts.append(('file2', transcript, utterance_idx))
                        break
                    elif msg_type == 'partial_transcript':
                        partial = data.get('text', '')
                        print(f"📄 partial: '{partial}'")
                    elif msg_type in ['endpoint_event', 'info', 'ingest_mode']:
                        print(f"🔔 {msg_type}: {data}")
                        
                except asyncio.TimeoutError:
                    print(".", end="", flush=True)  # Show we're waiting
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    return
            
            # Analyze results
            print(f"\n=== ANALYSIS ===")
            if len(transcripts) >= 2:
                file1_transcript = transcripts[0][1]
                file2_transcript = transcripts[1][1]
                
                print(f"File 1 transcript: '{file1_transcript}'")
                print(f"File 2 transcript: '{file2_transcript}'")
                
                if file1_transcript in file2_transcript and file1_transcript != file2_transcript:
                    print(f"🚨 CONVERSATION BLEEDING DETECTED!")
                    print(f"   File 1 text appears in File 2 transcript")
                    print(f"   Expected: Only File 2 content")
                    print(f"   Actual: File 1 + File 2 content")
                else:
                    print(f"✅ No conversation bleeding detected")
            else:
                print(f"❌ Insufficient transcripts received: {len(transcripts)}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(focused_bleeding_test())