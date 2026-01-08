#!/usr/bin/env python3
"""
WebSocket test client for PABX API
Tests real-time event broadcasting
"""

import asyncio
import websockets
import json
from datetime import datetime


async def test_websocket():
    """Connect to WebSocket and listen for events"""
    uri = "ws://192.168.100.39:8080/ws"
    
    print(f"[{datetime.now()}] Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"[{datetime.now()}] Connected!")
        print("Listening for events... (Press Ctrl+C to quit)")
        print("-" * 60)
        
        try:
            while True:
                # Receive message
                message = await websocket.recv()
                data = json.loads(message)
                
                # Handle different event types
                event_type = data.get("type", "unknown")
                timestamp = data.get("timestamp", datetime.now().isoformat())
                
                if event_type == "ping":
                    # Respond to ping with pong
                    pong_response = {"type": "pong", "timestamp": datetime.now().timestamp()}
                    await websocket.send(json.dumps(pong_response))
                    print(f"[{timestamp}] 🏓 Ping received, sent pong")
                
                elif event_type == "call_incoming":
                    call_data = data.get("data", {})
                    print(f"\n[{timestamp}] 📞 INCOMING CALL")
                    print(f"  From: {call_data.get('from_user')}")
                    print(f"  To: {call_data.get('to_user')}")
                    print(f"  Call ID: {call_data.get('call_id')}")
                    print(f"  State: {call_data.get('state')}")
                
                elif event_type == "call_answered":
                    call_data = data.get("data", {})
                    print(f"\n[{timestamp}] ✅ CALL ANSWERED")
                    print(f"  Call ID: {call_data.get('call_id')}")
                    print(f"  RTP Port: {call_data.get('local_rtp_port')}")
                    print(f"  Remote: {call_data.get('remote_rtp_ip')}:{call_data.get('remote_rtp_port')}")
                
                elif event_type == "call_ended":
                    call_data = data.get("data", {})
                    duration = 0
                    if call_data.get('answered_at') and call_data.get('ended_at'):
                        answered = datetime.fromisoformat(call_data['answered_at'].replace('Z', '+00:00'))
                        ended = datetime.fromisoformat(call_data['ended_at'].replace('Z', '+00:00'))
                        duration = (ended - answered).total_seconds()
                    
                    print(f"\n[{timestamp}] 🔚 CALL ENDED")
                    print(f"  Call ID: {call_data.get('call_id')}")
                    print(f"  Duration: {duration:.1f}s")
                    print(f"  Recording: {call_data.get('recording_file', 'N/A')}")
                
                else:
                    print(f"\n[{timestamp}] 📡 Event: {event_type}")
                    print(f"  Data: {json.dumps(data, indent=2)}")
                
                print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n\nDisconnecting...")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\nExiting...")
