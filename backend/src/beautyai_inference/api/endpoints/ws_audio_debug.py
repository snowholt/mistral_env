"""
Simple WebSocket Audio Debug Endpoint

Receives audio via WebSocket without WebRTC complexity.
"""

import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path
import time

logger = logging.getLogger(__name__)

ws_audio_debug_router = APIRouter()

@ws_audio_debug_router.websocket("/api/v1/ws/audio-debug")
async def websocket_audio_debug(websocket: WebSocket):
    """Simple WebSocket endpoint to receive audio data."""
    await websocket.accept()
    
    session_id = f"ws_debug_{int(time.time())}"
    logger.info(f"[WS-DEBUG] Session {session_id} connected")
    
    # Create output directory
    output_dir = Path("/home/lumi/beautyai/logs/websocket_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{session_id}.webm"
    bytes_received = 0
    chunks_received = 0
    
    try:
        await websocket.send_text(f"Connected! Session: {session_id}")
        
        with open(output_file, 'wb') as f:
            while True:
                # Receive binary audio data
                data = await websocket.receive_bytes()
                f.write(data)
                
                bytes_received += len(data)
                chunks_received += 1
                
                if chunks_received % 10 == 0:
                    logger.info(f"[WS-DEBUG] {session_id}: {chunks_received} chunks, {bytes_received} bytes")
                    await websocket.send_text(f"Received {chunks_received} chunks ({bytes_received} bytes)")
                
    except WebSocketDisconnect:
        logger.info(f"[WS-DEBUG] {session_id} disconnected: {chunks_received} chunks, {bytes_received} bytes total")
        await websocket.close()
    except Exception as e:
        logger.error(f"[WS-DEBUG] {session_id} error: {e}")
        await websocket.close()
    
    logger.info(f"[WS-DEBUG] {session_id} saved to: {output_file}")
    logger.info(f"[WS-DEBUG] {session_id} final stats: {chunks_received} chunks, {bytes_received} bytes")
