"""
FastAPI REST API
Web API for PABX system control and monitoring
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import json
import asyncio

from ..services.call_manager import CallManager, Call
from ..modules.sniffer import PacketCapture, SessionTracker, CaptureFilter
from ..modules.ht813 import HT813Device
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Create FastAPI app
app = FastAPI(
    title="BeautyAI PABX API",
    description="REST API for PABX system control and monitoring",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = Config()
call_manager: Optional[CallManager] = None
packet_capture: Optional[PacketCapture] = None
session_tracker: Optional[SessionTracker] = None
ht813_device: Optional[HT813Device] = None

# WebSocket connections
websocket_connections: List[WebSocket] = []


# Lifecycle events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global call_manager, packet_capture, session_tracker, ht813_device
    
    logger.info("Starting PABX API server")
    
    # Initialize call manager
    call_manager = CallManager()
    call_manager.start()
    
    # Set up call callbacks
    call_manager.on_call_incoming = lambda call: broadcast_event({
        'type': 'call_incoming',
        'data': call_to_dict(call)
    })
    
    call_manager.on_call_answered = lambda call: broadcast_event({
        'type': 'call_answered',
        'data': call_to_dict(call)
    })
    
    call_manager.on_call_ended = lambda call: broadcast_event({
        'type': 'call_ended',
        'data': call_to_dict(call)
    })
    
    # Initialize packet capture if configured
    capture_config = config.get('capture')
    if capture_config.get('enabled', False):
        capture_filter = CaptureFilter(
            target_ip=capture_config.get('target_ip'),
            sip_port=capture_config.get('sip_port', 5060),
            rtp_port_range=(
                capture_config.get('rtp_port_start', 10000),
                capture_config.get('rtp_port_end', 20000)
            )
        )
        
        packet_capture = PacketCapture(
            interface=capture_config.get('interface', 'any'),
            capture_filter=capture_filter
        )
        
        session_tracker = SessionTracker()
        packet_capture.start()
        
        logger.info("Packet capture started")
    
    # Initialize HT813 device if configured
    ht813_config = config.get('ht813')
    if ht813_config:
        ht813_device = HT813Device(
            ip_address=ht813_config.get('ip_address'),
            username=ht813_config.get('username', 'admin'),
            password=ht813_config.get('password', 'admin')
        )
        
        logger.info("HT813 device interface initialized")
    
    logger.info("PABX API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down PABX API server")
    
    if call_manager:
        call_manager.stop()
    
    if packet_capture:
        packet_capture.stop()
    
    logger.info("PABX API server shutdown complete")


# API endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "BeautyAI PABX API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# Call management endpoints

@app.get("/api/calls")
async def get_calls():
    """Get all active calls"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    calls = call_manager.get_active_calls()
    return {
        "count": len(calls),
        "calls": [call_to_dict(call) for call in calls]
    }


@app.get("/api/calls/{call_id}")
async def get_call(call_id: str):
    """Get call by ID"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    call = call_manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return call_to_dict(call)


@app.get("/api/calls/{call_id}/stats")
async def get_call_stats(call_id: str):
    """Get call statistics"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    stats = call_manager.get_call_stats(call_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return stats


@app.post("/api/calls/{call_id}/answer")
async def answer_call(call_id: str):
    """Answer incoming call"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    success = call_manager.answer_call(call_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to answer call")
    
    return {"success": True, "call_id": call_id}


@app.post("/api/calls/{call_id}/end")
async def end_call(call_id: str):
    """End active call"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    call_manager.end_call(call_id)
    return {"success": True, "call_id": call_id}


@app.post("/api/calls/{call_id}/play")
async def play_audio(call_id: str, audio_file: str):
    """Play audio file on call"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    success = call_manager.play_audio(call_id, audio_file)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to play audio")
    
    return {"success": True, "call_id": call_id, "file": audio_file}


@app.post("/api/calls/{call_id}/record")
async def start_recording(call_id: str):
    """Start recording call audio"""
    if not call_manager:
        raise HTTPException(status_code=503, detail="Call manager not initialized")
    
    success = call_manager.start_recording(call_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start recording")
    
    return {"success": True, "call_id": call_id}


# Packet capture endpoints

@app.get("/api/capture/status")
async def get_capture_status():
    """Get packet capture status"""
    if not packet_capture:
        return {"enabled": False}
    
    stats = packet_capture.get_statistics()
    return {
        "enabled": True,
        "running": packet_capture.running,
        "statistics": stats
    }


@app.get("/api/capture/sessions")
async def get_capture_sessions():
    """Get captured sessions"""
    if not session_tracker:
        raise HTTPException(status_code=503, detail="Session tracker not initialized")
    
    call_sessions = session_tracker.get_call_sessions()
    rtp_sessions = session_tracker.get_rtp_sessions()
    
    return {
        "call_sessions": [session_to_dict(s) for s in call_sessions],
        "rtp_sessions": [session_to_dict(s) for s in rtp_sessions]
    }


# HT813 device endpoints

@app.get("/api/ht813/status")
async def get_ht813_status():
    """Get HT813 device status"""
    if not ht813_device:
        raise HTTPException(status_code=503, detail="HT813 device not configured")
    
    status = ht813_device.get_status()
    if not status:
        raise HTTPException(status_code=500, detail="Failed to get device status")
    
    return {
        "mac_address": status.mac_address,
        "firmware_version": status.firmware_version,
        "uptime": status.uptime,
        "ip_address": status.ip_address,
        "fxs1_registered": status.fxs1_registered,
        "fxs2_registered": status.fxs2_registered,
        "active_calls": status.active_calls
    }


@app.get("/api/ht813/statistics")
async def get_ht813_statistics():
    """Get HT813 call statistics"""
    if not ht813_device:
        raise HTTPException(status_code=503, detail="HT813 device not configured")
    
    stats = ht813_device.get_call_statistics()
    if not stats:
        raise HTTPException(status_code=500, detail="Failed to get call statistics")
    
    return {
        "ports": [
            {
                "port": s.port_name,
                "total_calls": s.total_calls,
                "connected": s.connected_calls,
                "failed": s.failed_calls,
                "incoming": s.incoming_calls,
                "outgoing": s.outgoing_calls
            }
            for s in stats
        ]
    }


@app.post("/api/ht813/reboot")
async def reboot_ht813():
    """Reboot HT813 device"""
    if not ht813_device:
        raise HTTPException(status_code=503, detail="HT813 device not configured")
    
    success = ht813_device.reboot()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reboot device")
    
    return {"success": True}


# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time events"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    logger.info(f"WebSocket client connected: {websocket.client}")
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
        websocket_connections.remove(websocket)


# Helper functions

def call_to_dict(call: Call) -> dict:
    """Convert Call object to dictionary"""
    return {
        "call_id": call.call_id,
        "from_user": call.from_user,
        "to_user": call.to_user,
        "state": call.state,
        "started_at": call.started_at.isoformat() if call.started_at else None,
        "answered_at": call.answered_at.isoformat() if call.answered_at else None,
        "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        "local_rtp_port": call.local_rtp_port,
        "remote_rtp_ip": call.remote_rtp_ip,
        "remote_rtp_port": call.remote_rtp_port,
        "recording_file": call.recording_file
    }


def session_to_dict(session) -> dict:
    """Convert session object to dictionary"""
    data = {}
    for key, value in session.__dict__.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
        elif isinstance(value, list):
            # Skip large lists
            if key in ['packets', 'sequence_numbers']:
                data[key] = f"<{len(value)} items>"
            else:
                data[key] = value
        else:
            data[key] = value
    return data


def broadcast_event(event: dict):
    """Broadcast event to all WebSocket connections"""
    disconnected = []
    
    for websocket in websocket_connections:
        try:
            # Use asyncio to send
            asyncio.create_task(websocket.send_json(event))
        except Exception as e:
            logger.error(f"Error broadcasting to websocket: {e}")
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
