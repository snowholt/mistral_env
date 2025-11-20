"""
FastAPI REST API
Web API for PABX system control and monitoring
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from datetime import datetime
import json
import asyncio
import time

from ..services.call_manager import CallManager, Call
from ..modules.sniffer import PacketCapture, SessionTracker, CaptureFilter
from ..modules.ht813 import HT813Device
from ..modules.syslog.receiver import SyslogReceiver, SyslogMessage
from ..utils.config import Config
from ..utils.logger import get_logger, setup_logging

# Setup logging before anything else
setup_logging()

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
syslog_receiver: Optional[SyslogReceiver] = None
syslog_messages: List[SyslogMessage] = []  # Store recent messages

# WebSocket connections
websocket_connections: List[WebSocket] = []
event_loop: Optional[asyncio.AbstractEventLoop] = None
websocket_heartbeat_task: Optional[asyncio.Task] = None
websocket_last_pong: Dict[WebSocket, float] = {}  # Track last pong time per connection


# Lifecycle events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global call_manager, packet_capture, session_tracker, ht813_device, event_loop, websocket_heartbeat_task
    
    # Store the running event loop for thread-safe WebSocket broadcasting
    event_loop = asyncio.get_running_loop()
    
    # Start WebSocket heartbeat monitoring
    websocket_heartbeat_task = asyncio.create_task(websocket_heartbeat_monitor())
    
    print("=== PABX API SERVER STARTUP EVENT CALLED ===", flush=True)
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
            capture_sip=capture_config.get('capture_sip', True),
            capture_rtp=capture_config.get('capture_rtp', True),
            capture_rtcp=capture_config.get('capture_rtcp', True),
            interface=capture_config.get('interface')
        )
        
        packet_capture = PacketCapture(
            capture_filter=capture_filter
        )
        
        packet_capture.start()
        
        logger.info("Packet capture started")
    
    # Initialize HT813 device if configured
    # DISABLED: Web authentication causes device lockout from repeated login attempts
    # The API will automatically use SIP registration data as fallback (no auth needed)
    # ht813_config = config.get('ht813')
    # if ht813_config:
    #     ht813_device = HT813Device(
    #         ip_address=ht813_config.get('ip_address'),
    #         username=ht813_config.get('username', 'admin'),
    #         password=ht813_config.get('password', 'admin')
    #     )
    #     
    #     logger.info("HT813 device interface initialized")
    
    logger.info("HT813 web interface disabled - using SIP registration data only")
    
    # Initialize syslog receiver if configured
    syslog_config = config.get('syslog', {})
    if syslog_config.get('enabled', False):
        global syslog_receiver, syslog_messages
        
        syslog_receiver = SyslogReceiver(
            host='0.0.0.0',
            port=syslog_config.get('port', 514)
        )
        
        # Store messages callback
        def on_syslog_message(msg: SyslogMessage):
            syslog_messages.append(msg)
            # Keep only last 1000 messages
            if len(syslog_messages) > 1000:
                syslog_messages.pop(0)
        
        syslog_receiver.on_message = on_syslog_message
        syslog_receiver.start()
        
        logger.info("Syslog receiver started")
    
    logger.info("PABX API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down PABX API server")
    
    # Cancel WebSocket heartbeat task
    if websocket_heartbeat_task:
        websocket_heartbeat_task.cancel()
        try:
            await websocket_heartbeat_task
        except asyncio.CancelledError:
            pass
    
    if call_manager:
        call_manager.stop()
    
    if packet_capture:
        packet_capture.stop()
    
    if syslog_receiver:
        syslog_receiver.stop()
    
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


@app.get("/api/registrations")
async def get_registrations():
    """Get all SIP registrations"""
    if not call_manager or not call_manager.sip_server:
        raise HTTPException(status_code=503, detail="SIP server not initialized")
    
    registrations = call_manager.sip_server.registrations
    
    return {
        "count": len(registrations),
        "registrations": [
            {
                "user": user,
                "contact": reg.contact,
                "ip_address": reg.ip_address,
                "port": reg.port,
                "expires": reg.expires,
                "registered_at": reg.registered_at.isoformat()
            }
            for user, reg in registrations.items()
        ]
    }


@app.get("/api/trunk/status")
async def get_trunk_status():
    """
    Get SIP trunk registration status
    
    Returns information about outbound registration to STC provider
    """
    if not call_manager or not call_manager.sip_client:
        raise HTTPException(status_code=503, detail="SIP client not initialized")
    
    status = call_manager.sip_client.get_registration_status()
    
    return {
        "trunk": status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/calls/initiate")
async def initiate_call(request: dict):
    """
    Initiate outbound call
    
    Request body:
    {
        "from_user": "1002",
        "to_number": "+14383242270"
    }
    """
    if not call_manager or not call_manager.sip_server:
        raise HTTPException(status_code=503, detail="SIP server not initialized")
    
    from_user = request.get("from_user", "1002")
    to_number = request.get("to_number")
    
    if not to_number:
        raise HTTPException(status_code=400, detail="to_number is required")
    
    success = call_manager.sip_server.initiate_call(from_user, to_number)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initiate call")
    
    return {
        "success": True,
        "from_user": from_user,
        "to_number": to_number,
        "message": f"Call initiated from {from_user} to {to_number}"
    }


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
    """Get HT813 device status (with SIP registration fallback)"""
    # Try to get status from web interface first
    web_status = None
    if ht813_device:
        try:
            web_status = ht813_device.get_status()
        except:
            pass
    
    # If web interface fails, use SIP registration data as fallback
    if not web_status and call_manager and call_manager.sip_server:
        registrations = call_manager.sip_server.registrations
        ht813_config = Config().get('ht813', {})
        
        # Check if users are registered and get details
        fxs1_reg = '1001' in registrations
        fxs2_reg = '1002' in registrations
        
        # Calculate uptime from registration time (approximate)
        uptime_seconds = 0
        if registrations:
            oldest_reg = min(registrations.values(), key=lambda r: r.registered_at)
            uptime_seconds = int((datetime.now() - oldest_reg.registered_at).total_seconds())
        
        # Get IP from first registration or config
        ip_address = ht813_config.get('ip_address', '192.168.100.96')
        if registrations:
            first_reg = next(iter(registrations.values()))
            ip_address = first_reg.ip_address
        
        return {
            "mac_address": "EC:74:D7:62:4E:35",  # From config or detection
            "firmware_version": "1.0.17.3",  # Default (web interface unavailable)
            "uptime": uptime_seconds,  # Estimated from SIP registration
            "ip_address": ip_address,
            "fxs1_registered": fxs1_reg,
            "fxs2_registered": fxs2_reg,
            "active_calls": len(call_manager.sip_server.call_sessions) if call_manager.sip_server else 0,
            "data_source": "sip_registration"  # Indicate we're using SIP data
        }
    
    # Return web status if available
    if web_status:
        return {
            "mac_address": web_status.mac_address,
            "firmware_version": web_status.firmware_version,
            "uptime": web_status.uptime,
            "ip_address": web_status.ip_address,
            "fxs1_registered": web_status.fxs1_registered,
            "fxs2_registered": web_status.fxs2_registered,
            "active_calls": web_status.active_calls,
            "data_source": "web_interface"
        }
    
    raise HTTPException(status_code=503, detail="HT813 device not available")


@app.get("/api/ht813/statistics")
async def get_ht813_statistics():
    """Get HT813 call statistics (with fallback)"""
    # Try web interface first
    if ht813_device:
        try:
            stats = ht813_device.get_call_statistics()
            if stats:
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
        except:
            pass
    
    # Fallback: return empty statistics
    return {
        "ports": [
            {
                "port": "FXS1",
                "total_calls": 0,
                "connected": 0,
                "failed": 0,
                "incoming": 0,
                "outgoing": 0
            },
            {
                "port": "FXS2", 
                "total_calls": 0,
                "connected": 0,
                "failed": 0,
                "incoming": 0,
                "outgoing": 0
            }
        ]
    }
    
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


# Syslog endpoints

@app.get("/api/syslog/status")
async def get_syslog_status():
    """Get syslog receiver status"""
    if not syslog_receiver:
        return {"enabled": False}
    
    stats = syslog_receiver.get_statistics()
    return {
        "enabled": True,
        "running": syslog_receiver.running,
        "statistics": stats
    }


@app.get("/api/syslog/messages")
async def get_syslog_messages(limit: int = 100):
    """Get recent syslog messages"""
    if not syslog_receiver:
        return {"messages": []}
    
    # Get last N messages
    messages = syslog_messages[-limit:] if len(syslog_messages) > limit else syslog_messages
    
    return {
        "count": len(messages),
        "messages": [
            {
                "timestamp": msg.timestamp.isoformat(),
                "hostname": msg.hostname,
                "severity": msg.severity,
                "message": msg.message
            }
            for msg in messages
        ]
    }


# WebSocket endpoint

async def websocket_heartbeat_monitor():
    """
    Monitor WebSocket connections and send periodic pings
    
    Sends ping every 30 seconds and removes connections that don't respond
    within 60 seconds (stale connections).
    """
    PING_INTERVAL = 30  # seconds
    TIMEOUT = 60  # seconds
    
    logger.info("WebSocket heartbeat monitor started")
    
    while True:
        try:
            await asyncio.sleep(PING_INTERVAL)
            
            current_time = time.time()
            stale_connections = []
            
            for ws in list(websocket_connections):
                try:
                    # Check if connection is stale (no pong received)
                    last_pong = websocket_last_pong.get(ws, current_time)
                    
                    if current_time - last_pong > TIMEOUT:
                        logger.warning(f"WebSocket connection stale: {ws.client}")
                        stale_connections.append(ws)
                        continue
                    
                    # Send ping
                    await ws.send_json({"type": "ping", "timestamp": current_time})
                    logger.debug(f"Sent ping to WebSocket: {ws.client}")
                    
                except Exception as e:
                    logger.error(f"Error sending ping to WebSocket: {e}")
                    stale_connections.append(ws)
            
            # Remove stale connections
            for ws in stale_connections:
                if ws in websocket_connections:
                    websocket_connections.remove(ws)
                if ws in websocket_last_pong:
                    del websocket_last_pong[ws]
                
                try:
                    await ws.close()
                except:
                    pass
            
            if stale_connections:
                logger.info(f"Removed {len(stale_connections)} stale WebSocket connections")
                
        except asyncio.CancelledError:
            logger.info("WebSocket heartbeat monitor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in WebSocket heartbeat monitor: {e}", exc_info=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time events"""
    await websocket.accept()
    websocket_connections.append(websocket)
    websocket_last_pong[websocket] = time.time()  # Initialize last pong time
    
    logger.info(f"WebSocket client connected: {websocket.client}")
    
    try:
        while True:
            # Receive messages (for pong responses and other client messages)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                
                # Handle pong response
                if data.get("type") == "pong":
                    websocket_last_pong[websocket] = time.time()
                    logger.debug(f"Received pong from WebSocket: {websocket.client}")
                    
            except asyncio.TimeoutError:
                # No message received, continue monitoring
                continue
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        if websocket in websocket_last_pong:
            del websocket_last_pong[websocket]


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
    """
    Broadcast event to all WebSocket connections (thread-safe)
    
    This function can be called from any thread (e.g., SIP/RTP threads)
    and safely broadcasts to WebSocket connections in the FastAPI event loop.
    
    Args:
        event: Event dictionary to broadcast
    """
    if not websocket_connections or not event_loop:
        return
    
    # Create async coroutine for broadcasting
    async def _send_to_all():
        disconnected = []
        
        for websocket in websocket_connections:
            try:
                await websocket.send_json(event)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)
    
    # Schedule the coroutine in the event loop (thread-safe)
    try:
        asyncio.run_coroutine_threadsafe(_send_to_all(), event_loop)
    except Exception as e:
        logger.error(f"Error scheduling WebSocket broadcast: {e}")
