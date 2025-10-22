"""
WebRTC Connection Pool for Managing RTCPeerConnection Instances.

This module provides specialized connection pooling for WebRTC peer connections,
tracking ICE states, connection quality, and lifecycle management.

Created for WebRTC MVP Migration - Phase B
Author: BeautyAI Framework
Date: October 15, 2025
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from enum import Enum

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
    from aiortc.contrib.media import MediaRecorder, MediaRelay
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    MediaStreamTrack = None  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning("aiortc not available - WebRTC functionality disabled")

from .config_manager import get_config_manager

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_LANGUAGE = "ar"  # Default language when session info unavailable


class PeerConnectionState(Enum):
    """WebRTC peer connection states."""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


class ICEConnectionState(Enum):
    """ICE connection states."""
    NEW = "new"
    CHECKING = "checking"
    CONNECTED = "connected"
    COMPLETED = "completed"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"


class ICEGatheringState(Enum):
    """ICE gathering states."""
    NEW = "new"
    GATHERING = "gathering"
    COMPLETE = "complete"


@dataclass
class WebRTCConnectionData:
    """Data structure for WebRTC peer connection metadata."""
    
    peer_id: str
    peer_connection: Optional['RTCPeerConnection'] = None
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Connection states
    connection_state: str = PeerConnectionState.NEW.value
    ice_connection_state: str = ICEConnectionState.NEW.value
    ice_gathering_state: str = ICEGatheringState.NEW.value
    
    # ICE candidates
    ice_candidates: List[Dict[str, Any]] = field(default_factory=list)
    remote_ice_candidates_count: int = 0
    
    # SDP information
    local_sdp: Optional[str] = None
    remote_sdp: Optional[str] = None
    
    # Quality metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    
    # Data channel for sending transcriptions/responses
    data_channel: Optional[Any] = None  # RTCDataChannel instance
    data_channel_label: Optional[str] = None
    data_channel_ready_state: Optional[str] = None
    data_channel_last_updated: Optional[float] = None
    
    # Metadata
    client_info: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def update_connection_state(self, state: str):
        """Update connection state."""
        self.connection_state = state
        self.update_activity()
        logger.debug(f"[WebRTC] Peer {self.peer_id} connection state: {state}")
    
    def update_ice_connection_state(self, state: str):
        """Update ICE connection state."""
        self.ice_connection_state = state
        self.update_activity()
        logger.debug(f"[WebRTC] Peer {self.peer_id} ICE connection state: {state}")
    
    def update_ice_gathering_state(self, state: str):
        """Update ICE gathering state."""
        self.ice_gathering_state = state
        self.update_activity()
        logger.debug(f"[WebRTC] Peer {self.peer_id} ICE gathering state: {state}")
    
    def add_ice_candidate(self, candidate: Dict[str, Any]):
        """Add ICE candidate."""
        self.ice_candidates.append({
            **candidate,
            'timestamp': time.time()
        })
        self.remote_ice_candidates_count += 1
        logger.debug(f"[WebRTC] Added ICE candidate for peer {self.peer_id} (total: {self.remote_ice_candidates_count})")
    
    def attach_data_channel(self, channel: Any):
        """Attach client-created data channel to this connection."""
        self.data_channel = channel
        self.data_channel_label = getattr(channel, "label", None)
        # readyState may not be present immediately; use getattr to avoid AttributeError
        self.data_channel_ready_state = getattr(channel, "readyState", self.data_channel_ready_state)
        self.data_channel_last_updated = time.time()

    def update_data_channel_state(self, state: Optional[str]):
        """Persist the latest known data channel readyState."""
        self.data_channel_ready_state = state
        self.data_channel_last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        channel_state = "absent"
        if self.data_channel:
            channel_state = getattr(self.data_channel, "readyState", None) or self.data_channel_ready_state or "unknown"
        elif self.data_channel_ready_state:
            channel_state = self.data_channel_ready_state

        return {
            'peer_id': self.peer_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'connection_state': self.connection_state,
            'ice_connection_state': self.ice_connection_state,
            'ice_gathering_state': self.ice_gathering_state,
            'ice_candidates_count': len(self.ice_candidates),
            'remote_ice_candidates_count': self.remote_ice_candidates_count,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'client_info': self.client_info,
            'data_channel_present': bool(self.data_channel or self.data_channel_ready_state),
            'data_channel_state': channel_state,
            'data_channel_label': self.data_channel_label,
            'data_channel_last_updated': self.data_channel_last_updated
        }


class WebRTCConnectionPool:
    """
    Connection pool for WebRTC peer connections.
    
    Features:
    - RTCPeerConnection lifecycle management
    - ICE state tracking and candidate management
    - Connection quality monitoring
    - Graceful cleanup and resource management
    - State persistence and recovery
    """
    
    def __init__(
        self,
        max_connections: int = 100,
        connection_timeout_seconds: int = 1800,
        enable_metrics: bool = True
    ):
        """
        Initialize WebRTC connection pool.
        
        Args:
            max_connections: Maximum number of concurrent connections
            connection_timeout_seconds: Timeout for idle connections (default: 1800s = 30 minutes)
            enable_metrics: Enable metrics collection
        """
        self.max_connections = max_connections
        self.connection_timeout_seconds = connection_timeout_seconds
        self.enable_metrics = enable_metrics
        
        # Connection tracking
        self._connections: Dict[str, WebRTCConnectionData] = {}
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> peer_ids
        
        # Voice service adapters per peer (handles audio processing pipeline)
        self._voice_adapters: Dict[str, Any] = {}  # peer_id -> WebRTCVoiceServiceAdapter
        
        # Keep-alive tasks for active audio tracks
        self._keepalive_tasks: Dict[str, asyncio.Task] = {}  # peer_id -> keep-alive task

        # Track data channels received before connection registration completes
        self._pending_data_channels: Dict[str, Any] = {}  # peer_id -> RTCDataChannel instance
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"[WebRTC] Connection pool initialized (max_connections={max_connections})")
    
    async def start(self):
        """Start the connection pool and background tasks."""
        if self._running:
            logger.warning("[WebRTC] Connection pool already running")
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("[WebRTC] Connection pool started")
    
    async def stop(self):
        """Stop the connection pool and cleanup all connections."""
        if not self._running:
            return
        
        logger.info("[WebRTC] Stopping connection pool...")
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all connections
        async with self._lock:
            peer_ids = list(self._connections.keys())
            for peer_id in peer_ids:
                try:
                    await self._cleanup_connection(peer_id)
                except Exception as e:
                    logger.error(f"[WebRTC] Error cleaning up {peer_id} during shutdown: {e}")
        
        logger.info("[WebRTC] Connection pool stopped")
    
    async def _register_client_data_channel(self, peer_id: str, channel: Any) -> None:
        """Track data channel provided by the client once it is announced."""
        async with self._lock:
            if peer_id in self._connections:
                connection_data = self._connections[peer_id]
                connection_data.attach_data_channel(channel)
                self._pending_data_channels.pop(peer_id, None)
            else:
                # Store temporarily until the connection record is registered
                self._pending_data_channels[peer_id] = channel

    async def _handle_data_channel_open(self, peer_id: str, channel: Any) -> None:
        """Persist open state and bump activity when channel becomes ready."""
        async with self._lock:
            if peer_id in self._connections:
                connection_data = self._connections[peer_id]
                connection_data.attach_data_channel(channel)
                connection_data.update_data_channel_state("open")
                connection_data.update_activity()
                self._pending_data_channels.pop(peer_id, None)
            else:
                self._pending_data_channels[peer_id] = channel

    async def _handle_data_channel_close(self, peer_id: str, channel: Any) -> None:
        """Mark channel as closed and release pending references."""
        async with self._lock:
            if peer_id in self._connections:
                connection_data = self._connections[peer_id]
                connection_data.update_data_channel_state("closed")
            # Remove any pending reference for this peer
            self._pending_data_channels.pop(peer_id, None)

    async def create_peer_connection(
        self,
        peer_id: str,
        offer_sdp: str,
        user_id: Optional[str] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Create a new RTCPeerConnection and process SDP offer.
        
        Args:
            peer_id: Unique peer identifier
            offer_sdp: SDP offer from client
            user_id: Optional user identifier
            
        Returns:
            Tuple of (answer_sdp, ice_servers)
            
        Raises:
            RuntimeError: If aiortc not available
            ValueError: If max connections reached
            Exception: If connection creation fails
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc library not available - cannot create peer connection")
        
        async with self._lock:
            # Check max connections
            if len(self._connections) >= self.max_connections:
                raise ValueError(f"Maximum connections reached ({self.max_connections})")
            
            # Get STUN server from environment
            import os
            stun_server = os.getenv('WEBRTC_STUN_SERVER', 'stun:stun.l.google.com:19302')
            
            # Create RTCPeerConnection
            try:
                pc = RTCPeerConnection()
                
                # The client will create the data channel in the offer.
                # Server listens for it via @pc.on("datachannel") handler below.
                data_channel = None  # Will be set when client's channel is received
                logger.info(f"[WebRTC] RTCPeerConnection created for peer {peer_id}, waiting for client data channel")
                
                @pc.on("datachannel")
                def on_datachannel_from_client(channel):
                    """Handle data channel created by client (in offer)."""
                    nonlocal data_channel
                    data_channel = channel
                    logger.info(f"[WebRTC] ✓ Received data channel '{channel.label}' from client for peer {peer_id}")

                    asyncio.create_task(self._register_client_data_channel(peer_id, channel))

                    
                    @channel.on("open")
                    def on_dc_open():
                        logger.info(f"[WebRTC] ✓ Data channel '{channel.label}' OPENED for peer {peer_id}")
                    
                        if peer_id in self._connections:
                            self._connections[peer_id].update_activity()
                            # Update the stored reference now that it's open
                            self._connections[peer_id].data_channel = data_channel
                    
                    
                    @channel.on("close")
                    def on_dc_close():
                        logger.info(f"[WebRTC] Data channel '{channel.label}' CLOSED for peer {peer_id}")
                    
                    @channel.on("message")
                    def on_dc_message(message):
                        logger.debug(f"[WebRTC] Data channel message from {peer_id}: {message}")
                
                logger.info(f"[WebRTC] Data channel handler registered for peer {peer_id}")
                
                # Set up event handlers
                @pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    if peer_id in self._connections:
                        self._connections[peer_id].update_connection_state(pc.connectionState)
                
                @pc.on("iceconnectionstatechange")
                async def on_iceconnectionstatechange():
                    if peer_id in self._connections:
                        connection_data = self._connections[peer_id]
                        connection_data.update_ice_connection_state(pc.iceConnectionState)

                        # Mirror the ICE state into the general connection state so the
                        # status endpoint reflects connectivity changes promptly. aiortc's
                        # aggregate connectionState can lag or remain "connecting" when only
                        # trickle ICE is active, so we explicitly map terminal ICE states.
                        ice_state_normalized = (pc.iceConnectionState or "").lower()
                        if ice_state_normalized in (ICEConnectionState.CONNECTED.value, ICEConnectionState.COMPLETED.value):
                            connection_data.update_connection_state(PeerConnectionState.CONNECTED.value)
                        elif ice_state_normalized == ICEConnectionState.DISCONNECTED.value:
                            connection_data.update_connection_state(PeerConnectionState.DISCONNECTED.value)
                        elif ice_state_normalized == ICEConnectionState.FAILED.value:
                            connection_data.update_connection_state(PeerConnectionState.FAILED.value)
                        elif ice_state_normalized == ICEConnectionState.CLOSED.value:
                            connection_data.update_connection_state(PeerConnectionState.CLOSED.value)
                
                @pc.on("icegatheringstatechange")
                async def on_icegatheringstatechange():
                    if peer_id in self._connections:
                        self._connections[peer_id].update_ice_gathering_state(pc.iceGatheringState)
                
                @pc.on("track")
                async def on_track(track):
                    """
                    Handle incoming audio track from client.
                    
                    This is critical for WebRTC connection stability:
                    - Without consuming the track, the connection may timeout or disconnect
                    - We create a voice service adapter to process the audio stream
                    - The adapter wires: Track → AudioProcessor → VAD → Buffer → STT/LLM/TTS
                    """
                    logger.info(f"[WebRTC] Received {track.kind} track for peer {peer_id}")
                    
                    if track.kind == "audio":
                        # Start periodic activity updater to prevent idle timeout during audio processing
                        async def keep_alive_during_audio():
                            """
                            Update connection activity periodically while audio is active.
                            
                            This prevents the cleanup loop from removing the connection
                            during active audio streaming. Without this, connections appear
                            idle after initial setup and get cleaned up even though audio
                            frames are being processed.
                            """
                            try:
                                update_interval = 30  # Update every 30 seconds
                                while peer_id in self._connections:
                                    await asyncio.sleep(update_interval)
                                    if peer_id in self._connections:
                                        self._connections[peer_id].update_activity()
                                        logger.debug(
                                            f"[WebRTC] Updated activity for peer {peer_id} "
                                            f"during audio processing"
                                        )
                            except asyncio.CancelledError:
                                logger.debug(
                                    f"[WebRTC] Keep-alive task cancelled for peer {peer_id}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"[WebRTC] Error in keep-alive task for {peer_id}: {e}"
                                )
                        
                        # Start keep-alive task (runs in background, doesn't block)
                        task = asyncio.create_task(keep_alive_during_audio())
                        self._keepalive_tasks[peer_id] = task
                        logger.info(
                            f"[WebRTC] Started keep-alive task for peer {peer_id} "
                            f"to prevent idle timeout during audio"
                        )
                        try:
                            # Import voice service adapter (lazy import to avoid circular dependencies)
                            from ..services.voice.webrtc_voice_service_adapter import (
                                WebRTCVoiceServiceAdapter,
                                WebRTCVoiceConfig,
                                WebRTCVADConfig
                            )
                            from ..services.voice.conversation.simple_voice_service import SimpleVoiceService
                            from .webrtc_session_manager import get_webrtc_session_manager
                            
                            # Get session info from session manager
                            language = DEFAULT_LANGUAGE
                            session_id = None
                            try:
                                session_mgr = get_webrtc_session_manager()
                                session_info = await session_mgr.get_session_by_peer(peer_id)
                                if session_info:
                                    language = session_info.get('language', DEFAULT_LANGUAGE)
                                    session_id = session_info.get('session_id')
                            except Exception as e:
                                logger.warning(f"[WebRTC] Could not get session info for {peer_id}: {e}")
                            
                            if not session_id:
                                # Generate temporary session_id as fallback to maintain connection
                                import uuid
                                session_id = f"webrtc_temp_{uuid.uuid4().hex[:12]}"
                                logger.warning(
                                    f"[WebRTC] No session_id found for peer {peer_id}, "
                                    f"using temporary session {session_id}"
                                )
                            
                            # Create voice service for this peer if not exists
                            if peer_id not in self._voice_adapters:
                                logger.info(f"[WebRTC] Creating voice service adapter for peer {peer_id}, session {session_id}, language={language}")
                                
                                # Create SimpleVoiceService instance (language configured via voice registry)
                                simple_voice_service = SimpleVoiceService()
                                
                                # Create voice adapter with dual VAD configured for Silero-only mode
                                voice_config = WebRTCVoiceConfig(
                                    default_language=language,
                                    vad_config=WebRTCVADConfig(
                                        enable_browser_hints=False,
                                        require_silero_confirmation=False
                                    )
                                )
                                
                                # Define callback functions to send data via data channel
                                def send_transcription(p_id: str, text: str):
                                    """Send transcription via data channel"""
                                    logger.info(f"[WebRTC] send_transcription called for {p_id}: {text[:50]}...")
                                    if p_id in self._connections:
                                        dc = self._connections[p_id].data_channel
                                        logger.debug(f"[WebRTC] Data channel state for {p_id}: {dc.readyState if dc else 'None'}")
                                        if dc and dc.readyState == "open":
                                            import json
                                            try:
                                                dc.send(json.dumps({
                                                    "type": "transcription",
                                                    "text": text,
                                                    "timestamp": time.time()
                                                }))
                                                logger.info(f"[WebRTC] ✓ Sent transcription to {p_id}: {text[:50]}...")
                                            except Exception as e:
                                                logger.error(f"[WebRTC] Failed to send transcription: {e}", exc_info=True)
                                        else:
                                            logger.warning(f"[WebRTC] Data channel not open for {p_id} (state: {dc.readyState if dc else 'None'}), cannot send transcription")
                                    else:
                                        logger.warning(f"[WebRTC] Peer {p_id} not found in connections, cannot send transcription")
                                
                                def send_llm_response(p_id: str, text: str):
                                    """Send LLM response via data channel"""
                                    logger.info(f"[WebRTC] send_llm_response called for {p_id}: {text[:50]}...")
                                    if p_id in self._connections:
                                        dc = self._connections[p_id].data_channel
                                        logger.debug(f"[WebRTC] Data channel state for {p_id}: {dc.readyState if dc else 'None'}")
                                        if dc and dc.readyState == "open":
                                            import json
                                            try:
                                                dc.send(json.dumps({
                                                    "type": "assistant_response",
                                                    "text": text,
                                                    "timestamp": time.time()
                                                }))
                                                logger.info(f"[WebRTC] ✓ Sent LLM response to {p_id}: {text[:50]}...")
                                            except Exception as e:
                                                logger.error(f"[WebRTC] Failed to send LLM response: {e}", exc_info=True)
                                        else:
                                            logger.warning(f"[WebRTC] Data channel not open for {p_id} (state: {dc.readyState if dc else 'None'}), cannot send LLM response")
                                    else:
                                        logger.warning(f"[WebRTC] Peer {p_id} not found in connections, cannot send LLM response")
                                
                                def send_tts_audio(p_id: str, audio_bytes: bytes):
                                    """Send TTS audio via data channel"""
                                    if p_id in self._connections:
                                        dc = self._connections[p_id].data_channel
                                        if dc and dc.readyState == "open":
                                            import json
                                            import base64
                                            try:
                                                dc.send(json.dumps({
                                                    "type": "tts_audio",
                                                    "audio_base64": base64.b64encode(audio_bytes).decode(),
                                                    "timestamp": time.time()
                                                }))
                                                logger.info(f"[WebRTC] Sent TTS audio to {p_id}: {len(audio_bytes)} bytes")
                                            except Exception as e:
                                                logger.error(f"[WebRTC] Failed to send TTS audio: {e}")
                                        else:
                                            logger.warning(f"[WebRTC] Data channel not open for {p_id}, cannot send TTS audio")
                                
                                # Create adapter with callbacks wired
                                adapter = WebRTCVoiceServiceAdapter(
                                    peer_id=peer_id,
                                    session_id=session_id,
                                    language=language,
                                    voice_service=simple_voice_service,
                                    config=voice_config,
                                    on_transcription=send_transcription,
                                    on_llm_response=send_llm_response,
                                    on_tts_audio=send_tts_audio
                                )
                                
                                # Initialize the adapter
                                if await adapter.initialize():
                                    self._voice_adapters[peer_id] = adapter
                                    logger.info(f"[WebRTC] Voice adapter initialized for peer {peer_id}")
                                else:
                                    logger.error(f"[WebRTC] Failed to initialize voice adapter for {peer_id}")
                                    return
                            
                            # Start processing audio from the track
                            adapter = self._voice_adapters[peer_id]
                            if await adapter.start_voice_session(track):
                                logger.info(f"[WebRTC] Voice session started for peer {peer_id}")
                                
                                # Update session metadata to indicate audio track is active
                                try:
                                    session_mgr = get_webrtc_session_manager()
                                    await session_mgr.update_session_metadata(
                                        peer_id=peer_id,
                                        audio_track_active=True
                                    )
                                except Exception as e:
                                    logger.warning(f"[WebRTC] Could not update session metadata: {e}")
                            else:
                                logger.error(f"[WebRTC] Failed to start voice session for {peer_id}")
                        
                        except Exception as e:
                            logger.error(f"[WebRTC] Error handling audio track for {peer_id}: {e}", exc_info=True)
                    else:
                        logger.info(f"[WebRTC] Ignoring non-audio track ({track.kind}) for peer {peer_id}")
                
                # Process offer
                offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
                logger.debug(f"[WebRTC] Client offer SDP has {offer_sdp.count('m=application')} application sections")
                await pc.setRemoteDescription(offer)
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                logger.info(f"[WebRTC] Answer SDP created for {peer_id}")
                logger.debug(f"[WebRTC] Answer SDP has {answer.sdp.count('m=application')} application sections")
                if "a=sctp-port" in answer.sdp:
                    logger.info(f"[WebRTC] ✓ Answer SDP contains SCTP (data channel) for {peer_id}")
                else:
                    logger.warning(f"[WebRTC] ⚠️ Answer SDP does NOT contain SCTP for {peer_id} - data channel may not work!")
                
                # Store connection data
                connection_data = WebRTCConnectionData(
                    peer_id=peer_id,
                    peer_connection=pc,
                    user_id=user_id,
                    local_sdp=pc.localDescription.sdp,
                    remote_sdp=offer_sdp,
                    data_channel=data_channel
                )

                if not data_channel and peer_id in self._pending_data_channels:
                    pending_channel = self._pending_data_channels.pop(peer_id)
                    connection_data.attach_data_channel(pending_channel)
                elif data_channel:
                    connection_data.attach_data_channel(data_channel)
                
                self._connections[peer_id] = connection_data

                # Ensure initial connection states reflect the current RTCPeerConnection status.
                # The aiortc state change callbacks above may fire before the connection data is
                # registered in the pool (e.g., during setRemoteDescription/createAnswer). When
                # that happens, the guards in those callbacks skip updates because the peer_id
                # isn't tracked yet, leaving default "new" values. Snapshot the current state
                # here so the status endpoint immediately returns accurate information.
                try:
                    current_connection_state = getattr(pc, "connectionState", None)
                    if current_connection_state:
                        connection_data.update_connection_state(current_connection_state)

                    current_ice_state = getattr(pc, "iceConnectionState", None)
                    if current_ice_state:
                        connection_data.update_ice_connection_state(current_ice_state)

                    current_gathering_state = getattr(pc, "iceGatheringState", None)
                    if current_gathering_state:
                        connection_data.update_ice_gathering_state(current_gathering_state)
                except Exception as state_error:
                    logger.debug(
                        "[WebRTC] Failed to snapshot initial connection states for %s: %s",
                        peer_id,
                        state_error
                    )
                
                # Track user connections
                if user_id:
                    if user_id not in self._user_connections:
                        self._user_connections[user_id] = set()
                    self._user_connections[user_id].add(peer_id)
                
                # Get ICE servers configuration (use Google STUN servers as default)
                ice_servers = [
                    {"urls": "stun:stun.l.google.com:19302"},
                    {"urls": "stun:stun1.l.google.com:19302"}
                ]
                
                logger.info(f"[WebRTC] Created peer connection: peer_id={peer_id}, user_id={user_id}")
                
                return pc.localDescription.sdp, ice_servers
                
            except Exception as e:
                logger.error(f"[WebRTC] Failed to create peer connection for {peer_id}: {e}", exc_info=True)
                # Cleanup on failure
                if peer_id in self._connections:
                    await self._cleanup_connection(peer_id)
                raise
    
    async def add_ice_candidate(
        self,
        peer_id: str,
        candidate: str,
        sdp_mid: Optional[str] = None,
        sdp_m_line_index: Optional[int] = None
    ) -> int:
        """
        Add ICE candidate to peer connection.
        
        Args:
            peer_id: Peer connection identifier
            candidate: ICE candidate string
            sdp_mid: Media stream identification tag
            sdp_m_line_index: Media line index
            
        Returns:
            Candidate index
            
        Raises:
            ValueError: If peer connection not found
            Exception: If adding candidate fails
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc library not available")
        
        async with self._lock:
            if peer_id not in self._connections:
                raise ValueError(f"Peer connection not found: {peer_id}")
            
            connection_data = self._connections[peer_id]
            pc = connection_data.peer_connection
            
            if not pc:
                raise ValueError(f"No peer connection for {peer_id}")
            
            try:
                # Parse ICE candidate string into components
                # Browser format: "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host generation 0 ufrag abc network-cost 999"
                # Minimum format: "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host"
                parts = candidate.split()
                if len(parts) < 8 or not parts[0].startswith('candidate:'):
                    raise ValueError(f"Invalid candidate format (need at least 8 parts): {candidate}")
                
                foundation = parts[0][10:]  # Remove "candidate:" prefix
                component = int(parts[1])
                protocol = parts[2].upper()
                priority = int(parts[3])
                ip = parts[4]
                port = int(parts[5])
                
                # Find 'typ' keyword (can be at different positions due to extra fields)
                if 'typ' not in parts:
                    raise ValueError(f"Missing 'typ' keyword in candidate: {candidate}")
                
                typ_idx = parts.index('typ')
                if typ_idx + 1 >= len(parts):
                    raise ValueError(f"Missing candidate type after 'typ' in: {candidate}")
                
                candidate_type = parts[typ_idx + 1]
                
                # Create ICE candidate with parsed components
                ice_candidate = RTCIceCandidate(
                    component=component,
                    foundation=foundation,
                    ip=ip,
                    port=port,
                    priority=priority,
                    protocol=protocol,
                    type=candidate_type,
                    sdpMid=sdp_mid,
                    sdpMLineIndex=sdp_m_line_index
                )
                
                # Add to peer connection
                await pc.addIceCandidate(ice_candidate)
                
                # Track candidate
                connection_data.add_ice_candidate({
                    'candidate': candidate,
                    'sdp_mid': sdp_mid,
                    'sdp_m_line_index': sdp_m_line_index
                })
                
                return connection_data.remote_ice_candidates_count - 1
                
            except Exception as e:
                logger.error(f"[WebRTC] Failed to add ICE candidate for {peer_id}: {e}", exc_info=True)
                raise
    
    async def peer_exists(self, peer_id: str) -> bool:
        """Check if peer connection exists."""
        async with self._lock:
            return peer_id in self._connections
    
    async def get_connection_status(self, peer_id: str) -> Dict[str, Any]:
        """
        Get current status of peer connection.
        
        Args:
            peer_id: Peer connection identifier
            
        Returns:
            Dictionary with connection status
            
        Raises:
            ValueError: If peer connection not found
        """
        async with self._lock:
            if peer_id not in self._connections:
                raise ValueError(f"Peer connection not found: {peer_id}")

            connection_data = self._connections[peer_id]

            # Synchronize metadata with the live RTCPeerConnection states before returning
            pc = connection_data.peer_connection
            if pc:
                try:
                    current_connection_state = getattr(pc, "connectionState", None)
                    if current_connection_state and current_connection_state != connection_data.connection_state:
                        connection_data.update_connection_state(current_connection_state)

                    current_ice_state = getattr(pc, "iceConnectionState", None)
                    if current_ice_state and current_ice_state != connection_data.ice_connection_state:
                        connection_data.update_ice_connection_state(current_ice_state)

                    current_gathering_state = getattr(pc, "iceGatheringState", None)
                    if current_gathering_state and current_gathering_state != connection_data.ice_gathering_state:
                        connection_data.update_ice_gathering_state(current_gathering_state)
                except Exception as state_error:
                    logger.debug(
                        "[WebRTC] Failed to synchronize connection states for %s: %s",
                        peer_id,
                        state_error
                    )

            status = connection_data.to_dict()

            # Derive a consistent connection state from ICE state to avoid stale "connecting" values
            ice_state = (status.get('ice_connection_state') or "").lower()
            if ice_state in (ICEConnectionState.CONNECTED.value, ICEConnectionState.COMPLETED.value):
                status['connection_state'] = PeerConnectionState.CONNECTED.value
            elif ice_state == ICEConnectionState.FAILED.value:
                status['connection_state'] = PeerConnectionState.FAILED.value
            elif ice_state == ICEConnectionState.DISCONNECTED.value:
                status['connection_state'] = PeerConnectionState.DISCONNECTED.value
            elif ice_state == ICEConnectionState.CLOSED.value:
                status['connection_state'] = PeerConnectionState.CLOSED.value

            return status
    
    async def remove_peer_connection(self, peer_id: str):
        """
        Remove and cleanup peer connection.
        
        Args:
            peer_id: Peer connection identifier
        """
        async with self._lock:
            await self._cleanup_connection(peer_id)

    async def _cleanup_connection(self, peer_id: str):
        """Internal cleanup method (must be called with lock held)."""
        if peer_id not in self._connections:
            return

        connection_data = self._connections[peer_id]

        try:
            # Cancel keep-alive task if exists
            if peer_id in self._keepalive_tasks:
                try:
                    task = self._keepalive_tasks[peer_id]
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    del self._keepalive_tasks[peer_id]
                    logger.debug(f"[WebRTC] Cancelled keep-alive task for {peer_id}")
                except Exception as e:
                    logger.error(f"[WebRTC] Error cancelling keep-alive task for {peer_id}: {e}")
            
            # Stop voice adapter if exists
            if peer_id in self._voice_adapters:
                try:
                    adapter = self._voice_adapters[peer_id]
                    await adapter.stop_voice_session()
                    del self._voice_adapters[peer_id]
                    logger.info(f"[WebRTC] Stopped voice adapter for {peer_id}")
                except Exception as e:
                    logger.error(f"[WebRTC] Error stopping voice adapter for {peer_id}: {e}")
            
            # Close peer connection
            if connection_data.peer_connection:
                await connection_data.peer_connection.close()
            
            # Remove from tracking
            del self._connections[peer_id]

            # Clear any pending data channel reference for this peer
            if peer_id in self._pending_data_channels:
                del self._pending_data_channels[peer_id]
            
            # Remove from user connections
            if connection_data.user_id and connection_data.user_id in self._user_connections:
                self._user_connections[connection_data.user_id].discard(peer_id)
                if not self._user_connections[connection_data.user_id]:
                    del self._user_connections[connection_data.user_id]
            
            logger.info(f"[WebRTC] Cleaned up peer connection: {peer_id}")
            
        except Exception as e:
            logger.error(f"[WebRTC] Error cleaning up peer {peer_id}: {e}", exc_info=True)
    
    async def _cleanup_loop(self):
        """Background task to cleanup idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                idle_peers = []
                
                async with self._lock:
                    for peer_id, connection_data in self._connections.items():
                        idle_time = current_time - connection_data.last_activity
                        
                        if idle_time > self.connection_timeout_seconds:
                            idle_peers.append(peer_id)
                            logger.info(f"[WebRTC] Peer {peer_id} idle for {idle_time:.1f}s, cleaning up")
                    
                    # Cleanup idle connections
                    for peer_id in idle_peers:
                        await self._cleanup_connection(peer_id)
                
                if idle_peers:
                    logger.info(f"[WebRTC] Cleaned up {len(idle_peers)} idle connections")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WebRTC] Error in cleanup loop: {e}", exc_info=True)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        async with self._lock:
            return {
                'active_connections': len(self._connections),
                'total_connections': len(self._connections),
                'max_connections': self.max_connections,
                'users_connected': len(self._user_connections),
                'connections_by_state': self._get_connections_by_state()
            }
    
    def _get_connections_by_state(self) -> Dict[str, int]:
        """Get connection count by state."""
        states = {}
        for connection_data in self._connections.values():
            state = connection_data.connection_state
            states[state] = states.get(state, 0) + 1
        return states


# Singleton instance
_webrtc_pool: Optional[WebRTCConnectionPool] = None


def get_webrtc_pool() -> WebRTCConnectionPool:
    """
    Get the global WebRTC connection pool (singleton).
    
    Returns:
        The global WebRTC connection pool instance
    """
    global _webrtc_pool
    
    if _webrtc_pool is None:
        # Use environment variables for configuration
        import os
        max_connections = int(os.getenv('WEBRTC_MAX_CONNECTIONS', '100'))
        connection_timeout = int(os.getenv('WEBRTC_CONNECTION_TIMEOUT', '600'))
        enable_metrics = os.getenv('WEBRTC_ENABLE_METRICS', '1') == '1'
        
        _webrtc_pool = WebRTCConnectionPool(
            max_connections=max_connections,
            connection_timeout_seconds=connection_timeout,
            enable_metrics=enable_metrics
        )
        
        # Start the pool (will be managed by application lifecycle)
        # Note: This should ideally be called during app startup
    
    return _webrtc_pool


async def initialize_webrtc_pool():
    """Initialize and start the WebRTC connection pool (call during app startup)."""
    pool = get_webrtc_pool()
    await pool.start()
    logger.info("[WebRTC] Connection pool initialized and started")


async def shutdown_webrtc_pool():
    """Shutdown the WebRTC connection pool (call during app shutdown)."""
    global _webrtc_pool
    
    if _webrtc_pool:
        await _webrtc_pool.stop()
        _webrtc_pool = None
        logger.info("[WebRTC] Connection pool shut down")


__all__ = [
    'WebRTCConnectionPool',
    'WebRTCConnectionData',
    'PeerConnectionState',
    'ICEConnectionState',
    'ICEGatheringState',
    'get_webrtc_pool',
    'initialize_webrtc_pool',
    'shutdown_webrtc_pool'
]
