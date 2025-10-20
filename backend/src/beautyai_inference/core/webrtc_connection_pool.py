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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            'client_info': self.client_info
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
        connection_timeout_seconds: int = 300,
        enable_metrics: bool = True
    ):
        """
        Initialize WebRTC connection pool.
        
        Args:
            max_connections: Maximum number of concurrent connections
            connection_timeout_seconds: Timeout for idle connections
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
                
                # Set up event handlers
                @pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    if peer_id in self._connections:
                        self._connections[peer_id].update_connection_state(pc.connectionState)
                
                @pc.on("iceconnectionstatechange")
                async def on_iceconnectionstatechange():
                    if peer_id in self._connections:
                        self._connections[peer_id].update_ice_connection_state(pc.iceConnectionState)
                
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
                        try:
                            # Import voice service adapter (lazy import to avoid circular dependencies)
                            from ..services.voice.webrtc_voice_service_adapter import (
                                WebRTCVoiceServiceAdapter,
                                WebRTCVoiceConfig
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
                                
                                # Create SimpleVoiceService instance
                                simple_voice_service = SimpleVoiceService(language=language)
                                
                                # Create voice adapter with default config
                                voice_config = WebRTCVoiceConfig(
                                    default_language=language
                                )
                                
                                adapter = WebRTCVoiceServiceAdapter(
                                    peer_id=peer_id,
                                    session_id=session_id,
                                    language=language,
                                    voice_service=simple_voice_service,
                                    config=voice_config
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
                await pc.setRemoteDescription(offer)
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                # Store connection data
                connection_data = WebRTCConnectionData(
                    peer_id=peer_id,
                    peer_connection=pc,
                    user_id=user_id,
                    local_sdp=pc.localDescription.sdp,
                    remote_sdp=offer_sdp
                )
                
                self._connections[peer_id] = connection_data
                
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
            return connection_data.to_dict()
    
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
