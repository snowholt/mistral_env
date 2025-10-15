"""
WebRTC Session Manager - Extends Voice Session Management for WebRTC.

This module wraps VoiceSessionManager with WebRTC-specific metadata and
session tracking, providing unified session management for WebRTC voice calls.

Created for WebRTC MVP Migration - Phase B
Author: BeautyAI Framework
Date: October 15, 2025
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from .voice_session_manager import VoiceSessionManager, VoiceSessionState, VoiceConversationTurn
from .config_manager import get_config_manager

logger = logging.getLogger(__name__)


@dataclass
class WebRTCSessionMetadata:
    """WebRTC-specific session metadata."""
    
    peer_id: str
    ice_connection_state: str = "new"
    ice_gathering_state: str = "new"
    connection_state: str = "new"
    
    # Client capabilities
    client_codecs: List[str] = field(default_factory=list)
    client_transport: str = "unknown"
    client_user_agent: Optional[str] = None
    
    # Performance metrics
    connection_established_at: Optional[float] = None
    first_audio_received_at: Optional[float] = None
    audio_track_active: bool = False
    
    # Quality metrics
    total_packets_received: int = 0
    total_packets_lost: int = 0
    total_bytes_received: int = 0
    average_jitter_ms: float = 0.0
    
    def update_ice_state(self, state: str):
        """Update ICE connection state."""
        self.ice_connection_state = state
        if state == "connected" and not self.connection_established_at:
            self.connection_established_at = time.time()
    
    def update_audio_track(self, active: bool):
        """Update audio track status."""
        self.audio_track_active = active
        if active and not self.first_audio_received_at:
            self.first_audio_received_at = time.time()
    
    def update_quality_metrics(
        self,
        packets_received: int = 0,
        packets_lost: int = 0,
        bytes_received: int = 0,
        jitter_ms: float = 0.0
    ):
        """Update quality metrics."""
        self.total_packets_received += packets_received
        self.total_packets_lost += packets_lost
        self.total_bytes_received += bytes_received
        if jitter_ms > 0:
            # Running average
            self.average_jitter_ms = (self.average_jitter_ms + jitter_ms) / 2
    
    def get_packet_loss_rate(self) -> float:
        """Calculate packet loss rate."""
        total = self.total_packets_received + self.total_packets_lost
        if total == 0:
            return 0.0
        return self.total_packets_lost / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'peer_id': self.peer_id,
            'ice_connection_state': self.ice_connection_state,
            'ice_gathering_state': self.ice_gathering_state,
            'connection_state': self.connection_state,
            'client_codecs': self.client_codecs,
            'client_transport': self.client_transport,
            'client_user_agent': self.client_user_agent,
            'connection_established_at': self.connection_established_at,
            'first_audio_received_at': self.first_audio_received_at,
            'audio_track_active': self.audio_track_active,
            'total_packets_received': self.total_packets_received,
            'total_packets_lost': self.total_packets_lost,
            'total_bytes_received': self.total_bytes_received,
            'average_jitter_ms': self.average_jitter_ms,
            'packet_loss_rate': self.get_packet_loss_rate()
        }


class WebRTCSessionManager:
    """
    Session manager for WebRTC voice conversations.
    
    Features:
    - Wraps VoiceSessionManager for core session management
    - Adds WebRTC-specific metadata (ICE state, quality metrics)
    - Links peer_id to session_id for connection tracking
    - Provides WebRTC-aware session lifecycle
    - Tracks connection quality and performance
    """
    
    def __init__(
        self,
        voice_session_manager: Optional[VoiceSessionManager] = None,
        persist_sessions: bool = True,
        session_dir: Optional[Path] = None,
        auto_cleanup_files: bool = True
    ):
        """
        Initialize WebRTC session manager.
        
        Args:
            voice_session_manager: Existing voice session manager (or create new)
            persist_sessions: Whether to persist sessions to disk
            session_dir: Directory for session persistence
            auto_cleanup_files: Auto-cleanup session files on session end
        """
        # Use provided voice session manager or create new one
        if voice_session_manager is None:
            if session_dir is None:
                session_dir = Path("sessions/voice/webrtc")
            
            self.voice_session_manager = VoiceSessionManager(
                persist_sessions=persist_sessions,
                session_dir=session_dir,
                auto_cleanup_files=auto_cleanup_files
            )
        else:
            self.voice_session_manager = voice_session_manager
        
        # WebRTC-specific tracking
        self._webrtc_metadata: Dict[str, WebRTCSessionMetadata] = {}
        self._peer_to_session: Dict[str, str] = {}  # peer_id -> session_id
        self._session_to_peer: Dict[str, str] = {}  # session_id -> peer_id
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("[WebRTC] Session manager initialized")
    
    async def create_session(
        self,
        peer_id: str,
        language: str = "ar",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new WebRTC voice session.
        
        Args:
            peer_id: WebRTC peer connection identifier
            language: Conversation language (ar, en)
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Session identifier
        """
        async with self._lock:
            # Generate session ID
            session_id = f"webrtc_session_{uuid.uuid4().hex[:12]}"
            
            # Prepare session metadata
            session_metadata = metadata or {}
            session_metadata['peer_id'] = peer_id
            session_metadata['transport'] = 'webrtc'
            
            # Get voice type from config or default
            config_manager = get_config_manager()
            duplex_config = config_manager.get_value('duplex', {})
            voice_type = duplex_config.get('tts_voice', 'ar-SA-ZariyahNeural')
            
            # Create voice session
            self.voice_session_manager.create_session(
                session_id=session_id,
                connection_id=peer_id,  # Use peer_id as connection_id
                user_id=user_id,
                language=language,
                voice_type=voice_type,
                metadata=session_metadata
            )
            
            # Create WebRTC metadata
            webrtc_metadata = WebRTCSessionMetadata(
                peer_id=peer_id,
                client_user_agent=session_metadata.get('user_agent')
            )
            
            self._webrtc_metadata[session_id] = webrtc_metadata
            self._peer_to_session[peer_id] = session_id
            self._session_to_peer[session_id] = peer_id
            
            logger.info(f"[WebRTC] Created session: session_id={session_id}, peer_id={peer_id}, language={language}")
            
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[VoiceSessionState]:
        """
        Get voice session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            VoiceSessionState or None if not found
        """
        return self.voice_session_manager.get_session(session_id)
    
    async def get_session_by_peer(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information by peer_id.
        
        Args:
            peer_id: Peer connection identifier
            
        Returns:
            Dictionary with session info or None
        """
        async with self._lock:
            if peer_id not in self._peer_to_session:
                return None
            
            session_id = self._peer_to_session[peer_id]
            session_state = self.voice_session_manager.get_session(session_id)
            
            if not session_state:
                return None
            
            webrtc_metadata = self._webrtc_metadata.get(session_id)
            
            return {
                'session_id': session_id,
                'peer_id': peer_id,
                'user_id': session_state.user_id,
                'language': session_state.language,
                'voice_type': session_state.voice_type,
                'turn_count': session_state.turn_count,
                'created_at': session_state.created_at,
                'last_activity': session_state.last_activity,
                'webrtc_metadata': webrtc_metadata.to_dict() if webrtc_metadata else None
            }
    
    async def add_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        ai_response: str,
        processing_time_ms: int,
        audio_duration_ms: Optional[int] = None,
        transcription_quality: str = "ok"
    ):
        """
        Add a conversation turn to the session.
        
        Args:
            session_id: Session identifier
            user_input: User's transcribed input
            ai_response: AI's text response
            processing_time_ms: Total processing time
            audio_duration_ms: Duration of user's audio
            transcription_quality: Quality indicator
        """
        session_state = self.voice_session_manager.get_session(session_id)
        
        if not session_state:
            logger.warning(f"[WebRTC] Session not found: {session_id}")
            return
        
        # Create turn
        turn = VoiceConversationTurn(
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            user_input=user_input,
            ai_response=ai_response,
            language=session_state.language,
            voice_type=session_state.voice_type,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
            transcription_quality=transcription_quality
        )
        
        # Add to session
        self.voice_session_manager.add_turn(session_id, turn)
        
        logger.debug(f"[WebRTC] Added turn to session {session_id}: processing_time={processing_time_ms}ms")
    
    async def update_webrtc_metadata(
        self,
        session_id: str,
        **kwargs
    ):
        """
        Update WebRTC-specific metadata for a session.
        
        Args:
            session_id: Session identifier
            **kwargs: Metadata fields to update
        """
        async with self._lock:
            if session_id not in self._webrtc_metadata:
                logger.warning(f"[WebRTC] WebRTC metadata not found for session: {session_id}")
                return
            
            metadata = self._webrtc_metadata[session_id]
            
            # Update ICE states
            if 'ice_connection_state' in kwargs:
                metadata.update_ice_state(kwargs['ice_connection_state'])
            
            if 'ice_gathering_state' in kwargs:
                metadata.ice_gathering_state = kwargs['ice_gathering_state']
            
            if 'connection_state' in kwargs:
                metadata.connection_state = kwargs['connection_state']
            
            # Update audio track
            if 'audio_track_active' in kwargs:
                metadata.update_audio_track(kwargs['audio_track_active'])
            
            # Update quality metrics
            if any(k in kwargs for k in ['packets_received', 'packets_lost', 'bytes_received', 'jitter_ms']):
                metadata.update_quality_metrics(
                    packets_received=kwargs.get('packets_received', 0),
                    packets_lost=kwargs.get('packets_lost', 0),
                    bytes_received=kwargs.get('bytes_received', 0),
                    jitter_ms=kwargs.get('jitter_ms', 0.0)
                )
            
            logger.debug(f"[WebRTC] Updated metadata for session {session_id}")
    
    async def update_by_peer(
        self,
        peer_id: str,
        **kwargs
    ):
        """
        Update WebRTC metadata by peer_id.
        
        Args:
            peer_id: Peer connection identifier
            **kwargs: Metadata fields to update
        """
        async with self._lock:
            if peer_id not in self._peer_to_session:
                logger.warning(f"[WebRTC] No session found for peer: {peer_id}")
                return
            
            session_id = self._peer_to_session[peer_id]
        
        await self.update_webrtc_metadata(session_id, **kwargs)
    
    async def get_recent_context(
        self,
        session_id: str,
        max_turns: int = 3
    ) -> str:
        """
        Get recent conversation context for AI model.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns
            
        Returns:
            Formatted context string
        """
        session_state = self.voice_session_manager.get_session(session_id)
        
        if not session_state:
            return ""
        
        return session_state.get_recent_context(max_turns)
    
    async def cleanup_session(self, peer_id: str):
        """
        Cleanup session by peer_id.
        
        Args:
            peer_id: Peer connection identifier
        """
        async with self._lock:
            if peer_id not in self._peer_to_session:
                logger.debug(f"[WebRTC] No session found for cleanup: peer_id={peer_id}")
                return
            
            session_id = self._peer_to_session[peer_id]
            
            # End voice session
            self.voice_session_manager.end_session(session_id)
            
            # Remove WebRTC metadata
            if session_id in self._webrtc_metadata:
                del self._webrtc_metadata[session_id]
            
            # Remove mappings
            del self._peer_to_session[peer_id]
            if session_id in self._session_to_peer:
                del self._session_to_peer[session_id]
            
            logger.info(f"[WebRTC] Cleaned up session: session_id={session_id}, peer_id={peer_id}")
    
    async def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive session statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        session_state = self.voice_session_manager.get_session(session_id)
        
        if not session_state:
            return None
        
        async with self._lock:
            webrtc_metadata = self._webrtc_metadata.get(session_id)
            
            return {
                'session_id': session_id,
                'turn_count': session_state.turn_count,
                'average_response_time_ms': session_state.average_response_time_ms,
                'transcription_success_rate': session_state.transcription_success_rate,
                'total_processing_time_ms': session_state.total_processing_time_ms,
                'created_at': session_state.created_at,
                'last_activity': session_state.last_activity,
                'session_duration_seconds': time.time() - session_state.created_at,
                'webrtc_metadata': webrtc_metadata.to_dict() if webrtc_metadata else None
            }
    
    async def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all active WebRTC sessions.
        
        Returns:
            List of session dictionaries
        """
        async with self._lock:
            sessions = []
            
            for peer_id, session_id in self._peer_to_session.items():
                session_info = await self.get_session_by_peer(peer_id)
                if session_info:
                    sessions.append(session_info)
            
            return sessions
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._peer_to_session)


# Singleton instance
_webrtc_session_manager: Optional[WebRTCSessionManager] = None


def get_webrtc_session_manager() -> WebRTCSessionManager:
    """
    Get or create WebRTC session manager singleton.
    
    Returns:
        WebRTCSessionManager instance
    """
    global _webrtc_session_manager
    
    if _webrtc_session_manager is None:
        config_manager = get_config_manager()
        
        # Get session persistence settings
        # For now, use defaults - can be made configurable later
        session_dir = Path("sessions/voice/webrtc")
        
        _webrtc_session_manager = WebRTCSessionManager(
            persist_sessions=True,
            session_dir=session_dir,
            auto_cleanup_files=True
        )
        
        logger.info("[WebRTC] Session manager singleton created")
    
    return _webrtc_session_manager


def reset_webrtc_session_manager():
    """Reset the session manager singleton (for testing)."""
    global _webrtc_session_manager
    _webrtc_session_manager = None


__all__ = [
    'WebRTCSessionManager',
    'WebRTCSessionMetadata',
    'get_webrtc_session_manager',
    'reset_webrtc_session_manager'
]
