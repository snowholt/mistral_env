"""Manages streaming sessions with proper context isolation."""

from typing import Dict, Optional
import asyncio
import logging
from dataclasses import dataclass

from .streaming_session import StreamingSession
from .utterance_manager import UtteranceManager
from .transcription_filter import TranscriptionFilter

logger = logging.getLogger(__name__)

@dataclass
class StreamingSessionManager:
    """Manages streaming voice sessions with proper isolation."""
    
    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
        self.utterance_managers: Dict[str, UtteranceManager] = {}
        self.transcription_filters: Dict[str, TranscriptionFilter] = {}
        self.lock = asyncio.Lock()
        
    async def create_session(self, session_id: str, connection_id: str, language: str) -> StreamingSession:
        """Create a new streaming session with proper initialization."""
        async with self.lock:
            if session_id in self.sessions:
                # Clean up existing session first
                await self.cleanup_session(session_id)
            
            # Create new session
            session = StreamingSession(
                connection_id=connection_id,
                session_id=session_id,
                language=language
            )
            self.sessions[session_id] = session
            
            # Initialize managers for this session
            self.utterance_managers[session_id] = UtteranceManager()
            self.transcription_filters[session_id] = TranscriptionFilter()
            
            logger.info(f"Created new streaming session: {session_id}")
            return session
    
    async def cleanup_session(self, session_id: str):
        """Properly cleanup session to prevent leaking."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Cleanup ring buffer
            if hasattr(session, 'pcm_buffer') and session.pcm_buffer:
                await session.pcm_buffer.close()
            
            # Reset utterance manager
            if session_id in self.utterance_managers:
                self.utterance_managers[session_id].reset()
                del self.utterance_managers[session_id]
            
            # Reset transcription filter
            if session_id in self.transcription_filters:
                self.transcription_filters[session_id].reset()
                del self.transcription_filters[session_id]
            
            # Remove session
            del self.sessions[session_id]
            logger.info(f"Cleaned up streaming session: {session_id}")
    
    def reset_utterance_context(self, session_id: str):
        """Reset context between utterances to prevent leaking."""
        if session_id in self.utterance_managers:
            # Reset utterance manager for new utterance
            self.utterance_managers[session_id].reset()
            
        if session_id in self.transcription_filters:
            # Reset transcription filter
            self.transcription_filters[session_id].reset()
        
        logger.info(f"Reset utterance context for session: {session_id}")
    
    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def get_utterance_manager(self, session_id: str) -> Optional[UtteranceManager]:
        """Get utterance manager for session."""
        return self.utterance_managers.get(session_id)
    
    def get_transcription_filter(self, session_id: str) -> Optional[TranscriptionFilter]:
        """Get transcription filter for session."""
        return self.transcription_filters.get(session_id)