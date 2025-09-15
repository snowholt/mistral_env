"""
Voice Session Manager for tracking conversation context and state.

This module provides enhanced session management for voice conversations,
including context preservation, conversation history, and state tracking.

Author: BeautyAI Framework
Date: 2025-01-23
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class VoiceConversationTurn:
    """Represents a single turn in a voice conversation."""
    turn_id: str
    timestamp: float
    user_input: str
    ai_response: str
    language: str
    voice_type: str
    processing_time_ms: int
    audio_duration_ms: Optional[int] = None
    transcription_quality: str = "ok"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "ai_response": self.ai_response,
            "language": self.language,
            "voice_type": self.voice_type,
            "processing_time_ms": self.processing_time_ms,
            "audio_duration_ms": self.audio_duration_ms,
            "transcription_quality": self.transcription_quality
        }


@dataclass
class VoiceSessionState:
    """State information for a voice conversation session."""
    session_id: str
    connection_id: str
    user_id: Optional[str]
    language: str
    voice_type: str
    created_at: float
    last_activity: float
    turn_count: int = 0
    conversation_history: List[VoiceConversationTurn] = field(default_factory=list)
    context_summary: str = ""
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    total_processing_time_ms: int = 0
    average_response_time_ms: float = 0.0
    
    # Quality metrics
    transcription_success_rate: float = 1.0
    unclear_transcriptions: int = 0
    
    def update_metrics(self, turn: VoiceConversationTurn):
        """Update session metrics with new turn data."""
        self.turn_count += 1
        self.last_activity = time.time()
        self.total_processing_time_ms += turn.processing_time_ms
        self.average_response_time_ms = self.total_processing_time_ms / self.turn_count
        
        # Update quality metrics
        if turn.transcription_quality == "unclear":
            self.unclear_transcriptions += 1
        
        self.transcription_success_rate = (
            (self.turn_count - self.unclear_transcriptions) / self.turn_count
            if self.turn_count > 0 else 1.0
        )
    
    def get_recent_context(self, max_turns: int = 3) -> str:
        """
        Get recent conversation context for AI model.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted context string
        """
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-max_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_input}")
            context_parts.append(f"AI: {turn.ai_response}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "language": self.language,
            "voice_type": self.voice_type,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "turn_count": self.turn_count,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "context_summary": self.context_summary,
            "session_metadata": self.session_metadata,
            "total_processing_time_ms": self.total_processing_time_ms,
            "average_response_time_ms": self.average_response_time_ms,
            "transcription_success_rate": self.transcription_success_rate,
            "unclear_transcriptions": self.unclear_transcriptions
        }


class VoiceSessionManager:
    """
    Manager for voice conversation sessions with context preservation.
    
    Features:
    - Session lifecycle management
    - Conversation history tracking
    - Context preservation across turns
    - Performance and quality metrics
    - Session persistence (optional)
    """
    
    def __init__(self, persist_sessions: bool = False, session_dir: Optional[Path] = None, auto_cleanup_files: bool = True):
        """
        Initialize the Voice Session Manager.
        
        Args:
            persist_sessions: Whether to persist sessions to disk
            session_dir: Directory for session persistence
            auto_cleanup_files: Whether to automatically delete session files when sessions expire/close
        """
        self.persist_sessions = persist_sessions
        self.session_dir = session_dir or Path("sessions/voice")
        self.auto_cleanup_files = auto_cleanup_files
        
        # In-memory session storage
        self.active_sessions: Dict[str, VoiceSessionState] = {}
        
        # Session configuration
        self.max_history_turns = 10  # Maximum turns to keep in memory
        self.session_timeout_minutes = 30  # Session timeout
        self.context_window_turns = 3  # Turns to include in AI context
        
        # Create session directory if persistence enabled
        if self.persist_sessions:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VoiceSessionManager initialized (persist: {persist_sessions})")
    
    async def create_session(
        self,
        connection_id: str,
        language: str,
        voice_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> VoiceSessionState:
        """
        Create a new voice conversation session.
        
        Args:
            connection_id: WebSocket connection ID
            language: Session language (ar/en)
            voice_type: Voice type (male/female)
            user_id: Optional user identifier
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Created session state
        """
        if session_id is None:
            session_id = f"voice_{uuid.uuid4().hex[:12]}"
        
        # Check if session already exists
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} already exists, returning existing session")
            return self.active_sessions[session_id]
        
        # Create new session
        session = VoiceSessionState(
            session_id=session_id,
            connection_id=connection_id,
            user_id=user_id,
            language=language,
            voice_type=voice_type,
            created_at=time.time(),
            last_activity=time.time()
        )
        
        # Store in memory
        self.active_sessions[session_id] = session
        
        # Persist if enabled
        if self.persist_sessions:
            await self._persist_session(session)
        
        logger.info(f"Created voice session: {session_id} (connection: {connection_id})")
        return session
    
    async def get_session(self, session_id: str) -> Optional[VoiceSessionState]:
        """
        Get an existing session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state if found, None otherwise
        """
        # Check memory first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Check if session has timed out
            if self._is_session_expired(session):
                await self.close_session(session_id)
                return None
            
            return session
        
        # Try to load from disk if persistence enabled
        if self.persist_sessions:
            session = await self._load_session(session_id)
            if session and not self._is_session_expired(session):
                self.active_sessions[session_id] = session
                return session
        
        return None
    
    async def add_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        ai_response: str,
        processing_time_ms: int,
        audio_duration_ms: Optional[int] = None,
        transcription_quality: str = "ok"
    ) -> bool:
        """
        Add a conversation turn to the session.
        
        Args:
            session_id: Session identifier
            user_input: User's transcribed input
            ai_response: AI's response text
            processing_time_ms: Processing time in milliseconds
            audio_duration_ms: Audio duration in milliseconds
            transcription_quality: Quality of transcription (ok/unclear)
            
        Returns:
            True if turn was added successfully
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for turn addition")
            return False
        
        # Create turn
        turn = VoiceConversationTurn(
            turn_id=f"turn_{len(session.conversation_history) + 1}_{int(time.time() * 1000)}",
            timestamp=time.time(),
            user_input=user_input,
            ai_response=ai_response,
            language=session.language,
            voice_type=session.voice_type,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
            transcription_quality=transcription_quality
        )
        
        # Add to conversation history
        session.conversation_history.append(turn)
        
        # Update session metrics
        session.update_metrics(turn)
        
        # Trim history if too long
        if len(session.conversation_history) > self.max_history_turns:
            session.conversation_history = session.conversation_history[-self.max_history_turns:]
        
        # Update context summary if needed
        if len(session.conversation_history) % 5 == 0:  # Every 5 turns
            session.context_summary = self._generate_context_summary(session)
        
        # Persist if enabled
        if self.persist_sessions:
            await self._persist_session(session)
        
        logger.debug(f"Added turn to session {session_id}: {turn.turn_id}")
        return True
    
    async def get_conversation_context(self, session_id: str) -> str:
        """
        Get conversation context for AI model.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted context string
        """
        session = await self.get_session(session_id)
        if not session:
            return ""
        
        return session.get_recent_context(self.context_window_turns)
    
    async def close_session(self, session_id: str, delete_file: Optional[bool] = None) -> bool:
        """
        Close and cleanup a session.
        
        Args:
            session_id: Session identifier
            delete_file: Whether to delete the session file (defaults to auto_cleanup_files setting)
            
        Returns:
            True if session was closed successfully
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found for closure")
            return False
        
        session = self.active_sessions[session_id]
        
        # Final persistence if enabled and not deleting file
        should_delete_file = delete_file if delete_file is not None else self.auto_cleanup_files
        if self.persist_sessions and not should_delete_file:
            await self._persist_session(session)
        
        # Remove from memory
        del self.active_sessions[session_id]
        
        # Delete session file if auto-cleanup is enabled
        if should_delete_file and self.persist_sessions:
            await self._delete_session_file(session_id)
        
        logger.info(f"Closed voice session: {session_id} (turns: {session.turn_count}, file_deleted: {should_delete_file})")
        return True
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions and optionally orphaned session files.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = []
        
        # Clean up expired in-memory sessions
        for session_id, session in self.active_sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id, delete_file=True)  # Force file deletion for expired sessions
        
        # Clean up orphaned session files (files without active sessions)
        orphaned_files_count = 0
        if self.auto_cleanup_files and self.persist_sessions and self.session_dir.exists():
            try:
                current_time = time.time()
                timeout_seconds = self.session_timeout_minutes * 60
                
                for session_file in self.session_dir.glob("*.json"):
                    try:
                        # Check if file is older than timeout
                        file_age = current_time - session_file.stat().st_mtime
                        if file_age > timeout_seconds:
                            # Extract session_id from filename
                            session_id = session_file.stem
                            
                            # Only delete if not in active sessions
                            if session_id not in self.active_sessions:
                                session_file.unlink()
                                orphaned_files_count += 1
                                logger.debug(f"Deleted orphaned session file: {session_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to process session file {session_file}: {e}")
            except Exception as e:
                logger.error(f"Error during orphaned files cleanup: {e}")
        
        total_cleaned = len(expired_sessions) + orphaned_files_count
        if total_cleaned > 0:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions and {orphaned_files_count} orphaned files")
        
        return total_cleaned
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        total_sessions = len(self.active_sessions)
        total_turns = sum(session.turn_count for session in self.active_sessions.values())
        
        if total_sessions > 0:
            avg_turns_per_session = total_turns / total_sessions
            avg_response_time = sum(
                session.average_response_time_ms for session in self.active_sessions.values()
            ) / total_sessions
            avg_transcription_quality = sum(
                session.transcription_success_rate for session in self.active_sessions.values()
            ) / total_sessions
        else:
            avg_turns_per_session = 0
            avg_response_time = 0
            avg_transcription_quality = 1.0
        
        return {
            "active_sessions": total_sessions,
            "total_turns": total_turns,
            "average_turns_per_session": avg_turns_per_session,
            "average_response_time_ms": avg_response_time,
            "average_transcription_quality": avg_transcription_quality,
            "session_timeout_minutes": self.session_timeout_minutes,
            "persistence_enabled": self.persist_sessions
        }
    
    def _is_session_expired(self, session: VoiceSessionState) -> bool:
        """Check if a session has expired."""
        timeout_seconds = self.session_timeout_minutes * 60
        return (time.time() - session.last_activity) > timeout_seconds
    
    def _generate_context_summary(self, session: VoiceSessionState) -> str:
        """Generate a summary of the conversation context."""
        if not session.conversation_history:
            return ""
        
        # Simple summary - in a real implementation, this could use an LLM
        recent_topics = []
        for turn in session.conversation_history[-5:]:  # Last 5 turns
            if len(turn.user_input) > 10:  # Meaningful input
                recent_topics.append(turn.user_input[:50])  # First 50 chars
        
        if recent_topics:
            return f"Recent topics: {'; '.join(recent_topics)}"
        else:
            return ""
    
    async def _persist_session(self, session: VoiceSessionState):
        """Persist session to disk."""
        try:
            session_file = self.session_dir / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist session {session.session_id}: {e}")
    
    async def _delete_session_file(self, session_id: str):
        """Delete session file from disk."""
        try:
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                logger.debug(f"Deleted session file: {session_file.name}")
            else:
                logger.debug(f"Session file not found for deletion: {session_file.name}")
        except Exception as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")
    
    async def _load_session(self, session_id: str) -> Optional[VoiceSessionState]:
        """Load session from disk."""
        try:
            session_file = self.session_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct session state
            session = VoiceSessionState(
                session_id=data["session_id"],
                connection_id=data["connection_id"],
                user_id=data.get("user_id"),
                language=data["language"],
                voice_type=data["voice_type"],
                created_at=data["created_at"],
                last_activity=data["last_activity"],
                turn_count=data["turn_count"],
                context_summary=data.get("context_summary", ""),
                session_metadata=data.get("session_metadata", {}),
                total_processing_time_ms=data.get("total_processing_time_ms", 0),
                average_response_time_ms=data.get("average_response_time_ms", 0.0),
                transcription_success_rate=data.get("transcription_success_rate", 1.0),
                unclear_transcriptions=data.get("unclear_transcriptions", 0)
            )
            
            # Reconstruct conversation history
            for turn_data in data.get("conversation_history", []):
                turn = VoiceConversationTurn(
                    turn_id=turn_data["turn_id"],
                    timestamp=turn_data["timestamp"],
                    user_input=turn_data["user_input"],
                    ai_response=turn_data["ai_response"],
                    language=turn_data["language"],
                    voice_type=turn_data["voice_type"],
                    processing_time_ms=turn_data["processing_time_ms"],
                    audio_duration_ms=turn_data.get("audio_duration_ms"),
                    transcription_quality=turn_data.get("transcription_quality", "ok")
                )
                session.conversation_history.append(turn)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None


# Global session manager instance
_voice_session_manager: Optional[VoiceSessionManager] = None


def get_voice_session_manager(
    persist_sessions: bool = False, 
    session_dir: Optional[Path] = None,
    auto_cleanup_files: bool = True
) -> VoiceSessionManager:
    """
    Get the global voice session manager instance.
    
    Args:
        persist_sessions: Whether to enable session persistence
        session_dir: Directory for session persistence
        auto_cleanup_files: Whether to automatically delete session files when sessions expire/close
        
    Returns:
        Voice session manager instance
    """
    global _voice_session_manager
    
    if _voice_session_manager is None:
        _voice_session_manager = VoiceSessionManager(persist_sessions, session_dir, auto_cleanup_files)
    
    return _voice_session_manager


async def cleanup_voice_sessions():
    """Cleanup expired voice sessions (utility function)."""
    global _voice_session_manager
    
    if _voice_session_manager:
        return await _voice_session_manager.cleanup_expired_sessions()
    
    return 0