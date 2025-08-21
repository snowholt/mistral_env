"""
Session Management Service

This service handles chat session lifecycle management for BeautyAI,
including session creation, storage, retrieval, and cleanup.
"""
import logging
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Chat session data structure."""
    session_id: str
    model_name: str
    history: List[Dict[str, str]] = field(default_factory=list)
    system_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    language: str = "ar"
    last_activity: float = field(default_factory=time.time)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def add_message(self, role: str, content: str):
        """Add a message to the session history."""
        self.history.append({"role": role, "content": content})
        self.update_activity()
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history.clear()
        self.update_activity()
    
    def get_duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time
    
    def is_expired(self, max_age_hours: float = 24.0) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > (max_age_hours * 3600)


class SessionManagerService:
    """
    Service for managing chat sessions.
    
    Handles session creation, storage, retrieval, cleanup, and expiration.
    """
    
    def __init__(self, max_sessions: int = 1000, cleanup_interval_minutes: int = 60):
        self.active_sessions: Dict[str, ChatSession] = {}
        self.max_sessions = max_sessions
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def create_session(
        self,
        model_name: str,
        language: str = "ar",
        system_message: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            model_name: Name of the model for this session
            language: Language for the session
            system_message: Optional system message
            config: Optional configuration dict
            
        Returns:
            str: Session ID
        """
        with self._lock:
            # Generate unique session ID
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            session_id = f"chat_{model_name}_{timestamp}"
            
            # Create session
            session = ChatSession(
                session_id=session_id,
                model_name=model_name,
                language=language,
                system_message=system_message,
                config=config or {}
            )
            
            # Clean up old sessions if needed
            self._cleanup_if_needed()
            
            # Store session
            self.active_sessions[session_id] = session
            
            logger.info(f"Created new chat session: {session_id} for model: {model_name}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            ChatSession or None if not found
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    def update_session_history(
        self,
        session_id: str,
        history: List[Dict[str, str]]
    ) -> bool:
        """
        Update session conversation history.
        
        Args:
            session_id: Session ID
            history: New conversation history
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.history = history[:]
                session.update_activity()
                return True
            return False
    
    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Add a message to session history.
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            
        Returns:
            bool: True if added successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.add_message(role, content)
                return True
            return False
    
    def clear_session_history(self, session_id: str) -> bool:
        """
        Clear session conversation history.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if cleared successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.clear_history()
                return True
            return False
    
    def update_session_config(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Update session configuration.
        
        Args:
            session_id: Session ID
            config: New configuration dict
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.config.update(config)
                session.update_activity()
                return True
            return False
    
    def update_session_system_message(
        self,
        session_id: str,
        system_message: str
    ) -> bool:
        """
        Update session system message.
        
        Args:
            session_id: Session ID
            system_message: New system message
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.system_message = system_message
                session.update_activity()
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if deleted successfully
        """
        with self._lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of session summaries
        """
        with self._lock:
            sessions = []
            for session_id, session in self.active_sessions.items():
                sessions.append({
                    "session_id": session_id,
                    "model_name": session.model_name,
                    "language": session.language,
                    "message_count": len(session.history),
                    "duration": session.get_duration(),
                    "last_activity": session.last_activity,
                    "has_system_message": session.system_message is not None
                })
            return sessions
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.
        
        Returns:
            Dict with statistics
        """
        with self._lock:
            return {
                "total_sessions": len(self.active_sessions),
                "max_sessions": self.max_sessions,
                "cleanup_interval_minutes": self.cleanup_interval_minutes,
                "sessions_by_model": self._get_sessions_by_model(),
                "sessions_by_language": self._get_sessions_by_language()
            }
    
    def _get_sessions_by_model(self) -> Dict[str, int]:
        """Get count of sessions by model."""
        counts = {}
        for session in self.active_sessions.values():
            model = session.model_name
            counts[model] = counts.get(model, 0) + 1
        return counts
    
    def _get_sessions_by_language(self) -> Dict[str, int]:
        """Get count of sessions by language."""
        counts = {}
        for session in self.active_sessions.values():
            lang = session.language
            counts[lang] = counts.get(lang, 0) + 1
        return counts
    
    def _cleanup_if_needed(self):
        """Clean up sessions if max limit is reached."""
        if len(self.active_sessions) >= self.max_sessions:
            # Remove oldest sessions (by last activity)
            sessions_to_remove = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            # Remove oldest 10% of sessions
            remove_count = max(1, len(sessions_to_remove) // 10)
            for i in range(remove_count):
                session_id = sessions_to_remove[i][0]
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up old session: {session_id}")
    
    def _cleanup_expired_sessions(self, max_age_hours: float = 24.0):
        """Clean up expired sessions."""
        with self._lock:
            expired_sessions = []
            for session_id, session in self.active_sessions.items():
                if session.is_expired(max_age_hours):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_minutes * 60):
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in session cleanup: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started session cleanup thread")
    
    def shutdown(self):
        """Shutdown the session manager."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        with self._lock:
            session_count = len(self.active_sessions)
            self.active_sessions.clear()
            logger.info(f"Session manager shutdown. Cleared {session_count} sessions")


# Global instance for easy access
_shared_session_manager = None


def get_shared_session_manager() -> SessionManagerService:
    """
    Get the shared SessionManagerService instance.
    
    Returns:
        SessionManagerService: The shared singleton instance
    """
    global _shared_session_manager
    if _shared_session_manager is None:
        _shared_session_manager = SessionManagerService()
    return _shared_session_manager