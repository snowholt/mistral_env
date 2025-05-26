"""
Session service for managing chat session persistence.

This service handles session management including:
- Saving active sessions to files
- Loading sessions from files
- Session data serialization/deserialization
- Session metadata management
- Session cleanup and maintenance
"""
import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..base.base_service import BaseService

logger = logging.getLogger(__name__)


class SessionService(BaseService):
    """Service for session management and persistence."""
    
    def __init__(self):
        super().__init__()
        self.session_storage_dir = Path("sessions")
        self.session_storage_dir.mkdir(exist_ok=True)
    
    def save_session(self, session_id: str, session_data: Dict[str, Any], 
                    output_file: Optional[str] = None) -> bool:
        """
        Save a session to a file.
        
        Args:
            session_id: Unique identifier for the session
            session_data: Session data to save
            output_file: Optional specific output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not output_file:
                output_file = f"beautyai_session_{session_id}.json"
            
            # Ensure output path is absolute
            if not Path(output_file).is_absolute():
                output_file = self.session_storage_dir / output_file
            
            # Prepare session data for serialization
            serializable_data = self._prepare_session_for_serialization(session_data)
            
            # Add metadata
            serializable_data.update({
                "session_id": session_id,
                "saved_timestamp": int(time.time()),
                "version": "1.0"
            })
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session {session_id} saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {str(e)}")
            return False
    
    def load_session(self, input_file: str) -> Optional[Dict[str, Any]]:
        """
        Load a session from a file.
        
        Args:
            input_file: Path to the session file
            
        Returns:
            Dict containing session data or None if failed
        """
        try:
            # Ensure input path is absolute
            if not Path(input_file).is_absolute():
                input_file = self.session_storage_dir / input_file
            
            if not Path(input_file).exists():
                logger.error(f"Session file not found: {input_file}")
                return None
            
            with open(input_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Validate session data format
            if not self._validate_session_data(session_data):
                logger.error(f"Invalid session data format in {input_file}")
                return None
            
            logger.info(f"Session loaded from {input_file}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load session from {input_file}: {str(e)}")
            return None
    
    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions with metadata.
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        try:
            for session_file in self.session_storage_dir.glob("*.json"):
                try:
                    session_data = self.load_session(session_file)
                    if session_data:
                        metadata = {
                            "file_path": str(session_file),
                            "session_id": session_data.get("session_id", "unknown"),
                            "model": session_data.get("model", "unknown"),
                            "saved_timestamp": session_data.get("saved_timestamp", 0),
                            "message_count": len(session_data.get("history", [])),
                            "has_system_message": bool(session_data.get("system_message")),
                            "file_size": session_file.stat().st_size
                        }
                        sessions.append(metadata)
                except Exception as e:
                    logger.warning(f"Could not read session file {session_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
        
        # Sort by saved timestamp (newest first)
        sessions.sort(key=lambda x: x["saved_timestamp"], reverse=True)
        return sessions
    
    def delete_session(self, session_identifier: str) -> bool:
        """
        Delete a saved session.
        
        Args:
            session_identifier: Session ID or file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to find the session file
            if session_identifier.endswith('.json'):
                session_file = Path(session_identifier)
                if not session_file.is_absolute():
                    session_file = self.session_storage_dir / session_file
            else:
                # Search by session ID
                session_file = None
                for potential_file in self.session_storage_dir.glob("*.json"):
                    try:
                        with open(potential_file, 'r') as f:
                            data = json.load(f)
                            if data.get("session_id") == session_identifier:
                                session_file = potential_file
                                break
                    except:
                        continue
                
                if not session_file:
                    logger.error(f"Session not found: {session_identifier}")
                    return False
            
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Session deleted: {session_file}")
                return True
            else:
                logger.error(f"Session file not found: {session_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_identifier}: {str(e)}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up old session files.
        
        Args:
            days_old: Delete sessions older than this many days
            
        Returns:
            int: Number of sessions deleted
        """
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            deleted_count = 0
            
            for session_file in self.session_storage_dir.glob("*.json"):
                try:
                    # Check file modification time
                    if session_file.stat().st_mtime < cutoff_time:
                        session_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old session: {session_file}")
                except Exception as e:
                    logger.warning(f"Could not delete session file {session_file}: {str(e)}")
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {str(e)}")
            return 0
    
    def export_session_report(self, output_file: str) -> bool:
        """
        Export a report of all sessions.
        
        Args:
            output_file: Path for the report file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            sessions = self.list_saved_sessions()
            
            report = {
                "generated_timestamp": int(time.time()),
                "total_sessions": len(sessions),
                "sessions": sessions,
                "summary": {
                    "models_used": list(set(s["model"] for s in sessions)),
                    "total_messages": sum(s["message_count"] for s in sessions),
                    "total_storage_mb": sum(s["file_size"] for s in sessions) / (1024 * 1024),
                    "oldest_session": min((s["saved_timestamp"] for s in sessions), default=0),
                    "newest_session": max((s["saved_timestamp"] for s in sessions), default=0)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session report exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export session report: {str(e)}")
            return False
    
    def _prepare_session_for_serialization(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare session data for JSON serialization.
        
        Args:
            session_data: Raw session data
            
        Returns:
            Dict: Serializable session data
        """
        serializable = {}
        
        # Copy basic fields
        for key in ["model_name", "history", "system_message", "start_time", "config"]:
            if key in session_data:
                serializable[key] = session_data[key]
        
        # Handle special cases
        if "model_name" in session_data:
            serializable["model"] = session_data["model_name"]
        
        # Ensure timestamp is integer
        if "start_time" in serializable:
            serializable["start_time"] = int(serializable["start_time"])
        
        # Clean up any non-serializable objects
        serializable = self._clean_for_json(serializable)
        
        return serializable
    
    def _validate_session_data(self, session_data: Dict[str, Any]) -> bool:
        """
        Validate that session data has required fields.
        
        Args:
            session_data: Session data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ["session_id", "model"]
        
        for field in required_fields:
            if field not in session_data:
                logger.error(f"Missing required field in session data: {field}")
                return False
        
        # Validate history format if present
        if "history" in session_data:
            history = session_data["history"]
            if not isinstance(history, list):
                logger.error("Session history must be a list")
                return False
            
            for msg in history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    logger.error("Invalid message format in session history")
                    return False
        
        return True
    
    def _clean_for_json(self, obj: Any) -> Any:
        """
        Recursively clean an object to make it JSON serializable.
        
        Args:
            obj: Object to clean
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)
