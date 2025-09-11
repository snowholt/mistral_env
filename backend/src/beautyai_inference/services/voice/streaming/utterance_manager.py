"""Manages utterance boundaries and prevents context leaking."""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass 
class UtteranceContext:
    """Context for a single utterance."""
    utterance_id: int
    tokens: List[str] = field(default_factory=list)
    final_text: str = ""
    is_finalized: bool = False
    start_time: float = 0.0
    end_time: Optional[float] = None

class UtteranceManager:
    """Manages utterance lifecycle and prevents context leaking."""
    
    def __init__(self):
        self.current_utterance: Optional[UtteranceContext] = None
        self.utterance_history: List[UtteranceContext] = []
        self.max_history: int = 10
        self._lock = asyncio.Lock()
        
    async def start_new_utterance(self, utterance_id: int) -> UtteranceContext:
        """Start a new utterance with clean context."""
        async with self._lock:
            # Finalize current if exists
            if self.current_utterance and not self.current_utterance.is_finalized:
                await self.finalize_utterance()
            
            # Create new utterance with fresh context
            self.current_utterance = UtteranceContext(
                utterance_id=utterance_id,
                start_time=asyncio.get_event_loop().time()
            )
            
            logger.info(f"Started new utterance {utterance_id} with clean context")
            return self.current_utterance
    
    async def update_utterance(self, tokens: List[str], text: str):
        """Update current utterance without leaking previous context."""
        async with self._lock:
            if not self.current_utterance:
                logger.warning("No active utterance to update")
                return
            
            # Only update current utterance, don't mix with history
            self.current_utterance.tokens = tokens.copy()  # Copy to prevent reference issues
            self.current_utterance.final_text = text
    
    async def finalize_utterance(self) -> Optional[str]:
        """Finalize current utterance and clear its context."""
        async with self._lock:
            if not self.current_utterance:
                return None
            
            self.current_utterance.is_finalized = True
            self.current_utterance.end_time = asyncio.get_event_loop().time()
            
            # Add to history
            self.utterance_history.append(self.current_utterance)
            if len(self.utterance_history) > self.max_history:
                self.utterance_history.pop(0)
            
            final_text = self.current_utterance.final_text
            
            # Clear current utterance to prevent leaking
            self.current_utterance = None
            
            logger.info(f"Finalized utterance with text: {final_text[:50]}...")
            return final_text
    
    def get_clean_context(self) -> Dict:
        """Get clean context for current utterance only."""
        if not self.current_utterance:
            return {"tokens": [], "text": ""}
        
        return {
            "tokens": self.current_utterance.tokens.copy(),
            "text": self.current_utterance.final_text,
            "utterance_id": self.current_utterance.utterance_id
        }
    
    def reset(self):
        """Complete reset for new session."""
        self.current_utterance = None
        self.utterance_history.clear()
        logger.info("Utterance manager reset")