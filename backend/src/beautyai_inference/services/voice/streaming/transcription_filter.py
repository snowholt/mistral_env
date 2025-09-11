"""Filter for cleaning up transcription outputs and preventing phantom text."""

import re
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionFilter:
    """Filters and validates transcription outputs."""
    
    # Common phantom phrases to filter
    phantom_phrases: List[str] = field(default_factory=lambda: [
        "thank you",
        "yeah", 
        "no",
        "um",
        "uh",
        "okay",
        "yes"
    ])
    
    # Minimum confidence threshold for accepting transcription
    min_confidence: float = 0.7
    
    # Track recent transcriptions to detect repetitions
    recent_history: List[str] = field(default_factory=list)
    max_history: int = 10
    
    def filter_transcription(
        self, 
        text: str, 
        confidence: Optional[float] = None,
        is_final: bool = False
    ) -> Optional[str]:
        """
        Filter transcription to remove phantom text and validate quality.
        
        Returns None if transcription should be rejected, cleaned text otherwise.
        """
        if not text:
            return None
            
        # Normalize text for comparison
        normalized = text.lower().strip()
        
        # Check for single phantom phrases
        if normalized in self.phantom_phrases and not is_final:
            logger.debug(f"Filtering phantom phrase: {text}")
            return None
        
        # Check for low confidence (if provided)
        if confidence is not None and confidence < self.min_confidence:
            logger.debug(f"Filtering low confidence ({confidence:.2f}): {text}")
            return None
        
        # Check for nonsensical patterns
        if self._is_nonsensical(normalized):
            logger.debug(f"Filtering nonsensical text: {text}")
            return None
        
        # Check for excessive repetition
        if self._is_repetitive(normalized):
            logger.debug(f"Filtering repetitive text: {text}")
            return None
        
        # Add to history
        self.recent_history.append(normalized)
        if len(self.recent_history) > self.max_history:
            self.recent_history.pop(0)
        
        return text.strip()
    
    def _is_nonsensical(self, text: str) -> bool:
        """Detect nonsensical patterns like 'hello by you'."""
        nonsense_patterns = [
            r'hello\s+by\s+you',
            r'why\s+are\s+you.*i\s+would',  # "why are you I would"
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for repeated words (simple approach)
        words = text.split()
        if len(words) >= 2:
            # Check if more than half the words are the same word repeated
            from collections import Counter
            word_counts = Counter(words)
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) // 2 and len(words) > 2:
                return True
        
        return False
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text is excessively repetitive."""
        if len(self.recent_history) < 3:
            return False
        
        # Check if same text appears multiple times recently
        recent_count = self.recent_history[-5:].count(text)
        return recent_count >= 3
    
    def reset(self):
        """Reset filter state for new session."""
        self.recent_history.clear()