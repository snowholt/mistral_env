"""
Utterance Queue Service for Voice Pipeline.

Manages queued user utterances during tool execution or other blocking operations.
Provides intelligent merging, deduplication, and priority handling.

Author: BeautyAI Framework
Date: January 2026
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from collections import deque
from enum import Enum, auto

logger = logging.getLogger(__name__)


class UtterancePriority(Enum):
    """Priority levels for queued utterances."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3  # User correction or interrupt


@dataclass
class PendingUtterance:
    """Represents a pending user utterance waiting for processing."""
    text: str
    timestamp: float = field(default_factory=time.time)
    priority: UtterancePriority = UtterancePriority.NORMAL
    audio_data: Optional[bytes] = None
    language: str = "en"
    confidence: float = 1.0
    is_continuation: bool = False  # Part of ongoing speech
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "PendingUtterance") -> bool:
        """Enable priority queue sorting (higher priority first, then older)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


class MergeStrategy(Enum):
    """Strategies for merging multiple queued utterances."""
    CONCATENATE = auto()       # Simple concatenation with separator
    TAKE_LATEST = auto()       # Only use the latest utterance
    TAKE_COMPLETE = auto()     # Use the most complete (longest) utterance
    INTELLIGENT = auto()       # Use NLP to merge contextually


@dataclass
class QueueConfig:
    """Configuration for utterance queue behavior."""
    max_queue_size: int = 10
    max_queue_age_ms: int = 30000  # Max age before utterance is dropped
    merge_strategy: MergeStrategy = MergeStrategy.INTELLIGENT
    merge_separator: str = " "
    enable_deduplication: bool = True
    dedup_similarity_threshold: float = 0.8  # 80% similar = duplicate
    drop_low_confidence: bool = True
    min_confidence_threshold: float = 0.3


class UtteranceQueueService:
    """
    Service for managing queued user utterances during blocking operations.
    
    Features:
    - Priority-based queue ordering
    - Intelligent utterance merging
    - Deduplication of similar utterances
    - Age-based cleanup
    - Continuation detection for multi-part speech
    
    Usage:
        queue = UtteranceQueueService(session_id)
        
        # During tool execution, queue incoming utterances
        await queue.enqueue(text="What about option B?", priority=UtterancePriority.HIGH)
        
        # When tool completes, get merged/processed utterances
        merged_text = await queue.flush_and_merge()
    """
    
    def __init__(
        self,
        session_id: str,
        config: Optional[QueueConfig] = None,
        on_queue_full: Optional[Callable[[], Awaitable[None]]] = None
    ):
        """
        Initialize utterance queue service.
        
        Args:
            session_id: Unique session identifier
            config: Queue configuration
            on_queue_full: Callback when queue reaches max size
        """
        self.session_id = session_id
        self.config = config or QueueConfig()
        self.on_queue_full = on_queue_full
        
        self._queue: List[PendingUtterance] = []
        self._queue_lock = asyncio.Lock()
        self._last_enqueue_time = 0.0
        
        # Metrics
        self._total_enqueued = 0
        self._total_dropped = 0
        self._total_merged = 0
        
        logger.info(f"[UtteranceQueue] Initialized for session {session_id}")

    @property
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        """Check if queue is at maximum capacity."""
        return len(self._queue) >= self.config.max_queue_size

    async def enqueue(
        self,
        text: str,
        audio_data: Optional[bytes] = None,
        priority: UtterancePriority = UtterancePriority.NORMAL,
        language: str = "en",
        confidence: float = 1.0,
        is_continuation: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an utterance to the queue.
        
        Args:
            text: Transcribed speech text
            audio_data: Optional raw audio data
            priority: Utterance priority level
            language: Language code
            confidence: STT confidence score
            is_continuation: Whether this continues previous speech
            metadata: Optional additional metadata
            
        Returns:
            True if enqueued successfully
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Drop low confidence utterances
        if self.config.drop_low_confidence and confidence < self.config.min_confidence_threshold:
            logger.debug(f"[UtteranceQueue] Dropped low confidence utterance: '{text[:30]}...' ({confidence:.2f})")
            self._total_dropped += 1
            return False
        
        async with self._queue_lock:
            # Check for duplicates
            if self.config.enable_deduplication:
                for existing in self._queue:
                    similarity = self._compute_similarity(text, existing.text)
                    if similarity >= self.config.dedup_similarity_threshold:
                        # Update existing with higher confidence/priority if needed
                        if confidence > existing.confidence:
                            existing.confidence = confidence
                        if priority.value > existing.priority.value:
                            existing.priority = priority
                        logger.debug(f"[UtteranceQueue] Deduplicated: '{text[:30]}...'")
                        return True
            
            # Check queue capacity
            if self.is_full:
                if self.on_queue_full:
                    await self.on_queue_full()
                
                # Remove lowest priority, oldest item
                self._queue.sort()
                removed = self._queue.pop()
                logger.warning(f"[UtteranceQueue] Queue full, dropped: '{removed.text[:30]}...'")
                self._total_dropped += 1
            
            # Create and add utterance
            utterance = PendingUtterance(
                text=text,
                audio_data=audio_data,
                priority=priority,
                language=language,
                confidence=confidence,
                is_continuation=is_continuation,
                metadata=metadata or {}
            )
            
            self._queue.append(utterance)
            self._queue.sort()  # Maintain priority order
            self._last_enqueue_time = time.time()
            self._total_enqueued += 1
            
            logger.info(
                f"[UtteranceQueue] Enqueued ({priority.name}): '{text[:50]}...' "
                f"[queue size: {len(self._queue)}]"
            )
            
            return True

    async def flush_and_merge(self) -> Optional[str]:
        """
        Flush the queue and merge all utterances into a single text.
        
        Returns:
            Merged text or None if queue was empty
        """
        async with self._queue_lock:
            if not self._queue:
                return None
            
            # Remove expired utterances
            self._cleanup_expired()
            
            if not self._queue:
                return None
            
            utterances = list(self._queue)
            self._queue.clear()
        
        merged_text = await self._merge_utterances(utterances)
        
        if merged_text:
            self._total_merged += 1
            logger.info(f"[UtteranceQueue] Flushed and merged {len(utterances)} utterances")
        
        return merged_text

    async def peek(self) -> Optional[PendingUtterance]:
        """
        Peek at the highest priority utterance without removing it.
        
        Returns:
            Highest priority utterance or None
        """
        async with self._queue_lock:
            if self._queue:
                return self._queue[0]
            return None

    async def clear(self):
        """Clear all queued utterances."""
        async with self._queue_lock:
            count = len(self._queue)
            self._queue.clear()
            
        if count > 0:
            logger.info(f"[UtteranceQueue] Cleared {count} utterances")

    async def get_all(self) -> List[PendingUtterance]:
        """Get all queued utterances without clearing."""
        async with self._queue_lock:
            return list(self._queue)

    def _cleanup_expired(self):
        """Remove expired utterances from queue."""
        if not self._queue:
            return
        
        current_time = time.time()
        max_age_sec = self.config.max_queue_age_ms / 1000.0
        
        original_count = len(self._queue)
        self._queue = [
            u for u in self._queue
            if (current_time - u.timestamp) < max_age_sec
        ]
        
        removed_count = original_count - len(self._queue)
        if removed_count > 0:
            self._total_dropped += removed_count
            logger.debug(f"[UtteranceQueue] Expired {removed_count} utterances")

    async def _merge_utterances(self, utterances: List[PendingUtterance]) -> Optional[str]:
        """
        Merge multiple utterances based on configured strategy.
        
        Args:
            utterances: List of utterances to merge
            
        Returns:
            Merged text
        """
        if not utterances:
            return None
        
        if len(utterances) == 1:
            return utterances[0].text
        
        strategy = self.config.merge_strategy
        
        if strategy == MergeStrategy.TAKE_LATEST:
            # Sort by timestamp descending and take first
            sorted_by_time = sorted(utterances, key=lambda u: u.timestamp, reverse=True)
            return sorted_by_time[0].text
        
        elif strategy == MergeStrategy.TAKE_COMPLETE:
            # Take the longest (most complete) utterance
            longest = max(utterances, key=lambda u: len(u.text))
            return longest.text
        
        elif strategy == MergeStrategy.CONCATENATE:
            # Simple concatenation
            texts = [u.text for u in sorted(utterances, key=lambda u: u.timestamp)]
            return self.config.merge_separator.join(texts)
        
        elif strategy == MergeStrategy.INTELLIGENT:
            return await self._intelligent_merge(utterances)
        
        # Default fallback
        texts = [u.text for u in sorted(utterances, key=lambda u: u.timestamp)]
        return self.config.merge_separator.join(texts)

    async def _intelligent_merge(self, utterances: List[PendingUtterance]) -> str:
        """
        Intelligently merge utterances considering context and continuations.
        
        This removes redundant information and creates coherent combined speech.
        """
        if len(utterances) <= 1:
            return utterances[0].text if utterances else ""
        
        # Sort by timestamp
        sorted_utterances = sorted(utterances, key=lambda u: u.timestamp)
        
        # Group continuations
        merged_parts = []
        current_part = ""
        
        for i, utterance in enumerate(sorted_utterances):
            text = utterance.text.strip()
            
            if not text:
                continue
            
            if utterance.is_continuation and current_part:
                # Continuation of previous speech - append
                # Check for overlapping text
                overlap = self._find_overlap(current_part, text)
                if overlap:
                    # Remove overlap from new text
                    text = text[len(overlap):]
                
                if text:
                    current_part = current_part + " " + text
            else:
                # New utterance
                if current_part:
                    merged_parts.append(current_part.strip())
                current_part = text
        
        # Add last part
        if current_part:
            merged_parts.append(current_part.strip())
        
        # Check for semantic redundancy
        deduplicated = self._remove_semantic_duplicates(merged_parts)
        
        return " Also, ".join(deduplicated) if len(deduplicated) > 1 else deduplicated[0] if deduplicated else ""

    def _find_overlap(self, text1: str, text2: str) -> str:
        """Find overlapping text at the end of text1 and start of text2."""
        min_overlap = 3  # Minimum characters to consider overlap
        max_check = min(len(text1), len(text2), 50)  # Limit check length
        
        for length in range(max_check, min_overlap - 1, -1):
            if text1[-length:].lower() == text2[:length].lower():
                return text2[:length]
        
        return ""

    def _remove_semantic_duplicates(self, texts: List[str]) -> List[str]:
        """Remove semantically similar texts."""
        if len(texts) <= 1:
            return texts
        
        result = [texts[0]]
        
        for text in texts[1:]:
            is_duplicate = False
            for existing in result:
                similarity = self._compute_similarity(text, existing)
                if similarity >= self.config.dedup_similarity_threshold:
                    # Keep the longer one
                    if len(text) > len(existing):
                        result.remove(existing)
                        result.append(text)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(text)
        
        return result

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using simple word overlap.
        
        Returns value between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "session_id": self.session_id,
            "current_size": len(self._queue),
            "max_size": self.config.max_queue_size,
            "total_enqueued": self._total_enqueued,
            "total_dropped": self._total_dropped,
            "total_merged": self._total_merged,
            "last_enqueue_time": self._last_enqueue_time,
        }


# Factory function
def create_utterance_queue(
    session_id: str,
    max_size: int = 10,
    merge_strategy: MergeStrategy = MergeStrategy.INTELLIGENT
) -> UtteranceQueueService:
    """Create an utterance queue with default configuration."""
    config = QueueConfig(
        max_queue_size=max_size,
        merge_strategy=merge_strategy
    )
    return UtteranceQueueService(session_id, config)
