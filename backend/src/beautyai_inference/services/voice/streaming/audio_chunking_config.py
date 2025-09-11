"""
Audio chunking configuration for voice streaming.

Provides optimal chunk sizes to prevent word splitting and improve transcription accuracy.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioChunkingConfig:
    """Configuration for audio chunking to prevent word splitting."""
    
    # Chunk duration in milliseconds
    # Whisper models work best with 100-400ms chunks for streaming
    # Too small: splits words/phonemes
    # Too large: increases latency
    chunk_duration_ms: int = 200  # Increased from 20ms to 200ms
    
    # Overlap between chunks to capture word boundaries
    overlap_ms: int = 50  # 50ms overlap to avoid cutting words
    
    # Minimum chunk size to process (skip very small final chunks)
    min_chunk_duration_ms: int = 100
    
    # Buffer size for accumulating small chunks
    accumulation_buffer_ms: int = 400  # Accumulate up to 400ms before processing
    
    # Sample rate for audio processing
    sample_rate: int = 16000
    
    def get_samples_per_chunk(self) -> int:
        """Calculate number of samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    def get_overlap_samples(self) -> int:
        """Calculate number of samples for overlap."""
        return int(self.sample_rate * self.overlap_ms / 1000)
    
    def get_accumulation_samples(self) -> int:
        """Calculate number of samples for accumulation buffer."""
        return int(self.sample_rate * self.accumulation_buffer_ms / 1000)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.chunk_duration_ms < 100:
            logger.warning(f"Chunk duration {self.chunk_duration_ms}ms is too small, may split words")
            return False
        
        if self.chunk_duration_ms > 1000:
            logger.warning(f"Chunk duration {self.chunk_duration_ms}ms is too large, will increase latency")
            return False
        
        if self.overlap_ms >= self.chunk_duration_ms:
            logger.error("Overlap cannot be larger than chunk duration")
            return False
        
        return True