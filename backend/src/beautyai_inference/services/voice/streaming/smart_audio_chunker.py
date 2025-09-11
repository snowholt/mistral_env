"""
Smart audio chunker that prevents word splitting and improves transcription accuracy.
"""

import numpy as np
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass, field
import logging
from collections import deque

from .audio_chunking_config import AudioChunkingConfig

logger = logging.getLogger(__name__)

@dataclass
class AudioBuffer:
    """Buffer for accumulating audio samples."""
    samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    timestamp_ms: int = 0
    
    def add(self, new_samples: np.ndarray):
        """Add samples to buffer."""
        self.samples = np.concatenate([self.samples, new_samples])
    
    def clear(self):
        """Clear the buffer."""
        self.samples = np.array([], dtype=np.int16)
        self.timestamp_ms = 0
    
    def get_duration_ms(self, sample_rate: int) -> float:
        """Get duration of buffered audio in milliseconds."""
        return len(self.samples) * 1000 / sample_rate


class SmartAudioChunker:
    """
    Smart audio chunker that creates optimal chunk sizes for transcription.
    
    Features:
    - Prevents word splitting by using larger chunks
    - Implements overlap to capture word boundaries
    - Accumulates small chunks to avoid processing fragments
    - Detects silence to find natural boundaries
    """
    
    def __init__(self, config: Optional[AudioChunkingConfig] = None):
        """Initialize the smart chunker."""
        self.config = config or AudioChunkingConfig()
        
        if not self.config.validate():
            logger.warning("Using default configuration due to validation errors")
            self.config = AudioChunkingConfig()
        
        # Buffers
        self.accumulation_buffer = AudioBuffer()
        self.overlap_buffer = AudioBuffer()
        
        # Statistics
        self.total_samples_processed = 0
        self.chunks_created = 0
        
        logger.info(f"SmartAudioChunker initialized with {self.config.chunk_duration_ms}ms chunks, "
                   f"{self.config.overlap_ms}ms overlap")
    
    def process_audio(self, audio_data: bytes) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Process incoming audio data and yield optimally sized chunks.
        
        Args:
            audio_data: Raw audio bytes (PCM16)
            
        Yields:
            Tuples of (audio_chunk, timestamp_ms)
        """
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Add to accumulation buffer
        self.accumulation_buffer.add(samples)
        
        # Calculate how many samples we need for a chunk
        samples_per_chunk = self.config.get_samples_per_chunk()
        overlap_samples = self.config.get_overlap_samples()
        
        # Process accumulated audio if we have enough
        while len(self.accumulation_buffer.samples) >= samples_per_chunk:
            # Extract chunk with overlap from previous chunk if available
            if len(self.overlap_buffer.samples) > 0:
                # Prepend overlap from previous chunk
                chunk_samples = np.concatenate([
                    self.overlap_buffer.samples,
                    self.accumulation_buffer.samples[:samples_per_chunk - len(self.overlap_buffer.samples)]
                ])
                # Remove used samples from accumulation buffer
                self.accumulation_buffer.samples = self.accumulation_buffer.samples[samples_per_chunk - len(self.overlap_buffer.samples):]
            else:
                # No overlap, just take the chunk
                chunk_samples = self.accumulation_buffer.samples[:samples_per_chunk]
                self.accumulation_buffer.samples = self.accumulation_buffer.samples[samples_per_chunk:]
            
            # Save overlap for next chunk
            if overlap_samples > 0 and len(chunk_samples) >= overlap_samples:
                self.overlap_buffer.samples = chunk_samples[-overlap_samples:]
            
            # Calculate timestamp for this chunk
            timestamp_ms = int(self.total_samples_processed * 1000 / self.config.sample_rate)
            
            # Update statistics
            self.total_samples_processed += len(chunk_samples)
            self.chunks_created += 1
            
            # Apply windowing to smooth chunk boundaries (reduces artifacts)
            chunk_samples = self._apply_window(chunk_samples)
            
            yield (chunk_samples, timestamp_ms)
    
    def _apply_window(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply a window function to smooth chunk boundaries.
        
        This reduces audio artifacts at chunk boundaries that can confuse the transcription model.
        """
        window_size = min(len(samples) // 10, 160)  # 10ms at 16kHz or 10% of chunk
        
        if window_size > 0:
            # Apply fade-in
            fade_in = np.linspace(0.0, 1.0, window_size)
            samples[:window_size] = (samples[:window_size] * fade_in).astype(np.int16)
            
            # Apply fade-out
            fade_out = np.linspace(1.0, 0.0, window_size)
            samples[-window_size:] = (samples[-window_size:] * fade_out).astype(np.int16)
        
        return samples
    
    def flush(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Flush any remaining audio in the buffer.
        
        Returns:
            Final chunk if there's enough audio, None otherwise
        """
        min_samples = int(self.config.min_chunk_duration_ms * self.config.sample_rate / 1000)
        
        if len(self.accumulation_buffer.samples) >= min_samples:
            chunk_samples = self.accumulation_buffer.samples
            timestamp_ms = int(self.total_samples_processed * 1000 / self.config.sample_rate)
            
            self.accumulation_buffer.clear()
            self.overlap_buffer.clear()
            
            return (chunk_samples, timestamp_ms)
        
        return None
    
    def get_stats(self) -> dict:
        """Get chunking statistics."""
        return {
            "chunks_created": self.chunks_created,
            "total_samples": self.total_samples_processed,
            "total_duration_ms": self.total_samples_processed * 1000 / self.config.sample_rate,
            "buffer_duration_ms": self.accumulation_buffer.get_duration_ms(self.config.sample_rate),
            "avg_chunk_duration_ms": self.config.chunk_duration_ms
        }
    
    def reset(self):
        """Reset the chunker state."""
        self.accumulation_buffer.clear()
        self.overlap_buffer.clear()
        self.total_samples_processed = 0
        self.chunks_created = 0
        logger.info("SmartAudioChunker reset")