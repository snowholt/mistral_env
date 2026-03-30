"""Voice streaming utilities for real-time audio processing.

This module provides components for:
- Audio chunking and buffering (ring_buffer, audio_chunking_config)
- Sentence detection for progressive TTS (sentence_buffer)
- Endpointing and utterance management
- Streaming session management
"""

from .sentence_buffer import (
    SentenceStreamBuffer,
    SentenceStreamConfig,
    StreamedSentence,
    TTSStreamQueue,
)

# Optional imports - may not be needed in all contexts
try:
    from .ring_buffer import PCMInt16RingBuffer
except ImportError:
    PCMInt16RingBuffer = None

try:
    from .audio_chunking_config import AudioChunkingConfig
except ImportError:
    AudioChunkingConfig = None

try:
    from .endpointing import EndpointingConfig, EndpointingDetector
except ImportError:
    EndpointingConfig = None
    EndpointingDetector = None

__all__ = [
    # Sentence streaming (for progressive TTS)
    "SentenceStreamBuffer",
    "SentenceStreamConfig", 
    "StreamedSentence",
    "TTSStreamQueue",
    # Audio buffering (optional)
    "PCMInt16RingBuffer",
    "AudioChunkingConfig",
    # Endpointing (optional)
    "EndpointingConfig",
    "EndpointingDetector",
]
