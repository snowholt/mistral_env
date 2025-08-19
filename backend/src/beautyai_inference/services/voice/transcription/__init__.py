"""
Specialized Whisper Transcription Engines for BeautyAI Framework.

This module provides three optimized Whisper engines:
- WhisperLargeV3Engine: Maximum accuracy (1.55B params)
- WhisperLargeV3TurboEngine: Speed optimized (809M params, 4x faster)  
- WhisperArabicTurboEngine: Arabic specialized (809M params, fine-tuned)

Factory function automatically selects appropriate engine based on configuration.
"""

from .transcription_factory import (
    create_transcription_service,
    TranscriptionServiceProtocol,
    get_available_engines,
    validate_engine_availability
)

from .base_whisper_engine import BaseWhisperEngine
from .whisper_large_v3_engine import WhisperLargeV3Engine
from .whisper_large_v3_turbo_engine import WhisperLargeV3TurboEngine
from .whisper_arabic_turbo_engine import WhisperArabicTurboEngine

__all__ = [
    # Factory and protocol
    "create_transcription_service",
    "TranscriptionServiceProtocol", 
    "get_available_engines",
    "validate_engine_availability",
    
    # Base engine
    "BaseWhisperEngine",
    
    # Specialized engines
    "WhisperLargeV3Engine",
    "WhisperLargeV3TurboEngine", 
    "WhisperArabicTurboEngine"
]
