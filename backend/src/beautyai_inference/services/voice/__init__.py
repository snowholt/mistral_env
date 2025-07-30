"""
Voice services for speech processing functionality.

This module provides speech-to-text and text-to-speech capabilities
using optimized models and services.
"""

# Import transcription services
from .transcription.faster_whisper_service import FasterWhisperTranscriptionService

# Import conversation services
from .conversation.simple_voice_service import SimpleVoiceService

__all__ = [
    "FasterWhisperTranscriptionService", 
    "SimpleVoiceService"
]
