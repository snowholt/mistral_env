"""
BeautyAI Services Package.

This package contains all core business logic services organized by functionality:
- base: Common service functionality and base classes
- model: Model management services (registry, lifecycle, validation)
- inference: Inference services (chat, test, benchmark, session)
- config: Configuration services (config, validation, migration, backup)
- system: System services (memory, cache, status)
- voice: Voice services (transcription, synthesis, conversation)

The service layer is designed to be CLI-agnostic and API-ready for future
web service integration.
"""

# Import base service class for easy access
from .base.base_service import BaseService

# Import voice services (streamlined)
from .voice.transcription.faster_whisper_service import FasterWhisperTranscriptionService
from .voice.conversation.simple_voice_service import SimpleVoiceService

__all__ = [
    "BaseService",
    # Streamlined voice services
    "FasterWhisperTranscriptionService",
    "SimpleVoiceService"
]

__version__ = "1.0.0"
