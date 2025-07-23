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

# Import voice services for backward compatibility
from .voice.transcription.audio_transcription_service import AudioTranscriptionService
from .voice.synthesis.unified_tts_service import TextToSpeechService
from .voice.conversation.advanced_voice_service import VoiceToVoiceService

__all__ = [
    "BaseService",
    # Voice services for backward compatibility
    "AudioTranscriptionService",
    "TextToSpeechService", 
    "VoiceToVoiceService"
]

__version__ = "1.0.0"
