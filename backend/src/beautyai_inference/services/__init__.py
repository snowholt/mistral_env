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

# Import voice services with new names for backward compatibility
from .voice.transcription.audio_transcription_service import WhisperTranscriptionService
from .voice.synthesis.unified_tts_service import UnifiedTTSService
from .voice.conversation.advanced_voice_service import AdvancedVoiceConversationService

# Backward compatibility aliases - maintain old names for existing code
AudioTranscriptionService = WhisperTranscriptionService
TextToSpeechService = UnifiedTTSService
VoiceToVoiceService = AdvancedVoiceConversationService

__all__ = [
    "BaseService",
    # New descriptive voice service names
    "WhisperTranscriptionService",
    "UnifiedTTSService",
    "AdvancedVoiceConversationService",
    # Voice services for backward compatibility
    "AudioTranscriptionService",
    "TextToSpeechService", 
    "VoiceToVoiceService"
]

__version__ = "1.0.0"
