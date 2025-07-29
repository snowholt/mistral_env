"""
Voice Services Module for BeautyAI Framework.

Organized voice-related services:
- transcription: Audio-to-text services
- synthesis: Text-to-speech services  
- conversation: Voice conversation orchestration
"""

# Import all services for easy access
from .transcription.audio_transcription_service import WhisperTranscriptionService
from .synthesis.unified_tts_service import UnifiedTTSService
from .conversation.advanced_voice_service import AdvancedVoiceConversationService

# Backward compatibility aliases - maintain old names for existing code
AudioTranscriptionService = WhisperTranscriptionService
TextToSpeechService = UnifiedTTSService  
VoiceToVoiceService = AdvancedVoiceConversationService

__all__ = [
    # New descriptive names
    "WhisperTranscriptionService",
    "UnifiedTTSService", 
    "AdvancedVoiceConversationService",
    # Backward compatibility aliases
    "AudioTranscriptionService",
    "TextToSpeechService",
    "VoiceToVoiceService"
]
