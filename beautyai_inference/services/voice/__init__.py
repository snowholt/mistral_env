"""
Voice Services Module for BeautyAI Framework.

Organized voice-related services:
- transcription: Audio-to-text services
- synthesis: Text-to-speech services  
- conversation: Voice conversation orchestration
"""

# Import all services for easy access
from .transcription.audio_transcription_service import AudioTranscriptionService
from .synthesis.unified_tts_service import TextToSpeechService  # Keep original class name for compatibility
from .conversation.advanced_voice_service import VoiceToVoiceService  # Keep original class name for compatibility

__all__ = [
    "AudioTranscriptionService",
    "TextToSpeechService", 
    "VoiceToVoiceService"
]
