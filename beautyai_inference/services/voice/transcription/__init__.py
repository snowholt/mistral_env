"""
Audio Transcription Services Module.

Contains services for converting audio to text using various speech recognition models.
"""

from .audio_transcription_service import WhisperTranscriptionService

# Backward compatibility alias
AudioTranscriptionService = WhisperTranscriptionService

__all__ = [
    "WhisperTranscriptionService",
    "AudioTranscriptionService"  # Backward compatibility
]
