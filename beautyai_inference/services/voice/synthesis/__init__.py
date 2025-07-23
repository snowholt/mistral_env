"""
Text-to-Speech Synthesis Services Module.

Contains services for converting text to speech using various TTS engines.
"""

from .unified_tts_service import UnifiedTTSService

# Backward compatibility alias
TextToSpeechService = UnifiedTTSService

__all__ = [
    "UnifiedTTSService",
    "TextToSpeechService"  # Backward compatibility
]
