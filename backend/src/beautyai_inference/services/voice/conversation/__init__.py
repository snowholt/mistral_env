"""
Voice Conversation Services Module.

Contains services for orchestrating complete voice-to-voice conversations.
"""

from .advanced_voice_service import AdvancedVoiceConversationService
from .simple_voice_service import SimpleVoiceService

# Backward compatibility alias
VoiceToVoiceService = AdvancedVoiceConversationService

__all__ = [
    "AdvancedVoiceConversationService",
    "SimpleVoiceService",
    "VoiceToVoiceService"  # Backward compatibility
]
