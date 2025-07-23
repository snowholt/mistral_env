"""
Voice Conversation Services Module.

Contains services for orchestrating complete voice-to-voice conversations.
"""

from .advanced_voice_service import AdvancedVoiceConversationService

# Backward compatibility alias
VoiceToVoiceService = AdvancedVoiceConversationService

__all__ = [
    "AdvancedVoiceConversationService",
    "VoiceToVoiceService"  # Backward compatibility
]
