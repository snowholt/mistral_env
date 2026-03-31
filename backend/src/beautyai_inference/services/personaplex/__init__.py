"""
PersonaPlex Service Module for BeautyAI

This module provides integration with NVIDIA's PersonaPlex full-duplex
speech-to-speech conversational model.

PersonaPlex enables:
- Real-time, full-duplex voice conversations
- Voice and role control through prompts
- Natural conversational dynamics (interruptions, backchannels)
- Multiple voice personalities (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
"""

from .manager import PersonaPlexManager, get_personaplex_manager
from .constants import (
    VOICE_PROMPTS,
    DEFAULT_TEXT_PROMPTS,
    VoiceType,
    PersonaPlexConfig,
)

__all__ = [
    "PersonaPlexManager",
    "get_personaplex_manager",
    "VOICE_PROMPTS",
    "DEFAULT_TEXT_PROMPTS", 
    "VoiceType",
    "PersonaPlexConfig",
]
