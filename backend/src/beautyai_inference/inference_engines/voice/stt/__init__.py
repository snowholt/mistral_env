"""
Speech-to-Text (STT) engines for BeautyAI Framework.

This package contains Whisper-based transcription engines:
- BaseWhisperEngine: Abstract base class for all Whisper engines
- WhisperLargeV3Engine: Maximum accuracy (1.55B params)
- WhisperLargeV3TurboEngine: Speed optimized (809M params)
- WhisperArabicTurboEngine: Arabic specialized (mboushaba turbo fine-tune)
- WhisperFinetunedArabicEngine: BeautyAI fine-tuned Arabic
- WhisperGeniusArabicEngine: Genius AI Arabic fine-tuned
- WhisperByneArabicEngine: Byne Arabic fine-tuned (full Large v3, highest accuracy)
"""

from .base_whisper_engine import BaseWhisperEngine
from .whisper_large_v3_engine import WhisperLargeV3Engine
from .whisper_large_v3_turbo_engine import WhisperLargeV3TurboEngine
from .whisper_arabic_turbo_engine import WhisperArabicTurboEngine
from .whisper_finetuned_arabic_engine import WhisperFinetunedArabicEngine
from .whisper_genius_arabic_engine import WhisperGeniusArabicEngine
from .whisper_byne_arabic_engine import WhisperByneArabicEngine

__all__ = [
    "BaseWhisperEngine",
    "WhisperLargeV3Engine",
    "WhisperLargeV3TurboEngine",
    "WhisperArabicTurboEngine",
    "WhisperFinetunedArabicEngine",
    "WhisperGeniusArabicEngine",
    "WhisperByneArabicEngine",
]
