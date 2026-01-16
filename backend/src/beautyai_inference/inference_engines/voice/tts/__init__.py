"""
Text-to-Speech (TTS) engines for BeautyAI Framework.

This package contains TTS engines:
- EdgeTTSEngine: Microsoft Edge TTS (cloud-based, no GPU required)
- XTTSEngine: Coqui XTTS v2 (local, GPU-accelerated, voice cloning)
- SaudiXTTSEngine: Saudi Arabic XTTS v2 (fine-tuned for Saudi dialect)
"""

from .edge_tts_engine import EdgeTTSEngine
from .xtts_engine import XTTSEngine
from .saudi_xtts_engine import SaudiXTTSEngine

__all__ = [
    "EdgeTTSEngine",
    "XTTSEngine",
    "SaudiXTTSEngine",
]
