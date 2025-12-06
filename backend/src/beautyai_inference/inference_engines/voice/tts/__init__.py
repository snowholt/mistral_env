"""
Text-to-Speech (TTS) engines for BeautyAI Framework.

This package contains TTS engines:
- EdgeTTSEngine: Microsoft Edge TTS (cloud-based, no GPU required)
- XTTSEngine: Coqui XTTS v2 (local, GPU-accelerated, voice cloning)
"""

from .edge_tts_engine import EdgeTTSEngine
from .xtts_engine import XTTSEngine

__all__ = [
    "EdgeTTSEngine",
    "XTTSEngine",
]
