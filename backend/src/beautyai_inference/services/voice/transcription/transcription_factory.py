"""Transcription Service Factory

Chooses appropriate transcription backend (Faster-Whisper or Transformers)
based on voice registry configuration. This centralizes selection logic so
the rest of the codebase can remain agnostic to the underlying engine.
"""
from __future__ import annotations

import logging
from typing import Protocol

from ....config.voice_config_loader import get_voice_config
import os
from .faster_whisper_service import FasterWhisperTranscriptionService
from .transformers_whisper_service import TransformersWhisperService

logger = logging.getLogger(__name__)


class TranscriptionServiceProtocol(Protocol):  # structural typing only
    def load_whisper_model(self, model_name: str | None = None) -> bool: ...
    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str | None = None, language: str = "ar") -> str | None: ...  # noqa: E501
    def is_model_loaded(self) -> bool: ...
    def get_model_info(self) -> dict: ...


def create_transcription_service() -> TranscriptionServiceProtocol:
    """Instantiate the correct transcription backend per registry.

    Priority rules:
      1. engine_type == 'faster-whisper' → FasterWhisperTranscriptionService
      2. engine_type == 'transformers' (default) → TransformersWhisperService
      3. Fallback → TransformersWhisperService
    """
    vc = get_voice_config()
    stt_cfg = vc.get_stt_model_config()
    engine = (stt_cfg.engine_type or '').lower()
    force_transformers = os.getenv("FORCE_TRANSFORMERS_STT") == "1"
    if force_transformers:
        logger.warning("FORCE_TRANSFORMERS_STT=1 set – overriding engine_type '%s' -> transformers", engine)
        engine = "transformers"
    logger.debug("Transcription factory resolved engine_type='%s' force=%s", engine, force_transformers)
    if engine in ("faster-whisper", "faster_whisper"):
        logger.info("Transcription factory selecting Faster-Whisper backend (engine_type=%s)", engine)
        return FasterWhisperTranscriptionService()
    if engine == "transformers":
        logger.info("Transcription factory selecting Transformers backend (engine_type=%s)", engine)
        return TransformersWhisperService()
    logger.warning("Unknown engine_type '%s' – falling back to Transformers backend", engine)
    return TransformersWhisperService()


__all__ = ["create_transcription_service", "TranscriptionServiceProtocol"]
