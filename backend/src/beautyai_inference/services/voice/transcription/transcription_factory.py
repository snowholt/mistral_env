"""Transcription Service Factory - Updated for Specialized Whisper Engines

Chooses appropriate specialized Whisper engine based on voice registry configuration.
Supports three optimized engines:
- WhisperLargeV3Engine: Maximum accuracy (1.55B params)
- WhisperLargeV3TurboEngine: Speed optimized (809M params) 
- WhisperArabicTurboEngine: Arabic specialized (809M params, fine-tuned)

This centralizes engine selection logic so the rest of the codebase can remain
agnostic to the underlying engine implementation.
"""
from __future__ import annotations

import logging
from typing import Protocol
import os

from ....config.voice_config_loader import get_voice_config
from .whisper_large_v3_engine import WhisperLargeV3Engine
from .whisper_large_v3_turbo_engine import WhisperLargeV3TurboEngine  
from .whisper_arabic_turbo_engine import WhisperArabicTurboEngine

logger = logging.getLogger(__name__)


class TranscriptionServiceProtocol(Protocol):
    """Protocol defining the interface for all transcription engines."""
    def load_whisper_model(self, model_name: str | None = None) -> bool: ...
    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str | None = None, language: str = "ar") -> str | None: ...  # noqa: E501
    def is_model_loaded(self) -> bool: ...
    def get_model_info(self) -> dict: ...
    def cleanup(self) -> None: ...


def create_transcription_service() -> TranscriptionServiceProtocol:
    """
    Instantiate the correct specialized Whisper engine based on registry configuration.
    
    UPDATED: Now uses ModelManager for persistent Whisper model loading to improve
    performance and reduce memory usage across all voice services.

    Engine Selection Rules:
      1. Check ModelManager for existing persistent Whisper instance
      2. If not found, create new instance via ModelManager
      3. Fallback to direct instantiation only if ModelManager fails
      
    Environment Variables:
      - FORCE_ARABIC_ENGINE=1: Force Arabic engine regardless of config
      - FORCE_ACCURACY_ENGINE=1: Force Large v3 engine for maximum accuracy
      - DISABLE_PERSISTENT_WHISPER=1: Disable ModelManager integration (fallback to old behavior)
    """
    try:
        # Check if persistent Whisper loading is disabled
        disable_persistent = os.getenv("DISABLE_PERSISTENT_WHISPER") == "1"
        
        if not disable_persistent:
            # Try to get persistent Whisper model from ModelManager
            try:
                from ....core.model_manager import ModelManager
                model_manager = ModelManager()
                
                # Get persistent Whisper instance
                whisper_engine = model_manager.get_streaming_whisper()
                
                if whisper_engine is not None:
                    logger.info("✅ Using persistent Whisper model from ModelManager")
                    return whisper_engine
                else:
                    logger.warning("⚠️ Failed to get persistent Whisper model, falling back to direct instantiation")
                    
            except Exception as e:
                logger.warning(f"⚠️ ModelManager integration failed: {e}, falling back to direct instantiation")
        
        # Fallback to direct instantiation (original behavior)
        logger.info("🔄 Creating direct Whisper engine instance (non-persistent)")
        
        vc = get_voice_config()
        stt_cfg = vc.get_stt_model_config()
        engine_type = (stt_cfg.engine_type or '').lower()
        
        # Check environment overrides
        force_arabic = os.getenv("FORCE_ARABIC_ENGINE") == "1"
        force_accuracy = os.getenv("FORCE_ACCURACY_ENGINE") == "1"
        
        if force_arabic:
            logger.warning("FORCE_ARABIC_ENGINE=1 set – using Arabic Turbo engine")
            return WhisperArabicTurboEngine()
            
        if force_accuracy:
            logger.warning("FORCE_ACCURACY_ENGINE=1 set – using Large v3 engine for maximum accuracy")
            return WhisperLargeV3Engine()
        
        # Engine selection based on registry configuration
        engine_map = {
            "whisper_large_v3": WhisperLargeV3Engine,
            "whisper_large_v3_turbo": WhisperLargeV3TurboEngine,
            "whisper_arabic_turbo": WhisperArabicTurboEngine,
            # Legacy support
            "transformers": WhisperLargeV3TurboEngine,
            "faster-whisper": WhisperLargeV3TurboEngine,
            "faster_whisper": WhisperLargeV3TurboEngine
        }
        
        engine_class = engine_map.get(engine_type)
        
        if engine_class:
            logger.info(f"Transcription factory selecting {engine_class.__name__} (engine_type='{engine_type}')")
            return engine_class()
        else:
            logger.warning(f"Unknown engine_type '{engine_type}' – falling back to WhisperLargeV3TurboEngine")
            return WhisperLargeV3TurboEngine()
            
    except Exception as e:
        logger.error(f"Error in transcription factory: {e}")
        logger.info("Falling back to WhisperLargeV3TurboEngine due to configuration error")
        return WhisperLargeV3TurboEngine()


def get_available_engines() -> dict[str, str]:
    """
    Get a mapping of available engines and their descriptions.
    
    Returns:
        Dictionary mapping engine types to descriptions
    """
    return {
        "whisper_large_v3": "Maximum accuracy (1.55B params, 32 layers)",
        "whisper_large_v3_turbo": "Speed optimized (809M params, 4 layers, 4x faster)",
        "whisper_arabic_turbo": "Arabic specialized (809M params, 31% WER Arabic)",
    }


def validate_engine_availability() -> dict[str, bool]:
    """
    Check which engines can be instantiated successfully.
    
    Returns:
        Dictionary mapping engine types to availability status
    """
    engines = {
        "whisper_large_v3": WhisperLargeV3Engine,
        "whisper_large_v3_turbo": WhisperLargeV3TurboEngine,
        "whisper_arabic_turbo": WhisperArabicTurboEngine
    }
    
    availability = {}
    
    for engine_type, engine_class in engines.items():
        try:
            # Try to instantiate (but don't load model)
            engine = engine_class()
            availability[engine_type] = True
            # Clean up
            del engine
        except Exception as e:
            logger.warning(f"Engine {engine_type} not available: {e}")
            availability[engine_type] = False
    
    return availability


__all__ = [
    "create_transcription_service", 
    "TranscriptionServiceProtocol",
    "get_available_engines",
    "validate_engine_availability"
]
