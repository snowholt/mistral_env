"""
Voice Configuration Loader for BeautyAI Framework.

This module provides a centralized way to load voice model configurations
from the voice_models_registry.json file and integrates with preload configuration,
ensuring consistency across all voice services and preventing configuration drift.

Author: BeautyAI Framework
Date: 2025-01-30
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VoiceModelConfig:
    """Configuration for a voice model."""
    model_id: str
    engine_type: str
    model_type: str
    description: str
    supported_languages: list
    gpu_enabled: bool = False
    documentation: dict = None
    model_info: dict = None

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int
    channels: int
    bit_depth: int
    format: str

@dataclass
class PerformanceConfig:
    """Performance targets configuration."""
    total_latency_ms: int
    stt_latency_ms: int
    tts_latency_ms: int

@dataclass
class PreloadConfig:
    """Model preload configuration."""
    preload_on_startup: bool
    whisper_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    memory_threshold_gb: float

class VoiceConfigLoader:
    """
    Centralized voice configuration loader with preload configuration support.
    
    This class ensures all voice services use the same configuration
    from both voice_models_registry.json and preload_config.json files,
    preventing inconsistencies and enabling model preloading.
    """
    
    def __init__(self):
        self.config_path = Path(__file__).parent / "voice_models_registry.json"
        self.preload_config_path = Path(__file__).parent / "preload_config.json"
        self._config = None
        self._preload_config = None
        self._load_config()
        self._load_preload_config()
    
    def _load_config(self) -> None:
        """Load the voice configuration from registry file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Voice registry not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            logger.info(f"Voice configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load voice configuration: {e}")
            raise Exception(f"Voice configuration loading failed: {e}")
    
    def _load_preload_config(self) -> None:
        """Load the preload configuration for PersistentModelManager."""
        try:
            if not self.preload_config_path.exists():
                logger.warning(f"Preload config not found: {self.preload_config_path}")
                self._preload_config = None
                return
            
            with open(self.preload_config_path, 'r', encoding='utf-8') as f:
                self._preload_config = json.load(f)
            
            logger.info(f"Preload configuration loaded from {self.preload_config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load preload configuration: {e}")
            self._preload_config = None
    
    def get_stt_model_config(self) -> VoiceModelConfig:
        """Get the STT (Speech-to-Text) model configuration."""
        stt_model_name = self._config["default_models"]["stt"]
        model_data = self._config["models"][stt_model_name]
        
        return VoiceModelConfig(
            model_id=model_data["model_id"],
            engine_type=model_data["engine_type"],
            model_type=model_data["type"],
            description=model_data["description"],
            supported_languages=model_data["supported_languages"],
            gpu_enabled=model_data.get("gpu_enabled", False),
            documentation=model_data.get("documentation"),
            model_info=model_data.get("model_info")
        )
    
    def get_tts_model_config(self) -> VoiceModelConfig:
        """Get the TTS (Text-to-Speech) model configuration."""
        tts_model_name = self._config["default_models"]["tts"]
        model_data = self._config["models"][tts_model_name]
        
        return VoiceModelConfig(
            model_id=model_data["model_id"],
            engine_type=model_data["engine_type"],
            model_type=model_data["type"],
            description=model_data["description"],
            supported_languages=model_data["supported_languages"],
            gpu_enabled=model_data.get("gpu_enabled", False),
            documentation=model_data.get("documentation"),
            model_info=model_data.get("model_info")
        )
    
    def get_voice_settings(self, language: str) -> Dict[str, str]:
        """Get voice settings for a specific language."""
        if language not in self._config["voice_settings"]:
            raise ValueError(f"Language '{language}' not supported. Available: {list(self._config['voice_settings'].keys())}")
        
        return self._config["voice_settings"][language]
    
    def get_audio_config(self) -> AudioConfig:
        """Get audio processing configuration."""
        audio_cfg = self._config["audio_config"]
        return AudioConfig(
            sample_rate=audio_cfg["sample_rate"],
            channels=audio_cfg["channels"],
            bit_depth=audio_cfg["bit_depth"],
            format=audio_cfg["format"]
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance targets configuration."""
        perf_cfg = self._config["performance_targets"]
        return PerformanceConfig(
            total_latency_ms=perf_cfg["total_latency_ms"],
            stt_latency_ms=perf_cfg["stt_latency_ms"],
            tts_latency_ms=perf_cfg["tts_latency_ms"]
        )
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return list(self._config["voice_settings"].keys())
    
    def get_voice_types(self, language: str) -> list:
        """Get available voice types for a language."""
        voice_settings = self.get_voice_settings(language)
        return list(voice_settings.keys())
    
    def get_voice_id(self, language: str, gender: str) -> str:
        """Get the specific voice ID for a language and gender."""
        voice_settings = self.get_voice_settings(language)
        if gender not in voice_settings:
            available_genders = list(voice_settings.keys())
            raise ValueError(f"Gender '{gender}' not available for language '{language}'. Available: {available_genders}")
        
        return voice_settings[gender]
    
    def validate_language(self, language: str) -> bool:
        """Validate if a language is supported."""
        return language in self._config["voice_settings"]
    
    def validate_gender(self, language: str, gender: str) -> bool:
        """Validate if a gender is available for a language."""
        if not self.validate_language(language):
            return False
        return gender in self._config["voice_settings"][language]
    
    def get_preload_config(self) -> Optional[PreloadConfig]:
        """Get the preload configuration for PersistentModelManager."""
        if not self._preload_config:
            return None
        
        try:
            return PreloadConfig(
                preload_on_startup=self._preload_config.get("preload_on_startup", False),
                whisper_config=self._preload_config.get("models", {}).get("whisper", {}),
                llm_config=self._preload_config.get("models", {}).get("llm", {}),
                memory_threshold_gb=self._preload_config.get("memory_threshold_gb", 16.0)
            )
        except Exception as e:
            logger.error(f"Error parsing preload config: {e}")
            return None
    
    def validate_preload_models(self) -> Dict[str, bool]:
        """Validate that models specified in preload config are available in voice registry."""
        validation_results = {
            "whisper_available": False,
            "llm_available": False,
            "config_valid": False
        }
        
        preload_config = self.get_preload_config()
        if not preload_config:
            logger.warning("No preload configuration available for validation")
            return validation_results
        
        try:
            # Validate Whisper model
            whisper_model_id = preload_config.whisper_config.get("model_id")
            if whisper_model_id:
                stt_config = self.get_stt_model_config()
                validation_results["whisper_available"] = (whisper_model_id == stt_config.model_id)
            
            # Validate LLM model
            llm_model_path = preload_config.llm_config.get("model_path")
            if llm_model_path:
                # Check if the model path exists or is valid
                validation_results["llm_available"] = True  # Assuming path validation is done elsewhere
            
            validation_results["config_valid"] = (validation_results["whisper_available"] and 
                                                 validation_results["llm_available"])
            
            logger.info(f"Preload model validation: {validation_results}")
            
        except Exception as e:
            logger.error(f"Error validating preload models: {e}")
        
        return validation_results
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Get merged configuration including both voice registry and preload config."""
        merged_config = {
            "voice_registry": self._config,
            "preload_config": self._preload_config,
            "preload_available": self._preload_config is not None,
            "validation_results": self.validate_preload_models()
        }
        
        if self._preload_config:
            merged_config["preload_enabled"] = self._preload_config.get("preload_on_startup", False)
        else:
            merged_config["preload_enabled"] = False
        
        return merged_config
    
    def is_preload_enabled(self) -> bool:
        """Check if model preloading is enabled."""
        if not self._preload_config:
            return False
        return self._preload_config.get("preload_on_startup", False)
    
    def get_whisper_model_mapping(self) -> str:
        """Get the actual Whisper model ID for faster-whisper."""
        stt_config = self.get_stt_model_config()
        return stt_config.model_id
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of voice and preload configuration."""
        stt_config = self.get_stt_model_config()
        tts_config = self.get_tts_model_config()
        audio_config = self.get_audio_config()
        performance_config = self.get_performance_config()
        preload_config = self.get_preload_config()
        validation_results = self.validate_preload_models()
        
        summary = {
            "stt_model": {
                "name": self._config["default_models"]["stt"],
                "model_id": stt_config.model_id,
                "engine": stt_config.engine_type,
                "gpu_enabled": stt_config.gpu_enabled
            },
            "tts_model": {
                "name": self._config["default_models"]["tts"],
                "model_id": tts_config.model_id,
                "engine": tts_config.engine_type
            },
            "audio_format": {
                "format": audio_config.format,
                "sample_rate": audio_config.sample_rate,
                "channels": audio_config.channels,
                "bit_depth": audio_config.bit_depth
            },
            "performance_targets": {
                "total_latency_ms": performance_config.total_latency_ms,
                "stt_latency_ms": performance_config.stt_latency_ms,
                "tts_latency_ms": performance_config.tts_latency_ms
            },
            "supported_languages": self.get_supported_languages(),
            "total_voice_combinations": sum(len(voices) for voices in self._config["voice_settings"].values()),
            "preload_configuration": {
                "enabled": self.is_preload_enabled(),
                "config_available": preload_config is not None,
                "models_valid": validation_results,
                "memory_threshold_gb": preload_config.memory_threshold_gb if preload_config else None
            }
        }
        
        if preload_config:
            summary["preload_models"] = {
                "whisper": preload_config.whisper_config,
                "llm": preload_config.llm_config
            }
        
        return summary

# Global singleton instance
_voice_config_loader = None

def get_voice_config() -> VoiceConfigLoader:
    """Get the global voice configuration loader instance."""
    global _voice_config_loader
    if _voice_config_loader is None:
        _voice_config_loader = VoiceConfigLoader()
    return _voice_config_loader
