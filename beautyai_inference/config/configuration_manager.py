"""
Configuration Manager for BeautyAI Voice Services.

Provides centralized access to model registry configurations with validation,
fallback mechanisms, and service-specific configuration management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Centralized configuration management for voice services.
    
    This class provides access to model registry configurations including:
    - Edge TTS voice configurations
    - Coqui TTS model configurations  
    - Service-specific configurations
    - Validation and fallback mechanisms
    """
    
    _instance = None
    _config_cache = None
    _config_path = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if not hasattr(self, 'initialized'):
            self._config_path = Path(__file__).parent / "model_registry.json"
            self._config_cache = None
            self.initialized = True
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from model registry."""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
                logger.info(f"✅ Configuration loaded from {self._config_path}")
            else:
                logger.error(f"Configuration file not found: {self._config_path}")
                self._config_cache = {}
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config_cache = {}
    
    def reload_config(self) -> bool:
        """
        Reload configuration from disk.
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            self._load_config()
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        if self._config_cache is None:
            self._load_config()
        return self._config_cache or {}
    
    @staticmethod
    def get_edge_tts_voice(language: str, voice_type: str) -> str:
        """
        Get Edge TTS voice name from registry.
        
        Args:
            language: Language code (ar, en)
            voice_type: Voice type (male, female)
            
        Returns:
            str: Edge TTS voice name
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            edge_voices = config.get("edge_tts_voices", {})
            language_config = edge_voices.get(language, {})
            
            # Try to get primary voice
            voice_config = language_config.get("voices", {}).get(voice_type, {})
            if voice_config and "primary" in voice_config:
                return voice_config["primary"]
            
            # Fallback to fallback voices
            fallbacks = language_config.get("fallbacks", {}).get(voice_type, [])
            if fallbacks:
                return fallbacks[0]
            
            # Hard fallback based on language and voice type
            fallback_voices = {
                "ar": {
                    "male": "ar-SA-HamedNeural",
                    "female": "ar-SA-ZariyahNeural"
                },
                "en": {
                    "male": "en-US-AriaNeural", 
                    "female": "en-US-JennyNeural"
                }
            }
            
            return fallback_voices.get(language, {}).get(voice_type, "ar-SA-ZariyahNeural")
            
        except Exception as e:
            logger.error(f"Error getting Edge TTS voice for {language}/{voice_type}: {e}")
            # Ultimate fallback
            return "ar-SA-ZariyahNeural"
    
    @staticmethod
    def get_edge_tts_voices_for_language(language: str) -> Dict[str, str]:
        """
        Get all available Edge TTS voices for a language.
        
        Args:
            language: Language code (ar, en)
            
        Returns:
            Dict mapping voice types to voice names
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            edge_voices = config.get("edge_tts_voices", {})
            language_config = edge_voices.get(language, {})
            
            result = {}
            voices = language_config.get("voices", {})
            
            for voice_type, voice_config in voices.items():
                if "primary" in voice_config:
                    result[voice_type] = voice_config["primary"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Edge TTS voices for {language}: {e}")
            return {}
    
    @staticmethod
    def get_coqui_model_config() -> Dict[str, Any]:
        """
        Get Coqui TTS model configuration.
        
        Returns:
            Dict containing Coqui TTS model configuration
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            coqui_models = config.get("coqui_tts_models", {})
            # Return xtts_v2 configuration (primary model)
            return coqui_models.get("xtts_v2", {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "languages": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh-cn", "ja", "hu", "ko", "hi"],
                "features": ["voice_cloning", "multilingual", "emotion_control"]
            })
            
        except Exception as e:
            logger.error(f"Error getting Coqui model config: {e}")
            return {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "languages": ["ar", "en"],
                "features": ["multilingual"]
            }
    
    @staticmethod
    def get_service_config(service_name: str) -> Dict[str, Any]:
        """
        Get service-specific configuration.
        
        Args:
            service_name: Name of the service (simple_voice_service, advanced_voice_service)
            
        Returns:
            Dict containing service configuration
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            service_configs = config.get("service_configurations", {})
            service_defaults = config.get("service_defaults", {})
            
            # Get both service-specific config and defaults
            service_config = service_configs.get(service_name, {})
            default_config = service_defaults.get(service_name, {})
            
            # Merge configurations (service config takes precedence)
            merged_config = {**default_config, **service_config}
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Error getting service config for {service_name}: {e}")
            return {}
    
    @staticmethod
    def get_supported_languages(service_name: str) -> List[str]:
        """
        Get supported languages for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            List of supported language codes
        """
        service_config = ConfigurationManager.get_service_config(service_name)
        return service_config.get("supported_languages", ["ar", "en"])
    
    @staticmethod
    def get_default_voice(service_name: str, language: str, voice_type: str) -> str:
        """
        Get default voice for a service, language, and voice type.
        
        Args:
            service_name: Name of the service
            language: Language code
            voice_type: Voice type (male, female)
            
        Returns:
            str: Voice identifier
        """
        service_config = ConfigurationManager.get_service_config(service_name)
        
        try:
            default_voices = service_config.get("default_voices", {})
            language_voices = default_voices.get(language, {})
            return language_voices.get(voice_type, "female")
            
        except Exception as e:
            logger.error(f"Error getting default voice for {service_name}/{language}/{voice_type}: {e}")
            return "female"
    
    @staticmethod
    def validate_edge_tts_voice(voice_name: str) -> bool:
        """
        Validate that an Edge TTS voice name exists in configuration.
        
        Args:
            voice_name: Edge TTS voice name to validate
            
        Returns:
            bool: True if voice exists, False otherwise
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            edge_voices = config.get("edge_tts_voices", {})
            
            # Check all languages and voice types
            for language_config in edge_voices.values():
                voices = language_config.get("voices", {})
                for voice_config in voices.values():
                    if voice_config.get("primary") == voice_name:
                        return True
                
                # Check fallbacks
                fallbacks = language_config.get("fallbacks", {})
                for fallback_list in fallbacks.values():
                    if voice_name in fallback_list:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating Edge TTS voice {voice_name}: {e}")
            return False
    
    @staticmethod
    def validate_coqui_model(model_name: str) -> bool:
        """
        Validate that a Coqui TTS model exists in configuration.
        
        Args:
            model_name: Coqui TTS model name to validate
            
        Returns:
            bool: True if model exists, False otherwise
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            coqui_models = config.get("coqui_tts_models", {})
            
            for model_config in coqui_models.values():
                if model_config.get("model_name") == model_name:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating Coqui model {model_name}: {e}")
            return False
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive model information from registry.
        
        Args:
            model_name: Model name to look up
            
        Returns:
            Dict containing model information
        """
        instance = ConfigurationManager()
        config = instance.get_config()
        
        try:
            # Check in regular models section
            models = config.get("models", {})
            if model_name in models:
                return models[model_name]
            
            # Check in Coqui TTS models
            coqui_models = config.get("coqui_tts_models", {})
            for model_config in coqui_models.values():
                if model_config.get("model_name") == model_name:
                    return model_config
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {}
    
    @staticmethod
    def get_performance_config(service_name: str) -> Dict[str, Any]:
        """
        Get performance configuration for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dict containing performance configuration
        """
        service_config = ConfigurationManager.get_service_config(service_name)
        return service_config.get("performance_config", {})
    
    @staticmethod
    def is_service_feature_enabled(service_name: str, feature_name: str) -> bool:
        """
        Check if a feature is enabled for a service.
        
        Args:
            service_name: Name of the service
            feature_name: Name of the feature
            
        Returns:
            bool: True if feature is enabled, False otherwise
        """
        service_config = ConfigurationManager.get_service_config(service_name)
        features = service_config.get("features", {})
        return features.get(feature_name, False)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dict containing configuration summary
        """
        config = self.get_config()
        
        try:
            summary = {
                "edge_tts_languages": list(config.get("edge_tts_voices", {}).keys()),
                "coqui_models": list(config.get("coqui_tts_models", {}).keys()),
                "configured_services": list(config.get("service_configurations", {}).keys()),
                "total_llm_models": len(config.get("models", {})),
                "config_file_path": str(self._config_path),
                "config_loaded": self._config_cache is not None
            }
            
            # Add voice counts
            edge_voices = config.get("edge_tts_voices", {})
            for lang, lang_config in edge_voices.items():
                voices = lang_config.get("voices", {})
                summary[f"{lang}_voices_available"] = len(voices)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating configuration summary: {e}")
            return {"error": str(e)}


# Convenience functions for backward compatibility
def get_edge_tts_voice(language: str, voice_type: str) -> str:
    """Get Edge TTS voice - convenience function."""
    return ConfigurationManager.get_edge_tts_voice(language, voice_type)


def get_coqui_model_config() -> Dict[str, Any]:
    """Get Coqui model config - convenience function."""
    return ConfigurationManager.get_coqui_model_config()


def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get service config - convenience function."""
    return ConfigurationManager.get_service_config(service_name)
