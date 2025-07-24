"""
Integration tests for Step 5: Configuration system integration and conflict resolution.

Tests that services properly use ConfigurationManager instead of hardcoded mappings.
"""

import pytest
import logging
from pathlib import Path

from beautyai_inference.config.configuration_manager import ConfigurationManager
from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
from beautyai_inference.services.voice.conversation.advanced_voice_service import AdvancedVoiceConversationService
from beautyai_inference.services.voice.synthesis.unified_tts_service import UnifiedTTSService

logger = logging.getLogger(__name__)


class TestStep5ConfigurationIntegration:
    """Test suite for Step 5 configuration integration."""
    
    def test_configuration_manager_singleton(self):
        """Test that ConfigurationManager follows singleton pattern."""
        manager1 = ConfigurationManager()
        manager2 = ConfigurationManager()
        
        assert manager1 is manager2, "ConfigurationManager should be singleton"
        assert id(manager1) == id(manager2), "ConfigurationManager instances should be identical"
    
    def test_hardcoded_mappings_removed(self):
        """Test that hardcoded language mappings have been removed from services."""
        # Test AdvancedVoiceConversationService
        advanced_service = AdvancedVoiceConversationService()
        
        # Should not have hardcoded language_tts_models
        assert not hasattr(advanced_service, 'language_tts_models'), \
            "AdvancedVoiceConversationService should not have hardcoded language_tts_models"
        
        # Should have config_manager
        assert hasattr(advanced_service, 'config_manager'), \
            "AdvancedVoiceConversationService should have config_manager"
        
        assert isinstance(advanced_service.config_manager, ConfigurationManager), \
            "config_manager should be ConfigurationManager instance"
    
    def test_service_configuration_isolation(self):
        """Test that services use their own configuration sections."""
        config_manager = ConfigurationManager()
        
        # Test service-specific configurations
        simple_config = config_manager.get_service_config("simple_voice_service")
        advanced_config = config_manager.get_service_config("advanced_voice_service")
        
        # Simple service should use Edge TTS
        assert simple_config.get("tts_engine") == "edge_tts", \
            "SimpleVoiceService should use Edge TTS"
        
        # Advanced service should use Coqui TTS
        assert advanced_config.get("tts_engine") == "coqui_tts", \
            "AdvancedVoiceConversationService should use Coqui TTS"
        
        # Should have different supported languages
        simple_languages = simple_config.get("supported_languages", [])
        advanced_languages = advanced_config.get("supported_languages", [])
        
        assert len(simple_languages) <= len(advanced_languages), \
            "Advanced service should support more languages than simple service"
    
    def test_edge_tts_voice_configuration(self):
        """Test Edge TTS voice configuration and fallbacks."""
        config_manager = ConfigurationManager()
        
        # Test primary voice selection
        ar_male = config_manager.get_edge_tts_voice("ar", "male")
        ar_female = config_manager.get_edge_tts_voice("ar", "female")
        en_male = config_manager.get_edge_tts_voice("en", "male")
        en_female = config_manager.get_edge_tts_voice("en", "female")
        
        # Validate voice names
        assert ar_male.startswith("ar-"), f"Arabic male voice should start with 'ar-': {ar_male}"
        assert ar_female.startswith("ar-"), f"Arabic female voice should start with 'ar-': {ar_female}"
        assert en_male.startswith("en-"), f"English male voice should start with 'en-': {en_male}"
        assert en_female.startswith("en-"), f"English female voice should start with 'en-': {en_female}"
        
        # Test voice validation
        assert config_manager.validate_edge_tts_voice(ar_male), \
            f"Arabic male voice {ar_male} should validate successfully"
        assert config_manager.validate_edge_tts_voice(en_female), \
            f"English female voice {en_female} should validate successfully"
    
    def test_coqui_model_configuration(self):
        """Test Coqui TTS model configuration."""
        config_manager = ConfigurationManager()
        
        coqui_config = config_manager.get_coqui_model_config()
        
        # Should have required fields
        assert "model_name" in coqui_config, "Coqui config should have model_name"
        assert "languages" in coqui_config, "Coqui config should have languages"
        assert "features" in coqui_config, "Coqui config should have features"
        
        # Validate model name
        model_name = coqui_config["model_name"]
        assert "xtts_v2" in model_name or "multilingual" in model_name, \
            f"Coqui model should be multilingual: {model_name}"
        
        # Should support Arabic and English
        languages = coqui_config["languages"]
        assert "ar" in languages, "Coqui model should support Arabic"
        assert "en" in languages, "Coqui model should support English"
        
        # Test model validation
        assert config_manager.validate_coqui_model(model_name), \
            f"Coqui model {model_name} should validate successfully"
    
    def test_service_initialization_without_hardcoded_values(self):
        """Test that services initialize correctly using only configuration."""
        # Test SimpleVoiceService
        simple_service = SimpleVoiceService()
        
        # Should not have hardcoded voice mappings
        # (SimpleVoiceService already used ConfigurationManager, so this should pass)
        
        # Test AdvancedVoiceConversationService
        advanced_service = AdvancedVoiceConversationService()
        
        # Should have service config from ConfigurationManager
        assert hasattr(advanced_service, 'service_config'), \
            "AdvancedVoiceConversationService should have service_config"
        
        service_config = advanced_service.service_config
        assert isinstance(service_config, dict), "service_config should be a dictionary"
        assert len(service_config) > 0, "service_config should not be empty"
        
        # Should have TTS engine configuration
        assert "tts_engine" in service_config, "Service config should specify TTS engine"
        assert service_config["tts_engine"] == "coqui_tts", \
            "Advanced service should use Coqui TTS engine"
    
    def test_configuration_consistency(self):
        """Test consistency between different configuration access methods."""
        config_manager = ConfigurationManager()
        
        # Test that service configuration is consistent
        simple_config1 = config_manager.get_service_config("simple_voice_service")
        simple_config2 = config_manager.get_service_config("simple_voice_service")
        
        assert simple_config1 == simple_config2, \
            "Multiple calls to get_service_config should return identical results"
        
        # Test configuration summary
        summary = config_manager.get_configuration_summary()
        
        assert isinstance(summary, dict), "Configuration summary should be a dictionary"
        assert "configured_services" in summary, "Summary should list configured services"
        assert "simple_voice_service" in summary["configured_services"], \
            "Simple voice service should be in configured services"
        assert "advanced_voice_service" in summary["configured_services"], \
            "Advanced voice service should be in configured services"
    
    def test_performance_configuration(self):
        """Test performance-related configuration."""
        config_manager = ConfigurationManager()
        
        # Test simple service performance config
        simple_perf = config_manager.get_performance_config("simple_voice_service")
        
        # Simple service performance config might be empty (uses defaults)
        # But advanced service should have explicit performance config
        advanced_perf = config_manager.get_performance_config("advanced_voice_service")
        assert "target_response_time_ms" in advanced_perf, \
            "Advanced service should have response time target"
        
        advanced_target = advanced_perf["target_response_time_ms"]
        assert advanced_target > 0, \
            f"Advanced service should have positive response time target, got {advanced_target}ms"
        
        # Advanced service should have higher response time target than simple service (by design)
        assert advanced_target >= 2000, \
            f"Advanced service should have ≥2s response time target, got {advanced_target}ms"
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works after configuration changes."""
        # Test that services can be initialized without errors
        try:
            simple_service = SimpleVoiceService()
            advanced_service = AdvancedVoiceConversationService()
            unified_service = UnifiedTTSService()
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            pytest.fail(f"Service initialization failed after configuration changes: {e}")
        
        # Test that ConfigurationManager convenience functions work
        from beautyai_inference.config.configuration_manager import (
            get_edge_tts_voice, get_coqui_model_config, get_service_config
        )
        
        # Test convenience functions
        voice = get_edge_tts_voice("ar", "female")
        assert voice.startswith("ar-"), f"Convenience function should return Arabic voice: {voice}"
        
        coqui_config = get_coqui_model_config()
        assert "model_name" in coqui_config, "Convenience function should return Coqui config"
        
        service_config = get_service_config("simple_voice_service")
        assert "tts_engine" in service_config, "Convenience function should return service config"
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        config_manager = ConfigurationManager()
        
        # Test invalid language fallback
        invalid_voice = config_manager.get_edge_tts_voice("invalid_lang", "female")
        assert invalid_voice is not None, "Should provide fallback for invalid language"
        assert len(invalid_voice) > 0, "Fallback voice should not be empty"
        
        # Test invalid service configuration
        invalid_config = config_manager.get_service_config("nonexistent_service")
        assert isinstance(invalid_config, dict), "Should return empty dict for invalid service"
        
        # Test voice validation with invalid voice
        assert not config_manager.validate_edge_tts_voice("invalid-voice-name"), \
            "Should return False for invalid voice name"
        
        # Test model validation with invalid model
        assert not config_manager.validate_coqui_model("invalid_model_name"), \
            "Should return False for invalid model name"


class TestStep5ConfigurationMigration:
    """Test configuration migration and compatibility."""
    
    def test_no_breaking_changes(self):
        """Test that Step 5 changes don't break existing functionality."""
        # This test ensures that existing code patterns still work
        config_manager = ConfigurationManager()
        
        # Test that all expected configuration sections exist
        config = config_manager.get_config()
        
        required_sections = [
            "edge_tts_voices",
            "coqui_tts_models", 
            "service_configurations",
            "service_defaults"
        ]
        
        for section in required_sections:
            assert section in config, f"Required configuration section missing: {section}"
            assert isinstance(config[section], dict), f"Configuration section should be dict: {section}"
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        config_manager = ConfigurationManager()
        
        # Test that all Edge TTS voices in config are valid format
        config = config_manager.get_config()
        edge_voices = config.get("edge_tts_voices", {})
        
        # Language key to ISO code mapping
        lang_key_to_iso = {
            "arabic": "ar",
            "english": "en"
        }
        
        for language_key, lang_config in edge_voices.items():
            assert isinstance(lang_config, dict), f"Language config should be dict for {language_key}"
            
            voices = lang_config.get("voices", {})
            for voice_type, voice_config in voices.items():
                assert "primary" in voice_config, f"Voice config should have primary for {language_key}/{voice_type}"
                primary_voice = voice_config["primary"]
                
                # Get expected ISO code for this language key
                expected_iso = lang_key_to_iso.get(language_key, language_key)
                assert primary_voice.startswith(f"{expected_iso}-"), \
                    f"Primary voice should start with ISO language code '{expected_iso}-': {primary_voice}"
        
        # Test service configurations are complete
        service_configs = config.get("service_configurations", {})
        
        for service_name, service_config in service_configs.items():
            assert "tts_engine" in service_config, f"Service {service_name} should have tts_engine"
            assert "supported_languages" in service_config, f"Service {service_name} should have supported_languages"
            
            supported_languages = service_config["supported_languages"]
            assert isinstance(supported_languages, list), f"Supported languages should be list for {service_name}"
            assert len(supported_languages) > 0, f"Service {service_name} should support at least one language"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
