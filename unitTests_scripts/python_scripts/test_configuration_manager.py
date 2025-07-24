"""
Tests for Configuration Manager - Voice Services Configuration System.

This module tests the centralized configuration management system for voice services,
including Edge TTS voice configurations, Coqui TTS models, and service-specific settings.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from beautyai_inference.config.configuration_manager import ConfigurationManager


class TestConfigurationManager:
    """Test cases for the ConfigurationManager class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "edge_tts_voices": {
                "arabic": {
                    "voices": {
                        "male": {
                            "primary": "ar-SA-HamedNeural",
                            "quality": "high",
                            "region": "Saudi Arabia"
                        },
                        "female": {
                            "primary": "ar-SA-ZariyahNeural",
                            "quality": "high",
                            "region": "Saudi Arabia"
                        }
                    },
                    "fallbacks": {
                        "male": ["ar-EG-ShakirNeural"],
                        "female": ["ar-EG-SalmaNeural"]
                    }
                },
                "english": {
                    "voices": {
                        "male": {
                            "primary": "en-US-AriaNeural",
                            "quality": "high",
                            "region": "United States"
                        },
                        "female": {
                            "primary": "en-US-JennyNeural",
                            "quality": "high",
                            "region": "United States"
                        }
                    }
                }
            },
            "coqui_tts_models": {
                "xtts_v2": {
                    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                    "languages": ["ar", "en", "es"],
                    "features": ["voice_cloning", "multilingual"]
                }
            },
            "service_configurations": {
                "simple_voice_service": {
                    "tts_engine": "edge_tts",
                    "supported_languages": ["ar", "en"],
                    "default_voices": {
                        "ar": {
                            "male": "ar-SA-HamedNeural",
                            "female": "ar-SA-ZariyahNeural"
                        }
                    }
                }
            },
            "service_defaults": {
                "simple_voice_service": {
                    "audio_format": "wav",
                    "sample_rate": 22050
                }
            }
        }
    
    @pytest.fixture
    def config_manager(self, sample_config):
        """Create a ConfigurationManager instance with mocked config."""
        with patch.object(ConfigurationManager, '_load_config') as mock_load:
            manager = ConfigurationManager()
            manager._config_cache = sample_config
            return manager
    
    def test_singleton_pattern(self):
        """Test that ConfigurationManager follows singleton pattern."""
        manager1 = ConfigurationManager()
        manager2 = ConfigurationManager()
        assert manager1 is manager2
    
    def test_get_edge_tts_voice_primary(self, config_manager):
        """Test getting primary Edge TTS voice."""
        voice = ConfigurationManager.get_edge_tts_voice("arabic", "female")
        assert voice == "ar-SA-ZariyahNeural"
        
        voice = ConfigurationManager.get_edge_tts_voice("arabic", "male")
        assert voice == "ar-SA-HamedNeural"
    
    def test_get_edge_tts_voice_fallback(self, config_manager):
        """Test getting fallback Edge TTS voice when primary not available."""
        # Test with language that has fallbacks but no primary voices defined
        with patch.object(config_manager, 'get_config') as mock_config:
            mock_config.return_value = {
                "edge_tts_voices": {
                    "arabic": {
                        "fallbacks": {
                            "male": ["ar-EG-ShakirNeural"],
                            "female": ["ar-EG-SalmaNeural"]
                        }
                    }
                }
            }
            
            voice = ConfigurationManager.get_edge_tts_voice("arabic", "female")
            assert voice == "ar-EG-SalmaNeural"
    
    def test_get_edge_tts_voice_hard_fallback(self, config_manager):
        """Test hard fallback when no configuration available."""
        with patch.object(config_manager, 'get_config') as mock_config:
            mock_config.return_value = {}
            
            voice = ConfigurationManager.get_edge_tts_voice("arabic", "female")
            assert voice == "ar-SA-ZariyahNeural"  # Hard fallback
    
    def test_get_edge_tts_voices_for_language(self, config_manager):
        """Test getting all voices for a specific language."""
        voices = ConfigurationManager.get_edge_tts_voices_for_language("arabic")
        
        expected = {
            "male": "ar-SA-HamedNeural",
            "female": "ar-SA-ZariyahNeural"
        }
        assert voices == expected
    
    def test_get_coqui_model_config(self, config_manager):
        """Test getting Coqui TTS model configuration."""
        config = ConfigurationManager.get_coqui_model_config()
        
        assert config["model_name"] == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert "ar" in config["languages"]
        assert "voice_cloning" in config["features"]
    
    def test_get_service_config(self, config_manager):
        """Test getting service-specific configuration."""
        config = ConfigurationManager.get_service_config("simple_voice_service")
        
        # Should merge service config and defaults
        assert config["tts_engine"] == "edge_tts"
        assert config["audio_format"] == "wav"  # From defaults
        assert config["sample_rate"] == 22050   # From defaults
    
    def test_get_supported_languages(self, config_manager):
        """Test getting supported languages for a service."""
        languages = ConfigurationManager.get_supported_languages("simple_voice_service")
        assert languages == ["ar", "en"]
    
    def test_get_default_voice(self, config_manager):
        """Test getting default voice for service/language/type."""
        voice = ConfigurationManager.get_default_voice("simple_voice_service", "ar", "female")
        assert voice == "ar-SA-ZariyahNeural"
    
    def test_validate_edge_tts_voice(self, config_manager):
        """Test Edge TTS voice validation."""
        # Valid voice
        assert ConfigurationManager.validate_edge_tts_voice("ar-SA-ZariyahNeural") is True
        
        # Invalid voice
        assert ConfigurationManager.validate_edge_tts_voice("invalid-voice") is False
    
    def test_validate_coqui_model(self, config_manager):
        """Test Coqui TTS model validation."""
        # Valid model
        assert ConfigurationManager.validate_coqui_model("tts_models/multilingual/multi-dataset/xtts_v2") is True
        
        # Invalid model
        assert ConfigurationManager.validate_coqui_model("invalid-model") is False
    
    def test_get_performance_config(self, config_manager):
        """Test getting performance configuration."""
        # Add performance config to test data
        with patch.object(config_manager, 'get_config') as mock_config:
            mock_config.return_value = {
                "service_configurations": {
                    "simple_voice_service": {
                        "performance_config": {
                            "target_response_time_ms": 2000,
                            "enable_caching": True
                        }
                    }
                }
            }
            
            perf_config = ConfigurationManager.get_performance_config("simple_voice_service")
            assert perf_config["target_response_time_ms"] == 2000
            assert perf_config["enable_caching"] is True
    
    def test_is_service_feature_enabled(self, config_manager):
        """Test checking if service features are enabled."""
        # Add features to test data
        with patch.object(config_manager, 'get_config') as mock_config:
            mock_config.return_value = {
                "service_configurations": {
                    "advanced_voice_service": {
                        "features": {
                            "voice_cloning": True,
                            "emotion_control": False
                        }
                    }
                }
            }
            
            assert ConfigurationManager.is_service_feature_enabled("advanced_voice_service", "voice_cloning") is True
            assert ConfigurationManager.is_service_feature_enabled("advanced_voice_service", "emotion_control") is False
            assert ConfigurationManager.is_service_feature_enabled("advanced_voice_service", "nonexistent") is False
    
    def test_get_configuration_summary(self, config_manager):
        """Test getting configuration summary."""
        summary = config_manager.get_configuration_summary()
        
        assert "edge_tts_languages" in summary
        assert "arabic" in summary["edge_tts_languages"]
        assert "english" in summary["edge_tts_languages"]
        assert summary["coqui_models"] == ["xtts_v2"]
        assert "simple_voice_service" in summary["configured_services"]
    
    def test_reload_config(self, config_manager):
        """Test configuration reloading."""
        # Mock successful reload
        with patch.object(config_manager, '_load_config') as mock_load:
            result = config_manager.reload_config()
            assert result is True
            mock_load.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in configuration loading."""
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError("Config file not found")
            
            manager = ConfigurationManager()
            # Should not raise exception, should use empty config
            config = manager.get_config()
            assert isinstance(config, dict)
    
    def test_convenience_functions(self, config_manager):
        """Test convenience functions for backward compatibility."""
        from beautyai_inference.config.configuration_manager import (
            get_edge_tts_voice, get_coqui_model_config, get_service_config
        )
        
        voice = get_edge_tts_voice("arabic", "female")
        assert voice == "ar-SA-ZariyahNeural"
        
        coqui_config = get_coqui_model_config()
        assert "model_name" in coqui_config
        
        service_config = get_service_config("simple_voice_service")
        assert "tts_engine" in service_config


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_real_model_registry_loading(self):
        """Test loading from actual model registry file."""
        # This test will use the real model registry file
        manager = ConfigurationManager()
        config = manager.get_config()
        
        # Should have the basic structure we expect
        assert isinstance(config, dict)
        
        # Check if new sections exist
        if "edge_tts_voices" in config:
            assert isinstance(config["edge_tts_voices"], dict)
        
        if "service_configurations" in config:
            assert isinstance(config["service_configurations"], dict)
    
    def test_service_integration(self):
        """Test that services can use the configuration manager."""
        # Test that we can get configurations for both services
        simple_config = ConfigurationManager.get_service_config("simple_voice_service")
        advanced_config = ConfigurationManager.get_service_config("advanced_voice_service")
        
        # Configs should be dictionaries (even if empty)
        assert isinstance(simple_config, dict)
        assert isinstance(advanced_config, dict)
    
    def test_voice_selection_workflow(self):
        """Test complete voice selection workflow."""
        # Arabic female voice
        arabic_female = ConfigurationManager.get_edge_tts_voice("arabic", "female")
        assert isinstance(arabic_female, str)
        assert len(arabic_female) > 0
        
        # English male voice
        english_male = ConfigurationManager.get_edge_tts_voice("english", "male")
        assert isinstance(english_male, str)
        assert len(english_male) > 0
        
        # Should be different voices
        assert arabic_female != english_male


if __name__ == "__main__":
    pytest.main([__file__])
