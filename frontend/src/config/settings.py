"""
Enhanced Frontend Settings with Environment-Aware Configuration.

Supports hot-reloading, feature flags, and environment-specific settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .constants import CONFIG_DICT, ENVIRONMENT, DEBUG


class SettingsManager:
    """Frontend settings manager with hot-reloading support."""
    
    def __init__(self):
        self._settings = None
        self._settings_file = Path("frontend_settings.json")
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from file and environment."""
        # Base settings
        base_settings = {
            "SECRET_KEY": os.environ.get("WEBUI_SECRET_KEY", "change-me-in-prod-please"),
            "BEAUTYAI_API_URL": CONFIG_DICT["backend"]["api_url"],
            "API_VERSION": "v1",
            "ENVIRONMENT": ENVIRONMENT,
            "DEBUG": DEBUG,
            
            # Feature flags
            "FEATURE_FLAGS": {
                "VOICE_STREAMING": self._get_bool_env("VOICE_STREAMING", True),
                "DUPLEX_STREAMING": self._get_bool_env("DUPLEX_STREAMING", True),
                "ECHO_CANCELLATION": self._get_bool_env("ECHO_CANCELLATION", True),
                "DEVICE_SELECTION": self._get_bool_env("DEVICE_SELECTION", True),
                "METRICS_COLLECTION": self._get_bool_env("METRICS_COLLECTION", True),
                "AUTO_RECONNECT": self._get_bool_env("AUTO_RECONNECT", True),
                "HOT_RELOAD": self._get_bool_env("HOT_RELOAD", DEBUG),
                "ADVANCED_AUDIO_CONTROLS": self._get_bool_env("ADVANCED_AUDIO_CONTROLS", DEBUG),
            },
            
            # Audio settings
            "AUDIO_SETTINGS": {
                "DEFAULT_SAMPLE_RATE": CONFIG_DICT["audio"]["sample_rate"],
                "DEFAULT_CHUNK_SIZE": CONFIG_DICT["audio"]["chunk_size"],
                "ECHO_CANCELLATION": CONFIG_DICT["duplex"]["echo_cancellation"],
                "NOISE_SUPPRESSION": self._get_bool_env("NOISE_SUPPRESSION", True),
                "AUTO_GAIN_CONTROL": self._get_bool_env("AUTO_GAIN_CONTROL", True),
            },
            
            # UI settings
            "UI_SETTINGS": {
                "MAX_TRANSCRIPT_LENGTH": CONFIG_DICT["ui"]["max_transcript_length"],
                "MAX_CONVERSATION_TURNS": CONFIG_DICT["ui"]["max_conversation_turns"],
                "REFRESH_RATE_MS": CONFIG_DICT["ui"]["refresh_rate_ms"],
                "ENABLE_ANIMATIONS": self._get_bool_env("ENABLE_ANIMATIONS", True),
                "DARK_MODE": self._get_bool_env("DARK_MODE", False),
                "COMPACT_UI": self._get_bool_env("COMPACT_UI", False),
            },
            
            # Performance settings
            "PERFORMANCE_SETTINGS": {
                "API_TIMEOUT": CONFIG_DICT["performance"]["api_timeout"],
                "CONNECTION_TIMEOUT_MS": CONFIG_DICT["performance"]["connection_timeout_ms"],
                "RECONNECT_ATTEMPTS": CONFIG_DICT["performance"]["reconnect_attempts"],
                "RECONNECT_DELAY_MS": CONFIG_DICT["performance"]["reconnect_delay_ms"],
                "LAZY_LOADING": self._get_bool_env("LAZY_LOADING", True),
                "MEMORY_OPTIMIZATION": self._get_bool_env("MEMORY_OPTIMIZATION", True),
            },
            
            # Security settings
            "SECURITY_SETTINGS": {
                "SECURE_COOKIES": self._get_bool_env("SECURE_COOKIES", not DEBUG),
                "CSRF_PROTECTION": self._get_bool_env("CSRF_PROTECTION", True),
                "CONTENT_SECURITY_POLICY": self._get_bool_env("CSP_ENABLED", not DEBUG),
                "RATE_LIMITING": self._get_bool_env("RATE_LIMITING", True),
            },
            
            # Backend integration
            "BACKEND_SETTINGS": CONFIG_DICT["backend"].copy(),
            
            # Model settings
            "MODEL_SETTINGS": CONFIG_DICT["models"].copy(),
        }
        
        # Load file-based settings if exists
        if self._settings_file.exists():
            try:
                with open(self._settings_file, 'r') as f:
                    file_settings = json.load(f)
                base_settings = self._merge_settings(base_settings, file_settings)
            except Exception as e:
                print(f"Warning: Failed to load settings file: {e}")
        
        # Environment-specific overrides
        if ENVIRONMENT == "production":
            base_settings.update({
                "SECRET_KEY": os.environ.get("WEBUI_SECRET_KEY", 
                    "PLEASE-SET-A-SECURE-SECRET-KEY-IN-PRODUCTION"),
                "DEBUG": False,
                "FEATURE_FLAGS": {
                    **base_settings["FEATURE_FLAGS"],
                    "HOT_RELOAD": False,
                    "ADVANCED_AUDIO_CONTROLS": False,
                },
                "SECURITY_SETTINGS": {
                    **base_settings["SECURITY_SETTINGS"],
                    "SECURE_COOKIES": True,
                    "CONTENT_SECURITY_POLICY": True,
                }
            })
        elif ENVIRONMENT == "testing":
            base_settings.update({
                "BEAUTYAI_API_URL": "http://localhost:8001",  # Testing port
                "FEATURE_FLAGS": {
                    **base_settings["FEATURE_FLAGS"],
                    "METRICS_COLLECTION": False,  # Don't collect metrics in tests
                },
                "PERFORMANCE_SETTINGS": {
                    **base_settings["PERFORMANCE_SETTINGS"],
                    "API_TIMEOUT": 10,  # Shorter timeout for tests
                    "CONNECTION_TIMEOUT_MS": 2000,
                }
            })
        
        self._settings = base_settings
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def _merge_settings(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge settings dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        keys = key.split('.')
        value = self._settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.get_setting(f"FEATURE_FLAGS.{feature}", False)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]):
        """Save settings to file."""
        try:
            with open(self._settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Settings saved to {self._settings_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def reload_settings(self):
        """Reload settings from file and environment."""
        self._load_settings()
        print("Settings reloaded")


# Global settings manager instance
_settings_manager = SettingsManager()

def load_config() -> Dict[str, Any]:
    """Load application configuration from environment variables and files."""
    return _settings_manager.get_all_settings()

def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting value."""
    return _settings_manager.get_setting(key, default)

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled."""
    return _settings_manager.is_feature_enabled(feature)

def reload_settings():
    """Reload settings from sources."""
    _settings_manager.reload_settings()

def save_user_settings(settings: Dict[str, Any]):
    """Save user-specific settings."""
    _settings_manager.save_settings(settings)

# Export commonly used settings
BEAUTYAI_API_URL = get_setting("BEAUTYAI_API_URL")
FEATURE_FLAGS = get_setting("FEATURE_FLAGS", {})
AUDIO_SETTINGS = get_setting("AUDIO_SETTINGS", {})
UI_SETTINGS = get_setting("UI_SETTINGS", {})
PERFORMANCE_SETTINGS = get_setting("PERFORMANCE_SETTINGS", {})

# Environment detection helpers
def is_production() -> bool:
    """Check if running in production environment."""
    return ENVIRONMENT == "production"

def is_development() -> bool:
    """Check if running in development environment."""
    return ENVIRONMENT == "development"

def is_testing() -> bool:
    """Check if running in testing environment."""
    return ENVIRONMENT == "testing"
