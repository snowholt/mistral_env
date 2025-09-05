"""
Enhanced Configuration API Adapter.

Provides API-compatible interface for configuration management,
bridging enhanced config services with REST/GraphQL endpoints.
"""
from typing import Dict, Any, Optional, List
import logging
import time
import asyncio

from .base_adapter import APIServiceAdapter
from ..models import APIRequest, APIResponse
from ..errors import ValidationError, ConfigurationError
from ...core.config_manager import get_config_manager, ConfigManager

logger = logging.getLogger(__name__)


class ConfigAPIAdapter(APIServiceAdapter):
    """
    Enhanced API adapter for configuration management.
    
    Provides API-compatible interface for:
    - Environment-aware configuration management
    - Hot-reloading configuration updates
    - Encrypted secrets management
    - Configuration validation and health checks
    - Duplex streaming configuration
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize config API adapter with enhanced configuration manager."""
        self.config_manager = config_manager or get_config_manager()
        super().__init__(self.config_manager)
        
        # Register for configuration reload notifications
        self.config_manager.register_reload_callback(self._on_config_reload)
    
    async def _on_config_reload(self, config_manager: ConfigManager):
        """Handle configuration reload events."""
        logger.info("Configuration reloaded, updating adapter state")
        # Perform any necessary cleanup or reinitialization
    
    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "get_config": "Get current application configuration",
            "get_duplex_config": "Get duplex streaming configuration", 
            "get_system_config": "Get system configuration",
            "get_model_config": "Get model configuration",
            "update_config": "Update configuration section",
            "validate_config": "Validate configuration parameters",
            "health_check": "Perform configuration health check",
            "reload_config": "Reload configuration from files",
            "get_secrets": "Get encrypted secret values",
            "set_secret": "Set encrypted secret value"
        }
    
    async def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current application configuration.
        
        Args:
            section: Optional specific configuration section (duplex, system, models)
            
        Returns:
            Dictionary with configuration data
        """
        try:
            if section == "duplex":
                config_data = self.config_manager.get_duplex_config().dict()
            elif section == "system":
                config_data = self.config_manager.get_system_config().dict()
            elif section == "models":
                config_data = self.config_manager.get_model_config().dict()
            elif section is None:
                config_data = self.config_manager.get_all_config()
            else:
                raise ValidationError(f"Unknown configuration section: {section}")
            
            return {
                "config": config_data,
                "section": section,
                "timestamp": int(time.time()),
                "version": getattr(self.config_manager, '_config_version', '1.0.0')
            }
            
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            raise ConfigurationError(f"Configuration retrieval failed: {e}")
    
    async def get_duplex_config(self) -> Dict[str, Any]:
        """Get duplex streaming configuration."""
        try:
            duplex_config = self.config_manager.get_duplex_config()
            return {
                "duplex_config": duplex_config.dict(),
                "schema": duplex_config.schema(),
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get duplex configuration: {e}")
            raise ConfigurationError(f"Duplex configuration retrieval failed: {e}")
    
    async def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        try:
            system_config = self.config_manager.get_system_config()
            return {
                "system_config": system_config.dict(),
                "schema": system_config.schema(),
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get system configuration: {e}")
            raise ConfigurationError(f"System configuration retrieval failed: {e}")
    
    async def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        try:
            model_config = self.config_manager.get_model_config()
            return {
                "model_config": model_config.dict(),
                "schema": model_config.schema(),
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get model configuration: {e}")
            raise ConfigurationError(f"Model configuration retrieval failed: {e}")
    
    async def update_config(self, section: str, config_updates: Dict[str, Any], 
                           validate_only: bool = False) -> Dict[str, Any]:
        """
        Update configuration section.
        
        Args:
            section: Configuration section to update (duplex, system, models)
            config_updates: Configuration updates to apply
            validate_only: If True, only validate without applying changes
            
        Returns:
            Dictionary with update results
        """
        try:
            # Validate section
            valid_sections = ["duplex", "system", "models"]
            if section not in valid_sections:
                raise ValidationError(f"Invalid section. Must be one of: {valid_sections}")
            
            # Validate configuration updates
            validation_result = await self.validate_config(config_updates, section)
            
            if not validation_result["valid"]:
                raise ValidationError(f"Invalid configuration: {validation_result['errors']}")
            
            if validate_only:
                return {
                    "validated": True,
                    "changes_applied": False,
                    "validation": validation_result,
                    "timestamp": int(time.time())
                }
            
            # Apply configuration updates using transaction
            with self.config_manager.config_update_transaction():
                success = self.config_manager.update_config(section, config_updates)
                
                if not success:
                    raise ConfigurationError(f"Failed to update {section} configuration")
            
            # Get updated configuration
            updated_config = await self.get_config(section)
            
            return {
                "validated": True,
                "changes_applied": True,
                "updated_config": updated_config["config"],
                "validation": validation_result,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise ConfigurationError(f"Configuration update failed: {e}")
    
    async def validate_config(self, config_data: Dict[str, Any], 
                             section: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Args:
            config_data: Configuration data to validate
            section: Optional specific section being validated
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Use the enhanced configuration manager's validation
            if section:
                # Create temporary config object to validate
                if section == "duplex":
                    from ...core.config_manager import DuplexStreamingConfig
                    temp_config = self.config_manager.get_duplex_config().dict()
                    temp_config.update(config_data)
                    DuplexStreamingConfig(**temp_config)  # Validates with pydantic
                    
                elif section == "system":
                    from ...core.config_manager import SystemConfig
                    temp_config = self.config_manager.get_system_config().dict()
                    temp_config.update(config_data)
                    SystemConfig(**temp_config)  # Validates with pydantic
                    
                elif section == "models":
                    from ...core.config_manager import ModelConfig
                    temp_config = self.config_manager.get_model_config().dict()
                    temp_config.update(config_data)
                    ModelConfig(**temp_config)  # Validates with pydantic
            
            # Use configuration manager's validation
            overall_validation = self.config_manager.validate_config()
            
            return {
                "valid": True,
                "errors": [],
                "warnings": overall_validation.get("warnings", []),
                "section": section,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "section": section,
                "timestamp": int(time.time())
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform configuration health check."""
        try:
            health_status = self.config_manager.health_check()
            return {
                "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
                "details": health_status,
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Configuration health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time())
            }
    
    async def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from files."""
        try:
            # Trigger configuration reload
            self.config_manager._load_all_config()
            
            return {
                "reloaded": True,
                "message": "Configuration reloaded successfully",
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return {
                "reloaded": False,
                "error": str(e),
                "timestamp": int(time.time())
            }
    
    async def get_secret(self, key: str) -> Dict[str, Any]:
        """Get encrypted secret value."""
        try:
            secret_value = self.config_manager.get_secret(key)
            
            return {
                "secret_exists": secret_value is not None,
                "value": secret_value,
                "key": key,
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to get secret '{key}': {e}")
            raise ConfigurationError(f"Secret retrieval failed: {e}")
    
    async def set_secret(self, key: str, value: str) -> Dict[str, Any]:
        """Set encrypted secret value."""
        try:
            self.config_manager.save_encrypted_secret(key, value)
            
            return {
                "secret_saved": True,
                "key": key,
                "message": f"Secret '{key}' saved successfully",
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Failed to set secret '{key}': {e}")
            raise ConfigurationError(f"Secret storage failed: {e}")
    
    # Legacy methods for backward compatibility
    async def reset_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Reset configuration to defaults (legacy method)."""
        logger.warning("reset_config is deprecated, configuration is now file-based")
        return {
            "reset_successful": False,
            "message": "Configuration reset not supported with new file-based system",
            "recommendation": "Update configuration files directly",
            "timestamp": int(time.time())
        }
    
    async def get_model_registry(self) -> Dict[str, Any]:
        """Get model registry configuration (redirects to model config)."""
        return await self.get_model_config()
    
    async def update_model_registry(self, registry_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update model registry configuration (redirects to model config update)."""
        return await self.update_config("models", registry_updates)
    
    def cleanup(self):
        """Cleanup adapter resources."""
        if hasattr(self.config_manager, 'cleanup'):
            self.config_manager.cleanup()
            logger.info("Configuration adapter cleaned up")
