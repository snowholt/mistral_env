"""
Configuration API Adapter.

Provides API-compatible interface for configuration management,
bridging config services with REST/GraphQL endpoints.
"""
from typing import Dict, Any, Optional, List
import logging
import time

from .base_adapter import APIServiceAdapter
from ..models import APIRequest, APIResponse
from ..errors import ValidationError, ConfigurationError
from ...services.config.config_service import ConfigService

logger = logging.getLogger(__name__)


class ConfigAPIAdapter(APIServiceAdapter):
    """
    API adapter for configuration management.
    
    Provides API-compatible interface for:
    - Application configuration retrieval and updates
    - Model registry configuration
    - System settings management
    """
    
    def __init__(self, config_service: ConfigService):
        """Initialize config API adapter with required service."""
        self.config_service = config_service
        super().__init__(self.config_service)
    
    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "get_config": "Get current application configuration",
            "update_config": "Update application configuration", 
            "reset_config": "Reset configuration to defaults",
            "get_model_registry": "Get model registry configuration",
            "update_model_registry": "Update model registry",
            "validate_config": "Validate configuration parameters"
        }
    
    async def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current application configuration.
        
        Args:
            section: Optional specific configuration section
            
        Returns:
            Dictionary with configuration data
        """
        try:
            # Configure service
            self.config_service.configure({})
            
            # Get configuration
            if section:
                config_data = self.config_service.get_section(section)
            else:
                config_data = self.config_service.get_all_config()
            
            return {
                "config": config_data,
                "section": section,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            raise ConfigurationError(f"Configuration retrieval failed: {e}")
    
    async def update_config(self, config_updates: Dict[str, Any], 
                           section: Optional[str] = None,
                           validate_only: bool = False) -> Dict[str, Any]:
        """
        Update application configuration.
        
        Args:
            config_updates: Configuration updates to apply
            section: Optional specific section to update
            validate_only: If True, only validate without applying changes
            
        Returns:
            Dictionary with update results
        """
        try:
            # Validate configuration updates
            validation_result = self.validate_config(config_updates, section)
            
            if not validation_result["valid"]:
                raise ValidationError(f"Invalid configuration: {validation_result['errors']}")
            
            if validate_only:
                return {
                    "validated": True,
                    "changes_applied": False,
                    "validation": validation_result,
                    "timestamp": int(time.time())
                }
            
            # Apply configuration updates
            if section:
                result = self.config_service.update_section(section, config_updates)
            else:
                result = self.config_service.update_config(config_updates)
            
            return {
                "validated": True,
                "changes_applied": True,
                "updated_config": result,
                "validation": validation_result,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise ConfigurationError(f"Configuration update failed: {e}")
    
    async def reset_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Args:
            section: Optional specific section to reset
            
        Returns:
            Dictionary with reset results
        """
        try:
            # Reset configuration
            if section:
                result = self.config_service.reset_section(section)
            else:
                result = self.config_service.reset_to_defaults()
            
            return {
                "reset_successful": True,
                "section": section,
                "default_config": result,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            raise ConfigurationError(f"Configuration reset failed: {e}")
    
    async def get_model_registry(self) -> Dict[str, Any]:
        """
        Get model registry configuration.
        
        Returns:
            Dictionary with model registry data
        """
        try:
            # Get model registry
            registry_data = self.config_service.get_model_registry()
            
            return {
                "registry": registry_data,
                "model_count": len(registry_data.get("models", {})),
                "default_model": registry_data.get("default_model"),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get model registry: {e}")
            raise ConfigurationError(f"Model registry retrieval failed: {e}")
    
    async def update_model_registry(self, registry_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update model registry configuration.
        
        Args:
            registry_updates: Registry updates to apply
            
        Returns:
            Dictionary with update results
        """
        try:
            # Update model registry
            result = self.config_service.update_model_registry(registry_updates)
            
            return {
                "updated_registry": result,
                "model_count": len(result.get("models", {})),
                "default_model": result.get("default_model"),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to update model registry: {e}")
            raise ConfigurationError(f"Model registry update failed: {e}")
    
    def validate_config(self, config_data: Dict[str, Any], 
                       section: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Args:
            config_data: Configuration data to validate
            section: Optional specific section being validated
            
        Returns:
            Dictionary with validation results
        """
        validation_errors = []
        warnings = []
        
        try:
            # Basic structure validation
            if not isinstance(config_data, dict):
                validation_errors.append("Configuration must be a dictionary")
                return {
                    "valid": False,
                    "errors": validation_errors,
                    "warnings": warnings,
                    "timestamp": int(time.time())
                }
            
            # Section-specific validation
            if section == "models":
                validation_errors.extend(self._validate_model_config(config_data))
            elif section == "inference":
                validation_errors.extend(self._validate_inference_config(config_data))
            elif section == "system":
                validation_errors.extend(self._validate_system_config(config_data))
            else:
                # General validation for unknown sections
                for key, value in config_data.items():
                    if key.startswith("_"):
                        warnings.append(f"Configuration key '{key}' starts with underscore")
            
            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "warnings": warnings,
                "section": section,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": warnings,
                "section": section,
                "timestamp": int(time.time())
            }
    
    def _validate_model_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate model configuration section."""
        errors = []
        
        if "models" in config_data:
            models = config_data["models"]
            if not isinstance(models, dict):
                errors.append("models must be a dictionary")
            else:
                for model_name, model_config in models.items():
                    if not isinstance(model_config, dict):
                        errors.append(f"Model '{model_name}' config must be a dictionary")
                        continue
                    
                    required_fields = ["model_id", "engine_type"]
                    for field in required_fields:
                        if field not in model_config:
                            errors.append(f"Model '{model_name}' missing required field '{field}'")
        
        return errors
    
    def _validate_inference_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate inference configuration section."""
        errors = []
        
        if "max_tokens" in config_data:
            max_tokens = config_data["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                errors.append("max_tokens must be a positive integer")
        
        if "temperature" in config_data:
            temperature = config_data["temperature"]
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
                errors.append("temperature must be between 0 and 2")
        
        return errors
    
    def _validate_system_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate system configuration section."""
        errors = []
        
        if "gpu_memory_fraction" in config_data:
            fraction = config_data["gpu_memory_fraction"]
            if not isinstance(fraction, (int, float)) or fraction <= 0 or fraction > 1:
                errors.append("gpu_memory_fraction must be between 0 and 1")
        
        if "cache_dir" in config_data:
            cache_dir = config_data["cache_dir"]
            if not isinstance(cache_dir, str):
                errors.append("cache_dir must be a string")
        
        return errors
