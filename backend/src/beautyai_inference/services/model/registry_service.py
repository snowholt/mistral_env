"""
Model registry service for managing model registry CRUD operations.

This service focuses exclusively on model registry operations including
adding, updating, removing, listing, and managing default models.
"""
import logging
from typing import Dict, Any, Optional

from ..base.base_service import BaseService
from ...config.config_manager import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class RegistryService(BaseService):
    """Service for model registry CRUD operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        
    def list_models(self, registry_config: AppConfig) -> Dict[str, Any]:
        """
        List all models in the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            
        Returns:
            Dict containing models data and metadata
        """
        self.app_config = registry_config
        
        models = self.app_config.model_registry.models
        default_model = self.app_config.model_registry.default_model
        
        return {
            "models": models,
            "default_model": default_model,
            "total_count": len(models)
        }
    
    def add_model(self, registry_config: AppConfig, model_config: ModelConfig, set_as_default: bool = False) -> bool:
        """
        Add a new model to the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            model_config: Model configuration to add
            set_as_default: Whether to set this model as default
            
        Returns:
            bool: True if successful, False if model already exists
        """
        self.app_config = registry_config
        
        if model_config.name in self.app_config.model_registry.models:
            logger.warning(f"Model with name '{model_config.name}' already exists")
            return False
        
        # Add model to registry
        self.app_config.add_model_config(model_config)
        
        # Set as default if requested
        if set_as_default:
            self.app_config.model_registry.set_default_model(model_config.name)
            
        # Save registry
        self.app_config.save_model_registry()
        
        logger.info(f"Added model '{model_config.name}' to registry")
        return True
    
    def get_model(self, registry_config: AppConfig, model_name: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration from the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            model_name: Name of the model to retrieve
            
        Returns:
            ModelConfig if found, None otherwise
        """
        self.app_config = registry_config
        
        models = self.app_config.model_registry.models
        return models.get(model_name)
    
    def update_model(self, registry_config: AppConfig, model_name: str, updates: Dict[str, Any], set_as_default: bool = False) -> bool:
        """
        Update an existing model in the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            model_name: Name of the model to update
            updates: Dictionary of field updates
            set_as_default: Whether to set this model as default
            
        Returns:
            bool: True if successful, False if model not found
        """
        self.app_config = registry_config
        
        models = self.app_config.model_registry.models
        
        if model_name not in models:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
            
        model = models[model_name]
        
        # Update only the fields that were provided
        for field, value in updates.items():
            if hasattr(model, field) and value is not None:
                setattr(model, field, value)
        
        # Update the model in the registry
        self.app_config.model_registry.add_model(model)  # This will overwrite existing
        
        # Set as default if requested
        if set_as_default:
            self.app_config.model_registry.set_default_model(model_name)
            
        # Save registry
        self.app_config.save_model_registry()
        
        logger.info(f"Updated model '{model_name}' in registry")
        return True
    
    def remove_model(self, registry_config: AppConfig, model_name: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            model_name: Name of the model to remove
            
        Returns:
            bool: True if successful, False if model not found
        """
        self.app_config = registry_config
        
        models = self.app_config.model_registry.models
        
        if model_name not in models:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
        
        # Clear default if this model is the default
        if model_name == self.app_config.model_registry.default_model:
            logger.info(f"Clearing default model setting (was '{model_name}')")
            self.app_config.model_registry.default_model = None
            
        # Remove model
        self.app_config.model_registry.remove_model(model_name)
        self.app_config.save_model_registry()
        
        logger.info(f"Removed model '{model_name}' from registry")
        return True
    
    def set_default_model(self, registry_config: AppConfig, model_name: str) -> bool:
        """
        Set a model as the default.
        
        Args:
            registry_config: Application configuration containing model registry
            model_name: Name of the model to set as default
            
        Returns:
            bool: True if successful, False if model not found
        """
        self.app_config = registry_config
        
        models = self.app_config.model_registry.models
        
        if model_name not in models:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
            
        self.app_config.model_registry.set_default_model(model_name)
        self.app_config.save_model_registry()
        
        logger.info(f"Set '{model_name}' as the default model")
        return True
    
    def has_model(self, registry_config: AppConfig, model_name: str) -> bool:
        """
        Check if a model exists in the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            model_name: Name of the model to check
            
        Returns:
            bool: True if model exists, False otherwise
        """
        self.app_config = registry_config
        return model_name in self.app_config.model_registry.models
    
    def get_default_model(self, registry_config: AppConfig) -> Optional[str]:
        """
        Get the default model name.
        
        Args:
            registry_config: Application configuration containing model registry
            
        Returns:
            str: Default model name if set, None otherwise
        """
        self.app_config = registry_config
        return self.app_config.model_registry.default_model
    
    def get_model_count(self, registry_config: AppConfig) -> int:
        """
        Get the total number of models in the registry.
        
        Args:
            registry_config: Application configuration containing model registry
            
        Returns:
            int: Number of models in registry
        """
        self.app_config = registry_config
        return len(self.app_config.model_registry.models)

# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------
# Some legacy components/imports still reference `ModelRegistryService`. To avoid
# breaking those without refactoring all import sites immediately, we provide an
# alias subclass that inherits the complete behavior of `RegistryService`.
# This keeps existing code functional while signaling the preferred modern name.

class ModelRegistryService(RegistryService):  # pragma: no cover - thin alias
    """Backward-compatible alias for legacy imports expecting ModelRegistryService.

    Prefer importing and using `RegistryService` going forward. This alias can be
    removed after all references are migrated.
    """
    pass
