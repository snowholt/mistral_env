"""
Model Management API Adapter.

Provides API-compatible interface for model registry and lifecycle operations,
bridging model services with REST/GraphQL endpoints.
"""
from typing import Dict, Any, Optional, List
import logging
import time

from .base_adapter import APIServiceAdapter
from ..models import APIRequest, APIResponse, ModelAddRequest
from ..errors import ModelNotFoundError, ValidationError, ModelLoadError
from ...services.model.registry_service import RegistryService
from ...services.model.lifecycle_service import ModelLifecycleService
from ...services.model.validation_service import ModelValidationService
from ...config.config_manager import ModelConfig, AppConfig

logger = logging.getLogger(__name__)


class ModelAPIAdapter(APIServiceAdapter):
    """
    API adapter for model management operations.
    
    Provides API-compatible interface for:
    - Model registry operations (list, add, update, remove)
    - Model lifecycle management (load, unload, status)
    - Model validation and configuration
    """
    
    def __init__(self, registry_service: RegistryService, 
                 lifecycle_service: ModelLifecycleService,
                 validation_service: ModelValidationService):
        """Initialize model API adapter with required services."""
        self.registry_service = registry_service
        self.lifecycle_service = lifecycle_service
        self.validation_service = validation_service
        super().__init__(self.registry_service)  # Use registry as primary service
    """
    API adapter for model management operations.
    
    Provides API-compatible interface for:
    - Model registry operations (list, add, update, remove)
    - Model lifecycle management (load, unload, status)
    - Model validation and configuration
    """
    
    def __init__(self, registry_service: RegistryService, 
                 lifecycle_service: ModelLifecycleService,
                 validation_service: ModelValidationService):
        """Initialize model API adapter with required services."""
        self.registry_service = registry_service
        self.lifecycle_service = lifecycle_service
        self.validation_service = validation_service
        super().__init__(self.registry_service)  # Use registry as primary service
    
    async def list_models(self, limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        """
        List models with pagination support.
        
        Args:
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            Dictionary with models list and total count
        """
        start_time = time.time()
        
        try:
            # Get default app config with proper model registry path
            from pathlib import Path
            config_dir = Path(__file__).parent.parent.parent / "config"
            default_config_path = config_dir / "default_config.json"
            model_registry_path = config_dir / "model_registry.json"
            
            # Load app config
            if default_config_path.exists():
                app_config = AppConfig.load_from_file(default_config_path)
                # Override the models_file to use absolute path
                app_config.models_file = str(model_registry_path)
            else:
                app_config = AppConfig()
                app_config.models_file = str(model_registry_path)
            
            # Load the model registry - this is the missing step!
            app_config.load_model_registry()
            
            # Configure services with the corrected config
            self.registry_service.configure({})
            
            # Get models from registry
            result = self.registry_service.list_models(app_config)
            models_data = result["models"]
            
            # Apply pagination
            if limit is not None:
                end_idx = offset + limit
                paginated_models = dict(list(models_data.items())[offset:end_idx])
            else:
                paginated_models = models_data
            
            # Convert to API format
            api_models = []
            for name, model_config in paginated_models.items():
                api_models.append({
                    "name": name,
                    "model_id": model_config.model_id,
                    "engine_type": model_config.engine_type,
                    "quantization": model_config.quantization,
                    "dtype": model_config.dtype,
                    "description": model_config.description,
                    "is_default": name == result["default_model"]
                })
            
            return {
                "models": api_models,
                "total_count": len(models_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "list_models": "List all available models",
            "get_model_info": "Get detailed model information",
            "load_model": "Load a model for inference",
            "unload_model": "Unload a specific model",
            "unload_all_models": "Unload all models",
            "get_model_status": "Get model loading status",
            "get_system_status": "Get system memory and GPU status"
        }

    async def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Dictionary with detailed model information
        """
        return self.get_model_info(model_name)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Dictionary with detailed model information
        """
        try:
            # Configure services with default config
            self.registry_service.configure({})
            
            # Get default app config with proper model registry path
            from pathlib import Path
            config_dir = Path(__file__).parent.parent.parent / "config"
            default_config_path = config_dir / "default_config.json"
            model_registry_path = config_dir / "model_registry.json"
            
            # Load app config
            if default_config_path.exists():
                app_config = AppConfig.load_from_file(default_config_path)
                # Override the models_file to use absolute path
                app_config.models_file = str(model_registry_path)
            else:
                app_config = AppConfig()
                app_config.models_file = str(model_registry_path)
            
            # Get model configuration
            result = self.registry_service.list_models(app_config)
            models_data = result["models"]
            
            if model_name not in models_data:
                raise ModelNotFoundError(f"Model '{model_name}' not found in registry")
            
            model_config = models_data[model_name]
            
            # Get loading status
            loaded_models = self.lifecycle_service.list_loaded_models()
            is_loaded = any(model['name'] == model_name for model in loaded_models)
            
            return {
                "name": model_name,
                "model_id": model_config.model_id,
                "engine_type": model_config.engine_type,
                "quantization": model_config.quantization,
                "dtype": model_config.dtype,
                "description": model_config.description,
                "is_loaded": is_loaded,
                "is_default": model_name == result["default_model"],
                "loading_metadata": next(
                    (model for model in loaded_models if model['name'] == model_name),
                    None
                ) if is_loaded else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for '{model_name}': {e}")
            raise

    async def load_model(self, model_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a model for inference.
        
        Args:
            model_name: Name of the model to load
            force_reload: Whether to force reload if already loaded
            
        Returns:
            Dictionary with loading result
        """
        try:
            # Get default app config with proper model registry path
            from pathlib import Path
            config_dir = Path(__file__).parent.parent.parent / "config"
            default_config_path = config_dir / "default_config.json"
            model_registry_path = config_dir / "model_registry.json"
            
            # Load app config
            if default_config_path.exists():
                app_config = AppConfig.load_from_file(default_config_path)
                # Override the models_file to use absolute path
                app_config.models_file = str(model_registry_path)
            else:
                app_config = AppConfig()
                app_config.models_file = str(model_registry_path)
            
            # Load the model registry - this is the missing step!
            app_config.load_model_registry()
            
            # Configure services with the corrected config
            self.registry_service.configure({})
            
            # Get model configuration from registry
            model_config = self.registry_service.get_model(app_config, model_name)
            if not model_config:
                raise ModelNotFoundError(f"Model '{model_name}' not found in registry")
            
            # Check if already loaded and not forcing reload
            if not force_reload and self.lifecycle_service.model_manager.is_model_loaded(model_name):
                return {
                    "model_name": model_name,
                    "model_id": model_config.model_id,
                    "engine": model_config.engine_type,
                    "quantization": model_config.quantization,
                    "status": "already_loaded"
                }
            
            # Track loading time
            import time
            start_time = time.time()
            
            # Load the model using lifecycle service
            success, error_msg = self.lifecycle_service.load_model(model_config, show_progress=True)
            
            load_time = time.time() - start_time
            
            if not success:
                raise ModelLoadError(error_msg or f"Failed to load model '{model_name}'")
            
            return {
                "model_name": model_name,
                "model_id": model_config.model_id,
                "engine": model_config.engine_type,
                "quantization": model_config.quantization,
                "status": "loaded",
                "load_time_seconds": load_time
            }
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """
        Unload a specific model.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            Dictionary with unloading result
        """
        try:
            # Unload the model
            success, error_msg = self.lifecycle_service.unload_model(model_name, show_progress=True)
            
            if not success:
                raise ModelLoadError(error_msg or f"Failed to unload model '{model_name}'")
            
            return {
                "model_name": model_name,
                "status": "unloaded"
            }
            
        except Exception as e:
            logger.error(f"Failed to unload model '{model_name}': {e}")
            raise

    def unload_all_models(self) -> Dict[str, Any]:
        """
        Unload all loaded models.
        
        Returns:
            Dictionary with unloading results
        """
        try:
            result = self.lifecycle_service.unload_all_models()
            
            return {
                "unloaded_models": result.get("unloaded_models", []),
                "total_memory_freed": result.get("total_memory_freed"),
                "unloading_time_ms": result.get("unloading_time_ms"),
                "status": "all_unloaded"
            }
            
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
            raise

    async def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current model loading and system status.
        
        Args:
            model_name: Optional specific model to get status for
            
        Returns:
            Dictionary with status information
        """
        try:
            # Get loaded models
            loaded_models = self.lifecycle_service.list_loaded_models()
            
            # Get system status
            system_status = self.lifecycle_service.show_status()
            
            # Filter by specific model if requested
            if model_name:
                loaded_models = [
                    model for model in loaded_models 
                    if model['name'] == model_name
                ]
            
            return {
                "loaded_models": loaded_models,
                "system_status": system_status,
                "total_loaded": len(loaded_models),
                "timestamp": system_status.get("timestamp")
            }
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            raise
