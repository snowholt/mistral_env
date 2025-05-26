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
            # Configure services with default config
            self.registry_service.configure({})
            
            # Get default app config
            from pathlib import Path
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            app_config = AppConfig.load_from_file(default_config_path) if default_config_path.exists() else AppConfig()
            
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
            "get_model_status": "Get model loading status",
            "get_system_status": "Get system memory and GPU status"
        }
    
    @require_permission("model:read")
    def list_models(self, auth_context: AuthContext, 
                   request: ModelListRequest) -> ModelListResponse:
        """
        List available models with optional filtering.
        
        Args:
            auth_context: Authentication context
            request: Model list request with optional filters
            
        Returns:
            ModelListResponse with model list and metadata
        """
        def _list_models():
            # Get models from registry
            models = self.registry_service.list_models(
                model_type=request.model_type,
                engine=request.engine,
                quantization=request.quantization
            )
            
            # Get loading status for each model
            loaded_models = self.lifecycle_service.list_loaded_models()
            loaded_model_names = {model['name'] for model in loaded_models}
            
            # Enhance model data with loading status
            enhanced_models = []
            for model in models:
                model_data = model.copy()
                model_data['is_loaded'] = model['name'] in loaded_model_names
                enhanced_models.append(model_data)
            
            return {
                "models": enhanced_models,
                "total_count": len(enhanced_models),
                "loaded_count": len(loaded_model_names),
                "filters_applied": {
                    "model_type": request.model_type,
                    "engine": request.engine,
                    "quantization": request.quantization
                }
            }
        
        return self.execute_with_auth(auth_context, _list_models)
    
    @require_permission("model:read")
    def get_model_info(self, auth_context: AuthContext,
                      model_name: str) -> APIResponse:
        """
        Get detailed information about a specific model.
        
        Args:
            auth_context: Authentication context
            model_name: Name of the model to get info for
            
        Returns:
            APIResponse with detailed model information
        """
        def _get_model_info():
            # Get model configuration
            model_config = self.registry_service.get_model(model_name)
            
            # Get loading status
            loaded_models = self.lifecycle_service.list_loaded_models()
            is_loaded = any(model['name'] == model_name for model in loaded_models)
            
            # Get validation status
            validation_result = self.registry_service.validate_model_config(model_config)
            
            return {
                "model": model_config,
                "is_loaded": is_loaded,
                "validation": validation_result,
                "loading_metadata": next(
                    (model for model in loaded_models if model['name'] == model_name),
                    None
                ) if is_loaded else None
            }
        
        return self.execute_with_auth(auth_context, _get_model_info)
    
    @require_permission("model:manage")
    def load_model(self, auth_context: AuthContext,
                  request: ModelLoadRequest) -> ModelLoadResponse:
        """
        Load a model for inference.
        
        Args:
            auth_context: Authentication context
            request: Model load request with configuration
            
        Returns:
            ModelLoadResponse with loading result
        """
        def _load_model():
            # Load the model using lifecycle service
            result = self.lifecycle_service.load_model(
                model_name=request.model_name,
                engine=request.engine,
                quantization=request.quantization,
                **request.additional_config
            )
            
            return {
                "model_name": request.model_name,
                "engine": result.get("engine"),
                "quantization": result.get("quantization"),
                "loading_time_ms": result.get("loading_time_ms"),
                "memory_usage": result.get("memory_usage"),
                "status": "loaded"
            }
        
        return self.execute_with_auth(auth_context, _load_model)
    
    @require_permission("model:manage")
    def unload_model(self, auth_context: AuthContext,
                    request: ModelUnloadRequest) -> ModelUnloadResponse:
        """
        Unload a specific model.
        
        Args:
            auth_context: Authentication context
            request: Model unload request
            
        Returns:
            ModelUnloadResponse with unloading result
        """
        def _unload_model():
            # Unload the model
            result = self.lifecycle_service.unload_model(request.model_name)
            
            return {
                "model_name": request.model_name,
                "status": "unloaded",
                "memory_freed": result.get("memory_freed"),
                "unloading_time_ms": result.get("unloading_time_ms")
            }
        
        return self.execute_with_auth(auth_context, _unload_model)
    
    @require_permission("model:manage")
    def unload_all_models(self, auth_context: AuthContext) -> APIResponse:
        """
        Unload all loaded models.
        
        Args:
            auth_context: Authentication context
            
        Returns:
            APIResponse with unloading results
        """
        def _unload_all_models():
            result = self.lifecycle_service.unload_all_models()
            
            return {
                "unloaded_models": result.get("unloaded_models", []),
                "total_memory_freed": result.get("total_memory_freed"),
                "unloading_time_ms": result.get("unloading_time_ms"),
                "status": "all_unloaded"
            }
        
        return self.execute_with_auth(auth_context, _unload_all_models)
    
    @require_permission("model:read")
    def get_model_status(self, auth_context: AuthContext,
                        request: ModelStatusRequest) -> ModelStatusResponse:
        """
        Get current model loading and system status.
        
        Args:
            auth_context: Authentication context
            request: Status request with optional model filter
            
        Returns:
            ModelStatusResponse with status information
        """
        def _get_model_status():
            # Get loaded models
            loaded_models = self.lifecycle_service.list_loaded_models()
            
            # Get system status
            system_status = self.lifecycle_service.show_status()
            
            # Filter by specific model if requested
            if request.model_name:
                loaded_models = [
                    model for model in loaded_models 
                    if model['name'] == request.model_name
                ]
            
            return {
                "loaded_models": loaded_models,
                "system_status": system_status,
                "total_loaded": len(loaded_models),
                "timestamp": system_status.get("timestamp")
            }
        
        return self.execute_with_auth(auth_context, _get_model_status)
