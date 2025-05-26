"""
Model management API endpoints.

Provides REST API endpoints for model registry operations including:
- CRUD operations on model configurations
- Model lifecycle management (load/unload)
- Model validation and status checking
"""
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from ..models import (
    APIResponse, ModelListResponse, ModelAddRequest, ModelAddResponse,
    ModelLoadRequest, ModelLoadResponse, ErrorResponse
)
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import ModelNotFoundError, ValidationError, ModelLoadError
from ..adapters.model_adapter import ModelAPIAdapter
from ...services.model import RegistryService, ModelLifecycleService, ModelValidationService
from ...config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

models_router = APIRouter(prefix="/models", tags=["models"])

# Initialize services
registry_service = RegistryService()
lifecycle_service = ModelLifecycleService()
validation_service = ModelValidationService()

# Initialize adapter
model_adapter = ModelAPIAdapter(registry_service, lifecycle_service, validation_service)


@models_router.get("/", response_model=ModelListResponse)
async def list_models(
    auth: AuthContext = Depends(get_auth_context),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum number of models to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of models to skip")
):
    """
    List all models in the registry.
    
    Returns a paginated list of all model configurations with their basic information.
    Supports filtering and pagination for large model registries.
    """
    require_permissions(auth, ["model_read"])
    
    try:
        result = await model_adapter.list_models(limit=limit, offset=offset)
        return ModelListResponse(
            success=True,
            models=result["models"],
            total_count=result["total_count"]
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@models_router.post("/", response_model=ModelAddResponse)
async def add_model(
    request: ModelAddRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Add a new model to the registry.
    
    Creates a new model configuration and optionally sets it as the default.
    Validates the configuration before adding to ensure compatibility.
    """
    require_permissions(auth, ["model_write"])
    
    try:
        result = await model_adapter.add_model(request)
        return ModelAddResponse(
            success=True,
            model_name=result["model_name"],
            message=result["message"]
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to add model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add model: {str(e)}")


@models_router.get("/{model_name}", response_model=APIResponse)
async def get_model(
    model_name: str = Path(..., description="Name of the model to retrieve"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Get detailed information about a specific model.
    
    Returns complete model configuration including parameters, engine settings,
    and current status (loaded/unloaded).
    """
    require_permissions(auth, ["model_read"])
    
    try:
        result = await model_adapter.get_model(model_name)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@models_router.put("/{model_name}", response_model=APIResponse)
async def update_model(
    model_name: str = Path(..., description="Name of the model to update"),
    updates: Dict[str, Any] = None,
    set_as_default: bool = False,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Update an existing model configuration.
    
    Allows partial updates to model parameters while maintaining validation.
    Can optionally set the updated model as the new default.
    """
    require_permissions(auth, ["model_write"])
    
    try:
        result = await model_adapter.update_model(model_name, updates or {}, set_as_default)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to update model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@models_router.delete("/{model_name}", response_model=APIResponse)
async def remove_model(
    model_name: str = Path(..., description="Name of the model to remove"),
    clear_cache: bool = Query(False, description="Whether to clear the model's cache"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Remove a model from the registry.
    
    Permanently removes the model configuration and optionally clears its cache.
    Will unload the model from memory if currently loaded.
    """
    require_permissions(auth, ["model_delete"])
    
    try:
        result = await model_adapter.remove_model(model_name, clear_cache)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to remove model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove model: {str(e)}")


@models_router.post("/{model_name}/load", response_model=ModelLoadResponse)
async def load_model(
    model_name: str = Path(..., description="Name of the model to load"),
    request: ModelLoadRequest = None,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Load a model into memory for inference.
    
    Initializes the model with the specified engine and makes it available
    for inference operations. Returns memory usage and load time statistics.
    """
    require_permissions(auth, ["model_load"])
    
    try:
        force_reload = request.force_reload if request else False
        result = await model_adapter.load_model(model_name, force_reload)
        
        return ModelLoadResponse(
            success=True,
            model_name=result["model_name"],
            model_id=result["model_id"],
            memory_usage_mb=result.get("memory_usage_mb"),
            load_time_seconds=result.get("load_time_seconds")
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@models_router.post("/{model_name}/unload", response_model=APIResponse)
async def unload_model(
    model_name: str = Path(..., description="Name of the model to unload"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Unload a model from memory.
    
    Removes the model from memory and frees up GPU/system resources.
    The model configuration remains in the registry.
    """
    require_permissions(auth, ["model_load"])
    
    try:
        result = await model_adapter.unload_model(model_name)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@models_router.get("/{model_name}/status", response_model=APIResponse)
async def get_model_status(
    model_name: str = Path(..., description="Name of the model to check"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Get the current status of a model.
    
    Returns information about whether the model is loaded, memory usage,
    and other runtime statistics.
    """
    require_permissions(auth, ["model_read"])
    
    try:
        result = await model_adapter.get_model_status(model_name)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get model status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@models_router.post("/{model_name}/validate", response_model=APIResponse)
async def validate_model(
    model_name: str = Path(..., description="Name of the model to validate"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Validate a model configuration.
    
    Checks the model configuration for compatibility, validates parameters,
    and returns any issues or recommendations.
    """
    require_permissions(auth, ["model_read"])
    
    try:
        result = await model_adapter.validate_model(model_name)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to validate model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate model: {str(e)}")


@models_router.post("/default/{model_name}", response_model=APIResponse)
async def set_default_model(
    model_name: str = Path(..., description="Name of the model to set as default"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Set a model as the default.
    
    Changes the registry default to the specified model, which will be used
    for inference operations when no specific model is requested.
    """
    require_permissions(auth, ["model_write"])
    
    try:
        result = await model_adapter.set_default_model(model_name)
        return APIResponse(
            success=True,
            data=result
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to set default model to {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set default model: {str(e)}")
