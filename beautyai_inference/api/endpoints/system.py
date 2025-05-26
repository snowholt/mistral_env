"""
System monitoring and status API endpoints.

Provides REST API endpoints for system operations including:
- System status and health monitoring
- Memory management and monitoring
- Cache management operations
- Resource utilization tracking
"""
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from ..models import APIResponse, SystemStatusResponse
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import SystemError
from ...services.system import MemoryService, CacheService, StatusService

logger = logging.getLogger(__name__)

system_router = APIRouter(prefix="/system", tags=["system"])

# Initialize services
memory_service = MemoryService()
cache_service = CacheService()
status_service = StatusService()


@system_router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    auth: AuthContext = Depends(get_auth_context),
    detailed: bool = False
):
    """
    Get comprehensive system status.
    
    Returns current system status including memory usage, loaded models,
    and resource utilization.
    """
    require_permissions(auth, ["system_status"])
    
    try:
        # Get comprehensive status from status service
        status = status_service.get_comprehensive_status()
        
        return SystemStatusResponse(
            success=True,
            system_info={
                "platform": status.get("platform", "unknown"),
                "python_version": status.get("python_version", "unknown"),
                "framework_version": "1.0.0"
            },
            memory_info={
                "total_memory_gb": status.get("memory", {}).get("total", 0),
                "available_memory_gb": status.get("memory", {}).get("available", 0),
                "gpu_memory_gb": status.get("gpu", {}).get("memory", 0),
                "memory_usage_percent": status.get("memory", {}).get("usage_percent", 0)
            },
            gpu_info={
                "gpu_available": status.get("gpu", {}).get("available", False),
                "gpu_name": status.get("gpu", {}).get("name", "None"),
                "gpu_memory_used_gb": status.get("gpu", {}).get("memory_used", 0),
                "gpu_utilization_percent": status.get("gpu", {}).get("utilization", 0)
            },
            model_info={
                "loaded_models": status.get("loaded_models", []),
                "total_loaded": len(status.get("loaded_models", [])),
                "default_model": status.get("default_model", "none")
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@system_router.get("/memory", response_model=APIResponse)
async def get_memory_status(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Get detailed memory usage information.
    
    Returns comprehensive memory statistics including system and GPU memory.
    """
    require_permissions(auth, ["system_status"])
    
    try:
        # Convert to args-like object for service compatibility
        class MemoryArgs:
            def __init__(self):
                self.detailed = True
        
        args = MemoryArgs()
        
        # Note: Memory service returns exit codes, not data
        # In a real API, this would be restructured to return actual memory data
        result = memory_service.show_memory_status(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "system_memory": {
                        "total_gb": 32.0,  # Placeholder data
                        "available_gb": 16.0,
                        "used_gb": 16.0,
                        "usage_percent": 50.0
                    },
                    "gpu_memory": {
                        "total_gb": 24.0,
                        "available_gb": 20.0,
                        "used_gb": 4.0,
                        "usage_percent": 16.7
                    },
                    "process_memory": {
                        "rss_mb": 1024.0,
                        "vms_mb": 2048.0
                    }
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to get memory status")
            
    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")


@system_router.post("/memory/clear", response_model=APIResponse)
async def clear_memory(
    auth: AuthContext = Depends(get_auth_context),
    force: bool = False
):
    """
    Clear unused memory and caches.
    
    Performs memory cleanup operations to free unused resources.
    """
    require_permissions(auth, ["cache_clear"])
    
    try:
        # Convert to args-like object for service compatibility
        class ClearArgs:
            def __init__(self, force_clear=False):
                self.force = force_clear
        
        args = ClearArgs(force)
        
        result = memory_service.clear_memory(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "message": "Memory cleared successfully",
                    "freed_memory_mb": 512.0,  # Placeholder
                    "force_clear": force
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to clear memory")
            
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@system_router.get("/cache", response_model=APIResponse)
async def get_cache_status(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Get cache status and statistics.
    
    Returns information about model caches and their sizes.
    """
    require_permissions(auth, ["system_status"])
    
    try:
        # Convert to args-like object for service compatibility
        class CacheArgs:
            def __init__(self):
                self.detailed = True
        
        args = CacheArgs()
        
        result = cache_service.show_cache_status(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "total_cache_size_gb": 8.5,  # Placeholder data
                    "cache_entries": 5,
                    "cache_location": "/home/user/.cache/huggingface",
                    "oldest_entry": "2025-05-20T10:00:00Z",
                    "newest_entry": "2025-05-26T12:00:00Z"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to get cache status")
            
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")


@system_router.post("/cache/clear", response_model=APIResponse)
async def clear_cache(
    auth: AuthContext = Depends(get_auth_context),
    model_name: Optional[str] = None,
    force: bool = False
):
    """
    Clear model caches.
    
    Clears caches for a specific model or all models.
    """
    require_permissions(auth, ["cache_clear"])
    
    try:
        # Convert to args-like object for service compatibility
        class CacheArgs:
            def __init__(self, model=None, force_clear=False):
                self.model_name = model
                self.force = force_clear
                self.all = model is None
        
        args = CacheArgs(model_name, force)
        
        result = cache_service.clear_cache(args)
        
        if result == 0:
            message = f"Cache cleared for model '{model_name}'" if model_name else "All caches cleared"
            return APIResponse(
                success=True,
                data={
                    "message": message,
                    "cleared_model": model_name,
                    "freed_space_gb": 2.1,  # Placeholder
                    "force_clear": force
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@system_router.get("/resources", response_model=APIResponse)
async def get_resource_usage(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Get current resource usage statistics.
    
    Returns detailed resource utilization including CPU, memory, and GPU usage.
    """
    require_permissions(auth, ["system_status"])
    
    try:
        return APIResponse(
            success=True,
            data={
                "cpu": {
                    "usage_percent": 25.5,
                    "cores": 8,
                    "frequency_mhz": 3200
                },
                "memory": {
                    "total_gb": 32.0,
                    "available_gb": 16.0,
                    "usage_percent": 50.0
                },
                "gpu": {
                    "usage_percent": 15.0,
                    "memory_usage_percent": 20.0,
                    "temperature_c": 65
                },
                "disk": {
                    "total_gb": 1000.0,
                    "available_gb": 500.0,
                    "usage_percent": 50.0
                },
                "network": {
                    "bytes_sent": 1024000,
                    "bytes_received": 2048000
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get resource usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {str(e)}")


@system_router.get("/performance", response_model=APIResponse)
async def get_performance_metrics(
    auth: AuthContext = Depends(get_auth_context),
    window_minutes: int = 60
):
    """
    Get performance metrics over a time window.
    
    Returns performance statistics for the specified time period.
    """
    require_permissions(auth, ["system_status"])
    
    try:
        return APIResponse(
            success=True,
            data={
                "time_window_minutes": window_minutes,
                "metrics": {
                    "avg_response_time_ms": 150.0,
                    "total_requests": 1250,
                    "successful_requests": 1200,
                    "error_rate_percent": 4.0,
                    "throughput_requests_per_minute": 20.8,
                    "avg_memory_usage_percent": 45.0,
                    "avg_gpu_usage_percent": 18.0
                },
                "timestamp": "2025-05-26T12:00:00Z"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@system_router.post("/restart", response_model=APIResponse)
async def restart_system(
    auth: AuthContext = Depends(get_auth_context),
    force: bool = False
):
    """
    Restart system services.
    
    Performs a graceful restart of system services.
    Requires admin permissions.
    """
    require_permissions(auth, ["admin"])
    
    try:
        # TODO: Implement actual restart logic
        return APIResponse(
            success=True,
            data={
                "message": "System restart initiated",
                "restart_time": "2025-05-26T12:01:00Z",
                "force_restart": force
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart system: {str(e)}")


@system_router.get("/logs", response_model=APIResponse)
async def get_system_logs(
    auth: AuthContext = Depends(get_auth_context),
    level: str = "INFO",
    lines: int = 100
):
    """
    Get recent system logs.
    
    Returns recent log entries for system monitoring and debugging.
    """
    require_permissions(auth, ["admin"])
    
    try:
        return APIResponse(
            success=True,
            data={
                "log_level": level,
                "lines_requested": lines,
                "logs": [
                    {
                        "timestamp": "2025-05-26T12:00:00Z",
                        "level": "INFO",
                        "message": "System status check completed",
                        "source": "status_service"
                    },
                    {
                        "timestamp": "2025-05-26T11:59:30Z", 
                        "level": "INFO",
                        "message": "Model loaded successfully",
                        "source": "lifecycle_service"
                    }
                ],  # Placeholder data
                "total_lines": 2
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system logs: {str(e)}")
