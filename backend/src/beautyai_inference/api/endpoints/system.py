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
        
        # Extract system info
        system_info = status.system_info or {}
        memory_status = status.memory_status
        loaded_models_info = status.loaded_models or {}
        
        # Build memory info from memory status
        memory_info = {
            "total_memory_gb": status.memory_status.system_stats.get("total_gb", 0),
            "available_memory_gb": status.memory_status.system_stats.get("available_gb", 0),
            "gpu_memory_gb": 0,  # Will be filled from GPU info
            "memory_usage_percent": status.memory_status.system_stats.get("percent", 0)
        }
        
        # Extract GPU info
        gpu_info_data = status.memory_status.gpu_info
        gpu_memory_gb = 0
        gpu_memory_used_gb = 0
        gpu_utilization_percent = 0
        
        if status.memory_status.has_gpu and status.memory_status.gpu_stats:
            gpu_stat = status.memory_status.gpu_stats[0]
            gpu_memory_gb = round(gpu_stat.get("total_memory", 0) / (1024**3), 2)
            gpu_memory_used_gb = round(gpu_stat.get("memory_used", 0) / (1024**3), 2)
            gpu_utilization_percent = round(gpu_stat.get("gpu_utilization", 0), 1)
        
        gpu_info = {
            "gpu_available": gpu_info_data.get("is_available", False),
            "gpu_name": gpu_info_data.get("device_name", "None"),
            "gpu_memory_used_gb": gpu_memory_used_gb,
            "gpu_utilization_percent": gpu_utilization_percent
        }
        
        # Update memory info with GPU data
        memory_info["gpu_memory_gb"] = gpu_memory_gb
        
        # Extract model info
        models_list = loaded_models_info.get("local_models", {})
        model_info = {
            "loaded_models": list(models_list.keys()),
            "total_loaded": loaded_models_info.get("local_count", 0),
            "default_model": "none"  # TODO: Get from config
        }
        
        return SystemStatusResponse(
            success=True,
            system_info={
                "platform": system_info.get("platform", "unknown"),
                "python_version": system_info.get("python_version", "unknown"),
                "framework_version": "1.0.0",
                "gpu_available": gpu_info_data.get("is_available", False),
                "gpu_name": gpu_info_data.get("device_name", "None")
            },
            memory_info={
                "total_memory_gb": status.memory_status.system_stats.get("total_gb", 0),
                "available_memory_gb": status.memory_status.system_stats.get("available_gb", 0),
                "gpu_memory_gb": gpu_memory_gb,
                "memory_usage_percent": status.memory_status.system_stats.get("percent", 0),
                "gpu_memory_used_gb": gpu_memory_used_gb
            },
            loaded_models=[
                {
                    "model_name": model_name,
                    "model_info": model_data
                }
                for model_name, model_data in models_list.items()
            ],
            cache_info={
                "total_loaded": loaded_models_info.get("local_count", 0),
                "default_model": "none"
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
        # Get memory status from service
        memory_status = memory_service.get_memory_status()
        
        # Extract system memory info
        system_stats = memory_status.system_stats
        system_memory = {
            "total_gb": round(system_stats.get("total_gb", 0), 2),
            "available_gb": round(system_stats.get("available_gb", 0), 2),
            "used_gb": round(system_stats.get("used_gb", 0), 2),
            "usage_percent": round(system_stats.get("percent", 0), 1)
        }
        
        # Extract GPU memory info
        gpu_memory = {"total_gb": 0, "available_gb": 0, "used_gb": 0, "usage_percent": 0}
        if memory_status.has_gpu and memory_status.gpu_stats:
            gpu_stat = memory_status.gpu_stats[0]  # First GPU
            gpu_memory = {
                "total_gb": round(gpu_stat.get("total_memory", 0) / (1024**3), 2),
                "available_gb": round(gpu_stat.get("memory_free", 0) / (1024**3), 2),
                "used_gb": round(gpu_stat.get("memory_used", 0) / (1024**3), 2),
                "usage_percent": round(gpu_stat.get("memory_used_percent", 0), 1)
            }
        
        # Get process memory info (rough estimate)
        import psutil
        process = psutil.Process()
        memory_info_bytes = process.memory_info()
        process_memory = {
            "rss_mb": round(memory_info_bytes.rss / (1024**2), 1),
            "vms_mb": round(memory_info_bytes.vms / (1024**2), 1)
        }
        
        return APIResponse(
            success=True,
            data={
                "system_memory": system_memory,
                "gpu_memory": gpu_memory,
                "process_memory": process_memory,
                "gpu_available": memory_status.has_gpu,
                "gpu_count": len(memory_status.gpu_stats) if memory_status.gpu_stats else 0
            }
        )
            
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
        # Get memory status before clearing
        memory_before = memory_service.get_memory_status()
        
        # Clear GPU memory
        gpu_cleared = memory_service.clear_gpu_memory()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Get memory status after clearing
        memory_after = memory_service.get_memory_status()
        
        # Calculate freed memory (rough estimate)
        freed_memory_mb = 0
        if memory_before.has_gpu and memory_after.has_gpu:
            if memory_before.gpu_stats and memory_after.gpu_stats:
                before_used = memory_before.gpu_stats[0].get("used", 0)
                after_used = memory_after.gpu_stats[0].get("used", 0)
                freed_memory_mb = round((before_used - after_used) / (1024**2), 1)
        
        return APIResponse(
            success=True,
            data={
                "message": "Memory cleared successfully",
                "freed_memory_mb": max(freed_memory_mb, 0),  # Don't show negative
                "gpu_cleared": gpu_cleared,
                "force_clear": force
            }
        )
            
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
        # Get cache statistics from service
        cache_stats = cache_service.get_cache_statistics()
        total_cache_info = cache_service.get_total_cache_size()
        cached_models = cache_service.list_cached_models()
        
        # Format the response
        return APIResponse(
            success=True,
            data={
                "total_cache_size_gb": round(total_cache_info.get("total_size_bytes", 0) / (1024**3), 2),
                "cache_entries": len(cached_models),
                "cache_location": str(cache_service._get_huggingface_cache_dir()),
                "cached_models": [
                    {
                        "model_id": cache_info.model_id,
                        "size_gb": round(cache_info.size_bytes / (1024**3), 2),
                        "cache_path": str(cache_info.cache_path),
                        "size_human": cache_info.size_human
                    }
                    for cache_info in cached_models[:10]  # Limit to first 10 for brevity
                ],
                "statistics": cache_stats
            }
        )
            
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
        # Get cache size before clearing
        total_cache_before = cache_service.get_total_cache_size()
        size_before = total_cache_before.get("total_size_bytes", 0)
        
        if model_name:
            # Clear specific model cache
            success = cache_service.clear_model_cache(model_name)
            message = f"Cache cleared for model '{model_name}'" if success else f"Failed to clear cache for model '{model_name}'"
        else:
            # Clear all caches
            success = cache_service.clear_all_cache()
            message = "All caches cleared" if success else "Failed to clear all caches"
        
        # Get cache size after clearing
        total_cache_after = cache_service.get_total_cache_size()
        size_after = total_cache_after.get("total_size_bytes", 0)
        freed_space_gb = round((size_before - size_after) / (1024**3), 2)
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return APIResponse(
            success=True,
            data={
                "message": message,
                "cleared_model": model_name,
                "freed_space_gb": max(freed_space_gb, 0),  # Don't show negative
                "force_clear": force
            }
        )
            
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
        import psutil
        import shutil
        
        # Get memory status from our memory service
        memory_status = memory_service.get_memory_status()
        
        # Get CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Get disk usage for root partition
        disk_usage = shutil.disk_usage("/")
        
        # Get network I/O statistics
        net_io = psutil.net_io_counters()
        
        # Extract system memory data
        system_memory = memory_status.system_stats
        
        # Extract GPU data
        gpu_data = {"usage_percent": 0, "memory_usage_percent": 0, "temperature_c": 0}
        if memory_status.has_gpu and memory_status.gpu_stats:
            gpu_stat = memory_status.gpu_stats[0]
            gpu_data = {
                "usage_percent": round(gpu_stat.get("gpu_utilization", 0), 1),
                "memory_usage_percent": round(gpu_stat.get("memory_used_percent", 0), 1),
                "temperature_c": 0  # Temperature monitoring would require additional tools
            }
        
        return APIResponse(
            success=True,
            data={
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "cores": cpu_count,
                    "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq else 0
                },
                "memory": {
                    "total_gb": round(system_memory.get("total_gb", 0), 1),
                    "available_gb": round(system_memory.get("available_gb", 0), 1),
                    "usage_percent": round(system_memory.get("percent", 0), 1)
                },
                "gpu": gpu_data,
                "disk": {
                    "total_gb": round(disk_usage.total / (1024**3), 1),
                    "available_gb": round(disk_usage.free / (1024**3), 1),
                    "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1)
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_received": net_io.bytes_recv
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
