"""
System API Adapter.

Provides API-compatible interface for system monitoring and management,
bridging system services with REST/GraphQL endpoints.
"""
from typing import Dict, Any, Optional
import logging
import time

from .base_adapter import APIServiceAdapter
from ..models import APIRequest, APIResponse
from ..errors import SystemError
from ...services.system.status_service import StatusService
from ...services.system.memory_service import MemoryService
from ...services.system.cache_service import CacheService

logger = logging.getLogger(__name__)


class SystemAPIAdapter(APIServiceAdapter):
    """
    API adapter for system monitoring and management.
    
    Provides API-compatible interface for:
    - System status monitoring (GPU, memory, disk)
    - Health checks and diagnostics
    - Resource usage statistics
    """
    
    def __init__(self, status_service: StatusService, memory_service: MemoryService, cache_service: CacheService):
        """Initialize system API adapter with required services."""
        self.status_service = status_service
        self.memory_service = memory_service
        self.cache_service = cache_service
        super().__init__(self.status_service)
    
    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "get_system_status": "Get comprehensive system status",
            "get_memory_status": "Get memory usage statistics",
            "get_gpu_status": "Get GPU status and usage",
            "get_disk_status": "Get disk usage information",
            "health_check": "Perform system health check",
            "clear_cache": "Clear system caches"
        }
    
    async def get_system_status(self, include_detailed: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Args:
            include_detailed: Whether to include detailed metrics
            
        Returns:
            Dictionary with system status information
        """
        try:
            # Configure service
            self.status_service.configure({})
            
            # Get system status
            status_data = await self.status_service.get_system_status(
                detailed=include_detailed
            )
            
            return {
                "status": "healthy" if status_data.get("healthy", False) else "unhealthy",
                "system_info": status_data.get("system_info", {}),
                "memory": status_data.get("memory", {}),
                "gpu": status_data.get("gpu", {}),
                "disk": status_data.get("disk", {}),
                "processes": status_data.get("processes", []) if include_detailed else [],
                "uptime": status_data.get("uptime"),
                "load_average": status_data.get("load_average", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise SystemError(f"System status retrieval failed: {e}")
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory status
        """
        try:
            # Get memory information
            memory_data = await self.memory_service.get_memory_status()
            
            return {
                "system_memory": {
                    "total": memory_data.get("system", {}).get("total"),
                    "available": memory_data.get("system", {}).get("available"),
                    "used": memory_data.get("system", {}).get("used"),
                    "percentage": memory_data.get("system", {}).get("percentage")
                },
                "gpu_memory": memory_data.get("gpu", {}),
                "process_memory": memory_data.get("process", {}),
                "cached_models": memory_data.get("cached_models", []),
                "recommendations": memory_data.get("recommendations", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            raise SystemError(f"Memory status retrieval failed: {e}")
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get GPU status and usage information.
        
        Returns:
            Dictionary with GPU status
        """
        try:
            # Get GPU information
            gpu_data = await self.status_service.get_gpu_status()
            
            return {
                "gpu_available": gpu_data.get("available", False),
                "gpu_count": gpu_data.get("count", 0),
                "gpus": gpu_data.get("devices", []),
                "cuda_version": gpu_data.get("cuda_version"),
                "driver_version": gpu_data.get("driver_version"),
                "total_memory": gpu_data.get("total_memory"),
                "free_memory": gpu_data.get("free_memory"),
                "used_memory": gpu_data.get("used_memory"),
                "utilization": gpu_data.get("utilization", []),
                "temperature": gpu_data.get("temperature", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
            raise SystemError(f"GPU status retrieval failed: {e}")
    
    async def get_disk_status(self) -> Dict[str, Any]:
        """
        Get disk usage information.
        
        Returns:
            Dictionary with disk status
        """
        try:
            # Get disk information
            disk_data = await self.status_service.get_disk_status()
            
            return {
                "root_disk": disk_data.get("root", {}),
                "cache_disk": disk_data.get("cache", {}),
                "model_cache_size": disk_data.get("model_cache_size"),
                "disk_partitions": disk_data.get("partitions", []),
                "disk_io": disk_data.get("io_stats", {}),
                "recommendations": disk_data.get("recommendations", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get disk status: {e}")
            raise SystemError(f"Disk status retrieval failed: {e}")
    
    async def health_check(self, check_models: bool = True,
                          check_dependencies: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Args:
            check_models: Whether to check loaded models
            check_dependencies: Whether to check dependencies
            
        Returns:
            Dictionary with health check results
        """
        try:
            # Perform health check
            health_data = await self.status_service.health_check(
                check_models=check_models,
                check_dependencies=check_dependencies
            )
            
            return {
                "healthy": health_data.get("healthy", False),
                "checks": health_data.get("checks", {}),
                "warnings": health_data.get("warnings", []),
                "errors": health_data.get("errors", []),
                "model_status": health_data.get("model_status", {}) if check_models else {},
                "dependencies": health_data.get("dependencies", {}) if check_dependencies else {},
                "recommendations": health_data.get("recommendations", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise SystemError(f"Health check failed: {e}")
    
    async def clear_cache(self, cache_type: str = "all") -> Dict[str, Any]:
        """
        Clear system caches.
        
        Args:
            cache_type: Type of cache to clear (all, models, huggingface, torch)
            
        Returns:
            Dictionary with cache clearing results
        """
        try:
            # Clear cache
            result = await self.cache_service.clear_cache(cache_type=cache_type)
            
            return {
                "cache_type": cache_type,
                "cleared_successfully": result.get("success", False),
                "space_freed": result.get("space_freed", 0),
                "cleared_items": result.get("cleared_items", []),
                "errors": result.get("errors", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise SystemError(f"Cache clearing failed: {e}")
    
    async def get_resource_recommendations(self) -> Dict[str, Any]:
        """
        Get resource optimization recommendations.
        
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            # Get recommendations
            recommendations = await self.status_service.get_optimization_recommendations()
            
            return {
                "memory_recommendations": recommendations.get("memory", []),
                "gpu_recommendations": recommendations.get("gpu", []),
                "disk_recommendations": recommendations.get("disk", []),
                "performance_recommendations": recommendations.get("performance", []),
                "cost_recommendations": recommendations.get("cost", []),
                "priority_actions": recommendations.get("priority", []),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            raise SystemError(f"Recommendations retrieval failed: {e}")
