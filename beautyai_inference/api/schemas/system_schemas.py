"""
System API Schemas.

Defines request/response schemas for system monitoring and management endpoints.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class SystemStatusRequest:
    """Request schema for system status."""
    include_gpu: bool = True
    include_memory: bool = True
    include_models: bool = True


@dataclass
class SystemStatusResponse:
    """Response schema for system status."""
    system_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    gpu_info: Optional[Dict[str, Any]]
    loaded_models: List[Dict[str, Any]]
    timestamp: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class SystemHealthRequest:
    """Request schema for system health check."""
    deep_check: bool = False


@dataclass
class SystemHealthResponse:
    """Response schema for system health check."""
    status: str  # "healthy", "warning", "error"
    checks: Dict[str, Dict[str, Any]]
    overall_score: float
    timestamp: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class SystemCleanupRequest:
    """Request schema for system cleanup operations."""
    cleanup_cache: bool = True
    cleanup_temp: bool = True
    force_gc: bool = True
    confirm: bool = False


@dataclass
class SystemCleanupResponse:
    """Response schema for system cleanup operations."""
    cleanup_results: Dict[str, Any]
    freed_memory: int
    timestamp: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class SystemMetricsRequest:
    """Request schema for system metrics."""
    time_range: Optional[str] = None  # "1h", "24h", "7d"
    metrics_type: Optional[str] = None  # "cpu", "memory", "gpu"


@dataclass
class SystemMetricsResponse:
    """Response schema for system metrics."""
    metrics: Dict[str, List[Dict[str, Any]]]
    time_range: str
    timestamp: str
    success: bool = True
    message: Optional[str] = None
