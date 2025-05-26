"""
Model Management API Schemas.

Defines request and response schemas for model management operations.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelListRequest:
    """Request schema for listing models."""
    model_type: Optional[str] = None
    engine: Optional[str] = None
    quantization: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class ModelInfo:
    """Schema for model information."""
    name: str
    model_id: str
    engine_type: str
    quantization: Optional[str] = None
    dtype: Optional[str] = None
    description: Optional[str] = None
    is_loaded: bool = False
    is_default: bool = False


@dataclass
class ModelListResponse:
    """Response schema for model listing."""
    models: List[ModelInfo]
    total_count: int
    loaded_count: int = 0
    filters_applied: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelLoadRequest:
    """Request schema for loading a model."""
    model_name: str
    engine: Optional[str] = None
    quantization: Optional[str] = None
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelLoadResponse:
    """Response schema for model loading."""
    model_name: str
    engine: Optional[str] = None
    quantization: Optional[str] = None
    loading_time_ms: Optional[int] = None
    memory_usage: Optional[Dict[str, Any]] = None
    status: str = "loaded"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelUnloadRequest:
    """Request schema for unloading a model."""
    model_name: str


@dataclass
class ModelUnloadResponse:
    """Response schema for model unloading."""
    model_name: str
    status: str = "unloaded"
    memory_freed: Optional[Dict[str, Any]] = None
    unloading_time_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelStatusRequest:
    """Request schema for model status."""
    model_name: Optional[str] = None


@dataclass
class LoadedModel:
    """Schema for loaded model information."""
    name: str
    engine: str
    memory_usage: Dict[str, Any]
    load_time: datetime
    status: str = "loaded"


@dataclass
class SystemStatus:
    """Schema for system status information."""
    memory: Dict[str, Any]
    gpu: Dict[str, Any]
    disk: Dict[str, Any]
    timestamp: datetime


@dataclass
class ModelStatusResponse:
    """Response schema for model status."""
    loaded_models: List[LoadedModel]
    system_status: SystemStatus
    total_loaded: int
    timestamp: datetime = field(default_factory=datetime.now)
