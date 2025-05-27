"""
API Request/Response Data Models.

Defines the data structures used for API communication, designed to be
compatible with REST/GraphQL endpoints and JSON serialization.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json


@dataclass
class APIRequest:
    """Base class for all API requests."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create request from dictionary (from JSON deserialization)."""
        return cls(**data)


@dataclass
class APIResponse:
    """Base class for all API responses."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ErrorResponse(APIResponse):
    """Standard error response format."""
    success: bool = False
    error_code: str = ""
    error_message: str = ""
    error_details: Optional[Dict[str, Any]] = None


# Model Management API Models
@dataclass
class ModelListRequest(APIRequest):
    """Request to list models in registry."""
    pass


@dataclass
class ModelListResponse(APIResponse):
    """Response containing list of models."""
    success: bool = True
    models: List[Dict[str, Any]] = None
    total_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.models is None:
            self.models = []


@dataclass
class ModelAddRequest(APIRequest):
    """Request to add a model to registry."""
    model_name: str
    model_config: Dict[str, Any]
    set_as_default: bool = False


@dataclass
class ModelAddResponse(APIResponse):
    """Response for model addition."""
    success: bool = True
    model_name: str = ""
    message: str = ""


@dataclass
class ModelLoadRequest(APIRequest):
    """Request to load a model into memory."""
    model_name: str
    force_reload: bool = False


@dataclass
class ModelLoadResponse(APIResponse):
    """Response for model loading."""
    success: bool = True
    model_name: str = ""
    model_id: str = ""
    memory_usage_mb: Optional[float] = None
    load_time_seconds: Optional[float] = None


# Chat API Models
@dataclass
class ChatRequest(APIRequest):
    """Request for chat interaction."""
    model_name: str
    message: str
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    generation_config: Optional[Dict[str, Any]] = None
    stream: bool = False


@dataclass
class ChatResponse(APIResponse):
    """Response for chat interaction."""
    success: bool = True
    response: str = ""
    session_id: str = ""
    model_name: str = ""
    generation_stats: Optional[Dict[str, Any]] = None


# Configuration API Models
@dataclass
class ConfigGetRequest(APIRequest):
    """Request to get configuration."""
    section: Optional[str] = None


@dataclass
class ConfigGetResponse(APIResponse):
    """Response containing configuration."""
    success: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.config is None:
            self.config = {}


@dataclass
class ConfigSetRequest(APIRequest):
    """Request to set configuration value."""
    key: str
    value: Any
    section: Optional[str] = None


@dataclass
class ConfigSetResponse(APIResponse):
    """Response for configuration update."""
    success: bool = True
    key: str = ""
    old_value: Any = None
    new_value: Any = None


# System Status API Models
@dataclass
class SystemStatusRequest(APIRequest):
    """Request for system status."""
    include_memory: bool = True
    include_models: bool = True
    include_cache: bool = True


@dataclass
class SystemStatusResponse(APIResponse):
    """Response containing system status."""
    success: bool = True
    system_info: Dict[str, Any] = None
    memory_info: Optional[Dict[str, Any]] = None
    loaded_models: Optional[List[Dict[str, Any]]] = None
    cache_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.system_info is None:
            self.system_info = {}


# Test API Models
@dataclass
class TestRequest(APIRequest):
    """Request to test a model."""
    model_name: str
    prompt: str
    generation_config: Optional[Dict[str, Any]] = None
    validation_criteria: Optional[Dict[str, Any]] = None


@dataclass
class TestResponse(APIResponse):
    """Response for model test."""
    success: bool = True
    model_name: str = ""
    prompt: str = ""
    response: str = ""
    generation_stats: Dict[str, Any] = None
    validation_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.generation_stats is None:
            self.generation_stats = {}


# Benchmark API Models
@dataclass
class BenchmarkRequest(APIRequest):
    """Request to benchmark a model."""
    model_name: str
    benchmark_type: str  # "latency", "throughput", "comprehensive"
    config: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResponse(APIResponse):
    """Response for model benchmark."""
    success: bool = True
    model_name: str = ""
    benchmark_type: str = ""
    results: Dict[str, Any] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.results is None:
            self.results = {}
        if self.summary is None:
            self.summary = {}


# Session Management API Models
@dataclass
class SessionSaveRequest(APIRequest):
    """Request to save a chat session."""
    session_id: str
    session_data: Dict[str, Any]
    output_file: Optional[str] = None


@dataclass
class SessionSaveResponse(APIResponse):
    """Response for session save."""
    success: bool = True
    session_id: str = ""
    file_path: str = ""
    file_size_bytes: int = 0


@dataclass
class SessionLoadRequest(APIRequest):
    """Request to load a chat session."""
    input_file: str


@dataclass
class SessionLoadResponse(APIResponse):
    """Response for session load."""
    success: bool = True
    session_data: Dict[str, Any] = None
    session_id: str = ""
    message_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.session_data is None:
            self.session_data = {}


# Type aliases for convenience
ModelRequest = Union[ModelListRequest, ModelAddRequest, ModelLoadRequest]
ModelResponse = Union[ModelListResponse, ModelAddResponse, ModelLoadResponse]
ConfigRequest = Union[ConfigGetRequest, ConfigSetRequest]
ConfigResponse = Union[ConfigGetResponse, ConfigSetResponse]
SystemRequest = SystemStatusRequest
SystemResponse = SystemStatusResponse


# Backup/Restore API Models
@dataclass
class BackupRequest(APIRequest):
    """Request to create a configuration backup."""
    backup_name: str
    description: Optional[str] = None
    include_models: bool = True
    include_cache: bool = False


@dataclass
class BackupResponse(APIResponse):
    """Response for configuration backup."""
    success: bool = True
    backup_name: str = ""
    timestamp: str = ""
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    message: str = ""


@dataclass
class RestoreRequest(APIRequest):
    """Request to restore configuration from backup."""
    backup_name: str
    confirm: bool = False
    restore_models: bool = True


@dataclass
class RestoreResponse(APIResponse):
    """Response for configuration restore."""
    success: bool = True
    backup_name: str = ""
    restored_timestamp: str = ""
    message: str = ""
