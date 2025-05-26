"""
API Request/Response Schemas Package.

Defines the structured schemas for API requests and responses,
providing type safety and validation for API endpoints.
"""

from .model_schemas import *
from .inference_schemas import *
from .config_schemas import *
from .system_schemas import *

__all__ = [
    # Model schemas
    'ModelListRequest', 'ModelListResponse',
    'ModelLoadRequest', 'ModelLoadResponse',
    'ModelUnloadRequest', 'ModelUnloadResponse',
    'ModelStatusRequest', 'ModelStatusResponse',
    
    # Inference schemas
    'ChatRequest', 'ChatResponse',
    'TestRequest', 'TestResponse',
    'BenchmarkRequest', 'BenchmarkResponse',
    
    # Config schemas
    'ConfigGetRequest', 'ConfigGetResponse',
    'ConfigUpdateRequest', 'ConfigUpdateResponse',
    'ConfigValidateRequest', 'ConfigValidateResponse',
    'ConfigResetRequest', 'ConfigResetResponse',
    
    # System schemas
    'SystemStatusRequest', 'SystemStatusResponse',
    'SystemHealthRequest', 'SystemHealthResponse',
    'SystemCleanupRequest', 'SystemCleanupResponse',
    'SystemMetricsRequest', 'SystemMetricsResponse'
]
