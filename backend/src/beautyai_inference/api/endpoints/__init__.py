"""
API Endpoints Package.

This package contains all REST API endpoint implementations organized by functionality:
- models: Model management endpoints (CRUD operations, lifecycle)
- inference: Inference operation endpoints (chat, test, benchmark)
- config: Configuration management endpoints
- system: System monitoring and status endpoints
- health: Health check and service status endpoints

Each module provides FastAPI router definitions that can be included in the main API application.
"""

from .health import health_router
from .models import models_router
from .inference import inference_router
from .config import config_router
from .system import system_router
import os

# Always attempt to import streaming voice router; feature flag now enforced at runtime
# inside the endpoint handler so importing is safe and keeps OpenAPI documentation stable.
try:  # noqa: WPS501
    from .streaming_voice import streaming_voice_router  # type: ignore
except Exception:  # pragma: no cover - optional feature import failure path
    streaming_voice_router = None  # type: ignore

__all__ = [
    'health_router',
    'models_router', 
    'inference_router',
    'config_router',
    'system_router',
    'streaming_voice_router'
]

__version__ = "1.0.0"
