"""
API Service Adapters Package.

Provides adapter classes that bridge BeautyAI services with API endpoints.
These adapters handle request/response transformation and validation.
"""

from .base_adapter import APIServiceAdapter
from .model_adapter import ModelAPIAdapter
from .inference_adapter import InferenceAPIAdapter
from .config_adapter import ConfigAPIAdapter
from .system_adapter import SystemAPIAdapter

__all__ = [
    'APIServiceAdapter',
    'ModelAPIAdapter', 
    'InferenceAPIAdapter',
    'ConfigAPIAdapter',
    'SystemAPIAdapter'
]
