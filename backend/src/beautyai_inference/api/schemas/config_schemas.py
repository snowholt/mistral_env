"""
Configuration API Schemas.

Defines request/response schemas for configuration management endpoints.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ConfigGetRequest:
    """Request schema for getting configuration."""
    config_type: Optional[str] = None
    section: Optional[str] = None


@dataclass
class ConfigGetResponse:
    """Response schema for configuration retrieval."""
    config: Dict[str, Any]
    config_type: str
    timestamp: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class ConfigUpdateRequest:
    """Request schema for updating configuration."""
    config_type: str
    config_data: Dict[str, Any]
    merge: bool = True


@dataclass
class ConfigUpdateResponse:
    """Response schema for configuration updates."""
    updated_config: Dict[str, Any]
    config_type: str
    timestamp: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class ConfigValidateRequest:
    """Request schema for configuration validation."""
    config_type: str
    config_data: Dict[str, Any]


@dataclass
class ConfigValidateResponse:
    """Response schema for configuration validation."""
    is_valid: bool
    validation_errors: List[str]
    config_type: str
    success: bool = True
    message: Optional[str] = None


@dataclass
class ConfigResetRequest:
    """Request schema for resetting configuration to defaults."""
    config_type: str
    confirm: bool = False


@dataclass
class ConfigResetResponse:
    """Response schema for configuration reset."""
    reset_config: Dict[str, Any]
    config_type: str
    timestamp: str
    success: bool = True
    message: Optional[str] = None
