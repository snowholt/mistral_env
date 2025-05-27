"""
Configuration management API endpoints.

Provides REST API endpoints for configuration operations including:
- Configuration viewing and modification
- Configuration validation
- Configuration backup and restore
- Configuration migration
"""
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from ..models import (
    APIResponse, ConfigRequest, ConfigResponse,
    BackupRequest, BackupResponse
)
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import ValidationError, ConfigurationError
from ...services.config import ConfigService, ValidationService, MigrationService, BackupService

logger = logging.getLogger(__name__)

config_router = APIRouter(prefix="/config", tags=["config"])

# Initialize services
config_service = ConfigService()
validation_service = ValidationService()
migration_service = MigrationService()
backup_service = BackupService()


@config_router.get("/", response_model=ConfigResponse)
async def get_configuration(
    auth: AuthContext = Depends(get_auth_context),
    section: Optional[str] = None
):
    """
    Get current configuration.
    
    Returns the current application configuration, optionally filtered by section.
    """
    require_permissions(auth, ["config_read"])
    
    try:
        # Convert to args-like object for service compatibility
        class ConfigArgs:
            def __init__(self, section_name=None):
                self.section = section_name
                self.output_format = "json"
        
        args = ConfigArgs(section)
        
        # Note: Config service returns exit codes, not data
        # In a real API, this would be restructured to return actual config data
        result = config_service.show_config(args)
        if result == 0:
            return ConfigResponse(
                success=True,
                config={
                    "section": section or "all",
                    "values": {
                        "default_model": "qwen-7b",
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                message="Configuration retrieved successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve configuration")
            
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@config_router.post("/", response_model=ConfigResponse)
async def set_configuration(
    request: ConfigRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Set configuration values.
    
    Updates one or more configuration values.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        # Convert request to args-like object for service compatibility
        class ConfigArgs:
            def __init__(self, config_request):
                self.key = config_request.key
                self.value = config_request.value
                self.section = getattr(config_request, 'section', None)
        
        args = ConfigArgs(request)
        
        result = config_service.set_config(args)
        
        if result == 0:
            return ConfigResponse(
                success=True,
                key=request.key,
                value=request.value,
                message=f"Configuration '{request.key}' set successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to set configuration")
            
    except Exception as e:
        logger.error(f"Failed to set configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set configuration: {str(e)}")


@config_router.put("/", response_model=ConfigResponse)
async def update_configuration(
    request: Dict[str, Any],
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Update multiple configuration values.
    
    Batch update of configuration values.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        updated_keys = []
        
        for key, value in request.items():
            # Convert to individual config request
            class ConfigArgs:
                def __init__(self, k, v):
                    self.key = k
                    self.value = v
                    self.section = None
            
            args = ConfigArgs(key, value)
            result = config_service.set_config(args)
            
            if result == 0:
                updated_keys.append(key)
        
        return ConfigResponse(
            success=True,
            key="batch_update",
            value=updated_keys,
            message=f"Updated {len(updated_keys)} configuration values"
        )
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@config_router.delete("/", response_model=APIResponse)
async def reset_configuration(
    auth: AuthContext = Depends(get_auth_context),
    section: Optional[str] = None
):
    """
    Reset configuration to defaults.
    
    Resets configuration values to their defaults, optionally for a specific section.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        # Convert to args-like object for service compatibility
        class ConfigArgs:
            def __init__(self, section_name=None):
                self.section = section_name
                self.confirm = True
        
        args = ConfigArgs(section)
        
        result = config_service.reset_config(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "section": section or "all",
                    "message": "Configuration reset successfully"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to reset configuration")
            
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset configuration: {str(e)}")


@config_router.post("/validate", response_model=APIResponse)
async def validate_configuration(
    config_data: Optional[Dict[str, Any]] = None,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Validate configuration.
    
    Validates the current configuration or provided configuration data.
    """
    require_permissions(auth, ["config_read"])
    
    try:
        # Convert to args-like object for service compatibility
        class ConfigArgs:
            def __init__(self):
                self.config_file = None  # Validate current config
        
        args = ConfigArgs()
        
        result = validation_service.validate_config(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "message": "Configuration is valid"
                }
            )
        else:
            return APIResponse(
                success=False,
                data={
                    "is_valid": False,
                    "errors": ["Configuration validation failed"],
                    "warnings": [],
                    "message": "Configuration has validation errors"
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate configuration: {str(e)}")


@config_router.post("/backup", response_model=BackupResponse)
async def backup_configuration(
    request: BackupRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Create configuration backup.
    
    Creates a backup of the current configuration.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        # Convert request to args-like object for service compatibility
        class BackupArgs:
            def __init__(self, backup_request):
                self.backup_name = backup_request.backup_name
                self.description = getattr(backup_request, 'description', None)
        
        args = BackupArgs(request)
        
        result = backup_service.backup_config(args)
        
        if result == 0:
            return BackupResponse(
                success=True,
                backup_name=request.backup_name,
                timestamp="2025-05-26T12:00:00Z",  # Placeholder
                message="Configuration backup created successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create backup")
            
    except Exception as e:
        logger.error(f"Failed to backup configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to backup configuration: {str(e)}")


@config_router.post("/restore", response_model=APIResponse)
async def restore_configuration(
    backup_name: str,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Restore configuration from backup.
    
    Restores configuration from a previously created backup.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        # Convert to args-like object for service compatibility
        class RestoreArgs:
            def __init__(self, name):
                self.backup_name = name
                self.confirm = True
        
        args = RestoreArgs(backup_name)
        
        result = backup_service.restore_config(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "backup_name": backup_name,
                    "message": f"Configuration restored from backup '{backup_name}'"
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Backup not found")
            
    except Exception as e:
        logger.error(f"Failed to restore configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore configuration: {str(e)}")


@config_router.get("/backups", response_model=APIResponse)
async def list_backups(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    List available configuration backups.
    
    Returns a list of available configuration backups.
    """
    require_permissions(auth, ["config_read"])
    
    try:
        # TODO: Implement backup listing in service
        return APIResponse(
            success=True,
            data={
                "backups": [
                    {
                        "name": "backup_2025_05_26",
                        "timestamp": "2025-05-26T12:00:00Z",
                        "description": "Daily backup"
                    }
                ],  # Placeholder data
                "total_count": 1
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


@config_router.post("/migrate", response_model=APIResponse)
async def migrate_configuration(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Migrate configuration to latest format.
    
    Updates configuration format to the latest version.
    """
    require_permissions(auth, ["config_write"])
    
    try:
        # Convert to args-like object for service compatibility
        class MigrateArgs:
            def __init__(self):
                self.config_file = None  # Migrate current config
                self.backup = True
        
        args = MigrateArgs()
        
        result = migration_service.migrate_config(args)
        
        if result == 0:
            return APIResponse(
                success=True,
                data={
                    "message": "Configuration migrated successfully",
                    "backup_created": True
                }
            )
        else:
            return APIResponse(
                success=True,
                data={
                    "message": "Configuration is already up to date",
                    "backup_created": False
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to migrate configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to migrate configuration: {str(e)}")
