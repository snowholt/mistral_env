"""
Health check and service status endpoints.

Provides basic health monitoring and service availability endpoints.
These endpoints are typically used by load balancers and monitoring systems.
"""
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from ..models import APIResponse
from ..auth import AuthContext, get_auth_context
from ..adapters.base_adapter import APIServiceAdapter
from ...services.system import StatusService

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["health"])

# Initialize status service
status_service = StatusService()


@health_router.get("/", response_model=APIResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns basic service availability status without authentication requirements.
    Used by load balancers and monitoring systems.
    """
    try:
        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "service": "BeautyAI Inference API",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@health_router.get("/detailed", response_model=APIResponse)
async def detailed_health_check(auth: AuthContext = Depends(get_auth_context)):
    """
    Detailed health check with system information.
    
    Requires authentication and provides comprehensive system status including
    memory usage, GPU availability, and loaded models.
    """
    try:
        # Get comprehensive system status
        system_status = status_service.get_system_status()
        
        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "service": "BeautyAI Inference API",
                "version": "1.0.0", 
                "system": system_status,
                "authenticated_user": auth.username if auth.authenticated else "anonymous",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@health_router.get("/ready", response_model=APIResponse)
async def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies that the service is ready to handle requests by checking
    critical dependencies and configurations.
    """
    try:
        # Check if we can access basic services
        from ...config.config_manager import AppConfig
        from pathlib import Path
        
        issues = []
        
        # Check configuration
        try:
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.json"
            if not default_config_path.exists():
                issues.append("Default configuration file not found")
        except Exception as e:
            issues.append(f"Configuration check failed: {e}")
        
        # Check model registry
        try:
            status_service.configure({})  # Use default config
        except Exception as e:
            issues.append(f"Service configuration failed: {e}")
        
        if issues:
            return APIResponse(
                success=False,
                data={
                    "status": "not_ready",
                    "issues": issues,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        
        return APIResponse(
            success=True,
            data={
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Readiness check failed: {str(e)}")


@health_router.get("/live", response_model=APIResponse)
async def liveness_check():
    """
    Liveness check endpoint.
    
    Simple check to verify the service process is running and responsive.
    Used by orchestration systems for restart decisions.
    """
    return APIResponse(
        success=True,
        data={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )
