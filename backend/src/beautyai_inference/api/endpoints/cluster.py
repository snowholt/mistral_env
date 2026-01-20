"""
Cluster Management API Endpoints

Provides HTTP endpoints for:
- Slave registration with master
- Cluster status and health
- Server listing and management
- Routing decisions

Author: BeautyAI Framework
Date: 2026-01-19
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster", tags=["cluster"])


# ========== Request/Response Models ==========

class ServerRegistrationRequest(BaseModel):
    """Request model for slave server registration."""
    server_id: str = Field(..., description="Unique server identifier")
    host: str = Field(..., description="Server hostname/IP")
    port: int = Field(..., description="Server port")
    public_url: str = Field(..., description="Public URL for client redirects")
    mode: str = Field(default="slave", description="Server mode")
    capabilities: Dict[str, Any] = Field(
        default_factory=lambda: {"llm_slots": 1, "stt": True, "tts": True},
        description="Server capabilities"
    )
    gpu_memory_total_gb: float = Field(default=16.0, description="Total GPU memory in GB")


class ServerRegistrationResponse(BaseModel):
    """Response model for server registration."""
    success: bool
    message: str
    server_id: str
    cluster_info: Optional[Dict[str, Any]] = None


class ClusterStatusResponse(BaseModel):
    """Response model for cluster status."""
    mode: str
    server_id: str
    is_master: bool
    connected_to_redis: bool
    connected_to_master: bool
    registered_servers: int
    total_llm_slots: int
    total_active_requests: int
    total_sessions: int
    uptime_seconds: float


class RoutingDecisionRequest(BaseModel):
    """Request model for routing decision."""
    session_id: str = Field(..., description="Session identifier")
    prefer_local: bool = Field(default=True, description="Prefer local processing")


class RoutingDecisionResponse(BaseModel):
    """Response model for routing decision."""
    route_local: bool
    instance_id: Optional[str] = None
    server_id: Optional[str] = None
    redirect_url: Optional[str] = None
    reason: str
    wait_time_estimate_ms: float = 0.0


# ========== Endpoints ==========

@router.post("/register", response_model=ServerRegistrationResponse)
async def register_server(request: ServerRegistrationRequest):
    """
    Register a slave server with the master.
    
    This endpoint should be called by slave servers on startup to register
    themselves with the master for load balancing.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        from ..core.redis_client import ServerInfo
        
        coordinator = await get_cluster_coordinator()
        
        if not coordinator.is_master:
            raise HTTPException(
                status_code=400,
                detail="This server is not running as master"
            )
        
        # Create ServerInfo from request
        server_info = ServerInfo(
            server_id=request.server_id,
            host=request.host,
            port=request.port,
            public_url=request.public_url,
            mode=request.mode,
            capabilities=request.capabilities,
            gpu_memory_total_gb=request.gpu_memory_total_gb,
        )
        
        # Register with coordinator
        success = await coordinator.register_slave(server_info)
        
        if success:
            # Get cluster overview for response
            overview = await coordinator.get_cluster_overview()
            
            return ServerRegistrationResponse(
                success=True,
                message=f"Server {request.server_id} registered successfully",
                server_id=request.server_id,
                cluster_info={
                    "total_servers": overview.get("server_count", 1),
                    "total_llm_slots": overview.get("total_llm_slots", 1),
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to register server"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unregister/{server_id}")
async def unregister_server(server_id: str):
    """
    Unregister a server from the cluster.
    
    Called when a slave is shutting down gracefully.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        
        coordinator = await get_cluster_coordinator()
        
        if not coordinator.is_master:
            raise HTTPException(
                status_code=400,
                detail="This server is not running as master"
            )
        
        success = await coordinator.unregister_slave(server_id)
        
        return {
            "success": success,
            "message": f"Server {server_id} unregistered" if success else "Server not found"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=ClusterStatusResponse)
async def get_cluster_status():
    """
    Get current cluster status.
    
    Returns information about this server's role in the cluster,
    connectivity status, and basic metrics.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        
        coordinator = await get_cluster_coordinator()
        status = coordinator.get_status()
        
        return ClusterStatusResponse(
            mode=status.mode,
            server_id=status.server_id,
            is_master=status.is_master,
            connected_to_redis=status.connected_to_redis,
            connected_to_master=status.connected_to_master,
            registered_servers=status.registered_servers,
            total_llm_slots=status.total_llm_slots,
            total_active_requests=status.total_active_requests,
            total_sessions=status.total_sessions,
            uptime_seconds=status.uptime_seconds,
        )
        
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview")
async def get_cluster_overview():
    """
    Get complete cluster overview (master only).
    
    Returns detailed information about all servers in the cluster,
    including their current load and health status.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        
        coordinator = await get_cluster_coordinator()
        
        if not coordinator.is_master:
            # For slaves, return local status only
            return {
                "cluster_mode": "slave",
                "local_status": coordinator.get_status().__dict__,
            }
        
        return await coordinator.get_cluster_overview()
        
    except Exception as e:
        logger.error(f"Error getting cluster overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/servers")
async def list_servers():
    """
    List all registered servers in the cluster.
    
    Returns a list of all servers with their current status and capabilities.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        from ..core.redis_client import get_redis_client
        
        coordinator = await get_cluster_coordinator()
        
        if coordinator.is_standalone:
            # Return just this server
            return {
                "servers": [{
                    "server_id": coordinator.config.server_id,
                    "mode": "standalone",
                    "status": "active",
                }]
            }
        
        # Get from Redis
        redis = await get_redis_client()
        servers = await redis.get_all_servers()
        
        return {
            "servers": [s.to_dict() for s in servers]
        }
        
    except Exception as e:
        logger.error(f"Error listing servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route", response_model=RoutingDecisionResponse)
async def get_routing_decision(request: RoutingDecisionRequest):
    """
    Get routing decision for a session.
    
    Determines whether to process locally or redirect to another server.
    Used by WebRTC/WebSocket endpoints to decide routing.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        
        coordinator = await get_cluster_coordinator()
        decision = await coordinator.route_request(request.session_id)
        
        return RoutingDecisionResponse(
            route_local=decision.route_local,
            instance_id=decision.instance_id,
            server_id=decision.server_id,
            redirect_url=decision.redirect_url,
            reason=decision.reason,
            wait_time_estimate_ms=decision.wait_time_estimate_ms,
        )
        
    except Exception as e:
        logger.error(f"Error getting routing decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def cluster_health():
    """
    Cluster health check endpoint.
    
    Used by load balancers to determine if this server should receive traffic.
    Returns 200 if healthy, 503 if unhealthy or overloaded.
    """
    try:
        from ..core.cluster_coordinator import get_cluster_coordinator
        
        coordinator = await get_cluster_coordinator()
        
        # Check if we should accept new connections
        should_redirect = coordinator.should_redirect()
        status = coordinator.get_status()
        
        health_data = {
            "healthy": True,
            "accepting_connections": not should_redirect,
            "mode": status.mode,
            "server_id": status.server_id,
            "active_requests": status.total_active_requests,
            "llm_slots": status.total_llm_slots,
        }
        
        # Return 503 if overloaded (for load balancer to route elsewhere)
        if should_redirect:
            return Response(
                content=str(health_data),
                status_code=503,
                media_type="application/json"
            )
        
        return health_data
        
    except Exception as e:
        logger.error(f"Cluster health check failed: {e}")
        return Response(
            content=str({"healthy": False, "error": str(e)}),
            status_code=503,
            media_type="application/json"
        )


@router.get("/llm-pool")
async def get_llm_pool_status():
    """
    Get LLM pool status.
    
    Returns detailed information about local LLM instances and their load.
    """
    try:
        from ..core.persistent_model_manager import get_persistent_model_manager
        
        manager = get_persistent_model_manager()
        pool_status = manager.get_llm_pool_status()
        
        return pool_status
        
    except Exception as e:
        logger.error(f"Error getting LLM pool status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
