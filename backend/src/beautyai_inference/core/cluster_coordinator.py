"""
Cluster Coordinator for BeautyAI Distributed Architecture

Orchestrates master-slave communication:
- Master: Registers slaves, tracks load, routes overflow requests
- Slave: Registers with master, sends heartbeats, accepts routed sessions

Cross-Network Support:
- Uses public URLs for client redirects (WebRTC direct connection)
- Redis pub/sub for coordination (can be hosted on master or separate)
- HTTP API for registration and health checks

Author: BeautyAI Framework
Date: 2026-01-19
"""

import asyncio
import logging
import os
import time
import httpx
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .redis_client import (
    ClusterConfig,
    ClusterMode,
    RedisClient,
    ServerInfo,
    get_redis_client,
    initialize_redis,
)
from .inference_router import InferenceRouter, get_inference_router, RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class ClusterStatus:
    """Current cluster status."""
    mode: str
    server_id: str
    is_master: bool
    connected_to_redis: bool
    connected_to_master: bool  # For slaves
    registered_servers: int
    total_llm_slots: int
    total_active_requests: int
    total_sessions: int
    uptime_seconds: float


class ClusterCoordinator:
    """
    Coordinates distributed BeautyAI servers.
    
    Master Mode:
    - Accepts slave registrations
    - Maintains global load map
    - Provides routing decisions for overflow
    - Monitors slave health via heartbeats
    
    Slave Mode:
    - Registers with master on startup
    - Sends periodic heartbeats with load info
    - Accepts routed sessions from master
    - Falls back to standalone if master unreachable
    """
    
    _instance: Optional["ClusterCoordinator"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig.from_env()
        self._redis: Optional[RedisClient] = None
        self._router: Optional[InferenceRouter] = None
        self._server_info: Optional[ServerInfo] = None
        self._started = False
        self._start_time: float = 0.0
        
        # Master-specific
        self._registered_servers: Dict[str, ServerInfo] = {}
        self._server_health_tasks: Dict[str, asyncio.Task] = {}
        
        # Slave-specific  
        self._master_client: Optional[httpx.AsyncClient] = None
        self._registration_task: Optional[asyncio.Task] = None
        self._connected_to_master = False
        
        # Callbacks
        self._on_server_join: Optional[Callable] = None
        self._on_server_leave: Optional[Callable] = None
        self._on_routing_request: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    async def get_instance(cls, config: Optional[ClusterConfig] = None) -> "ClusterCoordinator":
        """Get singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    @property
    def is_master(self) -> bool:
        """Check if running as master."""
        return self.config.mode == ClusterMode.MASTER
    
    @property
    def is_slave(self) -> bool:
        """Check if running as slave."""
        return self.config.mode == ClusterMode.SLAVE
    
    @property
    def is_standalone(self) -> bool:
        """Check if running standalone (no clustering)."""
        return self.config.mode == ClusterMode.STANDALONE
    
    # ========== Lifecycle ==========
    
    async def start(self) -> bool:
        """
        Start cluster coordinator.
        
        Returns:
            bool: True if started successfully
        """
        if self._started:
            self.logger.info("Cluster coordinator already started")
            return True
        
        self._start_time = time.time()
        
        # Initialize components
        self._router = get_inference_router()
        self._router.set_cluster_coordinator(self)
        
        if self.is_standalone:
            self.logger.info("Running in standalone mode (no clustering)")
            self._started = True
            return True
        
        # Connect to Redis
        self._redis = await get_redis_client(self.config)
        redis_connected = await self._redis.connect()
        
        if not redis_connected:
            self.logger.warning("Redis connection failed, falling back to standalone mode")
            self.config.mode = ClusterMode.STANDALONE
            self._started = True
            return True
        
        # Initialize server info
        self._server_info = self._create_server_info()
        
        if self.is_master:
            await self._start_master()
        elif self.is_slave:
            await self._start_slave()
        
        self._started = True
        self.logger.info(f"✅ Cluster coordinator started in {self.config.mode.value} mode")
        return True
    
    async def stop(self) -> None:
        """Stop cluster coordinator."""
        if not self._started:
            return
        
        self.logger.info("Stopping cluster coordinator...")
        
        if self.is_slave and self._redis:
            # Unregister from cluster
            await self._redis.unregister_server(self.config.server_id)
        
        # Cancel tasks
        if self._registration_task:
            self._registration_task.cancel()
        
        for task in self._server_health_tasks.values():
            task.cancel()
        
        # Close connections
        if self._master_client:
            await self._master_client.aclose()
        
        self._started = False
        self.logger.info("Cluster coordinator stopped")
    
    def _create_server_info(self) -> ServerInfo:
        """Create server info for this instance."""
        # Get GPU memory info
        try:
            from ..utils.memory_utils import get_gpu_memory_stats
            gpu_stats = get_gpu_memory_stats()
            gpu_total = gpu_stats[0].get("memory_total_mb", 16000) / 1024 if gpu_stats else 16.0
            gpu_used = gpu_stats[0].get("memory_used_mb", 0) / 1024 if gpu_stats else 0.0
        except Exception:
            gpu_total = 16.0
            gpu_used = 0.0
        
        # Determine LLM slots based on mode and GPU
        llm_slots = int(os.getenv("LLM_POOL_SIZE", "2" if self.is_master else "1"))
        
        return ServerInfo(
            server_id=self.config.server_id,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            public_url=self.config.public_url or f"http://localhost:{os.getenv('PORT', '8000')}",
            mode=self.config.mode.value,
            capabilities={
                "llm_slots": llm_slots,
                "stt": True,
                "tts": True,
            },
            gpu_memory_total_gb=gpu_total,
            gpu_memory_used_gb=gpu_used,
        )
    
    # ========== Master Mode ==========
    
    async def _start_master(self) -> None:
        """Initialize master mode."""
        self.logger.info("Starting as MASTER node")
        
        # Register self in Redis
        await self._redis.register_server(self._server_info)
        
        # Subscribe to heartbeat channel
        await self._redis.subscribe("cluster:heartbeats", self._handle_heartbeat)
        
        # Start heartbeat listener
        asyncio.create_task(self._redis.listen_messages())
        
        # Start own heartbeat
        await self._redis.start_heartbeat_loop(
            self._server_info,
            self._update_server_stats,
        )
        
        self.logger.info("Master node initialized")
    
    async def _handle_heartbeat(self, data: Dict[str, Any]) -> None:
        """Handle incoming heartbeat from slave."""
        server_id = data.get("server_id")
        if not server_id or server_id == self.config.server_id:
            return
        
        # Update local cache
        if server_id in self._registered_servers:
            server = self._registered_servers[server_id]
            server.last_heartbeat = data.get("timestamp", time.time())
            server.llm_active = data.get("llm_active", 0)
            server.llm_queued = data.get("llm_queued", 0)
            server.active_sessions = data.get("active_sessions", 0)
        else:
            # New server, fetch full info
            server_info = await self._redis.get_server(server_id)
            if server_info:
                self._registered_servers[server_id] = server_info
                self.logger.info(f"New server joined cluster: {server_id}")
                
                if self._on_server_join:
                    await self._on_server_join(server_info)
    
    async def register_slave(self, server_info: ServerInfo) -> bool:
        """
        Register a slave server (called via API).
        
        Args:
            server_info: Slave server information
            
        Returns:
            bool: True if registered
        """
        if not self.is_master:
            return False
        
        # Store in Redis
        success = await self._redis.register_server(server_info)
        
        if success:
            self._registered_servers[server_info.server_id] = server_info
            self.logger.info(f"✅ Registered slave: {server_info.server_id} ({server_info.public_url})")
            
            if self._on_server_join:
                await self._on_server_join(server_info)
        
        return success
    
    async def unregister_slave(self, server_id: str) -> bool:
        """Unregister a slave server."""
        if not self.is_master:
            return False
        
        success = await self._redis.unregister_server(server_id)
        
        if server_id in self._registered_servers:
            server_info = self._registered_servers.pop(server_id)
            self.logger.info(f"Unregistered slave: {server_id}")
            
            if self._on_server_leave:
                await self._on_server_leave(server_info)
        
        return success
    
    # ========== Slave Mode ==========
    
    async def _start_slave(self) -> None:
        """Initialize slave mode."""
        self.logger.info(f"Starting as SLAVE node, master: {self.config.master_url}")
        
        # Create HTTP client for master communication
        self._master_client = httpx.AsyncClient(
            base_url=self.config.master_url,
            timeout=10.0,
        )
        
        # Register with master
        await self._register_with_master()
        
        # Register self in Redis
        await self._redis.register_server(self._server_info)
        
        # Start heartbeat
        await self._redis.start_heartbeat_loop(
            self._server_info,
            self._update_server_stats,
        )
        
        self.logger.info("Slave node initialized")
    
    async def _register_with_master(self) -> bool:
        """Register this slave with master via HTTP API."""
        if not self._master_client:
            return False
        
        try:
            response = await self._master_client.post(
                "/cluster/register",
                json=self._server_info.to_dict(),
            )
            
            if response.status_code == 200:
                self._connected_to_master = True
                self.logger.info(f"✅ Registered with master: {self.config.master_url}")
                return True
            else:
                self.logger.error(f"Failed to register with master: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering with master: {e}")
            self._connected_to_master = False
            return False
    
    async def _update_server_stats(self, server_info: ServerInfo) -> None:
        """Update server stats before heartbeat."""
        # Get current load from router
        if self._router:
            load_info = self._router.get_cluster_load_info()
            server_info.llm_active = load_info.get("total_active_requests", 0)
            server_info.llm_queued = load_info.get("total_queued_requests", 0)
        
        # Get GPU memory
        try:
            from ..utils.memory_utils import get_gpu_memory_stats
            gpu_stats = get_gpu_memory_stats()
            if gpu_stats:
                server_info.gpu_memory_used_gb = gpu_stats[0].get("memory_used_mb", 0) / 1024
        except Exception:
            pass
        
        # Get session count (if session manager available)
        try:
            from .webrtc_connection_pool import get_webrtc_pool
            pool = get_webrtc_pool()
            server_info.active_sessions = pool.get_connection_count() if pool else 0
        except Exception:
            pass
    
    # ========== Routing ==========
    
    async def get_available_servers(self) -> List[ServerInfo]:
        """
        Get servers with available capacity for routing.
        
        Returns:
            List of servers sorted by load (lowest first)
        """
        if self.is_standalone:
            return []
        
        # Get from Redis (includes self and all slaves)
        servers = await self._redis.get_available_servers()
        
        # Filter out self for routing decisions
        servers = [s for s in servers if s.server_id != self.config.server_id]
        
        return servers
    
    async def route_request(self, session_id: str) -> RoutingDecision:
        """
        Get routing decision for a new voice session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            RoutingDecision with local or redirect info
        """
        if self.is_standalone:
            return RoutingDecision(
                route_local=True,
                reason="Standalone mode",
            )
        
        # Check local capacity first
        if self._router:
            decision = await self._router.get_route(session_id, prefer_local=True)
            
            if decision.route_local and decision.instance_id:
                return decision
        
        # Check remote servers
        available_servers = await self.get_available_servers()
        
        if available_servers:
            best_server = available_servers[0]
            return RoutingDecision(
                route_local=False,
                server_id=best_server.server_id,
                redirect_url=best_server.public_url,
                reason=f"Redirecting to {best_server.server_id} (load: {best_server.llm_active}/{best_server.capabilities.get('llm_slots', 1)})",
            )
        
        # Fall back to local even if busy
        return RoutingDecision(
            route_local=True,
            reason="No remote capacity available, using local",
        )
    
    def should_redirect(self) -> bool:
        """
        Quick check if this server should redirect new sessions.
        
        Returns:
            bool: True if all local LLM slots are busy
        """
        if self.is_standalone or not self._router:
            return False
        
        load_info = self._router.get_cluster_load_info()
        return not load_info.get("has_capacity", True)
    
    # ========== Status & Health ==========
    
    def get_status(self) -> ClusterStatus:
        """Get current cluster status."""
        load_info = self._router.get_cluster_load_info() if self._router else {}
        
        return ClusterStatus(
            mode=self.config.mode.value,
            server_id=self.config.server_id,
            is_master=self.is_master,
            connected_to_redis=self._redis.is_connected if self._redis else False,
            connected_to_master=self._connected_to_master,
            registered_servers=len(self._registered_servers) if self.is_master else 0,
            total_llm_slots=load_info.get("instance_count", 0),
            total_active_requests=load_info.get("total_active_requests", 0),
            total_sessions=self._server_info.active_sessions if self._server_info else 0,
            uptime_seconds=time.time() - self._start_time if self._start_time else 0,
        )
    
    async def get_cluster_overview(self) -> Dict[str, Any]:
        """
        Get complete cluster overview (master only).
        
        Returns:
            Dictionary with all servers and their stats
        """
        if not self.is_master:
            return {"error": "Not master node"}
        
        servers = await self._redis.get_all_servers() if self._redis else []
        
        total_slots = sum(s.capabilities.get("llm_slots", 1) for s in servers)
        total_active = sum(s.llm_active for s in servers)
        total_sessions = sum(s.active_sessions for s in servers)
        
        return {
            "cluster_mode": "master",
            "server_count": len(servers),
            "total_llm_slots": total_slots,
            "total_active_requests": total_active,
            "total_sessions": total_sessions,
            "utilization_percent": (total_active / total_slots * 100) if total_slots > 0 else 0,
            "servers": [
                {
                    "server_id": s.server_id,
                    "public_url": s.public_url,
                    "mode": s.mode,
                    "status": s.status,
                    "llm_slots": s.capabilities.get("llm_slots", 1),
                    "llm_active": s.llm_active,
                    "llm_queued": s.llm_queued,
                    "active_sessions": s.active_sessions,
                    "gpu_memory_gb": f"{s.gpu_memory_used_gb:.1f}/{s.gpu_memory_total_gb:.1f}",
                    "last_heartbeat_ago_s": round(time.time() - s.last_heartbeat, 1),
                }
                for s in servers
            ],
        }
    
    # ========== Callbacks ==========
    
    def on_server_join(self, callback: Callable) -> None:
        """Set callback for when a server joins the cluster."""
        self._on_server_join = callback
    
    def on_server_leave(self, callback: Callable) -> None:
        """Set callback for when a server leaves the cluster."""
        self._on_server_leave = callback


# Global instance getter
_coordinator: Optional[ClusterCoordinator] = None


async def get_cluster_coordinator(config: Optional[ClusterConfig] = None) -> ClusterCoordinator:
    """Get global cluster coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = await ClusterCoordinator.get_instance(config)
    return _coordinator


async def initialize_cluster(config: Optional[ClusterConfig] = None) -> bool:
    """Initialize cluster coordinator."""
    coordinator = await get_cluster_coordinator(config)
    return await coordinator.start()


async def shutdown_cluster() -> None:
    """Shutdown cluster coordinator."""
    global _coordinator
    if _coordinator:
        await _coordinator.stop()
        _coordinator = None
