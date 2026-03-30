"""
Inference Router for BeautyAI Distributed Architecture

Provides intelligent routing of inference requests to:
- Multiple LLM instances (model pool) on same server
- Multiple servers in cluster (cross-network load balancing)

Features:
- Round-robin / least-loaded LLM selection
- Request tracking per instance
- Automatic overflow to slave servers
- Health-aware routing

Author: BeautyAI Framework
Date: 2026-01-19
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """LLM instance selection strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


@dataclass
class LLMInstanceStats:
    """Statistics for a single LLM instance."""
    instance_id: str
    model_name: str
    active_requests: int = 0
    queued_requests: int = 0
    total_requests: int = 0
    total_tokens_generated: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0
    is_healthy: bool = True
    error_count: int = 0
    
    @property
    def load(self) -> int:
        """Current load = active + queued requests."""
        return self.active_requests + self.queued_requests


@dataclass 
class RoutingDecision:
    """Result of routing decision."""
    route_local: bool = True           # True = use local instance, False = redirect
    instance_id: Optional[str] = None  # Local instance ID
    server_id: Optional[str] = None    # Remote server ID (for redirect)
    redirect_url: Optional[str] = None # Public URL for client redirect
    reason: str = ""                   # Explanation for decision
    wait_time_estimate_ms: float = 0.0 # Estimated wait if queued


class InferenceRouter:
    """
    Routes inference requests to available LLM instances.
    
    Supports:
    - Multiple LLM instances on same GPU (model pool)
    - Cross-server routing via cluster coordinator
    - Request tracking and load balancing
    """
    
    _instance: Optional["InferenceRouter"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        max_local_instances: int = 2,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        max_queue_per_instance: int = 5,
    ):
        self.max_local_instances = max_local_instances
        self.strategy = strategy
        self.max_queue_per_instance = max_queue_per_instance
        
        # LLM instance pool: instance_id -> (model_instance, stats)
        self._instances: Dict[str, Tuple[Any, LLMInstanceStats]] = {}
        self._instance_order: List[str] = []  # For round-robin
        self._round_robin_index: int = 0
        
        # Request tracking
        self._active_requests: Dict[str, str] = {}  # request_id -> instance_id
        self._request_start_times: Dict[str, float] = {}
        
        # Async lock for thread-safe operations
        self._async_lock = asyncio.Lock()
        
        # Cluster coordinator reference (set externally)
        self._cluster_coordinator = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"InferenceRouter initialized: max_instances={max_local_instances}, strategy={strategy.value}")
    
    @classmethod
    def get_instance(cls, **kwargs) -> "InferenceRouter":
        """Get singleton instance. kwargs only used on first creation."""
        with cls._lock:
            if cls._instance is None:
                # Filter to valid __init__ params only
                valid_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k in ('max_local_instances', 'strategy', 'max_queue_per_instance')
                }
                cls._instance = cls(**valid_kwargs)
            return cls._instance
    
    def set_cluster_coordinator(self, coordinator) -> None:
        """Set reference to cluster coordinator for cross-server routing."""
        self._cluster_coordinator = coordinator
    
    # ========== Instance Pool Management ==========
    
    def register_instance(self, instance_id: str, model_instance: Any, model_name: str) -> bool:
        """
        Register an LLM instance in the pool.
        
        Args:
            instance_id: Unique identifier (e.g., "llm:qwen3:0", "llm:qwen3:1")
            model_instance: The loaded model instance
            model_name: Model name for logging
            
        Returns:
            bool: True if registered successfully
        """
        if len(self._instances) >= self.max_local_instances:
            self.logger.warning(f"Cannot register instance {instance_id}: pool full ({self.max_local_instances} max)")
            return False
        
        if instance_id in self._instances:
            self.logger.warning(f"Instance {instance_id} already registered, updating")
        
        stats = LLMInstanceStats(
            instance_id=instance_id,
            model_name=model_name,
        )
        
        self._instances[instance_id] = (model_instance, stats)
        
        if instance_id not in self._instance_order:
            self._instance_order.append(instance_id)
        
        self.logger.info(f"✅ Registered LLM instance: {instance_id} ({model_name})")
        return True
    
    def unregister_instance(self, instance_id: str) -> bool:
        """Remove an instance from the pool."""
        if instance_id in self._instances:
            del self._instances[instance_id]
            if instance_id in self._instance_order:
                self._instance_order.remove(instance_id)
            self.logger.info(f"Unregistered LLM instance: {instance_id}")
            return True
        return False
    
    def get_instance_count(self) -> int:
        """Get number of registered instances."""
        return len(self._instances)
    
    def get_all_stats(self) -> List[LLMInstanceStats]:
        """Get stats for all instances."""
        return [stats for _, stats in self._instances.values()]
    
    # ========== Routing Logic ==========
    
    async def get_route(self, request_id: str, prefer_local: bool = True, auto_start: bool = True) -> RoutingDecision:
        """
        Determine where to route an inference request.
        
        Args:
            request_id: Unique request identifier
            prefer_local: Prefer local instances over remote
            auto_start: If True, automatically marks the instance as busy (atomically)
            
        Returns:
            RoutingDecision with routing information
        """
        async with self._async_lock:
            # Try to get local instance first
            local_instance = self._select_local_instance()
            
            if local_instance:
                # CRITICAL: Atomically mark as busy to prevent race conditions
                if auto_start:
                    self._mark_instance_busy_sync(request_id, local_instance)
                return RoutingDecision(
                    route_local=True,
                    instance_id=local_instance,
                    reason="Local instance available",
                )
            
            # Check if we can queue locally
            if self._can_queue_locally():
                least_loaded = self._get_least_loaded_instance()
                if least_loaded:
                    # CRITICAL: Atomically mark as busy (queued)
                    if auto_start:
                        self._mark_instance_busy_sync(request_id, least_loaded)
                    return RoutingDecision(
                        route_local=True,
                        instance_id=least_loaded,
                        reason="Queued on local instance",
                        wait_time_estimate_ms=self._estimate_wait_time(least_loaded),
                    )
            
            # Try remote routing if cluster coordinator available
            if self._cluster_coordinator and not prefer_local:
                remote = await self._get_remote_route()
                if remote:
                    return remote
            
            # Last resort: queue locally on least loaded
            least_loaded = self._get_least_loaded_instance()
            if least_loaded:
                # CRITICAL: Atomically mark as busy
                if auto_start:
                    self._mark_instance_busy_sync(request_id, least_loaded)
                return RoutingDecision(
                    route_local=True,
                    instance_id=least_loaded,
                    reason="All instances busy, queued locally",
                    wait_time_estimate_ms=self._estimate_wait_time(least_loaded),
                )
            
            # No instances available at all
            return RoutingDecision(
                route_local=True,
                instance_id=None,
                reason="No LLM instances available",
            )
    
    def _select_local_instance(self) -> Optional[str]:
        """Select an available local instance based on strategy."""
        available = [
            instance_id for instance_id, (_, stats) in self._instances.items()
            if stats.is_healthy and stats.active_requests == 0
        ]
        
        if not available:
            return None
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            # Find next available in round-robin order
            for _ in range(len(self._instance_order)):
                idx = self._round_robin_index % len(self._instance_order)
                instance_id = self._instance_order[idx]
                self._round_robin_index += 1
                
                if instance_id in available:
                    return instance_id
            return available[0] if available else None
        
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return min(available, key=lambda x: self._instances[x][1].load)
        
        else:  # RANDOM
            import random
            return random.choice(available)
    
    def _mark_instance_busy_sync(self, request_id: str, instance_id: str) -> None:
        """
        Synchronously mark an instance as busy (called within async lock).
        
        This MUST be called while holding _async_lock to prevent race conditions.
        """
        if instance_id not in self._instances:
            return
        
        _, stats = self._instances[instance_id]
        stats.active_requests += 1
        stats.total_requests += 1
        stats.last_used = time.time()
        
        self._active_requests[request_id] = instance_id
        self._request_start_times[request_id] = time.time()
        
        self.logger.info(f"🔒 Atomically assigned request {request_id} to {instance_id} (active: {stats.active_requests})")
    
    def _get_least_loaded_instance(self) -> Optional[str]:
        """Get instance with lowest load (can have active requests)."""
        healthy = [
            instance_id for instance_id, (_, stats) in self._instances.items()
            if stats.is_healthy
        ]
        
        if not healthy:
            return None
        
        return min(healthy, key=lambda x: self._instances[x][1].load)
    
    def _can_queue_locally(self) -> bool:
        """Check if any local instance can accept more queued requests."""
        for _, stats in self._instances.values():
            if stats.is_healthy and stats.queued_requests < self.max_queue_per_instance:
                return True
        return False
    
    def _estimate_wait_time(self, instance_id: str) -> float:
        """Estimate wait time for a queued request (ms)."""
        if instance_id not in self._instances:
            return 0.0
        
        _, stats = self._instances[instance_id]
        
        # Estimate based on average latency and queue depth
        if stats.avg_latency_ms > 0:
            return stats.avg_latency_ms * (stats.active_requests + stats.queued_requests)
        
        # Default estimate: 2 seconds per request in queue
        return 2000.0 * (stats.active_requests + stats.queued_requests)
    
    async def _get_remote_route(self) -> Optional[RoutingDecision]:
        """Get routing decision for remote server."""
        if not self._cluster_coordinator:
            return None
        
        try:
            # Get available servers from cluster
            available_servers = await self._cluster_coordinator.get_available_servers()
            
            if not available_servers:
                return None
            
            # Select least loaded server
            best_server = available_servers[0]  # Already sorted by load
            
            return RoutingDecision(
                route_local=False,
                server_id=best_server.server_id,
                redirect_url=best_server.public_url,
                reason=f"Redirecting to server {best_server.server_id} (load: {best_server.llm_active})",
            )
            
        except Exception as e:
            self.logger.error(f"Error getting remote route: {e}")
            return None
    
    # ========== Request Tracking ==========
    
    async def start_request(self, request_id: str, instance_id: str) -> bool:
        """
        Mark request as started on an instance.
        
        Note: If auto_start=True was used in get_route(), this is a no-op.
        
        Args:
            request_id: Unique request identifier
            instance_id: Instance handling the request
            
        Returns:
            bool: True if tracked successfully
        """
        async with self._async_lock:
            # Check if already started (via auto_start in get_route)
            if request_id in self._active_requests:
                self.logger.debug(f"Request {request_id} already started (auto_start)")
                return True
            
            if instance_id not in self._instances:
                return False
            
            _, stats = self._instances[instance_id]
            stats.active_requests += 1
            stats.total_requests += 1
            stats.last_used = time.time()
            
            self._active_requests[request_id] = instance_id
            self._request_start_times[request_id] = time.time()
            
            self.logger.debug(f"Started request {request_id} on {instance_id} (active: {stats.active_requests})")
            return True
    
    async def end_request(self, request_id: str, tokens_generated: int = 0, error: bool = False) -> bool:
        """
        Mark request as completed.
        
        Args:
            request_id: Request identifier
            tokens_generated: Number of tokens generated
            error: True if request failed
            
        Returns:
            bool: True if tracked successfully
        """
        async with self._async_lock:
            if request_id not in self._active_requests:
                return False
            
            instance_id = self._active_requests[request_id]
            
            if instance_id not in self._instances:
                return False
            
            _, stats = self._instances[instance_id]
            stats.active_requests = max(0, stats.active_requests - 1)
            stats.total_tokens_generated += tokens_generated
            
            if error:
                stats.error_count += 1
            
            # Update average latency
            if request_id in self._request_start_times:
                latency = (time.time() - self._request_start_times[request_id]) * 1000
                if stats.avg_latency_ms == 0:
                    stats.avg_latency_ms = latency
                else:
                    # Exponential moving average
                    stats.avg_latency_ms = 0.9 * stats.avg_latency_ms + 0.1 * latency
                del self._request_start_times[request_id]
            
            del self._active_requests[request_id]
            
            self.logger.debug(f"Ended request {request_id} on {instance_id} (active: {stats.active_requests})")
            return True
    
    # ========== Instance Access ==========
    
    def get_instance(self, instance_id: str) -> Optional[Any]:
        """Get model instance by ID."""
        if instance_id in self._instances:
            return self._instances[instance_id][0]
        return None
    
    def get_any_available_instance(self) -> Optional[Tuple[str, Any]]:
        """
        Get any available instance for immediate use.
        
        Returns:
            Tuple of (instance_id, model_instance) or None
        """
        for instance_id, (model, stats) in self._instances.items():
            if stats.is_healthy and stats.active_requests == 0:
                return (instance_id, model)
        
        # If none free, return least loaded
        least_loaded = self._get_least_loaded_instance()
        if least_loaded:
            return (least_loaded, self._instances[least_loaded][0])
        
        return None
    
    # ========== Health & Metrics ==========
    
    def mark_instance_unhealthy(self, instance_id: str) -> None:
        """Mark an instance as unhealthy."""
        if instance_id in self._instances:
            _, stats = self._instances[instance_id]
            stats.is_healthy = False
            self.logger.warning(f"Instance {instance_id} marked unhealthy")
    
    def mark_instance_healthy(self, instance_id: str) -> None:
        """Mark an instance as healthy."""
        if instance_id in self._instances:
            _, stats = self._instances[instance_id]
            stats.is_healthy = True
            stats.error_count = 0
            self.logger.info(f"Instance {instance_id} marked healthy")
    
    def get_cluster_load_info(self) -> Dict[str, Any]:
        """
        Get load information for cluster reporting.
        
        Returns:
            Dictionary with load metrics
        """
        total_active = sum(stats.active_requests for _, stats in self._instances.values())
        total_queued = sum(stats.queued_requests for _, stats in self._instances.values())
        healthy_count = sum(1 for _, stats in self._instances.values() if stats.is_healthy)
        
        return {
            "instance_count": len(self._instances),
            "healthy_count": healthy_count,
            "total_active_requests": total_active,
            "total_queued_requests": total_queued,
            "total_load": total_active + total_queued,
            "has_capacity": any(
                stats.is_healthy and stats.active_requests == 0
                for _, stats in self._instances.values()
            ),
            "instances": [
                {
                    "instance_id": stats.instance_id,
                    "model_name": stats.model_name,
                    "active": stats.active_requests,
                    "queued": stats.queued_requests,
                    "healthy": stats.is_healthy,
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                }
                for _, stats in self._instances.values()
            ],
        }


# Global instance getter
_router: Optional[InferenceRouter] = None


def get_inference_router(**kwargs) -> InferenceRouter:
    """Get global inference router instance."""
    global _router
    if _router is None:
        # Filter kwargs to valid __init__ params
        valid_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ('max_local_instances', 'strategy', 'max_queue_per_instance')
        }
        _router = InferenceRouter(**valid_kwargs)
    return _router
