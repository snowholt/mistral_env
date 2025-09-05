"""
Enhanced Connection Pool Management System for BeautyAI.

Provides scalable connection pooling for WebSocket connections with:
- Connection lifecycle management (create, acquire, release, cleanup)
- Connection health monitoring and automatic recovery
- Configurable pool limits and timeouts
- Metrics collection and performance monitoring
- Graceful shutdown and resource cleanup
- Integration with existing streaming infrastructure

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, Set
from collections import defaultdict
import threading
import weakref

from fastapi import WebSocket
from starlette.websockets import WebSocketState
import psutil

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states in the pool."""
    IDLE = "idle"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class ConnectionMetrics:
    """Metrics for individual connections."""
    connection_id: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    total_usage_count: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_response_time_ms: float = 0.0
    error_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_usage(self, bytes_sent: int = 0, bytes_received: int = 0, response_time_ms: float = 0):
        """Update connection usage metrics."""
        self.last_used_at = time.time()
        self.total_usage_count += 1
        self.total_bytes_sent += bytes_sent
        self.total_bytes_received += bytes_received
        
        if response_time_ms > 0:
            # Update rolling average
            if self.average_response_time_ms == 0:
                self.average_response_time_ms = response_time_ms
            else:
                # Exponential moving average with alpha=0.2
                self.average_response_time_ms = (0.8 * self.average_response_time_ms) + (0.2 * response_time_ms)

    def increment_errors(self):
        """Increment error count."""
        self.error_count += 1

    def get_age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    def get_idle_time_seconds(self) -> float:
        """Get time since last usage in seconds."""
        return time.time() - self.last_used_at


@dataclass
class PoolMetrics:
    """Overall pool metrics."""
    pool_name: str
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    unhealthy_connections: int = 0
    pending_connections: int = 0
    max_pool_size: int = 0
    total_created: int = 0
    total_destroyed: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    average_acquisition_time_ms: float = 0.0
    peak_usage: int = 0
    pool_utilization: float = 0.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def update_snapshot(self, connections: Dict[str, Any]):
        """Update metrics from current pool state."""
        self.last_updated = time.time()
        self.total_connections = len(connections)
        
        state_counts = defaultdict(int)
        for conn_data in connections.values():
            if hasattr(conn_data, 'metrics'):
                state_counts[conn_data.metrics.state] += 1
        
        self.active_connections = state_counts[ConnectionState.ACTIVE]
        self.idle_connections = state_counts[ConnectionState.IDLE]
        self.unhealthy_connections = state_counts[ConnectionState.UNHEALTHY]
        self.pending_connections = state_counts[ConnectionState.PENDING]
        
        if self.max_pool_size > 0:
            self.pool_utilization = self.total_connections / self.max_pool_size
        
        if self.total_connections > self.peak_usage:
            self.peak_usage = self.total_connections


class ConnectionPoolError(Exception):
    """Base exception for connection pool operations."""
    pass


class PoolExhaustedException(ConnectionPoolError):
    """Raised when the connection pool is exhausted."""
    pass


class ConnectionHealthError(ConnectionPoolError):
    """Raised when connection health check fails."""
    pass


class ConnectionPool(ABC):
    """
    Abstract base class for connection pooling.
    
    Provides common functionality for managing connection lifecycle,
    health monitoring, and metrics collection.
    """
    
    def __init__(
        self,
        pool_name: str,
        max_pool_size: int = 10,
        min_pool_size: int = 1,
        max_idle_time_seconds: int = 300,
        health_check_interval_seconds: int = 60,
        acquisition_timeout_seconds: int = 30,
        enable_metrics: bool = True
    ):
        self.pool_name = pool_name
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.max_idle_time_seconds = max_idle_time_seconds
        self.health_check_interval_seconds = health_check_interval_seconds
        self.acquisition_timeout_seconds = acquisition_timeout_seconds
        self.enable_metrics = enable_metrics
        
        # Connection storage
        self._connections: Dict[str, Any] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}
        self._pool_lock = asyncio.Lock()
        
        # Pool metrics
        self.metrics = PoolMetrics(pool_name=pool_name, max_pool_size=max_pool_size)
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self._connection_created_callbacks: List[Callable] = []
        self._connection_destroyed_callbacks: List[Callable] = []
        self._health_check_callbacks: List[Callable] = []
        
        logger.info(f"Initialized connection pool '{pool_name}' with max_size={max_pool_size}")

    @abstractmethod
    async def _create_connection(self, connection_id: str, **kwargs) -> Any:
        """Create a new connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _destroy_connection(self, connection_id: str, connection: Any) -> None:
        """Destroy a connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _health_check_connection(self, connection_id: str, connection: Any) -> bool:
        """Check if connection is healthy. Must be implemented by subclasses."""
        pass

    async def start(self):
        """Start the connection pool and background tasks."""
        logger.info(f"Starting connection pool '{self.pool_name}'")
        
        # Initialize minimum connections
        for i in range(self.min_pool_size):
            try:
                await self._create_and_add_connection()
            except Exception as e:
                logger.warning(f"Failed to create initial connection {i}: {e}")
        
        # Start health check task
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Connection pool '{self.pool_name}' started with {len(self._connections)} connections")

    async def stop(self):
        """Stop the connection pool and cleanup resources."""
        logger.info(f"Stopping connection pool '{self.pool_name}'")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Destroy all connections
        async with self._pool_lock:
            connection_ids = list(self._connections.keys())
            for connection_id in connection_ids:
                await self._destroy_and_remove_connection(connection_id)
        
        logger.info(f"Connection pool '{self.pool_name}' stopped")

    async def acquire_connection(self, **kwargs) -> tuple[str, Any]:
        """
        Acquire a connection from the pool.
        
        Returns:
            Tuple of (connection_id, connection_object)
        """
        start_time = time.time()
        
        try:
            async with asyncio.timeout(self.acquisition_timeout_seconds):
                async with self._pool_lock:
                    # Try to find an idle connection
                    for connection_id, connection_data in self._connections.items():
                        metrics = self._connection_metrics.get(connection_id)
                        if metrics and metrics.state == ConnectionState.IDLE:
                            # Mark as active
                            metrics.state = ConnectionState.ACTIVE
                            metrics.last_used_at = time.time()
                            
                            self.metrics.total_acquisitions += 1
                            
                            acquisition_time_ms = (time.time() - start_time) * 1000
                            if self.metrics.average_acquisition_time_ms == 0:
                                self.metrics.average_acquisition_time_ms = acquisition_time_ms
                            else:
                                self.metrics.average_acquisition_time_ms = (
                                    0.8 * self.metrics.average_acquisition_time_ms + 
                                    0.2 * acquisition_time_ms
                                )
                            
                            logger.debug(f"Acquired existing connection {connection_id} from pool '{self.pool_name}'")
                            return connection_id, connection_data['connection']
                    
                    # No idle connections, try to create a new one
                    if len(self._connections) < self.max_pool_size:
                        connection_id, connection = await self._create_and_add_connection(**kwargs)
                        
                        metrics = self._connection_metrics.get(connection_id)
                        if metrics:
                            metrics.state = ConnectionState.ACTIVE
                        
                        self.metrics.total_acquisitions += 1
                        
                        logger.debug(f"Created new connection {connection_id} for pool '{self.pool_name}'")
                        return connection_id, connection
                    
                    # Pool is exhausted
                    raise PoolExhaustedException(f"Connection pool '{self.pool_name}' is exhausted")
        
        except asyncio.TimeoutError:
            raise ConnectionPoolError(f"Timeout acquiring connection from pool '{self.pool_name}'")

    async def release_connection(self, connection_id: str, error: Optional[Exception] = None):
        """Release a connection back to the pool."""
        async with self._pool_lock:
            metrics = self._connection_metrics.get(connection_id)
            if not metrics:
                logger.warning(f"Attempted to release unknown connection {connection_id}")
                return
            
            if error:
                metrics.increment_errors()
                metrics.state = ConnectionState.UNHEALTHY
                logger.warning(f"Released connection {connection_id} with error: {error}")
                # Consider destroying unhealthy connections
                await self._destroy_and_remove_connection(connection_id)
            else:
                metrics.state = ConnectionState.IDLE
                metrics.last_used_at = time.time()
                self.metrics.total_releases += 1
                logger.debug(f"Released connection {connection_id} back to pool '{self.pool_name}'")

    async def _create_and_add_connection(self, **kwargs) -> tuple[str, Any]:
        """Create a new connection and add it to the pool."""
        connection_id = str(uuid.uuid4())
        
        try:
            connection = await self._create_connection(connection_id, **kwargs)
            
            # Store connection data
            self._connections[connection_id] = {
                'connection': connection,
                'created_at': time.time(),
                'kwargs': kwargs
            }
            
            # Initialize metrics
            metrics = ConnectionMetrics(
                connection_id=connection_id,
                state=ConnectionState.IDLE
            )
            self._connection_metrics[connection_id] = metrics
            
            self.metrics.total_created += 1
            
            # Execute callbacks
            for callback in self._connection_created_callbacks:
                try:
                    await callback(connection_id, connection)
                except Exception as e:
                    logger.warning(f"Connection created callback failed: {e}")
            
            logger.debug(f"Created connection {connection_id} in pool '{self.pool_name}'")
            return connection_id, connection
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise ConnectionPoolError(f"Failed to create connection: {e}")

    async def _destroy_and_remove_connection(self, connection_id: str):
        """Destroy a connection and remove it from the pool."""
        connection_data = self._connections.get(connection_id)
        if not connection_data:
            return
        
        try:
            await self._destroy_connection(connection_id, connection_data['connection'])
        except Exception as e:
            logger.warning(f"Error destroying connection {connection_id}: {e}")
        
        # Remove from pool
        self._connections.pop(connection_id, None)
        self._connection_metrics.pop(connection_id, None)
        
        self.metrics.total_destroyed += 1
        
        # Execute callbacks
        for callback in self._connection_destroyed_callbacks:
            try:
                await callback(connection_id)
            except Exception as e:
                logger.warning(f"Connection destroyed callback failed: {e}")
        
        logger.debug(f"Destroyed connection {connection_id} from pool '{self.pool_name}'")

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error in pool '{self.pool_name}': {e}")
                await asyncio.sleep(5)  # Brief delay on error

    async def _perform_health_checks(self):
        """Perform health checks on all connections."""
        async with self._pool_lock:
            unhealthy_connections = []
            idle_timeout_connections = []
            
            for connection_id, connection_data in list(self._connections.items()):
                metrics = self._connection_metrics.get(connection_id)
                if not metrics:
                    continue
                
                try:
                    # Check if connection is too old or idle for too long
                    if metrics.get_idle_time_seconds() > self.max_idle_time_seconds and metrics.state == ConnectionState.IDLE:
                        idle_timeout_connections.append(connection_id)
                        continue
                    
                    # Skip health check for active connections
                    if metrics.state == ConnectionState.ACTIVE:
                        continue
                    
                    # Perform health check
                    is_healthy = await self._health_check_connection(
                        connection_id, 
                        connection_data['connection']
                    )
                    
                    if not is_healthy:
                        metrics.state = ConnectionState.UNHEALTHY
                        unhealthy_connections.append(connection_id)
                    elif metrics.state == ConnectionState.UNHEALTHY:
                        # Connection recovered
                        metrics.state = ConnectionState.IDLE
                        logger.info(f"Connection {connection_id} recovered in pool '{self.pool_name}'")
                
                except Exception as e:
                    logger.warning(f"Health check failed for connection {connection_id}: {e}")
                    metrics.state = ConnectionState.UNHEALTHY
                    unhealthy_connections.append(connection_id)
            
            # Cleanup unhealthy and idle timeout connections
            for connection_id in unhealthy_connections + idle_timeout_connections:
                await self._destroy_and_remove_connection(connection_id)
            
            # Ensure minimum pool size
            current_healthy_count = len(self._connections) - len(unhealthy_connections)
            if current_healthy_count < self.min_pool_size:
                needed = self.min_pool_size - current_healthy_count
                for _ in range(needed):
                    try:
                        await self._create_and_add_connection()
                    except Exception as e:
                        logger.warning(f"Failed to create replacement connection: {e}")
                        break
            
            # Update metrics
            self.metrics.update_snapshot(self._connections)
            
            # Execute health check callbacks
            for callback in self._health_check_callbacks:
                try:
                    await callback(self.metrics)
                except Exception as e:
                    logger.warning(f"Health check callback failed: {e}")

    def add_connection_created_callback(self, callback: Callable):
        """Add callback for when connections are created."""
        self._connection_created_callbacks.append(callback)

    def add_connection_destroyed_callback(self, callback: Callable):
        """Add callback for when connections are destroyed."""
        self._connection_destroyed_callbacks.append(callback)

    def add_health_check_callback(self, callback: Callable):
        """Add callback for health check events."""
        self._health_check_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pool metrics."""
        self.metrics.update_snapshot(self._connections)
        
        connection_details = []
        for connection_id, metrics in self._connection_metrics.items():
            connection_details.append({
                'connection_id': connection_id,
                'state': metrics.state.value,
                'age_seconds': metrics.get_age_seconds(),
                'idle_time_seconds': metrics.get_idle_time_seconds(),
                'usage_count': metrics.total_usage_count,
                'error_count': metrics.error_count,
                'avg_response_time_ms': metrics.average_response_time_ms,
                'bytes_sent': metrics.total_bytes_sent,
                'bytes_received': metrics.total_bytes_received,
                'metadata': metrics.metadata
            })
        
        return {
            'pool': {
                'name': self.pool_name,
                'total_connections': self.metrics.total_connections,
                'active_connections': self.metrics.active_connections,
                'idle_connections': self.metrics.idle_connections,
                'unhealthy_connections': self.metrics.unhealthy_connections,
                'pending_connections': self.metrics.pending_connections,
                'max_pool_size': self.max_pool_size,
                'min_pool_size': self.min_pool_size,
                'pool_utilization': self.metrics.pool_utilization,
                'peak_usage': self.metrics.peak_usage,
                'total_created': self.metrics.total_created,
                'total_destroyed': self.metrics.total_destroyed,
                'total_acquisitions': self.metrics.total_acquisitions,
                'total_releases': self.metrics.total_releases,
                'avg_acquisition_time_ms': self.metrics.average_acquisition_time_ms,
                'last_updated': self.metrics.last_updated
            },
            'connections': connection_details,
            'system': {
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.Process().cpu_percent(),
                'timestamp': time.time()
            }
        }