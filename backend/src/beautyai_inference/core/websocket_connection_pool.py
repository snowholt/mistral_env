"""
WebSocket Connection Pool Implementation for BeautyAI.

Provides specialized connection pooling for WebSocket connections with:
- WebSocket-specific connection management and lifecycle
- Connection state tracking and health monitoring
- Message routing and session management
- Echo suppression and duplex streaming support
- Integration with existing streaming infrastructure
- Graceful connection cleanup and error handling

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable, Set
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .connection_pool import (
    ConnectionPool, ConnectionPoolError, ConnectionMetrics, 
    ConnectionState, PoolExhaustedException, CircuitBreakerConfig
)

logger = logging.getLogger(__name__)


class WebSocketConnectionState(Enum):
    """Extended WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketConnectionData:
    """Data structure for WebSocket connections in the pool."""
    websocket: WebSocket
    connection_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_info: Dict[str, Any] = field(default_factory=dict)
    streaming_config: Dict[str, Any] = field(default_factory=dict)
    echo_suppression_state: Dict[str, Any] = field(default_factory=dict)
    voice_session_data: Dict[str, Any] = field(default_factory=dict)
    last_activity: float = field(default_factory=time.time)
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    metrics: ConnectionMetrics = field(init=False)
    
    def __post_init__(self):
        if not hasattr(self, 'metrics'):
            self.metrics = ConnectionMetrics(connection_id=self.connection_id)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def add_to_queue(self, message: Dict[str, Any]):
        """Add message to queue for processing."""
        self.message_queue.append({
            **message,
            'queued_at': time.time()
        })

    def clear_queue(self):
        """Clear message queue."""
        self.message_queue.clear()

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.message_queue)


class WebSocketConnectionPool(ConnectionPool):
    """
    Specialized connection pool for WebSocket connections.
    
    Manages WebSocket lifecycle, routing, and streaming state with
    enhanced monitoring and health checks for real-time communication.
    """
    
    def __init__(
        self,
        pool_name: str = "websocket_pool",
        max_pool_size: int = 100,
        min_pool_size: int = 5,
        max_idle_time_seconds: int = 600,  # 10 minutes for WebSocket
        health_check_interval_seconds: int = 30,  # More frequent for real-time
        acquisition_timeout_seconds: int = 10,  # Shorter timeout for real-time
        enable_metrics: bool = True,
        max_message_queue_size: int = 100,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        super().__init__(
            pool_name=pool_name,
            max_pool_size=max_pool_size,
            min_pool_size=min_pool_size,
            max_idle_time_seconds=max_idle_time_seconds,
            health_check_interval_seconds=health_check_interval_seconds,
            acquisition_timeout_seconds=acquisition_timeout_seconds,
            enable_metrics=enable_metrics,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_config=circuit_breaker_config
        )
        
        self.max_message_queue_size = max_message_queue_size
        
        # WebSocket-specific tracking
        self._connection_data: Dict[str, WebSocketConnectionData] = {}
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._session_connections: Dict[str, str] = {}  # session_id -> connection_id
        
        # Message routing
        self._message_handlers: Dict[str, Callable] = {}
        self._broadcast_handlers: List[Callable] = []
        
        logger.info(f"Initialized WebSocket connection pool with max_size={max_pool_size}")

    async def _create_connection(self, connection_id: str, **kwargs) -> WebSocketConnectionData:
        """Create a new WebSocket connection entry (does not create actual WebSocket)."""
        # Note: This creates the pool entry, actual WebSocket is provided when registering
        connection_data = WebSocketConnectionData(
            websocket=None,  # Will be set during registration
            connection_id=connection_id,
            **kwargs
        )
        
        self._connection_data[connection_id] = connection_data
        
        logger.debug(f"Created WebSocket connection entry {connection_id}")
        return connection_data

    async def _destroy_connection(self, connection_id: str, connection: WebSocketConnectionData) -> None:
        """Destroy a WebSocket connection."""
        try:
            # Close WebSocket if connected
            if connection.websocket and connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close(code=1000, reason="Connection pool cleanup")
            
            # Clean up tracking data
            if connection.user_id and connection.user_id in self._user_connections:
                self._user_connections[connection.user_id].discard(connection_id)
                if not self._user_connections[connection.user_id]:
                    del self._user_connections[connection.user_id]
            
            if connection.session_id and connection.session_id in self._session_connections:
                del self._session_connections[connection.session_id]
            
            # Remove from connection data
            self._connection_data.pop(connection_id, None)
            
            logger.debug(f"Destroyed WebSocket connection {connection_id}")
            
        except Exception as e:
            logger.warning(f"Error destroying WebSocket connection {connection_id}: {e}")

    async def _health_check_connection(self, connection_id: str, connection: WebSocketConnectionData) -> bool:
        """Check if WebSocket connection is healthy."""
        try:
            if not connection.websocket:
                return False
            
            # Check WebSocket state
            if connection.websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            # Check activity timeout (more strict for WebSocket)
            idle_time = time.time() - connection.last_activity
            if idle_time > self.max_idle_time_seconds:
                logger.debug(f"Connection {connection_id} idle for {idle_time}s, marking unhealthy")
                return False
            
            # Check message queue size
            if connection.get_queue_size() > self.max_message_queue_size:
                logger.warning(f"Connection {connection_id} has oversized queue ({connection.get_queue_size()})")
                return False
            
            # For newer connections (< 60 seconds), skip ping check to avoid premature disconnects
            connection_age = time.time() - connection.metrics.created_at
            if connection_age < 60:
                logger.debug(f"Connection {connection_id} is new ({connection_age:.1f}s), skipping ping check")
                return True
            
            # Optional: Send ping frame to verify connection (only for older connections)
            try:
                await connection.websocket.ping()
                return True
            except Exception as ping_error:
                logger.debug(f"Ping failed for {connection_id}: {ping_error} - but connection may still be valid")
                # Don't immediately mark as unhealthy on ping failure - could be temporary
                # Only mark unhealthy if connection is also idle for a significant time
                if idle_time > 300:  # 5 minutes of idle time + ping failure = unhealthy
                    return False
                return True  # Give benefit of doubt for active connections
                
        except Exception as e:
            logger.warning(f"Health check failed for WebSocket connection {connection_id}: {e}")
            return False

    async def register_websocket(
        self, 
        websocket: WebSocket, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None,
        streaming_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new WebSocket connection with the pool.
        
        Args:
            websocket: The WebSocket connection instance
            user_id: Optional user identifier
            session_id: Optional session identifier
            client_info: Optional client information
            streaming_config: Optional streaming configuration
            
        Returns:
            connection_id: Unique identifier for the connection
        """
        connection_id, connection_data = await self.acquire_connection(
            user_id=user_id,
            session_id=session_id,
            client_info=client_info or {},
            streaming_config=streaming_config or {}
        )
        
        # Set the actual WebSocket
        connection_data.websocket = websocket
        connection_data.update_activity()
        
        # Update tracking
        if user_id:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(connection_id)
            
        if session_id:
            self._session_connections[session_id] = connection_id
        
        # Update metrics
        connection_data.metrics.state = ConnectionState.ACTIVE
        connection_data.metrics.last_used_at = time.time()
        
        logger.info(f"Registered WebSocket connection {connection_id} for user {user_id}, session {session_id}")
        return connection_id

    async def unregister_websocket(self, connection_id: str, error: Optional[Exception] = None):
        """Unregister a WebSocket connection from the pool."""
        await self.release_connection(connection_id, error=error)
        logger.info(f"Unregistered WebSocket connection {connection_id}")

    def get_connection(self, connection_id: str) -> Optional[WebSocketConnectionData]:
        """Get connection data by ID."""
        return self._connection_data.get(connection_id)

    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user."""
        return list(self._user_connections.get(user_id, set()))

    def get_session_connection(self, session_id: str) -> Optional[str]:
        """Get connection ID for a session."""
        return self._session_connections.get(session_id)

    async def send_to_connection(
        self, 
        connection_id: str, 
        message: Union[str, bytes, Dict[str, Any]],
        message_type: str = "text"
    ) -> bool:
        """
        Send message to a specific connection with circuit breaker protection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
            message_type: Type of message (text, binary, json)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        # Use circuit breaker if enabled
        if self._circuit_breaker:
            try:
                return await self._circuit_breaker.call(
                    self._send_to_connection_internal, connection_id, message, message_type
                )
            except Exception as e:
                logger.warning(f"Circuit breaker blocked send to connection {connection_id}: {e}")
                return False
        else:
            return await self._send_to_connection_internal(connection_id, message, message_type)

    async def _send_to_connection_internal(
        self, 
        connection_id: str, 
        message: Union[str, bytes, Dict[str, Any]],
        message_type: str = "text"
    ) -> bool:
        """
        Internal message sending logic without circuit breaker.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
            message_type: Type of message (text, binary, json)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        connection_data = self._connection_data.get(connection_id)
        if not connection_data or not connection_data.websocket:
            logger.warning(f"Connection {connection_id} not found or not ready")
            return False
        
        try:
            websocket = connection_data.websocket
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"Connection {connection_id} not in connected state")
                return False
            
            start_time = time.time()
            
            # Send based on message type
            if message_type == "json" or isinstance(message, dict):
                if isinstance(message, dict):
                    message = json.dumps(message)
                await websocket.send_text(message)
                bytes_sent = len(message.encode('utf-8'))
            elif message_type == "binary" or isinstance(message, bytes):
                await websocket.send_bytes(message)
                bytes_sent = len(message)
            else:
                await websocket.send_text(str(message))
                bytes_sent = len(str(message).encode('utf-8'))
            
            # Update metrics
            response_time_ms = (time.time() - start_time) * 1000
            connection_data.metrics.update_usage(
                bytes_sent=bytes_sent,
                response_time_ms=response_time_ms
            )
            connection_data.update_activity()
            
            logger.debug(f"Sent message to connection {connection_id} ({bytes_sent} bytes)")
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket {connection_id} disconnected during send")
            await self.unregister_websocket(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            connection_data.metrics.increment_errors()
            return False

    async def send_to_user(
        self, 
        user_id: str, 
        message: Union[str, bytes, Dict[str, Any]],
        message_type: str = "text"
    ) -> int:
        """
        Send message to all connections for a user.
        
        Returns:
            int: Number of successful sends
        """
        connection_ids = self.get_user_connections(user_id)
        successful_sends = 0
        
        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, message, message_type):
                successful_sends += 1
        
        logger.debug(f"Sent message to {successful_sends}/{len(connection_ids)} connections for user {user_id}")
        return successful_sends

    async def broadcast_message(
        self, 
        message: Union[str, bytes, Dict[str, Any]],
        message_type: str = "text",
        filter_func: Optional[Callable[[WebSocketConnectionData], bool]] = None
    ) -> int:
        """
        Broadcast message to all connections (optionally filtered).
        
        Args:
            message: Message to broadcast
            message_type: Type of message
            filter_func: Optional filter function for connections
            
        Returns:
            int: Number of successful sends
        """
        successful_sends = 0
        
        for connection_id, connection_data in self._connection_data.items():
            # Apply filter if provided
            if filter_func and not filter_func(connection_data):
                continue
            
            # Skip inactive connections
            if connection_data.metrics.state != ConnectionState.ACTIVE:
                continue
                
            if await self.send_to_connection(connection_id, message, message_type):
                successful_sends += 1
        
        logger.debug(f"Broadcast message to {successful_sends} connections")
        return successful_sends

    async def handle_connection_message(
        self, 
        connection_id: str, 
        message: Union[str, bytes],
        message_type: str = "text"
    ):
        """
        Handle incoming message from a connection.
        
        Args:
            connection_id: Source connection ID
            message: Received message
            message_type: Type of message
        """
        connection_data = self._connection_data.get(connection_id)
        if not connection_data:
            logger.warning(f"Received message from unknown connection {connection_id}")
            return
        
        try:
            # Update metrics
            bytes_received = len(message) if isinstance(message, (str, bytes)) else 0
            connection_data.metrics.update_usage(bytes_received=bytes_received)
            connection_data.update_activity()
            
            # Parse message if JSON
            parsed_message = None
            if message_type == "text" and isinstance(message, str):
                try:
                    parsed_message = json.loads(message)
                except json.JSONDecodeError:
                    parsed_message = {"text": message}
            else:
                parsed_message = {"data": message, "type": message_type}
            
            # Route to appropriate handler
            message_action = parsed_message.get("action") if isinstance(parsed_message, dict) else None
            
            if message_action and message_action in self._message_handlers:
                handler = self._message_handlers[message_action]
                await handler(connection_id, connection_data, parsed_message)
            else:
                # Default handling - add to queue
                connection_data.add_to_queue({
                    "message": parsed_message,
                    "type": message_type,
                    "received_at": time.time()
                })
            
            logger.debug(f"Handled message from connection {connection_id} ({bytes_received} bytes)")
            
        except Exception as e:
            logger.error(f"Error handling message from connection {connection_id}: {e}")
            connection_data.metrics.increment_errors()

    def register_message_handler(self, action: str, handler: Callable):
        """Register a message handler for specific action."""
        self._message_handlers[action] = handler
        logger.debug(f"Registered message handler for action '{action}'")

    def register_broadcast_handler(self, handler: Callable):
        """Register a broadcast handler."""
        self._broadcast_handlers.append(handler)
        logger.debug("Registered broadcast handler")

    async def process_queued_messages(self, connection_id: str, max_messages: int = 10):
        """Process queued messages for a connection."""
        connection_data = self._connection_data.get(connection_id)
        if not connection_data:
            return
        
        processed = 0
        while connection_data.message_queue and processed < max_messages:
            message_data = connection_data.message_queue.pop(0)
            
            try:
                # Execute broadcast handlers
                for handler in self._broadcast_handlers:
                    await handler(connection_id, connection_data, message_data)
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processing queued message for {connection_id}: {e}")
                connection_data.metrics.increment_errors()
                break
        
        if processed > 0:
            logger.debug(f"Processed {processed} queued messages for connection {connection_id}")

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        base_metrics = self.get_metrics()
        
        # Add WebSocket-specific metrics
        websocket_states = {}
        queue_sizes = []
        user_distribution = {}
        session_count = len(self._session_connections)
        
        for connection_data in self._connection_data.values():
            # WebSocket state distribution
            if connection_data.websocket:
                state = connection_data.websocket.client_state.name
                websocket_states[state] = websocket_states.get(state, 0) + 1
            
            # Queue size distribution
            queue_size = connection_data.get_queue_size()
            queue_sizes.append(queue_size)
            
            # User distribution
            if connection_data.user_id:
                user_distribution[connection_data.user_id] = user_distribution.get(connection_data.user_id, 0) + 1
        
        base_metrics.update({
            'websocket_specific': {
                'websocket_states': websocket_states,
                'average_queue_size': sum(queue_sizes) / len(queue_sizes) if queue_sizes else 0,
                'max_queue_size': max(queue_sizes) if queue_sizes else 0,
                'total_users': len(self._user_connections),
                'total_sessions': session_count,
                'user_connection_distribution': user_distribution,
                'message_handlers_registered': len(self._message_handlers),
                'broadcast_handlers_registered': len(self._broadcast_handlers)
            }
        })
        
        return base_metrics


# Singleton instance for global use
_websocket_pool_instance: Optional[WebSocketConnectionPool] = None


def get_websocket_pool() -> WebSocketConnectionPool:
    """Get the global WebSocket connection pool instance with configuration."""
    global _websocket_pool_instance
    if _websocket_pool_instance is None:
        # Load configuration from config system
        try:
            from ..config.config_loader import get_config
            config = get_config()
            
            # Connection pool configuration
            pool_config = config.get('connection_pool', {})
            circuit_breaker_config_dict = config.get('circuit_breaker', {})
            
            # Create circuit breaker config if enabled
            circuit_breaker_config = None
            if circuit_breaker_config_dict.get('enabled', True):
                circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=circuit_breaker_config_dict.get('failure_threshold', 5),
                    recovery_timeout_seconds=circuit_breaker_config_dict.get('recovery_timeout_seconds', 60.0),
                    success_threshold=circuit_breaker_config_dict.get('success_threshold', 3),
                    failure_rate_threshold=circuit_breaker_config_dict.get('failure_rate_threshold', 0.5),
                    minimum_requests=circuit_breaker_config_dict.get('minimum_requests', 10),
                    timeout_seconds=circuit_breaker_config_dict.get('timeout_seconds', 30.0),
                    max_retry_attempts=circuit_breaker_config_dict.get('max_retry_attempts', 3),
                    backoff_multiplier=circuit_breaker_config_dict.get('backoff_multiplier', 2.0),
                    max_backoff_seconds=circuit_breaker_config_dict.get('max_backoff_seconds', 300.0),
                    health_check_enabled=circuit_breaker_config_dict.get('health_check_enabled', True),
                    health_check_interval_seconds=circuit_breaker_config_dict.get('health_check_interval_seconds', 30.0)
                )
            
            _websocket_pool_instance = WebSocketConnectionPool(
                pool_name=pool_config.get('pool_name', 'websocket_pool'),
                max_pool_size=pool_config.get('max_pool_size', 100),
                min_pool_size=pool_config.get('min_pool_size', 5),
                max_idle_time_seconds=pool_config.get('max_idle_time_seconds', 600),
                health_check_interval_seconds=pool_config.get('health_check_interval_seconds', 30),
                acquisition_timeout_seconds=pool_config.get('acquisition_timeout_seconds', 10),
                enable_metrics=pool_config.get('enable_metrics', True),
                max_message_queue_size=pool_config.get('max_message_queue_size', 100),
                enable_circuit_breaker=pool_config.get('enable_circuit_breaker', True),
                circuit_breaker_config=circuit_breaker_config
            )
            
        except ImportError:
            # Fallback to default configuration if config system not available
            logger.warning("Config system not available, using default WebSocket pool configuration")
            _websocket_pool_instance = WebSocketConnectionPool()
        except Exception as e:
            logger.error(f"Error loading WebSocket pool configuration: {e}, using defaults")
            _websocket_pool_instance = WebSocketConnectionPool()
    
    return _websocket_pool_instance


async def initialize_websocket_pool(**kwargs) -> WebSocketConnectionPool:
    """Initialize the global WebSocket connection pool."""
    global _websocket_pool_instance
    if _websocket_pool_instance is not None:
        await _websocket_pool_instance.stop()
    
    _websocket_pool_instance = WebSocketConnectionPool(**kwargs)
    await _websocket_pool_instance.start()
    return _websocket_pool_instance


async def shutdown_websocket_pool():
    """Shutdown the global WebSocket connection pool."""
    global _websocket_pool_instance
    if _websocket_pool_instance is not None:
        await _websocket_pool_instance.stop()
        _websocket_pool_instance = None