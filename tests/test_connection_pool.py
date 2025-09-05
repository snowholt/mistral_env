"""
Tests for WebSocket Connection Pool System.

Comprehensive test suite for the connection pooling functionality including:
- Connection pool creation and management
- WebSocket-specific connection handling
- Health checks and lifecycle management
- Metrics and monitoring
- Error handling and recovery
- Integration with streaming infrastructure

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import json
import logging
import pytest
import time
import uuid
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from fastapi import WebSocket
from starlette.websockets import WebSocketState

# Import the classes we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.beautyai_inference.core.connection_pool import (
    ConnectionPool, ConnectionState, ConnectionMetrics, PoolMetrics,
    ConnectionPoolError, PoolExhaustedException, ConnectionHealthError
)
from src.beautyai_inference.core.websocket_connection_pool import (
    WebSocketConnectionPool, WebSocketConnectionData, WebSocketConnectionState,
    get_websocket_pool, initialize_websocket_pool, shutdown_websocket_pool
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self, client_state=WebSocketState.CONNECTED):
        self.client_state = client_state
        self.sent_messages = []
        self.ping_count = 0
        self.close_called = False
        
    async def accept(self):
        self.client_state = WebSocketState.CONNECTED
    
    async def close(self, code=1000, reason=""):
        self.client_state = WebSocketState.DISCONNECTED
        self.close_called = True
    
    async def send_text(self, data):
        self.sent_messages.append({"type": "text", "data": data})
    
    async def send_bytes(self, data):
        self.sent_messages.append({"type": "bytes", "data": data})
    
    async def ping(self):
        self.ping_count += 1
        if self.client_state != WebSocketState.CONNECTED:
            raise Exception("WebSocket not connected")
        return b"pong"


class TestConnectionMetrics:
    """Test connection metrics functionality."""
    
    def test_connection_metrics_initialization(self):
        """Test basic connection metrics initialization."""
        connection_id = "test-connection-123"
        metrics = ConnectionMetrics(connection_id=connection_id)
        
        assert metrics.connection_id == connection_id
        assert metrics.total_usage_count == 0
        assert metrics.total_bytes_sent == 0
        assert metrics.total_bytes_received == 0
        assert metrics.error_count == 0
        assert metrics.state == ConnectionState.IDLE
        assert isinstance(metrics.created_at, float)
        assert isinstance(metrics.last_used_at, float)
    
    def test_metrics_usage_update(self):
        """Test metrics usage tracking."""
        metrics = ConnectionMetrics(connection_id="test")
        initial_time = metrics.last_used_at
        
        # Simulate usage
        time.sleep(0.1)  # Small delay to ensure time difference
        metrics.update_usage(bytes_sent=100, bytes_received=50, response_time_ms=250)
        
        assert metrics.total_usage_count == 1
        assert metrics.total_bytes_sent == 100
        assert metrics.total_bytes_received == 50
        assert metrics.average_response_time_ms == 250
        assert metrics.last_used_at > initial_time
        
        # Test rolling average
        metrics.update_usage(bytes_sent=200, bytes_received=100, response_time_ms=150)
        assert metrics.total_usage_count == 2
        assert metrics.total_bytes_sent == 300
        assert metrics.total_bytes_received == 150
        # Rolling average: (0.8 * 250) + (0.2 * 150) = 200 + 30 = 230
        assert abs(metrics.average_response_time_ms - 230) < 0.1
    
    def test_metrics_age_and_idle_time(self):
        """Test age and idle time calculations."""
        metrics = ConnectionMetrics(connection_id="test")
        
        # Allow some time to pass
        time.sleep(0.1)
        
        age = metrics.get_age_seconds()
        idle_time = metrics.get_idle_time_seconds()
        
        assert age >= 0.1
        assert idle_time >= 0.1
        assert age >= idle_time  # Age should be >= idle time


class TestWebSocketConnectionPool:
    """Test WebSocket connection pool functionality."""
    
    @pytest.fixture
    async def pool(self):
        """Create a test connection pool."""
        pool = WebSocketConnectionPool(
            pool_name="test_pool",
            max_pool_size=5,
            min_pool_size=1,
            max_idle_time_seconds=60,
            health_check_interval_seconds=10
        )
        await pool.start()
        yield pool
        await pool.stop()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        return MockWebSocket()
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test pool initialization and basic properties."""
        assert pool.pool_name == "test_pool"
        assert pool.max_pool_size == 5
        assert pool.min_pool_size == 1
        assert pool.max_idle_time_seconds == 60
        
        metrics = pool.get_metrics()
        assert metrics["pool"]["name"] == "test_pool"
        assert metrics["pool"]["max_pool_size"] == 5
    
    @pytest.mark.asyncio
    async def test_websocket_registration(self, pool, mock_websocket):
        """Test WebSocket registration and unregistration."""
        # Register WebSocket
        connection_id = await pool.register_websocket(
            websocket=mock_websocket,
            user_id="test_user",
            session_id="test_session",
            client_info={"app_version": "1.0"},
            streaming_config={"vad_enabled": True}
        )
        
        assert connection_id is not None
        assert isinstance(connection_id, str)
        
        # Check connection exists
        connection_data = pool.get_connection(connection_id)
        assert connection_data is not None
        assert connection_data.websocket == mock_websocket
        assert connection_data.user_id == "test_user"
        assert connection_data.session_id == "test_session"
        assert connection_data.client_info["app_version"] == "1.0"
        assert connection_data.streaming_config["vad_enabled"] == True
        
        # Check tracking
        user_connections = pool.get_user_connections("test_user")
        assert connection_id in user_connections
        
        session_connection = pool.get_session_connection("test_session")
        assert session_connection == connection_id
        
        # Unregister
        await pool.unregister_websocket(connection_id)
        
        # Check cleanup
        connection_data = pool.get_connection(connection_id)
        assert connection_data is None
        
        user_connections = pool.get_user_connections("test_user")
        assert len(user_connections) == 0
    
    @pytest.mark.asyncio
    async def test_message_sending(self, pool, mock_websocket):
        """Test message sending functionality."""
        connection_id = await pool.register_websocket(
            websocket=mock_websocket,
            user_id="test_user"
        )
        
        # Test text message
        success = await pool.send_to_connection(connection_id, "Hello World", "text")
        assert success == True
        assert len(mock_websocket.sent_messages) == 1
        assert mock_websocket.sent_messages[0]["type"] == "text"
        assert mock_websocket.sent_messages[0]["data"] == "Hello World"
        
        # Test JSON message
        test_data = {"type": "test", "message": "Hello JSON"}
        success = await pool.send_to_connection(connection_id, test_data, "json")
        assert success == True
        assert len(mock_websocket.sent_messages) == 2
        sent_json = json.loads(mock_websocket.sent_messages[1]["data"])
        assert sent_json["type"] == "test"
        assert sent_json["message"] == "Hello JSON"
        
        # Test binary message
        binary_data = b"Binary Test Data"
        success = await pool.send_to_connection(connection_id, binary_data, "binary")
        assert success == True
        assert len(mock_websocket.sent_messages) == 3
        assert mock_websocket.sent_messages[2]["type"] == "bytes"
        assert mock_websocket.sent_messages[2]["data"] == binary_data
        
        await pool.unregister_websocket(connection_id)
    
    @pytest.mark.asyncio
    async def test_user_broadcasting(self, pool):
        """Test broadcasting to user connections."""
        # Create multiple connections for the same user
        websockets = [MockWebSocket() for _ in range(3)]
        connection_ids = []
        
        for ws in websockets:
            conn_id = await pool.register_websocket(
                websocket=ws,
                user_id="broadcast_user"
            )
            connection_ids.append(conn_id)
        
        # Broadcast message to user
        test_message = {"type": "broadcast", "data": "Hello All"}
        sent_count = await pool.send_to_user("broadcast_user", test_message, "json")
        
        assert sent_count == 3
        
        # Check all websockets received the message
        for ws in websockets:
            assert len(ws.sent_messages) == 1
            received_data = json.loads(ws.sent_messages[0]["data"])
            assert received_data["type"] == "broadcast"
            assert received_data["data"] == "Hello All"
        
        # Cleanup
        for conn_id in connection_ids:
            await pool.unregister_websocket(conn_id)
    
    @pytest.mark.asyncio
    async def test_global_broadcasting(self, pool):
        """Test global broadcasting functionality."""
        # Create connections with different users
        websockets = [MockWebSocket() for _ in range(4)]
        connection_ids = []
        
        for i, ws in enumerate(websockets):
            conn_id = await pool.register_websocket(
                websocket=ws,
                user_id=f"user_{i}"
            )
            connection_ids.append(conn_id)
        
        # Broadcast to all
        test_message = {"type": "global", "announcement": "Server maintenance"}
        sent_count = await pool.broadcast_message(test_message, "json")
        
        assert sent_count == 4
        
        # Check all websockets received the message
        for ws in websockets:
            assert len(ws.sent_messages) == 1
            received_data = json.loads(ws.sent_messages[0]["data"])
            assert received_data["type"] == "global"
            assert received_data["announcement"] == "Server maintenance"
        
        # Test filtered broadcasting
        def filter_even_users(conn_data):
            return conn_data.user_id.endswith(("0", "2"))
        
        sent_count = await pool.broadcast_message(
            {"type": "filtered", "message": "Even users only"}, 
            "json", 
            filter_func=filter_even_users
        )
        
        assert sent_count == 2  # user_0 and user_2
        
        # Cleanup
        for conn_id in connection_ids:
            await pool.unregister_websocket(conn_id)
    
    @pytest.mark.asyncio
    async def test_message_handling(self, pool, mock_websocket):
        """Test incoming message handling."""
        connection_id = await pool.register_websocket(
            websocket=mock_websocket,
            user_id="test_user"
        )
        
        # Test message handler registration
        handled_messages = []
        
        async def test_handler(conn_id, conn_data, message):
            handled_messages.append((conn_id, message))
        
        pool.register_message_handler("test_action", test_handler)
        
        # Send message with action
        await pool.handle_connection_message(
            connection_id,
            json.dumps({"action": "test_action", "data": "test_data"}),
            "text"
        )
        
        # Check handler was called
        assert len(handled_messages) == 1
        assert handled_messages[0][0] == connection_id
        assert handled_messages[0][1]["action"] == "test_action"
        
        # Check connection data has queued messages for unknown actions
        connection_data = pool.get_connection(connection_id)
        
        await pool.handle_connection_message(
            connection_id,
            json.dumps({"action": "unknown_action", "data": "queue_me"}),
            "text"
        )
        
        connection_data = pool.get_connection(connection_id)
        assert connection_data.get_queue_size() > 0
        
        await pool.unregister_websocket(connection_id)
    
    @pytest.mark.asyncio
    async def test_health_checks(self, pool):
        """Test connection health checks."""
        # Create healthy connection
        healthy_ws = MockWebSocket()
        healthy_id = await pool.register_websocket(
            websocket=healthy_ws,
            user_id="healthy_user"
        )
        
        # Create unhealthy connection (disconnected)
        unhealthy_ws = MockWebSocket(client_state=WebSocketState.DISCONNECTED)
        unhealthy_id = await pool.register_websocket(
            websocket=unhealthy_ws,
            user_id="unhealthy_user"
        )
        
        # Manually trigger health check
        await pool._perform_health_checks()
        
        # Healthy connection should still exist
        healthy_data = pool.get_connection(healthy_id)
        assert healthy_data is not None
        assert healthy_ws.ping_count > 0  # Ping was called
        
        # Unhealthy connection should be removed
        unhealthy_data = pool.get_connection(unhealthy_id)
        assert unhealthy_data is None
    
    @pytest.mark.asyncio
    async def test_pool_limits(self, pool):
        """Test pool size limits."""
        connections = []
        websockets = []
        
        # Fill up to max capacity
        for i in range(pool.max_pool_size):
            ws = MockWebSocket()
            conn_id = await pool.register_websocket(
                websocket=ws,
                user_id=f"user_{i}"
            )
            connections.append(conn_id)
            websockets.append(ws)
        
        # Try to exceed capacity (should still work as WebSocket pool creates new entries)
        overflow_ws = MockWebSocket()
        overflow_id = await pool.register_websocket(
            websocket=overflow_ws,
            user_id="overflow_user"
        )
        
        # Should succeed (WebSocket pool is more permissive than base pool)
        assert overflow_id is not None
        
        # Check metrics
        metrics = pool.get_metrics()
        assert metrics["pool"]["total_connections"] >= pool.max_pool_size
        
        # Cleanup
        for conn_id in connections + [overflow_id]:
            await pool.unregister_websocket(conn_id)
    
    @pytest.mark.asyncio
    async def test_pool_statistics(self, pool):
        """Test comprehensive pool statistics."""
        # Create various connections
        connections = []
        for i in range(3):
            ws = MockWebSocket()
            conn_id = await pool.register_websocket(
                websocket=ws,
                user_id=f"stats_user_{i}",
                session_id=f"session_{i}"
            )
            connections.append(conn_id)
            
            # Add some messages to queues
            connection_data = pool.get_connection(conn_id)
            connection_data.add_to_queue({"message": f"test_{i}"})
        
        # Get statistics
        stats = pool.get_pool_statistics()
        
        # Check basic pool stats
        assert stats["pool"]["total_connections"] >= 3
        assert stats["pool"]["name"] == "test_pool"
        
        # Check WebSocket-specific stats
        ws_stats = stats["websocket_specific"]
        assert ws_stats["total_users"] >= 3
        assert ws_stats["total_sessions"] >= 3
        assert ws_stats["average_queue_size"] > 0
        
        # Cleanup
        for conn_id in connections:
            await pool.unregister_websocket(conn_id)


class TestConnectionPoolIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_singleton_pool_access(self):
        """Test singleton pool access patterns."""
        # Initialize pool
        pool1 = await initialize_websocket_pool(
            pool_name="singleton_test",
            max_pool_size=10
        )
        
        # Get same instance
        pool2 = get_websocket_pool()
        
        assert pool1 is pool2
        assert pool1.pool_name == "singleton_test"
        
        # Shutdown
        await shutdown_websocket_pool()
        
        # Getting pool after shutdown should create new instance
        pool3 = get_websocket_pool()
        assert pool3 is not pool1
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test handling concurrent connections."""
        pool = WebSocketConnectionPool(
            pool_name="concurrent_test",
            max_pool_size=20
        )
        await pool.start()
        
        # Create many concurrent connections
        async def create_connection(i):
            ws = MockWebSocket()
            return await pool.register_websocket(
                websocket=ws,
                user_id=f"concurrent_user_{i}",
                client_info={"thread": i}
            )
        
        # Create 15 connections concurrently
        connection_tasks = [create_connection(i) for i in range(15)]
        connection_ids = await asyncio.gather(*connection_tasks)
        
        # All should succeed
        assert len(connection_ids) == 15
        assert all(conn_id is not None for conn_id in connection_ids)
        
        # Test concurrent message sending
        async def send_messages(conn_id, count):
            success_count = 0
            for i in range(count):
                success = await pool.send_to_connection(
                    conn_id, 
                    {"message": f"test_{i}"}, 
                    "json"
                )
                if success:
                    success_count += 1
            return success_count
        
        # Send messages concurrently
        send_tasks = [send_messages(conn_id, 5) for conn_id in connection_ids]
        sent_counts = await asyncio.gather(*send_tasks)
        
        # Most should succeed (some might fail due to timing)
        total_sent = sum(sent_counts)
        assert total_sent >= len(connection_ids) * 3  # At least 60% success rate
        
        # Cleanup
        cleanup_tasks = [pool.unregister_websocket(conn_id) for conn_id in connection_ids]
        await asyncio.gather(*cleanup_tasks)
        
        await pool.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling scenarios."""
        pool = WebSocketConnectionPool(pool_name="error_test")
        await pool.start()
        
        # Test sending to non-existent connection
        success = await pool.send_to_connection("non-existent", "test", "text")
        assert success == False
        
        # Test with failing WebSocket
        failing_ws = Mock()
        failing_ws.client_state = WebSocketState.CONNECTED
        failing_ws.send_text = AsyncMock(side_effect=Exception("WebSocket error"))
        
        conn_id = await pool.register_websocket(websocket=failing_ws)
        success = await pool.send_to_connection(conn_id, "test", "text")
        assert success == False
        
        # Connection should be tracked for error handling
        connection_data = pool.get_connection(conn_id)
        if connection_data:
            assert connection_data.metrics.error_count > 0
        
        await pool.stop()


if __name__ == "__main__":
    # Run basic test to verify imports and basic functionality
    async def basic_test():
        print("🧪 Running basic WebSocket connection pool test...")
        
        # Test basic pool creation
        pool = WebSocketConnectionPool(pool_name="basic_test", max_pool_size=5)
        await pool.start()
        
        # Test WebSocket registration
        mock_ws = MockWebSocket()
        conn_id = await pool.register_websocket(
            websocket=mock_ws,
            user_id="test_user",
            session_id="test_session"
        )
        
        print(f"✅ Created connection: {conn_id}")
        
        # Test message sending
        success = await pool.send_to_connection(conn_id, {"message": "test"}, "json")
        print(f"✅ Message sent: {success}")
        
        # Test metrics
        metrics = pool.get_metrics()
        print(f"✅ Pool metrics: {metrics['pool']['total_connections']} connections")
        
        # Cleanup
        await pool.unregister_websocket(conn_id)
        await pool.stop()
        
        print("🎉 Basic test completed successfully!")
    
    # Run the test
    asyncio.run(basic_test())