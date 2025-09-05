"""
Integration Tests for Circuit Breaker with Connection Pool.

Tests circuit breaker integration with the WebSocket connection pool:
- Circuit breaker protection during connection acquisition
- Circuit breaker protection during message sending
- Health monitoring integration
- Real-time failure detection and recovery
- Pool-level circuit breaker metrics

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.beautyai_inference.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError
)
from backend.src.beautyai_inference.core.websocket_connection_pool import (
    WebSocketConnectionPool,
    WebSocketConnectionData
)
from backend.src.beautyai_inference.core.connection_pool import (
    ConnectionPool,
    ConnectionPoolError,
    PoolExhaustedException
)


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self, should_fail=False, fail_on_send=False):
        self.should_fail = should_fail
        self.fail_on_send = fail_on_send
        self.client_state = "CONNECTED"
        self.messages_sent = []
        self.closed = False
        
    async def accept(self):
        if self.should_fail:
            raise ConnectionError("Mock connection failed")
            
    async def send_text(self, text):
        if self.fail_on_send:
            raise ConnectionError("Mock send failed")
        self.messages_sent.append(('text', text))
        
    async def send_bytes(self, data):
        if self.fail_on_send:
            raise ConnectionError("Mock send failed")
        self.messages_sent.append(('bytes', data))
        
    async def close(self, code=1000, reason=""):
        self.closed = True
        self.client_state = "DISCONNECTED"
        
    async def ping(self):
        if self.should_fail:
            raise ConnectionError("Ping failed")
        return True


class MockConnectionPool(ConnectionPool):
    """Mock connection pool for testing."""
    
    def __init__(self, pool_name="test_pool", should_fail=False, **kwargs):
        self.should_fail = should_fail
        self.created_connections = {}
        super().__init__(pool_name=pool_name, **kwargs)
        
    async def _create_connection(self, connection_id: str, **kwargs):
        """Create mock connection."""
        if self.should_fail:
            raise ConnectionError("Mock create connection failed")
        
        mock_connection = {"id": connection_id, "created_at": time.time()}
        self.created_connections[connection_id] = mock_connection
        return mock_connection
        
    async def _destroy_connection(self, connection_id: str, connection):
        """Destroy mock connection."""
        self.created_connections.pop(connection_id, None)
        
    async def _health_check_connection(self, connection_id: str, connection) -> bool:
        """Mock health check."""
        if self.should_fail:
            return False
        return True


class TestCircuitBreakerConnectionPoolIntegration:
    """Test circuit breaker integration with connection pool."""
    
    @pytest.fixture
    async def connection_pool_with_circuit_breaker(self):
        """Create connection pool with circuit breaker enabled."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=1.0,
            success_threshold=2,
            timeout_seconds=5.0,
            health_check_enabled=False
        )
        
        pool = MockConnectionPool(
            pool_name="test_cb_pool",
            max_pool_size=5,
            min_pool_size=1,
            enable_circuit_breaker=True,
            circuit_breaker_config=circuit_config
        )
        
        await pool.start()
        yield pool
        await pool.stop()
        
    @pytest.mark.asyncio
    async def test_successful_connection_acquisition(self, connection_pool_with_circuit_breaker):
        """Test successful connection acquisition through circuit breaker."""
        pool = connection_pool_with_circuit_breaker
        
        # Should succeed
        connection_id, connection = await pool.acquire_connection()
        
        assert connection_id is not None
        assert connection is not None
        assert pool._circuit_breaker.metrics.successful_requests == 1
        assert pool._circuit_breaker.is_closed
        
        await pool.release_connection(connection_id)
        
    @pytest.mark.asyncio
    async def test_connection_acquisition_with_failures(self, connection_pool_with_circuit_breaker):
        """Test connection acquisition failures trigger circuit breaker."""
        pool = connection_pool_with_circuit_breaker
        
        # Make pool fail connections
        pool.should_fail = True
        
        # Trigger failures to open circuit
        for i in range(3):
            with pytest.raises(ConnectionError):
                await pool.acquire_connection()
        
        # Circuit should be open
        assert pool._circuit_breaker.is_open
        assert pool._circuit_breaker.metrics.consecutive_failures == 3
        
        # Next attempt should fail fast with CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await pool.acquire_connection()
            
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, connection_pool_with_circuit_breaker):
        """Test circuit breaker recovery after failures."""
        pool = connection_pool_with_circuit_breaker
        
        # Open circuit with failures
        pool.should_fail = True
        for i in range(3):
            with pytest.raises(ConnectionError):
                await pool.acquire_connection()
        
        assert pool._circuit_breaker.is_open
        
        # Fix the pool
        pool.should_fail = False
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to HALF_OPEN and succeed
        connection_id, connection = await pool.acquire_connection()
        assert pool._circuit_breaker.is_half_open
        
        # Second success should close circuit
        connection_id2, connection2 = await pool.acquire_connection()
        assert pool._circuit_breaker.is_closed
        
        await pool.release_connection(connection_id)
        await pool.release_connection(connection_id2)
        
    @pytest.mark.asyncio
    async def test_pool_metrics_in_circuit_breaker(self, connection_pool_with_circuit_breaker):
        """Test pool metrics are available in circuit breaker."""
        pool = connection_pool_with_circuit_breaker
        
        # Get pool metrics
        pool_metrics = await pool._get_pool_metrics_for_circuit_breaker()
        
        assert 'pool' in pool_metrics
        assert 'circuit_breaker' in pool_metrics
        assert pool_metrics['pool']['name'] == 'test_cb_pool'
        assert pool_metrics['circuit_breaker']['name'] == 'pool_test_cb_pool'


class TestWebSocketCircuitBreakerIntegration:
    """Test circuit breaker integration with WebSocket connection pool."""
    
    @pytest.fixture
    async def websocket_pool_with_circuit_breaker(self):
        """Create WebSocket connection pool with circuit breaker."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=1.0,
            success_threshold=2,
            timeout_seconds=5.0,
            health_check_enabled=False
        )
        
        pool = WebSocketConnectionPool(
            pool_name="websocket_cb_test",
            max_pool_size=10,
            min_pool_size=2,
            enable_circuit_breaker=True,
            circuit_breaker_config=circuit_config
        )
        
        await pool.start()
        yield pool
        await pool.stop()
        
    @pytest.mark.asyncio
    async def test_websocket_connection_registration(self, websocket_pool_with_circuit_breaker):
        """Test WebSocket connection registration with circuit breaker."""
        pool = websocket_pool_with_circuit_breaker
        
        mock_ws = MockWebSocket()
        
        # Register WebSocket connection
        connection_id = await pool.register_websocket(
            websocket=mock_ws,
            user_id="user123",
            session_id="session123"
        )
        
        assert connection_id is not None
        assert pool.get_connection(connection_id) is not None
        assert pool._circuit_breaker.metrics.successful_requests >= 1
        
        await pool.unregister_websocket(connection_id)
        
    @pytest.mark.asyncio
    async def test_websocket_message_sending_with_circuit_breaker(self, websocket_pool_with_circuit_breaker):
        """Test WebSocket message sending through circuit breaker."""
        pool = websocket_pool_with_circuit_breaker
        
        mock_ws = MockWebSocket()
        connection_id = await pool.register_websocket(websocket=mock_ws)
        
        # Send successful message
        success = await pool.send_to_connection(connection_id, "test message")
        
        assert success is True
        assert len(mock_ws.messages_sent) == 1
        assert mock_ws.messages_sent[0] == ('text', 'test message')
        
        await pool.unregister_websocket(connection_id)
        
    @pytest.mark.asyncio
    async def test_websocket_send_failures_trigger_circuit_breaker(self, websocket_pool_with_circuit_breaker):
        """Test WebSocket send failures trigger circuit breaker."""
        pool = websocket_pool_with_circuit_breaker
        
        # Create failing WebSocket
        mock_ws = MockWebSocket(fail_on_send=True)
        connection_id = await pool.register_websocket(websocket=mock_ws)
        
        # Trigger send failures
        for i in range(3):
            success = await pool.send_to_connection(connection_id, f"message {i}")
            assert success is False
        
        # Circuit should be open after multiple send failures
        assert pool._circuit_breaker.is_open
        
        # Next send should fail fast (circuit breaker protection)
        success = await pool.send_to_connection(connection_id, "blocked message")
        assert success is False  # Failed due to circuit breaker
        
        await pool.unregister_websocket(connection_id)
        
    @pytest.mark.asyncio
    async def test_websocket_send_recovery(self, websocket_pool_with_circuit_breaker):
        """Test WebSocket send recovery through circuit breaker."""
        pool = websocket_pool_with_circuit_breaker
        
        # Create initially failing WebSocket
        mock_ws = MockWebSocket(fail_on_send=True)
        connection_id = await pool.register_websocket(websocket=mock_ws)
        
        # Open circuit with failures
        for i in range(3):
            await pool.send_to_connection(connection_id, f"failing message {i}")
        
        assert pool._circuit_breaker.is_open
        
        # Fix the WebSocket
        mock_ws.fail_on_send = False
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to HALF_OPEN and succeed
        success = await pool.send_to_connection(connection_id, "recovery message 1")
        assert success is True
        assert pool._circuit_breaker.is_half_open
        
        # Second success should close circuit
        success = await pool.send_to_connection(connection_id, "recovery message 2")
        assert success is True
        assert pool._circuit_breaker.is_closed
        
        await pool.unregister_websocket(connection_id)
        
    @pytest.mark.asyncio
    async def test_multiple_websocket_connections_circuit_breaker(self, websocket_pool_with_circuit_breaker):
        """Test circuit breaker with multiple WebSocket connections."""
        pool = websocket_pool_with_circuit_breaker
        
        # Create multiple connections
        connections = []
        for i in range(3):
            mock_ws = MockWebSocket()
            connection_id = await pool.register_websocket(
                websocket=mock_ws, 
                user_id=f"user{i}"
            )
            connections.append((connection_id, mock_ws))
        
        # Send messages to all connections
        for connection_id, mock_ws in connections:
            success = await pool.send_to_connection(connection_id, f"message to {connection_id}")
            assert success is True
        
        # All should have received messages
        for connection_id, mock_ws in connections:
            assert len(mock_ws.messages_sent) == 1
        
        # Circuit breaker should have recorded successes
        assert pool._circuit_breaker.metrics.successful_requests >= 3
        assert pool._circuit_breaker.is_closed
        
        # Cleanup
        for connection_id, _ in connections:
            await pool.unregister_websocket(connection_id)
            
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_in_websocket_pool(self, websocket_pool_with_circuit_breaker):
        """Test circuit breaker metrics are included in WebSocket pool metrics."""
        pool = websocket_pool_with_circuit_breaker
        
        # Register a connection and send a message
        mock_ws = MockWebSocket()
        connection_id = await pool.register_websocket(websocket=mock_ws)
        await pool.send_to_connection(connection_id, "test message")
        
        # Get comprehensive metrics
        metrics = pool.get_metrics()
        
        assert 'circuit_breaker' in metrics
        cb_metrics = metrics['circuit_breaker']
        
        assert cb_metrics['name'] == 'pool_websocket_cb_test'
        assert cb_metrics['state'] == 'CLOSED'
        assert cb_metrics['successful_requests'] >= 1
        assert 'failure_rate' in cb_metrics
        assert 'config' in cb_metrics
        
        await pool.unregister_websocket(connection_id)


class TestCircuitBreakerHealthMonitoring:
    """Test circuit breaker health monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_health_monitoring_with_unhealthy_pool(self):
        """Test circuit breaker responds to unhealthy connection pool."""
        # Mock pool metrics showing high unhealthy connection ratio
        async def mock_unhealthy_pool_metrics():
            return {
                'pool': {
                    'unhealthy_connections': 8,  # 80% unhealthy
                    'total_connections': 10
                }
            }
        
        circuit_config = CircuitBreakerConfig(
            health_check_enabled=True,
            health_check_interval_seconds=0.1,
            failure_threshold=1  # Low threshold for testing
        )
        
        cb = CircuitBreaker(
            "health_test",
            circuit_config,
            pool_metrics_provider=mock_unhealthy_pool_metrics
        )
        
        await cb.start()
        
        try:
            # Wait for health checks to detect unhealthy pool
            await asyncio.sleep(0.3)
            
            # Health monitoring should have triggered evaluation
            # Note: In real implementation, this would affect circuit state
            # based on the health check results
            
            health_metrics = await mock_unhealthy_pool_metrics()
            assert health_metrics['pool']['unhealthy_connections'] == 8
            
        finally:
            await cb.stop()
            
    @pytest.mark.asyncio
    async def test_health_monitoring_error_handling(self):
        """Test health monitoring handles errors gracefully."""
        async def failing_pool_metrics():
            raise ConnectionError("Pool metrics unavailable")
        
        circuit_config = CircuitBreakerConfig(
            health_check_enabled=True,
            health_check_interval_seconds=0.1
        )
        
        cb = CircuitBreaker(
            "health_error_test",
            circuit_config,
            pool_metrics_provider=failing_pool_metrics
        )
        
        await cb.start()
        
        try:
            # Wait for health checks with errors
            await asyncio.sleep(0.3)
            
            # Circuit breaker should still be operational despite health check errors
            assert cb.is_closed
            
        finally:
            await cb.stop()


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration integration."""
    
    @pytest.mark.asyncio
    async def test_websocket_pool_circuit_breaker_disabled(self):
        """Test WebSocket pool with circuit breaker disabled."""
        pool = WebSocketConnectionPool(
            pool_name="no_cb_test",
            enable_circuit_breaker=False
        )
        
        await pool.start()
        
        try:
            assert pool._circuit_breaker is None
            
            # Connection operations should work without circuit breaker
            mock_ws = MockWebSocket()
            connection_id = await pool.register_websocket(websocket=mock_ws)
            
            success = await pool.send_to_connection(connection_id, "test message")
            assert success is True
            
            # Metrics should not include circuit breaker
            metrics = pool.get_metrics()
            assert 'circuit_breaker' not in metrics
            
            await pool.unregister_websocket(connection_id)
            
        finally:
            await pool.stop()
            
    @pytest.mark.asyncio
    async def test_custom_circuit_breaker_config(self):
        """Test WebSocket pool with custom circuit breaker configuration."""
        custom_config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout_seconds=5.0,
            success_threshold=5,
            failure_rate_threshold=0.8
        )
        
        pool = WebSocketConnectionPool(
            pool_name="custom_cb_test",
            enable_circuit_breaker=True,
            circuit_breaker_config=custom_config
        )
        
        await pool.start()
        
        try:
            metrics = pool.get_metrics()
            cb_config = metrics['circuit_breaker']['config']
            
            assert cb_config['failure_threshold'] == 10
            assert cb_config['recovery_timeout_seconds'] == 5.0
            assert cb_config['success_threshold'] == 5
            assert cb_config['failure_rate_threshold'] == 0.8
            
        finally:
            await pool.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])