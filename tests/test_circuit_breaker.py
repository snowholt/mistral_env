"""
Test Suite for Circuit Breaker Implementation.

Tests circuit breaker functionality including:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold monitoring and recovery
- Integration with connection pool
- Metrics collection and reporting
- Configuration and callback handling
- Concurrent request handling

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.beautyai_inference.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    circuit_breaker_registry
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 60.0
        assert config.success_threshold == 3
        assert config.failure_rate_threshold == 0.5
        assert config.minimum_requests == 10
        assert config.timeout_seconds == 30.0
        assert config.health_check_enabled is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout_seconds=30.0,
            success_threshold=5,
            failure_rate_threshold=0.8
        )
        
        assert config.failure_threshold == 10
        assert config.recovery_timeout_seconds == 30.0
        assert config.success_threshold == 5
        assert config.failure_rate_threshold == 0.8


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        
        assert metrics.name == "test_circuit"
        assert metrics.state == CircuitBreakerState.CLOSED
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.consecutive_failures == 0
        
    def test_update_success(self):
        """Test success metrics update."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        
        metrics.update_success(response_time_ms=100.0)
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert metrics.failure_rate == 0.0
        assert metrics.average_response_time_ms == 100.0
        assert metrics.last_success_time is not None
        
    def test_update_failure(self):
        """Test failure metrics update."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        
        metrics.update_failure("Test error")
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.consecutive_successes == 0
        assert metrics.consecutive_failures == 1
        assert metrics.failure_rate == 1.0
        assert metrics.last_error == "Test error"
        assert metrics.last_failure_time is not None
        
    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        
        # 3 successes, 2 failures = 40% failure rate
        metrics.update_success()
        metrics.update_success()
        metrics.update_success()
        metrics.update_failure("Error 1")
        metrics.update_failure("Error 2")
        
        assert metrics.total_requests == 5
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 2
        assert metrics.failure_rate == 0.4
        
    def test_state_transitions(self):
        """Test state transition tracking."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        initial_time = metrics.state_change_time
        
        # Transition to OPEN
        time.sleep(0.1)  # Small delay to ensure time difference
        metrics.update_state(CircuitBreakerState.OPEN)
        
        assert metrics.state == CircuitBreakerState.OPEN
        assert metrics.total_state_changes == 1
        assert metrics.state_change_time > initial_time
        
        # Transition to HALF_OPEN
        time.sleep(0.1)
        open_time = metrics.state_change_time
        metrics.update_state(CircuitBreakerState.HALF_OPEN)
        
        assert metrics.state == CircuitBreakerState.HALF_OPEN
        assert metrics.total_state_changes == 2
        assert metrics.state_change_time > open_time
        assert metrics.open_duration_seconds > 0
        
    def test_uptime_percentage(self):
        """Test uptime percentage calculation."""
        metrics = CircuitBreakerMetrics(name="test_circuit")
        
        # Initially should be 100% uptime
        uptime = metrics.get_uptime_percentage()
        assert uptime == 100.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest_asyncio.fixture
    async def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,  # Lower for testing
            recovery_timeout_seconds=1.0,  # Shorter for testing
            success_threshold=2,  # Lower for testing
            timeout_seconds=5.0,
            health_check_enabled=False  # Disable for testing
        )
        
        cb = CircuitBreaker("test_circuit", config)
        await cb.start()
        yield cb
        await cb.stop()
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, circuit_breaker):
        """Test successful operation execution."""
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        
        assert result == "success"
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.state == CircuitBreakerState.CLOSED
        
    @pytest.mark.asyncio
    async def test_failed_operation(self, circuit_breaker):
        """Test failed operation handling."""
        async def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.failed_requests == 1
        assert circuit_breaker.metrics.consecutive_failures == 1
        assert circuit_breaker.metrics.state == CircuitBreakerState.CLOSED
        
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""
        async def failing_operation():
            raise ValueError("Test error")
        
        # Trigger failures to reach threshold (3)
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)
        
        # Circuit should be open now
        assert circuit_breaker.metrics.state == CircuitBreakerState.OPEN
        assert circuit_breaker.metrics.consecutive_failures == 3
        
        # Next call should fail fast
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(failing_operation)
            
    @pytest.mark.asyncio
    async def test_circuit_recovery(self, circuit_breaker):
        """Test circuit breaker recovery process."""
        async def failing_operation():
            raise ValueError("Test error")
        
        async def successful_operation():
            return "success"
        
        # Open circuit with failures
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.metrics.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # First successful call should transition to HALF_OPEN
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.metrics.state == CircuitBreakerState.HALF_OPEN
        
        # Second successful call should transition to CLOSED
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.metrics.state == CircuitBreakerState.CLOSED
        
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test operation timeout handling."""
        async def slow_operation():
            await asyncio.sleep(10)  # Longer than timeout
            return "too_slow"
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_operation)
        
        assert circuit_breaker.metrics.failed_requests == 1
        
    @pytest.mark.asyncio
    async def test_failure_rate_threshold(self):
        """Test circuit opens based on failure rate."""
        config = CircuitBreakerConfig(
            failure_threshold=100,  # High threshold
            failure_rate_threshold=0.5,  # 50% failure rate
            minimum_requests=10,
            health_check_enabled=False
        )
        
        cb = CircuitBreaker("rate_test", config)
        await cb.start()
        
        try:
            async def success_op():
                return "success"
            
            async def fail_op():
                raise ValueError("Error")
            
            # 10 requests: 6 failures (60% failure rate) should open circuit
            for i in range(4):
                await cb.call(success_op)
            for i in range(6):
                with pytest.raises(ValueError):
                    await cb.call(fail_op)
            
            assert cb.metrics.failure_rate == 0.6
            assert cb.metrics.state == CircuitBreakerState.OPEN
            
        finally:
            await cb.stop()
            
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, circuit_breaker):
        """Test concurrent request handling."""
        async def slow_success():
            await asyncio.sleep(0.1)
            return "success"
        
        # Execute multiple concurrent requests
        tasks = [circuit_breaker.call(slow_success) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == "success" for result in results)
        assert circuit_breaker.metrics.total_requests == 5
        assert circuit_breaker.metrics.successful_requests == 5
        
    @pytest.mark.asyncio
    async def test_callbacks(self, circuit_breaker):
        """Test state change and failure callbacks."""
        state_changes = []
        failures = []
        
        async def state_callback(name, state, metrics):
            state_changes.append((name, state))
            
        async def failure_callback(name, error, metrics):
            failures.append((name, error))
        
        circuit_breaker.add_state_change_callback(state_callback)
        circuit_breaker.add_failure_callback(failure_callback)
        
        # Trigger failures to open circuit
        async def failing_op():
            raise ValueError("Test error")
        
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_op)
        
        # Check callbacks were called
        assert len(failures) == 3
        assert len(state_changes) == 1
        assert state_changes[0] == ("test_circuit", CircuitBreakerState.OPEN)
        
    def test_metrics_export(self, circuit_breaker):
        """Test metrics export functionality."""
        metrics = circuit_breaker.get_metrics()
        
        assert metrics['name'] == 'test_circuit'
        assert metrics['state'] == 'CLOSED'
        assert 'total_requests' in metrics
        assert 'config' in metrics
        assert 'uptime_percentage' in metrics
        
    def test_state_properties(self, circuit_breaker):
        """Test state property methods."""
        assert circuit_breaker.is_closed is True
        assert circuit_breaker.is_open is False
        assert circuit_breaker.is_half_open is False


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""
    
    @pytest.mark.asyncio
    async def test_registry_get_or_create(self):
        """Test registry get or create functionality."""
        registry = CircuitBreakerRegistry()
        
        # Create first circuit breaker
        cb1 = await registry.get_or_create("test1")
        assert cb1.name == "test1"
        
        # Get same circuit breaker
        cb2 = await registry.get_or_create("test1")
        assert cb1 is cb2
        
        # Create different circuit breaker
        cb3 = await registry.get_or_create("test2")
        assert cb3.name == "test2"
        assert cb1 is not cb3
        
        await registry.stop_all()
        
    @pytest.mark.asyncio
    async def test_registry_remove(self):
        """Test registry remove functionality."""
        registry = CircuitBreakerRegistry()
        
        cb = await registry.get_or_create("test")
        assert cb.name == "test"
        
        await registry.remove("test")
        
        # Should create new instance
        cb2 = await registry.get_or_create("test")
        assert cb is not cb2
        
        await registry.stop_all()
        
    @pytest.mark.asyncio
    async def test_registry_metrics(self):
        """Test registry metrics collection."""
        registry = CircuitBreakerRegistry()
        
        await registry.get_or_create("test1")
        await registry.get_or_create("test2")
        
        metrics = registry.get_all_metrics()
        
        assert "test1" in metrics
        assert "test2" in metrics
        assert metrics["test1"]["name"] == "test1"
        assert metrics["test2"]["name"] == "test2"
        
        await registry.stop_all()


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_pool_metrics_integration(self):
        """Test integration with pool metrics provider."""
        pool_metrics_calls = []
        
        async def mock_pool_metrics():
            pool_metrics_calls.append(time.time())
            return {
                'pool': {
                    'unhealthy_connections': 5,
                    'total_connections': 10
                }
            }
        
        config = CircuitBreakerConfig(
            health_check_enabled=True,
            health_check_interval_seconds=0.1
        )
        
        cb = CircuitBreaker("pool_test", config, mock_pool_metrics)
        await cb.start()
        
        # Wait for health checks
        await asyncio.sleep(0.3)
        
        assert len(pool_metrics_calls) > 0
        
        await cb.stop()
        
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test proper error propagation through circuit breaker."""
        config = CircuitBreakerConfig(health_check_enabled=False)
        cb = CircuitBreaker("error_test", config)
        await cb.start()
        
        try:
            # Custom exception should propagate
            async def custom_error():
                raise ConnectionError("Custom error")
            
            with pytest.raises(ConnectionError) as exc_info:
                await cb.call(custom_error)
            
            assert str(exc_info.value) == "Custom error"
            assert cb.metrics.last_error == "Custom error"
            
        finally:
            await cb.stop()
            
    @pytest.mark.asyncio
    async def test_circuit_breaker_error_details(self):
        """Test circuit breaker error contains proper details."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=10.0,
            health_check_enabled=False
        )
        
        cb = CircuitBreaker("error_details_test", config)
        await cb.start()
        
        try:
            async def failing_op():
                raise ValueError("Test error")
            
            # Trigger failures to open circuit
            for i in range(2):
                with pytest.raises(ValueError):
                    await cb.call(failing_op)
            
            assert cb.is_open
            
            # Next call should raise CircuitBreakerError
            with pytest.raises(CircuitBreakerError) as exc_info:
                await cb.call(failing_op)
            
            error = exc_info.value
            assert error.state == CircuitBreakerState.OPEN
            assert error.last_failure_time is not None
            assert "OPEN" in str(error)
            
        finally:
            await cb.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])