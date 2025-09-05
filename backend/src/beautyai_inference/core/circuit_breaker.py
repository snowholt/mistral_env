"""
Circuit Breaker Implementation for BeautyAI Connection Pool.

Provides circuit breaker pattern for fault tolerance and system protection:
- Three-state circuit breaker (CLOSED, OPEN, HALF_OPEN)
- Integration with connection pool health metrics
- Configurable failure thresholds and recovery timeouts
- Real-time monitoring and automatic recovery
- Backpressure and load shedding capabilities
- Comprehensive metrics and diagnostics

The circuit breaker wraps connection operations and monitors their success/failure
rates. When failure rate exceeds configured thresholds, it "opens" the circuit
to prevent further damage and allow the system to recover.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Union, Set
import statistics
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states following the classic pattern."""
    CLOSED = auto()    # Normal operation, requests pass through
    OPEN = auto()      # Circuit is open, requests fail fast
    HALF_OPEN = auto() # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, state: CircuitBreakerState, last_failure_time: float):
        super().__init__(message)
        self.state = state
        self.last_failure_time = last_failure_time


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    name: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_change_time: float = field(default_factory=time.time)
    open_duration_seconds: float = 0.0
    half_open_duration_seconds: float = 0.0
    total_state_changes: int = 0
    failure_rate: float = 0.0
    average_response_time_ms: float = 0.0
    last_error: Optional[str] = None
    
    def update_success(self, response_time_ms: float = 0):
        """Update metrics for successful operation."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Update average response time
        if response_time_ms > 0:
            if self.average_response_time_ms == 0:
                self.average_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                self.average_response_time_ms = (
                    0.8 * self.average_response_time_ms + 0.2 * response_time_ms
                )
        
        self._update_failure_rate()

    def update_failure(self, error_message: str = ""):
        """Update metrics for failed operation."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
        self.last_error = error_message
        
        self._update_failure_rate()

    def update_state(self, new_state: CircuitBreakerState):
        """Update circuit breaker state."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            current_time = time.time()
            
            # Track state duration
            state_duration = current_time - self.state_change_time
            if old_state == CircuitBreakerState.OPEN:
                self.open_duration_seconds += state_duration
            elif old_state == CircuitBreakerState.HALF_OPEN:
                self.half_open_duration_seconds += state_duration
            
            self.state_change_time = current_time
            self.total_state_changes += 1
            
            logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.name} -> {new_state.name}")

    def _update_failure_rate(self):
        """Calculate current failure rate."""
        if self.total_requests > 0:
            self.failure_rate = self.failed_requests / self.total_requests
        else:
            self.failure_rate = 0.0

    def get_uptime_percentage(self) -> float:
        """Get percentage of time circuit breaker was in CLOSED state."""
        total_time = time.time() - (self.state_change_time - 
                                   (self.open_duration_seconds + self.half_open_duration_seconds))
        if total_time <= 0:
            return 100.0
        
        closed_time = total_time - self.open_duration_seconds - self.half_open_duration_seconds
        return (closed_time / total_time) * 100.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of consecutive failures to open circuit
    recovery_timeout_seconds: float = 60.0  # Time to wait before attempting recovery
    success_threshold: int = 3  # Number of consecutive successes to close circuit
    failure_rate_threshold: float = 0.5  # Failure rate (0-1) to open circuit
    minimum_requests: int = 10  # Minimum requests before calculating failure rate
    timeout_seconds: float = 30.0  # Timeout for individual operations
    max_retry_attempts: int = 3  # Maximum retry attempts
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    max_backoff_seconds: float = 300.0  # Maximum backoff delay
    health_check_enabled: bool = True  # Enable periodic health checks
    health_check_interval_seconds: float = 30.0  # Health check interval


class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting against cascading failures.
    
    The circuit breaker monitors operation success/failure rates and automatically
    switches between CLOSED (normal), OPEN (failing fast), and HALF_OPEN (testing
    recovery) states to protect system resources and enable recovery.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        pool_metrics_provider: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration settings
            pool_metrics_provider: Function to get connection pool metrics
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.pool_metrics_provider = pool_metrics_provider
        
        # State management
        self.metrics = CircuitBreakerMetrics(name=name)
        self._state_lock = asyncio.Lock()
        
        # Request tracking
        self._recent_requests = deque(maxlen=100)  # Track recent request outcomes
        self._pending_requests: Set[asyncio.Task] = set()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self._state_change_callbacks: List[Callable] = []
        self._failure_callbacks: List[Callable] = []
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {self.config}")

    async def start(self):
        """Start circuit breaker and background tasks."""
        logger.info(f"Starting circuit breaker '{self.name}'")
        
        if self.config.health_check_enabled and not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Circuit breaker '{self.name}' started")

    async def stop(self):
        """Stop circuit breaker and cleanup resources."""
        logger.info(f"Stopping circuit breaker '{self.name}'")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel pending requests
        for task in self._pending_requests.copy():
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._pending_requests, return_exceptions=True)
        
        logger.info(f"Circuit breaker '{self.name}' stopped")

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation through the circuit breaker.
        
        Args:
            operation: Async function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result
            
        Raises:
            CircuitBreakerError: When circuit is open
            asyncio.TimeoutError: When operation times out
        """
        async with self._state_lock:
            # Check if circuit should be opened based on current state
            await self._evaluate_state()
            
            # Fail fast if circuit is open
            if self.metrics.state == CircuitBreakerState.OPEN:
                time_since_failure = time.time() - (self.metrics.last_failure_time or 0)
                if time_since_failure < self.config.recovery_timeout_seconds:
                    error_msg = (f"Circuit breaker '{self.name}' is OPEN. "
                               f"Recovery timeout: {self.config.recovery_timeout_seconds - time_since_failure:.1f}s")
                    raise CircuitBreakerError(error_msg, self.metrics.state, self.metrics.last_failure_time)
                else:
                    # Time to try recovery
                    await self._transition_to_half_open()
        
        # Execute operation with timeout and error handling
        start_time = time.time()
        task = asyncio.create_task(self._execute_with_timeout(operation, *args, **kwargs))
        self._pending_requests.add(task)
        
        try:
            result = await task
            
            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            await self._record_success(response_time_ms)
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(str(e))
            raise
            
        finally:
            self._pending_requests.discard(task)

    async def _execute_with_timeout(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with timeout."""
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                return await operation(*args, **kwargs)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Operation timed out after {self.config.timeout_seconds}s")

    async def _record_success(self, response_time_ms: float):
        """Record successful operation."""
        async with self._state_lock:
            self.metrics.update_success(response_time_ms)
            self._recent_requests.append({'success': True, 'timestamp': time.time()})
            
            # Check if circuit should be closed from half-open state
            if (self.metrics.state == CircuitBreakerState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.config.success_threshold):
                await self._transition_to_closed()

    async def _record_failure(self, error_message: str):
        """Record failed operation."""
        async with self._state_lock:
            self.metrics.update_failure(error_message)
            self._recent_requests.append({'success': False, 'timestamp': time.time()})
            
            # Execute failure callbacks
            for callback in self._failure_callbacks:
                try:
                    await callback(self.name, error_message, self.metrics)
                except Exception as e:
                    logger.warning(f"Failure callback error: {e}")
            
            # Check if circuit should be opened
            await self._evaluate_state()

    async def _evaluate_state(self):
        """Evaluate whether circuit breaker state should change."""
        current_state = self.metrics.state
        
        if current_state == CircuitBreakerState.CLOSED:
            # Check if should open circuit
            should_open = (
                # Consecutive failures threshold
                self.metrics.consecutive_failures >= self.config.failure_threshold or
                
                # Failure rate threshold (if enough requests)
                (self.metrics.total_requests >= self.config.minimum_requests and 
                 self.metrics.failure_rate >= self.config.failure_rate_threshold)
            )
            
            if should_open:
                await self._transition_to_open()
                
        elif current_state == CircuitBreakerState.HALF_OPEN:
            # In half-open state, any failure should open the circuit
            if self.metrics.consecutive_failures > 0:
                await self._transition_to_open()

    async def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self.metrics.update_state(CircuitBreakerState.OPEN)
        await self._notify_state_change(CircuitBreakerState.OPEN)
        
    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self.metrics.update_state(CircuitBreakerState.HALF_OPEN)
        await self._notify_state_change(CircuitBreakerState.HALF_OPEN)
        
    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self.metrics.update_state(CircuitBreakerState.CLOSED)
        await self._notify_state_change(CircuitBreakerState.CLOSED)

    async def _notify_state_change(self, new_state: CircuitBreakerState):
        """Notify callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                await callback(self.name, new_state, self.metrics)
            except Exception as e:
                logger.warning(f"State change callback error: {e}")

    async def _health_check_loop(self):
        """Background health checking."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error in circuit breaker '{self.name}': {e}")
                await asyncio.sleep(5)

    async def _perform_health_check(self):
        """Perform health check and update metrics."""
        # Get pool metrics if available
        if self.pool_metrics_provider:
            try:
                pool_metrics = await self.pool_metrics_provider()
                
                # Analyze pool health
                if pool_metrics and 'pool' in pool_metrics:
                    pool_data = pool_metrics['pool']
                    unhealthy_ratio = (pool_data.get('unhealthy_connections', 0) / 
                                     max(pool_data.get('total_connections', 1), 1))
                    
                    # If too many connections are unhealthy, consider opening circuit
                    if unhealthy_ratio > 0.5 and self.metrics.state == CircuitBreakerState.CLOSED:
                        logger.warning(f"High unhealthy connection ratio ({unhealthy_ratio:.2f}) "
                                     f"in pool, evaluating circuit breaker state")
                        async with self._state_lock:
                            await self._evaluate_state()
                            
            except Exception as e:
                logger.warning(f"Error retrieving pool metrics for health check: {e}")

    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)

    def add_failure_callback(self, callback: Callable):
        """Add callback for failures."""
        self._failure_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        recent_success_rate = 0.0
        if self._recent_requests:
            recent_successes = sum(1 for req in self._recent_requests if req['success'])
            recent_success_rate = recent_successes / len(self._recent_requests)
        
        return {
            'name': self.name,
            'state': self.metrics.state.name,
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'failure_rate': self.metrics.failure_rate,
            'recent_success_rate': recent_success_rate,
            'consecutive_failures': self.metrics.consecutive_failures,
            'consecutive_successes': self.metrics.consecutive_successes,
            'last_failure_time': self.metrics.last_failure_time,
            'last_success_time': self.metrics.last_success_time,
            'state_change_time': self.metrics.state_change_time,
            'total_state_changes': self.metrics.total_state_changes,
            'uptime_percentage': self.metrics.get_uptime_percentage(),
            'average_response_time_ms': self.metrics.average_response_time_ms,
            'last_error': self.metrics.last_error,
            'pending_requests': len(self._pending_requests),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout_seconds': self.config.recovery_timeout_seconds,
                'success_threshold': self.config.success_threshold,
                'failure_rate_threshold': self.config.failure_rate_threshold,
                'timeout_seconds': self.config.timeout_seconds
            }
        }

    @property 
    def is_closed(self) -> bool:
        """Check if circuit is in CLOSED state."""
        return self.metrics.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is in OPEN state."""
        return self.metrics.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in HALF_OPEN state."""
        return self.metrics.state == CircuitBreakerState.HALF_OPEN


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None,
        pool_metrics_provider: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        async with self._lock:
            if name not in self._circuit_breakers:
                circuit_breaker = CircuitBreaker(name, config, pool_metrics_provider)
                await circuit_breaker.start()
                self._circuit_breakers[name] = circuit_breaker
            
            return self._circuit_breakers[name]
    
    async def remove(self, name: str):
        """Remove circuit breaker from registry."""
        async with self._lock:
            if name in self._circuit_breakers:
                await self._circuit_breakers[name].stop()
                del self._circuit_breakers[name]
    
    async def stop_all(self):
        """Stop all circuit breakers."""
        async with self._lock:
            for circuit_breaker in self._circuit_breakers.values():
                await circuit_breaker.stop()
            self._circuit_breakers.clear()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return circuit_breaker_registry