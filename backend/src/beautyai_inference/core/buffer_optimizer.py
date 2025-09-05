"""
Buffer Optimization Engine for BeautyAI Framework.

Provides intelligent buffer management with adaptive sizing, memory pooling,
and performance optimization based on real-time metrics and circuit breaker states.

Author: BeautyAI Framework  
Date: September 5, 2025
"""

import asyncio
import logging
import time
import statistics
import psutil
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

from .buffer_types import (
    BufferType, BufferStrategy, SizingAlgorithm, BufferState, 
    BufferMetrics, BufferConfiguration, BufferSizingResult,
    AudioBufferConfig, MemoryPoolBlock, BufferOptimizationConfig
)
from .performance_monitor import get_performance_monitor
from .circuit_breaker import get_circuit_breaker_registry
from .websocket_connection_pool import get_websocket_pool

logger = logging.getLogger(__name__)


class MemoryPool:
    """
    Efficient memory pool for buffer reuse and allocation optimization.
    
    Manages pre-allocated memory blocks to reduce allocation overhead
    and fragmentation during high-throughput operations.
    """
    
    def __init__(self, block_size: int, max_blocks: int, cleanup_interval: float = 60.0):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cleanup_interval = cleanup_interval
        
        # Memory management
        self._available_blocks: deque = deque()
        self._allocated_blocks: Dict[str, MemoryPoolBlock] = {}
        self._lock = threading.RLock()
        
        # Metrics
        self.total_allocations = 0
        self.total_deallocations = 0  
        self.cache_hits = 0
        self.cache_misses = 0
        self.peak_usage = 0
        
        # Background cleanup
        self._cleanup_task = None
        self._shutdown = False
        
        logger.info(f"Initialized memory pool: {block_size} bytes per block, {max_blocks} max blocks")
        
    def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def stop(self):
        """Stop the memory pool and cleanup."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all blocks
        with self._lock:
            self._available_blocks.clear()
            self._allocated_blocks.clear()
            
    def acquire_block(self) -> Optional[MemoryPoolBlock]:
        """Acquire a memory block from the pool."""
        with self._lock:
            # Try to reuse an available block
            if self._available_blocks:
                block = self._available_blocks.popleft()
                block.acquire()
                self._allocated_blocks[block.block_id] = block
                self.cache_hits += 1
                self.total_allocations += 1
                return block
            
            # Create new block if under limit
            if len(self._allocated_blocks) < self.max_blocks:
                block_id = str(uuid.uuid4())
                block = MemoryPoolBlock(
                    block_id=block_id,
                    size_bytes=self.block_size,
                    allocated_at=time.time(),
                    last_used=time.time(),
                    data=bytearray(self.block_size)
                )
                block.acquire()
                self._allocated_blocks[block_id] = block
                self.cache_misses += 1
                self.total_allocations += 1
                
                # Update peak usage
                current_usage = len(self._allocated_blocks)
                if current_usage > self.peak_usage:
                    self.peak_usage = current_usage
                    
                return block
            
            # Pool exhausted
            logger.warning(f"Memory pool exhausted: {len(self._allocated_blocks)} blocks allocated")
            return None
            
    def release_block(self, block: MemoryPoolBlock):
        """Release a memory block back to the pool."""
        with self._lock:
            if block.block_id not in self._allocated_blocks:
                logger.warning(f"Attempting to release unknown block: {block.block_id}")
                return
                
            block.release()
            self.total_deallocations += 1
            
            if block.is_available():
                # Move to available pool
                del self._allocated_blocks[block.block_id]
                
                # Reset the block data
                if hasattr(block.data, 'clear'):
                    block.data[:] = bytes(self.block_size)  # Clear but keep allocation
                
                self._available_blocks.append(block)
                
                # Limit available pool size
                while len(self._available_blocks) > self.max_blocks // 2:
                    self._available_blocks.popleft()
                    
    async def _cleanup_loop(self):
        """Background cleanup of unused blocks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_unused_blocks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory pool cleanup: {e}")
                
    async def _cleanup_unused_blocks(self):
        """Clean up blocks that haven't been used recently."""
        current_time = time.time()
        cleanup_threshold = 300.0  # 5 minutes
        
        with self._lock:
            # Clean up available blocks that are too old
            cleaned_available = deque()
            while self._available_blocks:
                block = self._available_blocks.popleft()
                if current_time - block.last_used < cleanup_threshold:
                    cleaned_available.append(block)
                # Else: let block be garbage collected
                    
            self._available_blocks = cleaned_available
            
        logger.debug(f"Memory pool cleanup: {len(self._available_blocks)} blocks remaining")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                "block_size": self.block_size,
                "max_blocks": self.max_blocks,
                "allocated_blocks": len(self._allocated_blocks),
                "available_blocks": len(self._available_blocks),
                "total_allocations": self.total_allocations,
                "total_deallocations": self.total_deallocations,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "peak_usage": self.peak_usage,
                "current_memory_mb": (len(self._allocated_blocks) * self.block_size) / (1024 * 1024)
            }


class AdaptiveBufferSizer:
    """
    Intelligent buffer sizing based on performance metrics and patterns.
    
    Uses multiple algorithms to determine optimal buffer sizes based on
    latency, throughput, memory pressure, and circuit breaker states.
    """
    
    def __init__(self, config: BufferConfiguration):
        self.config = config
        
        # Historical data for analysis
        self._latency_history = deque(maxlen=100)
        self._utilization_history = deque(maxlen=50)
        self._throughput_history = deque(maxlen=50)
        
        # Algorithm-specific state
        self._last_resize_time = 0
        self._resize_count = 0
        self._current_size = config.initial_size_bytes
        
        # Performance tracking
        self._performance_baseline = None
        self._optimization_attempts = 0
        self._successful_optimizations = 0
        
    def calculate_optimal_size(self, metrics: BufferMetrics, 
                             circuit_breaker_state: Optional[str] = None,
                             memory_pressure: float = 0.0) -> BufferSizingResult:
        """Calculate optimal buffer size based on current metrics."""
        
        # Update historical data
        self._latency_history.append(metrics.average_latency_ms)
        self._utilization_history.append(metrics.utilization_percentage)
        self._throughput_history.append(metrics.throughput_mbps)
        
        # Check resize cooldown
        current_time = time.time()
        if current_time - self._last_resize_time < self.config.resize_cooldown_seconds:
            return BufferSizingResult(
                recommended_size=metrics.current_size,
                size_change_bytes=0,
                size_change_percentage=0.0,
                reasoning="Resize cooldown period active",
                confidence=0.0,
                estimated_impact={}
            )
        
        # Apply sizing algorithm
        if self.config.algorithm == SizingAlgorithm.PERCENTILE_BASED:
            result = self._percentile_based_sizing(metrics)
        elif self.config.algorithm == SizingAlgorithm.MOVING_AVERAGE:
            result = self._moving_average_sizing(metrics)
        elif self.config.algorithm == SizingAlgorithm.LINEAR:
            result = self._linear_sizing(metrics)
        else:
            result = self._adaptive_sizing(metrics)
        
        # Apply circuit breaker adjustments
        if circuit_breaker_state and self.config.circuit_breaker_aware:
            result = self._apply_circuit_breaker_adjustment(result, circuit_breaker_state)
            
        # Apply memory pressure adjustments
        if memory_pressure > 0.8:
            result = self._apply_memory_pressure_adjustment(result, memory_pressure)
            
        # Enforce constraints
        result = self._apply_constraints(result)
        
        return result
        
    def _percentile_based_sizing(self, metrics: BufferMetrics) -> BufferSizingResult:
        """Size buffer based on latency percentiles."""
        if len(self._latency_history) < 10:
            return self._no_change_result(metrics, "Insufficient latency data")
        
        # Calculate latency statistics
        latencies = list(self._latency_history)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        median_latency = statistics.median(latencies)
        
        target_latency = self.config.target_latency_ms
        tolerance = self.config.latency_tolerance_ms
        
        # Determine if resize is needed
        if p95_latency > target_latency + tolerance:
            # High latency, increase buffer
            size_factor = min(1.5, (p95_latency / target_latency) ** 0.5)
            new_size = int(metrics.current_size * size_factor)
            reasoning = f"High P95 latency ({p95_latency:.1f}ms > {target_latency + tolerance:.1f}ms)"
            confidence = min(0.9, (p95_latency - target_latency) / target_latency)
            
        elif p95_latency < target_latency - tolerance and metrics.utilization_percentage < 0.6:
            # Low latency and utilization, decrease buffer  
            size_factor = max(0.8, target_latency / p95_latency)
            new_size = int(metrics.current_size * size_factor)
            reasoning = f"Low P95 latency ({p95_latency:.1f}ms) and utilization ({metrics.utilization_percentage:.1%})"
            confidence = min(0.8, (target_latency - p95_latency) / target_latency)
            
        else:
            return self._no_change_result(metrics, "Latency within target range")
            
        return self._create_sizing_result(metrics.current_size, new_size, reasoning, confidence)
        
    def _moving_average_sizing(self, metrics: BufferMetrics) -> BufferSizingResult:
        """Size buffer based on moving averages of key metrics."""
        if len(self._utilization_history) < 5:
            return self._no_change_result(metrics, "Insufficient utilization data")
            
        avg_utilization = statistics.mean(self._utilization_history)
        target_utilization = self.config.utilization_target
        tolerance = self.config.utilization_tolerance
        
        if avg_utilization > target_utilization + tolerance:
            # High utilization, increase buffer
            size_factor = 1 + ((avg_utilization - target_utilization) * 2)
            new_size = int(metrics.current_size * size_factor)
            reasoning = f"High average utilization ({avg_utilization:.1%})"
            confidence = min(0.8, (avg_utilization - target_utilization) / target_utilization)
            
        elif avg_utilization < target_utilization - tolerance:
            # Low utilization, decrease buffer
            size_factor = max(0.7, avg_utilization / target_utilization)
            new_size = int(metrics.current_size * size_factor)
            reasoning = f"Low average utilization ({avg_utilization:.1%})"
            confidence = min(0.7, (target_utilization - avg_utilization) / target_utilization)
            
        else:
            return self._no_change_result(metrics, "Utilization within target range")
            
        return self._create_sizing_result(metrics.current_size, new_size, reasoning, confidence)
        
    def _linear_sizing(self, metrics: BufferMetrics) -> BufferSizingResult:
        """Simple linear scaling based on utilization."""
        utilization = metrics.utilization_percentage
        target = self.config.utilization_target
        
        if abs(utilization - target) < self.config.utilization_tolerance:
            return self._no_change_result(metrics, "Utilization at target")
            
        # Linear adjustment
        adjustment_factor = (target / utilization) if utilization > 0 else 1.0
        new_size = int(metrics.current_size * adjustment_factor)
        
        reasoning = f"Linear adjustment for utilization {utilization:.1%} -> target {target:.1%}"
        confidence = min(0.6, abs(utilization - target) / target)
        
        return self._create_sizing_result(metrics.current_size, new_size, reasoning, confidence)
        
    def _adaptive_sizing(self, metrics: BufferMetrics) -> BufferSizingResult:
        """Advanced adaptive sizing using multiple factors."""
        # Combine multiple signals
        latency_score = self._calculate_latency_score(metrics)
        utilization_score = self._calculate_utilization_score(metrics)  
        throughput_score = self._calculate_throughput_score(metrics)
        overflow_penalty = self._calculate_overflow_penalty(metrics)
        
        # Weighted combination
        composite_score = (
            0.4 * latency_score + 
            0.3 * utilization_score +
            0.2 * throughput_score +
            0.1 * overflow_penalty
        )
        
        # Map score to size adjustment
        if composite_score > 0.2:
            size_factor = 1 + (composite_score * 0.5)
            reasoning = "Adaptive increase based on composite metrics"
        elif composite_score < -0.2:
            size_factor = 1 + (composite_score * 0.3)  # More conservative decrease
            reasoning = "Adaptive decrease based on composite metrics"
        else:
            return self._no_change_result(metrics, "Composite score within acceptable range")
            
        new_size = int(metrics.current_size * size_factor)
        confidence = min(0.9, abs(composite_score))
        
        return self._create_sizing_result(metrics.current_size, new_size, reasoning, confidence)
        
    def _calculate_latency_score(self, metrics: BufferMetrics) -> float:
        """Calculate latency-based adjustment score (-1 to 1)."""
        target = self.config.target_latency_ms
        current = metrics.average_latency_ms
        
        if current > target * 1.2:
            return min(1.0, (current - target) / target)  # Increase buffer
        elif current < target * 0.8:
            return max(-1.0, (current - target) / target)  # Decrease buffer
        return 0.0
        
    def _calculate_utilization_score(self, metrics: BufferMetrics) -> float:
        """Calculate utilization-based adjustment score (-1 to 1)."""
        target = self.config.utilization_target
        current = metrics.utilization_percentage
        
        if current > target + 0.2:
            return min(1.0, (current - target) / target)  # Increase buffer
        elif current < target - 0.2:  
            return max(-1.0, (current - target) / target)  # Decrease buffer
        return 0.0
        
    def _calculate_throughput_score(self, metrics: BufferMetrics) -> float:
        """Calculate throughput-based adjustment score (-1 to 1)."""
        if len(self._throughput_history) < 3:
            return 0.0
            
        recent_throughput = list(self._throughput_history)[-3:]
        if len(set(recent_throughput)) == 1:  # All same values
            return 0.0
            
        trend = recent_throughput[-1] - recent_throughput[0]
        if abs(trend) < 0.1:  # Minimal change
            return 0.0
            
        # Positive trend = good, negative trend = bad
        return max(-0.5, min(0.5, trend / max(recent_throughput)))
        
    def _calculate_overflow_penalty(self, metrics: BufferMetrics) -> float:
        """Calculate penalty for buffer overflows/underruns."""
        penalty = 0.0
        
        if metrics.buffer_overflows > 0:
            penalty += min(0.5, metrics.buffer_overflows * 0.1)
            
        if metrics.buffer_underruns > 2:
            penalty -= min(0.3, metrics.buffer_underruns * 0.05)
            
        return penalty
        
    def _apply_circuit_breaker_adjustment(self, result: BufferSizingResult, 
                                        circuit_state: str) -> BufferSizingResult:
        """Apply circuit breaker state adjustments."""
        if circuit_state == "OPEN":
            # Reduce buffer size when circuit is open
            adjusted_size = int(result.recommended_size * self.config.circuit_open_size_factor)
            result.recommended_size = adjusted_size
            result.reasoning += f" (Circuit breaker OPEN - reduced by {self.config.circuit_open_size_factor}x)"
            
        elif circuit_state == "HALF_OPEN":
            # Slightly increase buffer during recovery
            adjusted_size = int(result.recommended_size * self.config.circuit_recovery_size_factor)
            result.recommended_size = adjusted_size
            result.reasoning += f" (Circuit breaker HALF_OPEN - increased by {self.config.circuit_recovery_size_factor}x)"
            
        return result
        
    def _apply_memory_pressure_adjustment(self, result: BufferSizingResult,
                                        memory_pressure: float) -> BufferSizingResult:
        """Apply memory pressure adjustments."""
        if memory_pressure > 0.9:
            # Severe memory pressure, reduce aggressively
            pressure_factor = max(0.5, 1.0 - ((memory_pressure - 0.9) * 5))
            result.recommended_size = int(result.recommended_size * pressure_factor)
            result.reasoning += f" (High memory pressure {memory_pressure:.1%} - reduced)"
            
        return result
        
    def _apply_constraints(self, result: BufferSizingResult) -> BufferSizingResult:
        """Apply size constraints and validation."""
        # Enforce min/max bounds
        result.recommended_size = max(self.config.min_size_bytes, result.recommended_size)
        result.recommended_size = min(self.config.max_size_bytes, result.recommended_size)
        
        # Enforce resize increment
        if result.size_change_bytes != 0:
            increment = self.config.resize_increment_bytes
            remainder = abs(result.size_change_bytes) % increment
            if remainder != 0:
                if result.size_change_bytes > 0:
                    result.recommended_size += increment - remainder
                else:
                    result.recommended_size -= remainder
                    
        # Enforce maximum resize factor
        original_size = result.recommended_size - result.size_change_bytes
        if original_size > 0:
            resize_factor = result.recommended_size / original_size
            if resize_factor > self.config.max_resize_factor:
                result.recommended_size = int(original_size * self.config.max_resize_factor)
                result.reasoning += f" (Limited by max resize factor {self.config.max_resize_factor}x)"
                
        # Recalculate changes after constraints
        current_size = result.recommended_size - result.size_change_bytes
        result.size_change_bytes = result.recommended_size - current_size
        if current_size > 0:
            result.size_change_percentage = (result.size_change_bytes / current_size) * 100
            
        return result
        
    def _no_change_result(self, metrics: BufferMetrics, reasoning: str) -> BufferSizingResult:
        """Create a no-change sizing result."""
        return BufferSizingResult(
            recommended_size=metrics.current_size,
            size_change_bytes=0,
            size_change_percentage=0.0,
            reasoning=reasoning,
            confidence=0.0,
            estimated_impact={}
        )
        
    def _create_sizing_result(self, current_size: int, new_size: int, 
                            reasoning: str, confidence: float) -> BufferSizingResult:
        """Create a sizing result with calculated changes."""
        size_change = new_size - current_size
        size_change_pct = (size_change / current_size * 100) if current_size > 0 else 0.0
        
        # Estimate performance impact
        estimated_impact = {
            "latency_change_ms": -size_change * 0.001,  # Rough estimate
            "memory_change_mb": size_change / (1024 * 1024),
            "throughput_change_pct": size_change_pct * 0.1  # Rough correlation
        }
        
        return BufferSizingResult(
            recommended_size=new_size,
            size_change_bytes=size_change,
            size_change_percentage=size_change_pct,
            reasoning=reasoning,
            confidence=confidence,
            estimated_impact=estimated_impact
        )


# Global buffer manager instance
_buffer_manager_instance = None


class BufferManager:
    """
    Central buffer management system for the BeautyAI framework.
    
    Provides intelligent buffer optimization, memory pooling, and adaptive
    sizing based on real-time performance metrics and system state.
    """
    
    def __init__(self, config: BufferOptimizationConfig):
        self.config = config
        
        # Core components
        self._memory_pools: Dict[int, MemoryPool] = {}  # size -> pool
        self._buffer_configs: Dict[str, BufferConfiguration] = {}
        self._buffer_sizers: Dict[str, AdaptiveBufferSizer] = {}
        self._buffer_metrics: Dict[str, BufferMetrics] = {}
        
        # System integration
        self._performance_monitor = None
        self._circuit_breaker_registry = None
        self._websocket_pool = None
        
        # Background tasks
        self._optimization_task = None
        self._metrics_collection_task = None
        self._shutdown = False
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="buffer-opt")
        
        # Optimization tracking
        self._optimization_history = deque(maxlen=100)
        self._last_optimization_time = 0
        
        logger.info("Initialized BufferManager with optimization enabled")
        
    async def start(self):
        """Start the buffer manager and background tasks."""
        # Initialize system integrations
        self._performance_monitor = get_performance_monitor()
        self._circuit_breaker_registry = get_circuit_breaker_registry()
        self._websocket_pool = get_websocket_pool()
        
        # Initialize default memory pools
        await self._initialize_default_pools()
        
        # Start background tasks
        if self.config.optimization_interval_seconds > 0:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            
        if self.config.metrics_collection_enabled:
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
        # Start memory pool cleanup tasks
        for pool in self._memory_pools.values():
            pool.start_cleanup_task()
            
        logger.info("BufferManager started successfully")
        
    async def stop(self):
        """Stop the buffer manager and cleanup resources."""
        self._shutdown = True
        
        # Cancel background tasks
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
                
        if self._metrics_collection_task:
            self._metrics_collection_task.cancel()
            try:
                await self._metrics_collection_task
            except asyncio.CancelledError:
                pass
                
        # Stop memory pools
        for pool in self._memory_pools.values():
            await pool.stop()
            
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        logger.info("BufferManager stopped successfully")
        
    async def register_buffer(self, buffer_id: str, config: BufferConfiguration) -> bool:
        """Register a new buffer for optimization."""
        try:
            self._buffer_configs[buffer_id] = config
            self._buffer_sizers[buffer_id] = AdaptiveBufferSizer(config)
            self._buffer_metrics[buffer_id] = BufferMetrics(
                buffer_id=buffer_id,
                buffer_type=config.buffer_type,
                current_size=config.initial_size_bytes,
                optimal_size=config.initial_size_bytes
            )
            
            logger.info(f"Registered buffer '{buffer_id}' of type {config.buffer_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register buffer '{buffer_id}': {e}")
            return False
            
    async def unregister_buffer(self, buffer_id: str) -> bool:
        """Unregister a buffer from optimization."""
        try:
            self._buffer_configs.pop(buffer_id, None)
            self._buffer_sizers.pop(buffer_id, None)
            self._buffer_metrics.pop(buffer_id, None)
            
            logger.info(f"Unregistered buffer '{buffer_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister buffer '{buffer_id}': {e}")
            return False
            
    async def get_optimal_buffer_size(self, buffer_id: str) -> Optional[int]:
        """Get the current optimal size for a buffer."""
        if buffer_id not in self._buffer_metrics:
            return None
            
        return self._buffer_metrics[buffer_id].optimal_size
        
    async def update_buffer_metrics(self, buffer_id: str, metrics: Dict[str, Any]) -> bool:
        """Update metrics for a specific buffer."""
        if buffer_id not in self._buffer_metrics:
            logger.warning(f"Attempted to update metrics for unregistered buffer: {buffer_id}")
            return False
            
        try:
            buffer_metrics = self._buffer_metrics[buffer_id]
            
            # Update metrics from provided data
            for key, value in metrics.items():
                if hasattr(buffer_metrics, key):
                    setattr(buffer_metrics, key, value)
                    
            buffer_metrics.timestamp = time.time()
            
            # Trigger optimization if significant change
            await self._maybe_trigger_optimization(buffer_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for buffer '{buffer_id}': {e}")
            return False
            
    async def acquire_memory_block(self, size_bytes: int) -> Optional[MemoryPoolBlock]:
        """Acquire a memory block from the appropriate pool."""
        # Find the best matching pool
        pool = self._find_best_memory_pool(size_bytes)
        if not pool:
            # Create new pool if none exists
            pool = await self._create_memory_pool(size_bytes)
            
        if pool:
            return pool.acquire_block()
            
        return None
        
    async def release_memory_block(self, block: MemoryPoolBlock):
        """Release a memory block back to its pool."""
        pool = self._memory_pools.get(block.size_bytes)
        if pool:
            pool.release_block(block)
        else:
            logger.warning(f"No pool found for block size {block.size_bytes}")
            
    async def optimize_buffer(self, buffer_id: str) -> Optional[BufferSizingResult]:
        """Optimize a specific buffer size."""
        if buffer_id not in self._buffer_configs:
            return None
            
        try:
            config = self._buffer_configs[buffer_id]
            metrics = self._buffer_metrics[buffer_id]
            sizer = self._buffer_sizers[buffer_id]
            
            # Get system state
            circuit_state = await self._get_circuit_breaker_state(buffer_id)
            memory_pressure = await self._get_memory_pressure()
            
            # Calculate optimal size
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                sizer.calculate_optimal_size,
                metrics,
                circuit_state,
                memory_pressure
            )
            
            # Apply optimization if significant change
            if abs(result.size_change_percentage) >= config.resize_threshold_percentage:
                await self._apply_buffer_optimization(buffer_id, result)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to optimize buffer '{buffer_id}': {e}")
            return None
            
    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer optimization statistics."""
        stats = {
            "total_buffers": len(self._buffer_configs),
            "active_pools": len(self._memory_pools),
            "optimization_enabled": self.config.enabled,
            "optimization_interval": self.config.optimization_interval_seconds,
            "buffers": {},
            "memory_pools": {},
            "system": {}
        }
        
        # Buffer statistics
        for buffer_id, metrics in self._buffer_metrics.items():
            config = self._buffer_configs.get(buffer_id)
            stats["buffers"][buffer_id] = {
                "type": config.buffer_type.value if config else "unknown",
                "current_size_kb": metrics.current_size / 1024,
                "optimal_size_kb": metrics.optimal_size / 1024,
                "utilization": f"{metrics.utilization_percentage:.1%}",
                "latency_ms": metrics.average_latency_ms,
                "throughput_mbps": metrics.throughput_mbps,
                "overflows": metrics.buffer_overflows,
                "underruns": metrics.buffer_underruns
            }
            
        # Memory pool statistics
        for size_bytes, pool in self._memory_pools.items():
            pool_stats = pool.get_stats()
            stats["memory_pools"][f"{size_bytes}_bytes"] = pool_stats
            
        # System statistics
        stats["system"] = {
            "memory_pressure": await self._get_memory_pressure(),
            "optimization_count": len(self._optimization_history),
            "last_optimization": self._last_optimization_time
        }
        
        return stats
        
    async def _initialize_default_pools(self):
        """Initialize default memory pools."""
        default_sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB to 256KB
        
        for size in default_sizes:
            pool = MemoryPool(
                block_size=size,
                max_blocks=self.config.global_memory_limit_mb * 1024 * 1024 // (size * 10),
                cleanup_interval=60.0
            )
            self._memory_pools[size] = pool
            
        logger.info(f"Initialized {len(default_sizes)} default memory pools")
        
    async def _optimization_loop(self):
        """Main optimization loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.optimization_interval_seconds)
                await self._run_optimization_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(30.0)  # Collect every 30 seconds
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                
    async def _run_optimization_cycle(self):
        """Run a complete optimization cycle for all buffers."""
        if not self.config.enabled:
            return
            
        optimization_start = time.time()
        optimized_count = 0
        
        for buffer_id in list(self._buffer_configs.keys()):
            try:
                result = await self.optimize_buffer(buffer_id)
                if result and result.size_change_bytes != 0:
                    optimized_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to optimize buffer '{buffer_id}': {e}")
                
        optimization_time = time.time() - optimization_start
        
        # Record optimization cycle
        self._optimization_history.append({
            "timestamp": optimization_start,
            "duration_seconds": optimization_time,
            "buffers_optimized": optimized_count,
            "total_buffers": len(self._buffer_configs)
        })
        
        self._last_optimization_time = optimization_start
        
        logger.debug(f"Optimization cycle completed: {optimized_count}/{len(self._buffer_configs)} buffers optimized in {optimization_time:.2f}s")
        
    async def _collect_system_metrics(self):
        """Collect system-level metrics for optimization decisions."""
        try:
            # Get system memory usage
            memory = psutil.virtual_memory()
            
            # Update buffer metrics with system data
            for buffer_id, metrics in self._buffer_metrics.items():
                # Add system context to buffer metrics
                metrics.timestamp = time.time()
                
                # Estimate memory waste
                if metrics.current_size > metrics.optimal_size:
                    metrics.memory_waste_bytes = metrics.current_size - metrics.optimal_size
                else:
                    metrics.memory_waste_bytes = 0
                    
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            
    async def _get_circuit_breaker_state(self, buffer_id: str) -> Optional[str]:
        """Get circuit breaker state for buffer-related services."""
        if not self._circuit_breaker_registry:
            return None
            
        try:
            # Map buffer to potential circuit breaker
            if "websocket" in buffer_id.lower() or "audio" in buffer_id.lower():
                cb = self._circuit_breaker_registry.get_circuit_breaker("websocket_pool")
                if cb:
                    return cb.state.name
                    
        except Exception as e:
            logger.warning(f"Failed to get circuit breaker state: {e}")
            
        return None
        
    async def _get_memory_pressure(self) -> float:
        """Get current system memory pressure (0.0 to 1.0)."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0
            
    def _find_best_memory_pool(self, size_bytes: int) -> Optional[MemoryPool]:
        """Find the best matching memory pool for the requested size."""
        # Find the smallest pool that can accommodate the size
        best_size = None
        for pool_size in self._memory_pools.keys():
            if pool_size >= size_bytes:
                if best_size is None or pool_size < best_size:
                    best_size = pool_size
                    
        return self._memory_pools.get(best_size) if best_size else None
        
    async def _create_memory_pool(self, size_bytes: int) -> Optional[MemoryPool]:
        """Create a new memory pool for the given size."""
        try:
            # Round up to next power of 2 or standard size
            pool_size = self._round_to_pool_size(size_bytes)
            
            if pool_size in self._memory_pools:
                return self._memory_pools[pool_size]
                
            # Calculate max blocks based on global memory limit
            max_blocks = min(100, self.config.global_memory_limit_mb * 1024 * 1024 // (pool_size * 10))
            
            pool = MemoryPool(pool_size, max_blocks)
            pool.start_cleanup_task()
            
            self._memory_pools[pool_size] = pool
            
            logger.info(f"Created new memory pool: {pool_size} bytes, {max_blocks} max blocks")
            return pool
            
        except Exception as e:
            logger.error(f"Failed to create memory pool for size {size_bytes}: {e}")
            return None
            
    def _round_to_pool_size(self, size_bytes: int) -> int:
        """Round size to appropriate pool size."""
        if size_bytes <= 1024:
            return 1024
        elif size_bytes <= 4096:
            return 4096
        elif size_bytes <= 16384:
            return 16384
        elif size_bytes <= 65536:
            return 65536
        else:
            # Round up to next 64KB boundary
            return ((size_bytes + 65535) // 65536) * 65536
            
    async def _apply_buffer_optimization(self, buffer_id: str, result: BufferSizingResult):
        """Apply buffer optimization result."""
        try:
            metrics = self._buffer_metrics[buffer_id]
            old_size = metrics.current_size
            
            # Update metrics
            metrics.current_size = result.recommended_size
            metrics.optimal_size = result.recommended_size
            
            # Log the change
            logger.info(f"Optimized buffer '{buffer_id}': {old_size} -> {result.recommended_size} bytes "
                       f"({result.size_change_percentage:+.1f}%) - {result.reasoning}")
                       
            # Publish metrics if enabled
            if self.config.publish_metrics_to_monitoring and self._performance_monitor:
                await self._publish_optimization_metrics(buffer_id, result)
                
        except Exception as e:
            logger.error(f"Failed to apply optimization for buffer '{buffer_id}': {e}")
            
    async def _publish_optimization_metrics(self, buffer_id: str, result: BufferSizingResult):
        """Publish optimization metrics to performance monitoring."""
        try:
            if not self._performance_monitor:
                return
                
            # Add custom metric for buffer optimization
            await self._performance_monitor.add_custom_metric(
                name=f"buffer_optimization_{buffer_id}",
                value=result.size_change_percentage,
                labels={
                    "buffer_id": buffer_id,
                    "reasoning": result.reasoning[:50],  # Truncate for label
                    "confidence": f"{result.confidence:.2f}"
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to publish optimization metrics: {e}")
            
    async def _maybe_trigger_optimization(self, buffer_id: str):
        """Trigger optimization if conditions are met."""
        metrics = self._buffer_metrics[buffer_id]
        config = self._buffer_configs[buffer_id]
        
        # Check for trigger conditions
        should_optimize = (
            metrics.buffer_overflows >= config.overflow_threshold or
            metrics.buffer_underruns >= config.underrun_threshold or
            metrics.average_latency_ms > config.latency_spike_threshold_ms
        )
        
        if should_optimize:
            logger.info(f"Triggering immediate optimization for buffer '{buffer_id}'")
            await self.optimize_buffer(buffer_id)


# Singleton access functions
async def initialize_buffer_manager(config: Optional[BufferOptimizationConfig] = None) -> BufferManager:
    """Initialize the global buffer manager."""
    global _buffer_manager_instance
    
    if _buffer_manager_instance is not None:
        await _buffer_manager_instance.stop()
        
    _buffer_manager_instance = BufferManager(config or BufferOptimizationConfig())
    await _buffer_manager_instance.start()
    
    return _buffer_manager_instance


def get_buffer_manager() -> Optional[BufferManager]:
    """Get the global buffer manager instance."""
    return _buffer_manager_instance


async def shutdown_buffer_manager():
    """Shutdown the global buffer manager."""
    global _buffer_manager_instance
    
    if _buffer_manager_instance:
        await _buffer_manager_instance.stop()
        _buffer_manager_instance = None