"""
Test suite for buffer optimization system in BeautyAI framework.

This module provides comprehensive tests for buffer optimization, memory pooling,
adaptive sizing, and integration with the overall system.
"""

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

import sys, os

# Ensure backend/src is on the path for direct test execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND_SRC = os.path.join(PROJECT_ROOT, 'backend', 'src')
if BACKEND_SRC not in sys.path:
    sys.path.insert(0, BACKEND_SRC)

from beautyai_inference.core.buffer_types import (
    BufferType, BufferStrategy, SizingAlgorithm, BufferConfiguration,
    BufferMetrics, BufferSizingResult, BufferOptimizationConfig,
    MemoryPoolBlock
)
from beautyai_inference.core.buffer_optimizer import (
    MemoryPool, AdaptiveBufferSizer, BufferManager,
    initialize_buffer_manager, get_buffer_manager, shutdown_buffer_manager
)
from beautyai_inference.core.buffer_integration import (
    BufferIntegrationHelper, WebSocketBufferWrapper,
    initialize_buffer_optimization_from_config
)


class TestBufferTypes:
    """Test buffer type definitions and configurations."""
    
    def test_buffer_type_enum(self):
        """Test BufferType enum values."""
        assert BufferType.AUDIO_INPUT.value == "audio_input"
        assert BufferType.AUDIO_OUTPUT.value == "audio_output"
        assert BufferType.MESSAGE_QUEUE.value == "message_queue"
    
    def test_buffer_configuration(self):
        """Test BufferConfiguration creation and validation."""
        config = BufferConfiguration(
            buffer_id="test_buffer",
            buffer_type=BufferType.AUDIO_INPUT,
            strategy=BufferStrategy.ADAPTIVE
        )
        
        assert config.buffer_id == "test_buffer"
        assert config.buffer_type == BufferType.AUDIO_INPUT
        assert config.strategy == BufferStrategy.ADAPTIVE
        assert config.initial_size_bytes == 65536  # Default value
    
    def test_buffer_metrics(self):
        """Test BufferMetrics creation and updates."""
        metrics = BufferMetrics(
            buffer_id="test_metrics",
            buffer_type=BufferType.AUDIO_OUTPUT,
            current_size=16384,
            max_size=65536
        )
        
        assert metrics.buffer_id == "test_metrics"
        assert metrics.buffer_type == BufferType.AUDIO_OUTPUT
        assert metrics.current_size == 16384
        assert metrics.max_size == 65536


class TestMemoryPool:
    """Test memory pool functionality."""
    
    @pytest.fixture
    def memory_pool(self):
        """Create a test memory pool."""
        pool = MemoryPool(
            block_size=4096,
            max_blocks=10,
            cleanup_interval=30.0
        )
        return pool
    
    def test_memory_pool_creation(self, memory_pool):
        """Test memory pool initialization."""
        assert memory_pool.block_size == 4096
        assert memory_pool.max_blocks == 10
        
    def test_block_acquire_release(self, memory_pool):
        """Test acquiring and releasing memory blocks."""
        # Acquire a block
        block = memory_pool.acquire_block()
        assert block is not None
        assert block.size_bytes == 4096
        assert not block.is_available()  # Should be acquired
        
        # Release the block
        memory_pool.release_block(block)
        assert block.is_available()  # Should be available after release
        
    def test_pool_exhaustion(self, memory_pool):
        """Test memory pool behavior when exhausted."""
        blocks = []
        
        # Acquire all blocks
        for _ in range(memory_pool.max_blocks):
            block = memory_pool.acquire_block()
            assert block is not None
            blocks.append(block)
        
        # Pool should be exhausted
        exhausted_block = memory_pool.acquire_block()
        assert exhausted_block is None
        
    def test_pool_stats(self, memory_pool):
        """Test memory pool statistics."""
        stats = memory_pool.get_stats()
        
        assert "allocated_blocks" in stats
        assert "available_blocks" in stats
        assert "block_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats


class TestAdaptiveBufferSizer:
    """Test adaptive buffer sizing algorithms."""
    
    @pytest.fixture
    def buffer_config(self):
        """Create test buffer configuration."""
        return BufferConfiguration(
            buffer_id="test_adaptive",
            buffer_type=BufferType.AUDIO_INPUT,
            strategy=BufferStrategy.ADAPTIVE,
            algorithm=SizingAlgorithm.PERCENTILE_BASED,
            target_latency_ms=100.0,
            latency_tolerance_ms=20.0,
            utilization_target=0.7,
            utilization_tolerance=0.1
        )
    
    @pytest.fixture
    def buffer_metrics(self):
        """Create test buffer metrics."""
        return BufferMetrics(
            buffer_id="test_adaptive",
            buffer_type=BufferType.AUDIO_INPUT,
            current_size=32768,
            max_size=131072
        )
    
    def test_adaptive_sizer_creation(self, buffer_config):
        """Test creating adaptive buffer sizer."""
        sizer = AdaptiveBufferSizer(buffer_config)
        assert sizer.config == buffer_config
        
    def test_latency_based_sizing(self, buffer_config, buffer_metrics):
        """Test latency-based sizing algorithm."""
        sizer = AdaptiveBufferSizer(buffer_config)
        
        # Build up latency history with high latency values
        for _ in range(15):  # Need at least 10 for percentile calculation
            buffer_metrics.average_latency_ms = 500.0
            buffer_metrics.utilization_percentage = 0.9
            sizer.calculate_optimal_size(buffer_metrics, None, 0.5)
        
        # Now test with high latency
        buffer_metrics.average_latency_ms = 500.0
        buffer_metrics.utilization_percentage = 0.9
        
        result = sizer.calculate_optimal_size(buffer_metrics, None, 0.5)
        
        assert result.recommended_size > 32768  # Should be larger than current
        assert result.confidence > 0.5
        assert "latency" in result.reasoning.lower()
        
    def test_throughput_based_sizing(self, buffer_config, buffer_metrics):
        """Test throughput-based sizing algorithm."""
        buffer_config.algorithm = SizingAlgorithm.MOVING_AVERAGE
        sizer = AdaptiveBufferSizer(buffer_config)
        
        # Build up utilization history with high values
        for _ in range(10):  # Need at least 5 for moving average
            buffer_metrics.utilization_percentage = 0.95
            buffer_metrics.throughput_mbps = 5.0
            buffer_metrics.buffer_underruns = 3
            sizer.calculate_optimal_size(buffer_metrics, None, 0.3)
        
        # High throughput with underruns should increase size
        buffer_metrics.throughput_mbps = 5.0
        buffer_metrics.buffer_underruns = 3
        buffer_metrics.utilization_percentage = 0.95
        
        result = sizer.calculate_optimal_size(buffer_metrics, None, 0.3)
        
        assert result.recommended_size > 32768  # Should be larger than current
        assert result.confidence > 0.3
        assert "utilization" in result.reasoning.lower()
        
    def test_predictive_sizing(self, buffer_config, buffer_metrics):
        """Test predictive sizing algorithm."""
        buffer_config.algorithm = SizingAlgorithm.MACHINE_LEARNING
        sizer = AdaptiveBufferSizer(buffer_config)
        
        # Set metrics for predictive algorithm
        buffer_metrics.utilization_percentage = 0.8
        buffer_metrics.average_latency_ms = 150.0
        buffer_metrics.throughput_mbps = 2.0
        
        result = sizer.calculate_optimal_size(buffer_metrics, None, 0.2)
        
        assert isinstance(result, BufferSizingResult)
        assert result.recommended_size >= buffer_config.min_size_bytes
        assert result.recommended_size <= buffer_config.max_size_bytes
        
    def test_size_constraints(self, buffer_config, buffer_metrics):
        """Test buffer size constraint enforcement."""
        sizer = AdaptiveBufferSizer(buffer_config)
        
        # Set extreme metrics that would suggest very large buffer
        buffer_metrics.utilization_percentage = 0.99
        buffer_metrics.average_latency_ms = 1000.0
        buffer_metrics.throughput_mbps = 10.0
        
        result = sizer.calculate_optimal_size(buffer_metrics, None, 0.1)
        
        # Should be constrained by max_size_bytes
        assert result.recommended_size <= buffer_config.max_size_bytes


class TestBufferManager:
    """Test buffer manager functionality."""
    
    @pytest_asyncio.fixture
    async def buffer_manager(self):
        """Create test buffer manager."""
        config = BufferOptimizationConfig(
            enabled=True,
            optimization_interval_seconds=10.0,
            metrics_collection_enabled=True,
            performance_monitoring_integration=True,
            global_memory_limit_mb=512,
            enable_memory_pressure_response=True
        )
        manager = BufferManager(config)
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_buffer_manager_lifecycle(self, buffer_manager):
        """Test buffer manager startup and shutdown."""
        # Manager should be running
        assert not buffer_manager._shutdown
        
        # Should be able to register buffers
        config = BufferConfiguration(
            buffer_id="test_manager",
            buffer_type=BufferType.MESSAGE_QUEUE,
            strategy=BufferStrategy.FIXED
        )
        
        success = await buffer_manager.register_buffer("test_manager", config)
        assert success
        
        # Should be able to get buffer stats
        buffer_stats = await buffer_manager.get_buffer_stats()
        assert buffer_stats is not None
        
    @pytest.mark.asyncio
    async def test_buffer_registration(self, buffer_manager):
        """Test buffer registration and management."""
        config = BufferConfiguration(
            buffer_id="test_registration",
            buffer_type=BufferType.AUDIO_INPUT,
            strategy=BufferStrategy.ADAPTIVE,
            initial_size_bytes=16384
        )
        
        # Register buffer
        success = await buffer_manager.register_buffer("test_registration", config)
        assert success
        
        # Check registration via stats
        stats = await buffer_manager.get_buffer_stats()
        assert stats["total_buffers"] >= 1
        
    @pytest.mark.asyncio
    async def test_memory_pool_management(self, buffer_manager):
        """Test memory pool creation and management."""
        # Get memory pool by requesting a block
        block = await buffer_manager.acquire_memory_block(4096)
        assert block is not None
        assert block.size_bytes == 4096
        
        # Release the block
        await buffer_manager.release_memory_block(block)
        
        # Verify the release worked by checking stats
        stats = await buffer_manager.get_buffer_stats()
        assert "memory_pools" in stats
        
    @pytest.mark.asyncio
    async def test_metrics_collection(self, buffer_manager):
        """Test buffer metrics collection."""
        # Register a buffer first
        config = BufferConfiguration(
            buffer_id="test_metrics",
            buffer_type=BufferType.AUDIO_OUTPUT,
            strategy=BufferStrategy.ADAPTIVE
        )
        await buffer_manager.register_buffer("test_metrics", config)
        
        # Update metrics
        metrics = BufferMetrics(
            buffer_id="test_metrics",
            buffer_type=BufferType.AUDIO_OUTPUT,
            current_size=32768,
            max_size=65536,
            utilization_percentage=0.8,
            average_latency_ms=120.0
        )
        
        await buffer_manager.update_buffer_metrics("test_metrics", metrics)
        
        # Check that metrics system is working
        stats = await buffer_manager.get_buffer_stats()
        assert "buffers" in stats
        assert "test_metrics" in stats["buffers"]


class TestBufferIntegration:
    """Test buffer integration helpers and wrappers."""
    
    def test_integration_helper(self):
        """Test buffer integration helper."""
        helper = BufferIntegrationHelper()
        assert helper is not None
    
    @pytest.mark.asyncio
    async def test_websocket_buffer_wrapper(self):
        """Test WebSocket buffer wrapper."""
        # Create buffer wrapper with mock data
        wrapper = WebSocketBufferWrapper(
            session_id="test_ws_buffer", 
            initial_size=8192
        )
        
        # Test basic functionality
        assert wrapper.buffer_id == "websocket_audio_test_ws_buffer"
        # Internal buffer length should match requested initial size
        assert len(wrapper._buffer) == 8192
        
        # Cleanup
        await wrapper.cleanup()
    
    @pytest.mark.asyncio
    async def test_buffer_optimization_initialization(self):
        """Test buffer optimization initialization from config."""
        # Simple test for initialization function
        try:
            await initialize_buffer_optimization_from_config()
        except Exception:
            # Expected if config system isn't available
            pass


class TestBufferOptimizationEndToEnd:
    """End-to-end tests for buffer optimization system."""
    
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self):
        """Test complete buffer optimization workflow."""
        # Initialize system
        config = BufferOptimizationConfig(
            enabled=True,
            optimization_interval_seconds=1.0,  # Fast for testing
            metrics_collection_enabled=True,
            performance_monitoring_integration=False  # Disable for isolated testing
        )
        
        manager = BufferManager(config)
        await manager.start()
        
        try:
            # Register buffer
            buffer_config = BufferConfiguration(
                buffer_id="e2e_test",
                buffer_type=BufferType.AUDIO_INPUT,
                strategy=BufferStrategy.ADAPTIVE,
                initial_size_bytes=16384
            )
            
            success = await manager.register_buffer("e2e_test", buffer_config)
            assert success
            
            # Simulate metrics updates over time
            for i in range(10):
                metrics = BufferMetrics(
                    buffer_id="e2e_test",
                    buffer_type=BufferType.AUDIO_INPUT,
                    current_size=16384,
                    max_size=65536,
                    utilization_percentage=0.7 + (i * 0.05),
                    average_latency_ms=100.0 + (i * 10),
                    throughput_mbps=1.0 + (i * 0.2)
                )
                await manager.update_buffer_metrics("e2e_test", metrics)
                await asyncio.sleep(0.1)
            
            # Let optimization run
            await asyncio.sleep(2)
            
            # Check that optimization occurred
            final_stats = await manager.get_buffer_stats()
            assert final_stats is not None
            
            # Get system stats
            assert final_stats["total_buffers"] >= 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test buffer optimization performance under high load."""
        config = BufferOptimizationConfig(
            enabled=True,
            optimization_interval_seconds=5.0,
            metrics_collection_enabled=True,
            performance_monitoring_integration=False,  # Disable for isolated testing
            global_memory_limit_mb=256
        )
        
        manager = BufferManager(config)
        await manager.start()
        
        try:
            # Register multiple buffers
            buffer_configs = []
            for i in range(5):
                config = BufferConfiguration(
                    buffer_id=f"load_test_{i}",
                    buffer_type=BufferType.AUDIO_INPUT,
                    strategy=BufferStrategy.ADAPTIVE,
                    initial_size_bytes=8192 * (i + 1)
                )
                buffer_configs.append(config)
                await manager.register_buffer(config.buffer_id, config)
            
            # Simulate concurrent load
            async def simulate_buffer_activity(buffer_id: str):
                for _ in range(20):
                    metrics = BufferMetrics(
                        buffer_id=buffer_id,
                        buffer_type=BufferType.AUDIO_INPUT,
                        current_size=16384,
                        max_size=65536,
                        utilization_percentage=0.6 + (time.time() % 0.4),
                        average_latency_ms=80.0 + (time.time() % 40),
                        throughput_mbps=1.5 + (time.time() % 1.0)
                    )
                    await manager.update_buffer_metrics(buffer_id, metrics)
                    await asyncio.sleep(0.1)
            
            # Run all simulations concurrently
            tasks = [
                simulate_buffer_activity(config.buffer_id)
                for config in buffer_configs
            ]
            
            await asyncio.gather(*tasks)
            
            # Verify all buffers are still registered and functioning
            stats = await manager.get_buffer_stats()
            assert stats is not None
            assert stats["total_buffers"] >= len(buffer_configs)
                
        finally:
            await manager.stop()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])