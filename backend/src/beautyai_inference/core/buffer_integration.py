"""
Buffer optimization integration utilities for BeautyAI framework.

This module provides utilities for integrating buffer optimization with
existing system components like WebSocket endpoints, TTS engines, and
performance monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .buffer_types import (
    BufferType, BufferStrategy, SizingAlgorithm, BufferConfiguration,
    BufferMetrics, AudioBufferConfig, BufferOptimizationConfig
)
from .buffer_optimizer import BufferManager, get_buffer_manager
from .config_manager import get_config_manager


logger = logging.getLogger(__name__)


class BufferIntegrationHelper:
    """
    Helper class for integrating buffer optimization with system components.
    """
    
    @staticmethod
    async def register_websocket_buffers(manager: BufferManager, session_id: str) -> bool:
        """Register WebSocket-related buffers for a session."""
        try:
            config_manager = get_config_manager()
            if not config_manager:
                logger.error("Config manager not available")
                return False
                
            config_dict = await config_manager.get_config()
            buffer_config = config_dict.get("buffer_optimization", {})
            
            # Register WebSocket audio buffer
            ws_buffer_config = buffer_config.get("websocket_buffer", {})
            websocket_buffer_config = BufferConfiguration(
                buffer_id=f"websocket_audio_{session_id}",
                buffer_type=BufferType(ws_buffer_config.get("buffer_type", "audio_stream")),
                strategy=BufferStrategy(ws_buffer_config.get("strategy", "adaptive")),
                sizing_algorithm=SizingAlgorithm(ws_buffer_config.get("sizing_algorithm", "predictive_based")),
                initial_size_bytes=ws_buffer_config.get("initial_size_bytes", 65536),
                min_size_bytes=ws_buffer_config.get("min_size_bytes", 16384),
                max_size_bytes=ws_buffer_config.get("max_size_bytes", 262144),
                growth_factor=ws_buffer_config.get("growth_factor", 1.5),
                shrink_factor=ws_buffer_config.get("shrink_factor", 0.8),
                resize_threshold_percentage=ws_buffer_config.get("resize_threshold_percentage", 10.0),
                overflow_threshold=ws_buffer_config.get("overflow_threshold", 3),
                underrun_threshold=ws_buffer_config.get("underrun_threshold", 5),
                latency_spike_threshold_ms=ws_buffer_config.get("latency_spike_threshold_ms", 500.0),
                utilization_threshold=ws_buffer_config.get("utilization_threshold", 0.8)
            )
            
            success = await manager.register_buffer(
                f"websocket_audio_{session_id}",
                websocket_buffer_config
            )
            
            if success:
                logger.info(f"Registered WebSocket buffers for session {session_id}")
            else:
                logger.error(f"Failed to register WebSocket buffers for session {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error registering WebSocket buffers for session {session_id}: {e}")
            return False
            
    @staticmethod
    async def register_audio_buffers(manager: BufferManager, component_id: str) -> bool:
        """Register audio processing buffers for a component."""
        try:
            config_manager = get_config_manager()
            if not config_manager:
                logger.error("Config manager not available")
                return False
                
            config_dict = await config_manager.get_config()
            buffer_config = config_dict.get("buffer_optimization", {})
            
            # Register audio buffer
            audio_buffer_config_dict = buffer_config.get("audio_buffer", {})
            audio_buffer_config = BufferConfiguration(
                buffer_id=f"audio_{component_id}",
                buffer_type=BufferType(audio_buffer_config_dict.get("buffer_type", "audio_stream")),
                strategy=BufferStrategy(audio_buffer_config_dict.get("strategy", "performance_optimized")),
                sizing_algorithm=SizingAlgorithm(audio_buffer_config_dict.get("sizing_algorithm", "latency_based")),
                initial_size_bytes=audio_buffer_config_dict.get("initial_size_bytes", 32768),
                min_size_bytes=audio_buffer_config_dict.get("min_size_bytes", 8192),
                max_size_bytes=audio_buffer_config_dict.get("max_size_bytes", 131072),
                growth_factor=audio_buffer_config_dict.get("growth_factor", 1.3),
                shrink_factor=audio_buffer_config_dict.get("shrink_factor", 0.75),
                resize_threshold_percentage=audio_buffer_config_dict.get("resize_threshold_percentage", 15.0),
                overflow_threshold=audio_buffer_config_dict.get("overflow_threshold", 2),
                underrun_threshold=audio_buffer_config_dict.get("underrun_threshold", 3),
                latency_spike_threshold_ms=audio_buffer_config_dict.get("latency_spike_threshold_ms", 200.0),
                utilization_threshold=audio_buffer_config_dict.get("utilization_threshold", 0.75)
            )
            
            success = await manager.register_buffer(
                f"audio_{component_id}",
                audio_buffer_config
            )
            
            if success:
                logger.info(f"Registered audio buffers for component {component_id}")
            else:
                logger.error(f"Failed to register audio buffers for component {component_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error registering audio buffers for component {component_id}: {e}")
            return False
            
    @staticmethod
    async def register_message_queue_buffers(manager: BufferManager, component_id: str) -> bool:
        """Register message queue buffers for a component."""
        try:
            config_manager = get_config_manager()
            if not config_manager:
                logger.error("Config manager not available")
                return False
                
            config_dict = await config_manager.get_config()
            buffer_config = config_dict.get("buffer_optimization", {})
            
            # Register message queue buffer
            mq_buffer_config_dict = buffer_config.get("message_queue", {})
            message_queue_config = BufferConfiguration(
                buffer_id=f"message_queue_{component_id}",
                buffer_type=BufferType(mq_buffer_config_dict.get("buffer_type", "message_queue")),
                strategy=BufferStrategy(mq_buffer_config_dict.get("strategy", "memory_efficient")),
                sizing_algorithm=SizingAlgorithm(mq_buffer_config_dict.get("sizing_algorithm", "throughput_based")),
                initial_size_bytes=mq_buffer_config_dict.get("initial_size_bytes", 16384),
                min_size_bytes=mq_buffer_config_dict.get("min_size_bytes", 4096),
                max_size_bytes=mq_buffer_config_dict.get("max_size_bytes", 65536),
                growth_factor=mq_buffer_config_dict.get("growth_factor", 1.4),
                shrink_factor=mq_buffer_config_dict.get("shrink_factor", 0.7),
                resize_threshold_percentage=mq_buffer_config_dict.get("resize_threshold_percentage", 20.0),
                overflow_threshold=mq_buffer_config_dict.get("overflow_threshold", 5),
                underrun_threshold=mq_buffer_config_dict.get("underrun_threshold", 10),
                latency_spike_threshold_ms=mq_buffer_config_dict.get("latency_spike_threshold_ms", 1000.0),
                utilization_threshold=mq_buffer_config_dict.get("utilization_threshold", 0.85)
            )
            
            success = await manager.register_buffer(
                f"message_queue_{component_id}",
                message_queue_config
            )
            
            if success:
                logger.info(f"Registered message queue buffers for component {component_id}")
            else:
                logger.error(f"Failed to register message queue buffers for component {component_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error registering message queue buffers for component {component_id}: {e}")
            return False
            
    @staticmethod
    async def update_buffer_metrics_from_websocket(
        buffer_id: str,
        bytes_processed: int,
        processing_time_ms: float,
        queue_size: int,
        overflows: int = 0,
        underruns: int = 0
    ) -> bool:
        """Update buffer metrics from WebSocket component."""
        try:
            manager = get_buffer_manager()
            if not manager:
                return False
                
            metrics_update = {
                "bytes_processed": bytes_processed,
                "processing_time_ms": processing_time_ms,
                "queue_size": queue_size,
                "buffer_overflows": overflows,
                "buffer_underruns": underruns,
                "utilization_percentage": min(1.0, queue_size / 1000.0),  # Estimate
                "throughput_mbps": (bytes_processed * 8) / (processing_time_ms * 1000) if processing_time_ms > 0 else 0,
                "average_latency_ms": processing_time_ms
            }
            
            return await manager.update_buffer_metrics(buffer_id, metrics_update)
            
        except Exception as e:
            logger.error(f"Error updating WebSocket buffer metrics: {e}")
            return False
            
    @staticmethod
    async def update_buffer_metrics_from_audio(
        buffer_id: str,
        audio_bytes: int,
        latency_ms: float,
        buffer_size: int,
        dropouts: int = 0
    ) -> bool:
        """Update buffer metrics from audio component."""
        try:
            manager = get_buffer_manager()
            if not manager:
                return False
                
            metrics_update = {
                "bytes_processed": audio_bytes,
                "average_latency_ms": latency_ms,
                "current_size": buffer_size,
                "buffer_underruns": dropouts,
                "utilization_percentage": min(1.0, audio_bytes / buffer_size) if buffer_size > 0 else 0,
                "throughput_mbps": (audio_bytes * 8) / (latency_ms * 1000) if latency_ms > 0 else 0
            }
            
            return await manager.update_buffer_metrics(buffer_id, metrics_update)
            
        except Exception as e:
            logger.error(f"Error updating audio buffer metrics: {e}")
            return False
            
    @staticmethod
    async def get_optimal_buffer_size(buffer_id: str) -> Optional[int]:
        """Get optimal buffer size for a component."""
        try:
            manager = get_buffer_manager()
            if not manager:
                return None
                
            return await manager.get_optimal_buffer_size(buffer_id)
            
        except Exception as e:
            logger.error(f"Error getting optimal buffer size: {e}")
            return None
            
    @staticmethod
    async def unregister_session_buffers(session_id: str) -> bool:
        """Unregister all buffers for a session."""
        try:
            manager = get_buffer_manager()
            if not manager:
                return False
                
            # Unregister buffers associated with the session
            buffer_ids = [
                f"websocket_audio_{session_id}",
                f"audio_{session_id}",
                f"message_queue_{session_id}"
            ]
            
            success = True
            for buffer_id in buffer_ids:
                result = await manager.unregister_buffer(buffer_id)
                if not result:
                    success = False
                    
            if success:
                logger.info(f"Unregistered all buffers for session {session_id}")
            else:
                logger.warning(f"Some buffers failed to unregister for session {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error unregistering session buffers: {e}")
            return False


class WebSocketBufferWrapper:
    """
    Buffer wrapper for WebSocket components with optimization integration.
    """
    
    def __init__(self, session_id: str, initial_size: int = 65536):
        self.session_id = session_id
        self.buffer_id = f"websocket_audio_{session_id}"
        self._buffer = bytearray(initial_size)
        self._write_pos = 0
        self._read_pos = 0
        self._overflows = 0
        self._underruns = 0
        self._last_metrics_update = 0
        
    async def write(self, data: bytes) -> bool:
        """Write data to buffer with overflow detection."""
        try:
            if len(data) + self._write_pos > len(self._buffer):
                # Check if we can get optimal size
                optimal_size = await BufferIntegrationHelper.get_optimal_buffer_size(self.buffer_id)
                
                if optimal_size and optimal_size > len(self._buffer):
                    # Resize buffer
                    new_buffer = bytearray(optimal_size)
                    available_data = self._write_pos - self._read_pos
                    if available_data > 0:
                        new_buffer[:available_data] = self._buffer[self._read_pos:self._write_pos]
                    self._buffer = new_buffer
                    self._write_pos = available_data
                    self._read_pos = 0
                    logger.debug(f"Resized buffer to {optimal_size} bytes")
                else:
                    # Overflow
                    self._overflows += 1
                    return False
                    
            # Write data
            end_pos = self._write_pos + len(data)
            self._buffer[self._write_pos:end_pos] = data
            self._write_pos = end_pos
            
            # Update metrics periodically
            await self._maybe_update_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Buffer write error: {e}")
            return False
            
    async def read(self, size: int) -> Optional[bytes]:
        """Read data from buffer with underrun detection."""
        try:
            available = self._write_pos - self._read_pos
            
            if available < size:
                self._underruns += 1
                if available == 0:
                    return None
                size = available
                
            # Read data
            data = bytes(self._buffer[self._read_pos:self._read_pos + size])
            self._read_pos += size
            
            # Reset positions if buffer is empty
            if self._read_pos >= self._write_pos:
                self._read_pos = 0
                self._write_pos = 0
                
            return data
            
        except Exception as e:
            logger.error(f"Buffer read error: {e}")
            return None
            
    async def _maybe_update_metrics(self):
        """Update buffer metrics periodically."""
        import time
        
        now = time.time()
        if now - self._last_metrics_update > 5.0:  # Every 5 seconds
            await BufferIntegrationHelper.update_buffer_metrics_from_websocket(
                self.buffer_id,
                bytes_processed=self._write_pos,
                processing_time_ms=50.0,  # Estimate
                queue_size=self._write_pos - self._read_pos,
                overflows=self._overflows,
                underruns=self._underruns
            )
            self._last_metrics_update = now
            
    def get_available_data(self) -> int:
        """Get amount of available data."""
        return self._write_pos - self._read_pos
        
    def get_available_space(self) -> int:
        """Get amount of available space."""
        return len(self._buffer) - self._write_pos
        
    async def cleanup(self):
        """Cleanup buffer resources."""
        await BufferIntegrationHelper.unregister_session_buffers(self.session_id)


async def initialize_buffer_optimization_from_config() -> Optional[BufferManager]:
    """Initialize buffer optimization system from configuration."""
    try:
        config_manager = get_config_manager()
        if not config_manager:
            logger.error("Config manager not available for buffer optimization")
            return None
            
        config_dict = await config_manager.get_config()
        buffer_config = config_dict.get("buffer_optimization", {})
        
        if not buffer_config.get("enabled", False):
            logger.info("Buffer optimization is disabled")
            return None
            
        # Create configuration object
        optimization_config = BufferOptimizationConfig(
            enabled=buffer_config.get("enabled", True),
            optimization_interval_seconds=buffer_config.get("optimization_interval_seconds", 60.0),
            metrics_collection_enabled=buffer_config.get("metrics_collection_enabled", True),
            publish_metrics_to_monitoring=buffer_config.get("publish_metrics_to_monitoring", True),
            global_memory_limit_mb=buffer_config.get("global_memory_limit_mb", 256)
        )
        
        # Initialize buffer manager
        from .buffer_optimizer import initialize_buffer_manager
        manager = await initialize_buffer_manager(optimization_config)
        
        logger.info("Buffer optimization system initialized successfully")
        return manager
        
    except Exception as e:
        logger.error(f"Failed to initialize buffer optimization: {e}")
        return None