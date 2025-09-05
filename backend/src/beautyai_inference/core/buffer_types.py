"""
Buffer Optimization Types and Data Structures.

This module contains shared types, enums, and data classes used across
the buffer optimization system to provide efficient memory management,
adaptive sizing, and performance optimization.

Author: BeautyAI Framework
Date: September 5, 2025
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Callable
from collections import deque


class BufferType(Enum):
    """Types of buffers that can be managed."""
    AUDIO_INPUT = "audio_input"         # Audio input buffers from clients
    AUDIO_OUTPUT = "audio_output"       # Audio output buffers to clients  
    MESSAGE_QUEUE = "message_queue"     # WebSocket message queues
    MEMORY_POOL = "memory_pool"         # Reusable memory pools
    STREAM_BUFFER = "stream_buffer"     # Streaming data buffers
    JITTER_BUFFER = "jitter_buffer"     # Network jitter compensation


class BufferStrategy(Enum):
    """Buffer sizing strategies."""
    FIXED = "fixed"                     # Fixed buffer size
    ADAPTIVE = "adaptive"               # Adaptive based on metrics
    PREDICTIVE = "predictive"           # Predictive based on patterns
    CIRCUIT_AWARE = "circuit_aware"     # Circuit breaker aware sizing
    LOAD_BALANCED = "load_balanced"     # Load-based optimization


class SizingAlgorithm(Enum):
    """Algorithms for buffer size calculation."""
    LINEAR = "linear"                   # Linear scaling
    EXPONENTIAL = "exponential"         # Exponential scaling
    PERCENTILE_BASED = "percentile_based"  # Based on latency percentiles
    MOVING_AVERAGE = "moving_average"   # Moving average based
    MACHINE_LEARNING = "ml_based"       # ML-based prediction


class BufferState(Enum):
    """Buffer operational states."""
    OPTIMAL = "optimal"                 # Operating within optimal range
    UNDERUTILIZED = "underutilized"     # Buffer too large, memory waste
    CONGESTED = "congested"             # Buffer too small, performance impact
    UNSTABLE = "unstable"               # Frequent size adjustments needed
    CIRCUIT_PROTECTED = "circuit_protected"  # Protected by circuit breaker


@dataclass
class BufferMetrics:
    """Metrics for buffer performance tracking."""
    buffer_id: str
    buffer_type: BufferType
    
    # Size metrics
    current_size: int = 0
    max_size: int = 0
    min_size: int = 0
    optimal_size: int = 0
    
    # Usage metrics
    utilization_percentage: float = 0.0
    peak_utilization: float = 0.0
    average_utilization: float = 0.0
    
    # Performance metrics
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    
    # Efficiency metrics
    memory_waste_bytes: int = 0
    resize_frequency: float = 0.0
    last_resize_time: float = field(default_factory=time.time)
    
    # Health metrics  
    buffer_overflows: int = 0
    buffer_underruns: int = 0
    allocation_failures: int = 0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    

@dataclass
class BufferConfiguration:
    """Configuration for buffer optimization."""
    buffer_id: str
    buffer_type: BufferType
    strategy: BufferStrategy = BufferStrategy.ADAPTIVE
    algorithm: SizingAlgorithm = SizingAlgorithm.PERCENTILE_BASED
    
    # Size constraints
    min_size_bytes: int = 1024           # 1KB minimum
    max_size_bytes: int = 16777216       # 16MB maximum  
    initial_size_bytes: int = 65536      # 64KB initial
    
    # Adaptive parameters
    target_latency_ms: float = 100.0     # Target latency
    latency_tolerance_ms: float = 50.0   # Latency tolerance
    utilization_target: float = 0.75     # Target utilization (75%)
    utilization_tolerance: float = 0.15  # Utilization tolerance (±15%)
    
    # Resize behavior
    resize_threshold_percentage: float = 0.2   # Resize if 20% change needed
    resize_cooldown_seconds: float = 5.0       # Minimum time between resizes
    max_resize_factor: float = 2.0             # Maximum resize multiplier
    resize_increment_bytes: int = 4096         # Minimum resize increment (4KB)
    
    # Memory pool settings
    enable_memory_pooling: bool = True
    pool_block_size_bytes: int = 4096          # 4KB blocks
    pool_max_blocks: int = 1000                # Maximum pooled blocks
    pool_cleanup_interval_seconds: float = 60.0  # Pool cleanup interval
    
    # Circuit breaker integration
    circuit_breaker_aware: bool = True
    circuit_open_size_factor: float = 0.5      # Reduce size when circuit open
    circuit_recovery_size_factor: float = 1.2  # Increase size during recovery
    
    # Performance thresholds
    overflow_threshold: int = 5                # Max overflows before resize
    underrun_threshold: int = 3                # Max underruns before resize
    latency_spike_threshold_ms: float = 500.0  # Latency spike threshold
    

@dataclass 
class BufferSizingResult:
    """Result of buffer sizing calculation."""
    recommended_size: int
    size_change_bytes: int
    size_change_percentage: float
    reasoning: str
    confidence: float
    estimated_impact: Dict[str, float]  # Expected performance impact
    

@dataclass
class AudioBufferConfig:
    """Specialized configuration for audio buffers."""
    sample_rate: int = 16000             # Audio sample rate
    bit_depth: int = 16                  # Audio bit depth
    channels: int = 1                    # Audio channels (mono)
    chunk_size_ms: int = 20              # Audio chunk size in milliseconds
    
    # Jitter buffer settings
    jitter_buffer_target_ms: int = 60    # Target jitter buffer size
    jitter_buffer_max_ms: int = 200      # Maximum jitter buffer size
    jitter_buffer_adaptive: bool = True  # Enable adaptive jitter buffering
    
    # Quality settings
    quality_adaptation_enabled: bool = True
    min_quality_factor: float = 0.5      # Minimum quality during adaptation
    quality_recovery_time_ms: int = 2000 # Time to recover quality
    

@dataclass
class MemoryPoolBlock:
    """A block of memory in the pool."""
    block_id: str
    size_bytes: int
    allocated_at: float
    last_used: float
    reference_count: int = 0
    data: Optional[bytes] = None
    
    def is_available(self) -> bool:
        """Check if block is available for reuse."""
        return self.reference_count == 0
    
    def acquire(self):
        """Acquire a reference to this block."""
        self.reference_count += 1
        self.last_used = time.time()
    
    def release(self):
        """Release a reference to this block."""
        if self.reference_count > 0:
            self.reference_count -= 1


@dataclass
class BufferOptimizationConfig:
    """Overall configuration for buffer optimization system."""
    enabled: bool = True
    optimization_interval_seconds: float = 10.0
    metrics_collection_enabled: bool = True
    performance_monitoring_integration: bool = True
    
    # Global settings
    global_memory_limit_mb: int = 1024    # 1GB memory limit
    enable_memory_pressure_response: bool = True  
    memory_pressure_threshold: float = 0.85  # 85% memory usage threshold
    
    # Monitoring integration
    publish_metrics_to_monitoring: bool = True
    anomaly_detection_enabled: bool = True
    alert_on_optimization_failures: bool = True
    
    # Default buffer configurations
    default_audio_buffer: AudioBufferConfig = field(default_factory=AudioBufferConfig)
    default_message_queue_size: int = 100
    default_stream_buffer_size_kb: int = 64  # 64KB
    
    # Adaptive algorithms
    enable_predictive_sizing: bool = True
    learning_rate: float = 0.1
    prediction_window_seconds: int = 300  # 5 minutes
    
    # Performance targets
    target_p95_latency_ms: float = 200.0
    target_memory_efficiency: float = 0.8  # 80% efficiency target
    target_throughput_improvement: float = 0.15  # 15% improvement target