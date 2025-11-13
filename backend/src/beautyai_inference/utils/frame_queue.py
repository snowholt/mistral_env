"""
Frame Queue System for WebRTC Audio Processing

Bounded queue with drop-oldest policy and frame reordering for deterministic async processing.
Designed for low-latency real-time audio with predictable backpressure.

Author: BeautyAI Framework
Date: November 13, 2025
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import time


@dataclass
class FramePacket:
    """Minimal frame data for hot-path enqueue"""
    frame_index: int
    timestamp_mono: float  # Monotonic clock timestamp
    sample_rate: int
    
    # Raw audio data
    audio_48k_int16: np.ndarray  # Raw 48kHz PCM
    audio_16k_float32: np.ndarray  # Downsampled 16kHz (anti-alias filtered)
    
    # Lightweight metadata
    samples_48k: int
    samples_16k: int
    
    # For reordering results
    result: Optional[Dict[str, Any]] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None


@dataclass
class FrameQueueStats:
    """Statistics for monitoring queue health"""
    enqueued_count: int = 0
    dequeued_count: int = 0
    dropped_count: int = 0  # Dropped due to full queue
    peak_depth: int = 0
    current_depth: int = 0
    
    # Timing stats
    recv_deltas_ms: List[float] = field(default_factory=list)
    worker_service_times_ms: List[float] = field(default_factory=list)
    underrun_count: int = 0  # Inter-frame gap >30ms


class BoundedFrameQueue:
    """
    Thread-safe bounded queue with drop-oldest policy for audio frames.
    
    Design:
    - Fixed capacity (3-8 frames typical for 60-160ms buffering)
    - Drop-oldest when full (never block producer/recv loop)
    - Lock-free read for stats (atomic counters)
    - Reordering support via frame_index
    """
    
    def __init__(self, max_size: int = 5):
        """
        Args:
            max_size: Maximum queue depth (recommend 3-8 for <160ms buffer)
        """
        self.max_size = max_size
        self._queue = deque(maxlen=max_size)  # Auto-drops oldest when full
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = FrameQueueStats()
        
        # Reordering: Track results by frame_index
        self._results: Dict[int, FramePacket] = {}
        self._results_lock = threading.Lock()
        self._commit_cursor = 0  # Next frame_index to commit to disk
        
        # Last frame time for underrun detection
        self._last_enqueue_time: Optional[float] = None
    
    def enqueue(self, packet: FramePacket) -> bool:
        """
        Enqueue a frame packet (hot path - must be fast).
        
        Returns:
            True if enqueued, False if dropped (should not happen with maxlen deque)
        """
        now = time.monotonic()
        
        # Detect underruns (>30ms gap)
        if self._last_enqueue_time is not None:
            delta_ms = (now - self._last_enqueue_time) * 1000
            self.stats.recv_deltas_ms.append(delta_ms)
            if delta_ms > 30.0:
                self.stats.underrun_count += 1
        
        self._last_enqueue_time = now
        
        with self._lock:
            # Check if we're about to drop
            if len(self._queue) >= self.max_size:
                self.stats.dropped_count += 1
            
            self._queue.append(packet)
            self.stats.enqueued_count += 1
            self.stats.current_depth = len(self._queue)
            self.stats.peak_depth = max(self.stats.peak_depth, self.stats.current_depth)
        
        return True
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        """
        Dequeue a frame packet (worker path).
        
        Args:
            timeout: Max wait time in seconds (None = no wait)
        
        Returns:
            FramePacket or None if empty/timeout
        """
        start_time = time.monotonic()
        deadline = start_time + timeout if timeout else start_time
        
        while True:
            with self._lock:
                if self._queue:
                    packet = self._queue.popleft()
                    self.stats.dequeued_count += 1
                    self.stats.current_depth = len(self._queue)
                    packet.processing_start_time = time.monotonic()
                    return packet
            
            # Queue empty
            if timeout is None or time.monotonic() >= deadline:
                return None
            
            # Brief sleep before retry
            time.sleep(0.001)  # 1ms
    
    def commit_result(self, packet: FramePacket):
        """
        Commit a processed result for frame reordering.
        
        Args:
            packet: Processed packet with result dict populated
        """
        packet.processing_end_time = time.monotonic()
        
        # Track service time
        if packet.processing_start_time:
            service_time_ms = (packet.processing_end_time - packet.processing_start_time) * 1000
            self.stats.worker_service_times_ms.append(service_time_ms)
        
        with self._results_lock:
            self._results[packet.frame_index] = packet
    
    def get_contiguous_results(self) -> List[FramePacket]:
        """
        Get contiguous results from commit cursor onward.
        
        Returns:
            List of packets ready for disk flush (ordered by frame_index)
        """
        ready = []
        
        with self._results_lock:
            # Skip missing frames (dropped by queue)
            while self._commit_cursor not in self._results and self._results:
                # Find the minimum frame index available
                if self._results:
                    min_idx = min(self._results.keys())
                    if min_idx > self._commit_cursor:
                        # Frames were dropped, skip to first available
                        self._commit_cursor = min_idx
                        break
            
            # Collect contiguous frames
            while self._commit_cursor in self._results:
                packet = self._results.pop(self._commit_cursor)
                ready.append(packet)
                self._commit_cursor += 1
        
        return ready
    
    def get_stats_snapshot(self) -> Dict[str, Any]:
        """Get current queue statistics (safe to call from any thread)"""
        with self._lock:
            current_depth = len(self._queue)
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p
            f = int(k)
            c = min(f + 1, len(sorted_data) - 1)
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
        
        recv_deltas = self.stats.recv_deltas_ms[-1000:]  # Last 1000 frames
        service_times = self.stats.worker_service_times_ms[-1000:]
        
        return {
            "enqueued": self.stats.enqueued_count,
            "dequeued": self.stats.dequeued_count,
            "dropped": self.stats.dropped_count,
            "underruns": self.stats.underrun_count,
            "current_depth": current_depth,
            "peak_depth": self.stats.peak_depth,
            "commit_cursor": self._commit_cursor,
            "pending_results": len(self._results),
            "recv_delta_p50_ms": percentile(recv_deltas, 0.50),
            "recv_delta_p90_ms": percentile(recv_deltas, 0.90),
            "recv_delta_p99_ms": percentile(recv_deltas, 0.99),
            "worker_service_p50_ms": percentile(service_times, 0.50),
            "worker_service_p90_ms": percentile(service_times, 0.90),
            "worker_service_p99_ms": percentile(service_times, 0.99),
        }
    
    def clear(self):
        """Clear queue and reset stats"""
        with self._lock:
            self._queue.clear()
        with self._results_lock:
            self._results.clear()
            self._commit_cursor = 0
        
        self.stats = FrameQueueStats()
        self._last_enqueue_time = None
