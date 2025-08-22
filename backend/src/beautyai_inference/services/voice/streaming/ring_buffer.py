"""PCM Int16 Ring Buffer (Phase 2+)

Adds basic telemetry & lifecycle methods used by the streaming endpoint.

Features:
 - Fixed maximum duration (seconds) based on sample rate.
 - Write Int16 frames (bytes or memoryview) and maintain rolling position.
 - Read last N samples window for decode window assembly.
 - Compute rolling RMS (naive implementation, optimized later if needed).
 - Simple usage ratio + drop accounting (oversized writes collapse history).
 - Async close semantics so endpoint can signal shutdown to analyzers.

Design notes:
 - Raw int16 stored in bytearray.
 - Single producer, single consumer; protected by asyncio.Lock.
 - Metrics intentionally minimal until Phase 10 hardening.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class RingBufferStats:
    capacity_samples: int
    total_write_calls: int = 0
    total_samples_written: int = 0
    total_dropped_events: int = 0  # count of times an oversized write truncated history
    max_usage_ratio: float = 0.0
    total_rms_reads: int = 0


class PCMInt16RingBuffer:
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 40.0) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.capacity_samples = int(sample_rate * max_seconds)
        self.buffer = bytearray(self.capacity_samples * 2)  # int16 => 2 bytes
        self.write_index = 0  # sample index (0..capacity-1)
        self.total_written = 0  # cumulative samples written
        self.lock = asyncio.Lock()
        self._closed = False
        self.stats = RingBufferStats(capacity_samples=self.capacity_samples)

    @property
    def duration_filled_seconds(self) -> float:
        return min(self.total_written, self.capacity_samples) / self.sample_rate

    @property
    def is_full(self) -> bool:
        return self.total_written >= self.capacity_samples

    async def write(self, pcm_int16: bytes) -> None:
        if self._closed:
            return
        if len(pcm_int16) % 2 != 0:
            raise ValueError("PCM data length must be multiple of 2")
        samples = len(pcm_int16) // 2
        async with self.lock:
            if samples >= self.capacity_samples:
                # Keep only last capacity worth
                pcm_int16 = pcm_int16[-self.capacity_samples * 2 :]
                samples = self.capacity_samples
                self.stats.total_dropped_events += 1
            start = self.write_index % self.capacity_samples
            end = start + samples
            if end <= self.capacity_samples:
                self.buffer[start * 2 : end * 2] = pcm_int16
            else:  # wrap
                first = (self.capacity_samples - start) * 2
                self.buffer[start * 2 :] = pcm_int16[:first]
                self.buffer[: (samples * 2) - first] = pcm_int16[first:]
            self.write_index = (self.write_index + samples) % self.capacity_samples
            self.total_written += samples
            self.stats.total_write_calls += 1
            self.stats.total_samples_written += samples
            usage = self.usage_ratio()
            if usage > self.stats.max_usage_ratio:
                self.stats.max_usage_ratio = usage

    async def read_last_window(self, seconds: float) -> bytes:
        if seconds <= 0:
            return b""
        samples = int(seconds * self.sample_rate)
        async with self.lock:
            available = min(self.total_written, self.capacity_samples)
            samples = min(samples, available)
            if samples <= 0:
                return b""
            end = self.write_index % self.capacity_samples
            start = (end - samples) % self.capacity_samples
            if start < end:
                return bytes(self.buffer[start * 2 : end * 2])
            return bytes(self.buffer[start * 2 :] + self.buffer[: end * 2])

    async def read_all(self) -> bytes:
        return await self.read_last_window(self.max_seconds)

    async def clear(self) -> None:
        async with self.lock:
            self.write_index = 0
            self.total_written = 0

    async def reset_for_new_utterance(self) -> None:
        """
        Reset ring buffer for new utterance to prevent conversation bleeding.
        
        This method clears the audio history while preserving stats for debugging.
        Should be called after final transcript processing to ensure clean
        utterance boundaries.
        """
        async with self.lock:
            # Clear audio data
            self.write_index = 0
            self.total_written = 0
            # Keep stats for debugging but reset max usage
            self.stats.max_usage_ratio = 0.0
            # Note: Don't reset total stats as they're useful for session diagnostics

    async def rms(self, seconds: float) -> float:
        window = await self.read_last_window(seconds)
        if not window:
            return 0.0
        count = len(window) // 2
        if count == 0:
            return 0.0
        import struct
        samples = struct.unpack('<' + 'h' * count, window)
        acc = 0.0
        for s in samples:
            f = s / 32768.0
            acc += f * f
        self.stats.total_rms_reads += 1
        return math.sqrt(acc / count)

    def usage_ratio(self) -> float:
        # proportion of capacity currently containing valid historical data
        filled = min(self.total_written, self.capacity_samples)
        return filled / self.capacity_samples if self.capacity_samples else 0.0

    @property
    def closed(self) -> bool:  # used by endpoint cleanup
        return self._closed

    async def close(self) -> None:
        async with self.lock:
            self._closed = True

__all__ = ["PCMInt16RingBuffer", "RingBufferStats"]
