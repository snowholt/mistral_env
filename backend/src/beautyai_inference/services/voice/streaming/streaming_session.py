"""Streaming Session State (Phase 2)

Separated session state from endpoint file to prepare for incremental
decode loop and endpoint detection modules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .ring_buffer import PCMInt16RingBuffer


@dataclass
class StreamingSession:
    connection_id: str
    session_id: str
    language: str
    sample_rate: int = 16000
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    pcm_buffer: PCMInt16RingBuffer = field(default_factory=lambda: PCMInt16RingBuffer(16000, 40.0))
    total_frames_received: int = 0
    bytes_received: int = 0
    closed: bool = False

    def touch(self) -> None:
        self.last_activity = time.time()

    async def ingest_pcm(self, pcm_int16: bytes) -> None:
        await self.pcm_buffer.write(pcm_int16)
        self.bytes_received += len(pcm_int16)
        self.total_frames_received += 1
        self.touch()

__all__ = ["StreamingSession"]
