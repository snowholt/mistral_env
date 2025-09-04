"""Streaming Voice Metrics Aggregation (Phase 10 + Duplex Extensions)

Lightweight per-session rolling statistics. Not a global store.

Environment flag VOICE_STREAMING_METRICS_JSON=1 enables structured JSON
logging of periodic snapshots (emitted on final transcripts or at
interval events).

Phase 11 additions:
- TTS streaming metrics (first byte latency, duration, stalls)
- Duplex session metrics (barge-in count, echo correlation)
- Playback performance metrics (buffer underruns, jitter)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import time


@dataclass
class _RollingStat:
    name: str
    window: int = 50
    values: List[float] = field(default_factory=list)
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def update(self, v: float) -> None:
        self.count += 1
        self.total += v
        self.total_sq += v * v
        if v < self.min_v:
            self.min_v = v
        if v > self.max_v:
            self.max_v = v
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p90": None}
        mean = self.total / self.count
        vals = sorted(self.values)
        def _percentile(p: float) -> Optional[float]:
            if not vals:
                return None
            k = max(0, min(len(vals) - 1, int(round(p * (len(vals) - 1)))))
            return vals[k]
        return {
            "count": self.count,
            "mean": round(mean, 2),
            "min": round(self.min_v, 2),
            "max": round(self.max_v, 2),
            "p50": round(_percentile(0.5), 2) if vals else None,
            "p90": round(_percentile(0.9), 2) if vals else None,
        }


@dataclass 
class DuplexMetrics:
    """Additional metrics for full duplex streaming."""
    tts_first_byte_ms: _RollingStat = field(default_factory=lambda: _RollingStat("tts_first_byte_ms"))
    tts_stream_duration_ms: _RollingStat = field(default_factory=lambda: _RollingStat("tts_stream_duration_ms"))
    playback_stalls: int = 0
    barge_in_count: int = 0
    echo_correlation_score: _RollingStat = field(default_factory=lambda: _RollingStat("echo_correlation_score"))
    duplex_session_duration_s: Optional[float] = None
    tts_chunks_sent: int = 0
    tts_bytes_sent: int = 0
    buffer_underruns: int = 0
    jitter_buffer_size_ms: _RollingStat = field(default_factory=lambda: _RollingStat("jitter_buffer_size_ms"))
    
    def update_tts_first_byte(self, latency_ms: float) -> None:
        """Update TTS first byte latency metric."""
        self.tts_first_byte_ms.update(latency_ms)
    
    def update_tts_stream_duration(self, duration_ms: float) -> None:
        """Update TTS streaming duration metric."""
        self.tts_stream_duration_ms.update(duration_ms)
    
    def update_echo_correlation(self, correlation: float) -> None:
        """Update echo correlation score."""
        self.echo_correlation_score.update(correlation)
    
    def inc_playback_stall(self) -> None:
        """Increment playback stall counter."""
        self.playback_stalls += 1
    
    def inc_barge_in(self) -> None:
        """Increment barge-in counter."""
        self.barge_in_count += 1
    
    def inc_buffer_underrun(self) -> None:
        """Increment buffer underrun counter."""
        self.buffer_underruns += 1
    
    def update_tts_chunk(self, chunk_size_bytes: int) -> None:
        """Update TTS chunk metrics."""
        self.tts_chunks_sent += 1
        self.tts_bytes_sent += chunk_size_bytes
    
    def update_jitter_buffer_size(self, size_ms: float) -> None:
        """Update jitter buffer size metric."""
        self.jitter_buffer_size_ms.update(size_ms)
    
    def set_session_duration(self, duration_s: float) -> None:
        """Set total duplex session duration."""
        self.duplex_session_duration_s = duration_s
    
    def snapshot(self) -> Dict[str, Any]:
        """Get snapshot of duplex metrics."""
        return {
            "tts_first_byte_ms": self.tts_first_byte_ms.snapshot(),
            "tts_stream_duration_ms": self.tts_stream_duration_ms.snapshot(),
            "echo_correlation_score": self.echo_correlation_score.snapshot(),
            "jitter_buffer_size_ms": self.jitter_buffer_size_ms.snapshot(),
            "counts": {
                "playback_stalls": self.playback_stalls,
                "barge_ins": self.barge_in_count,
                "buffer_underruns": self.buffer_underruns,
                "tts_chunks_sent": self.tts_chunks_sent,
            },
            "totals": {
                "tts_bytes_sent": self.tts_bytes_sent,
                "duplex_session_duration_s": self.duplex_session_duration_s,
            }
        }


@dataclass
class SessionMetrics:
    session_id: str
    started_at: float = field(default_factory=time.time)
    decode_ms: _RollingStat = field(default_factory=lambda: _RollingStat("decode_ms"))
    cycle_latency_ms: _RollingStat = field(default_factory=lambda: _RollingStat("cycle_latency_ms"))
    end_silence_gap_ms: _RollingStat = field(default_factory=lambda: _RollingStat("end_silence_gap_ms"))
    final_transcripts: int = 0
    partial_transcripts: int = 0
    endpoints: int = 0
    
    # Duplex streaming extensions
    duplex_metrics: Optional[DuplexMetrics] = None
    duplex_enabled: bool = False

    def __post_init__(self):
        """Initialize duplex metrics if duplex mode is enabled."""
        if self.duplex_enabled:
            self.duplex_metrics = DuplexMetrics()

    def enable_duplex(self) -> None:
        """Enable duplex mode and initialize duplex metrics."""
        if not self.duplex_enabled:
            self.duplex_enabled = True
            self.duplex_metrics = DuplexMetrics()

    def update_perf_cycle(self, decode_ms: float, cycle_latency_ms: float) -> None:
        self.decode_ms.update(decode_ms)
        self.cycle_latency_ms.update(cycle_latency_ms)

    def update_endpoint(self, end_gap: Optional[float]) -> None:
        self.endpoints += 1
        if end_gap is not None:
            self.end_silence_gap_ms.update(end_gap)

    def inc_partial(self) -> None:
        self.partial_transcripts += 1

    def inc_final(self) -> None:
        self.final_transcripts += 1

    # Duplex metric update methods
    def update_tts_first_byte(self, latency_ms: float) -> None:
        """Update TTS first byte latency."""
        if self.duplex_metrics:
            self.duplex_metrics.update_tts_first_byte(latency_ms)
    
    def update_tts_stream_duration(self, duration_ms: float) -> None:
        """Update TTS stream duration."""
        if self.duplex_metrics:
            self.duplex_metrics.update_tts_stream_duration(duration_ms)
    
    def update_echo_correlation(self, correlation: float) -> None:
        """Update echo correlation score."""
        if self.duplex_metrics:
            self.duplex_metrics.update_echo_correlation(correlation)
    
    def inc_playback_stall(self) -> None:
        """Increment playback stall counter."""
        if self.duplex_metrics:
            self.duplex_metrics.inc_playback_stall()
    
    def inc_barge_in(self) -> None:
        """Increment barge-in counter."""
        if self.duplex_metrics:
            self.duplex_metrics.inc_barge_in()
    
    def inc_buffer_underrun(self) -> None:
        """Increment buffer underrun counter."""
        if self.duplex_metrics:
            self.duplex_metrics.inc_buffer_underrun()
    
    def update_tts_chunk(self, chunk_size_bytes: int) -> None:
        """Update TTS chunk sent metrics."""
        if self.duplex_metrics:
            self.duplex_metrics.update_tts_chunk(chunk_size_bytes)
    
    def update_jitter_buffer_size(self, size_ms: float) -> None:
        """Update jitter buffer size."""
        if self.duplex_metrics:
            self.duplex_metrics.update_jitter_buffer_size(size_ms)

    def snapshot(self) -> Dict[str, Any]:
        base_snapshot = {
            "session_id": self.session_id,
            "uptime_s": round(time.time() - self.started_at, 2),
            "decode_ms": self.decode_ms.snapshot(),
            "cycle_latency_ms": self.cycle_latency_ms.snapshot(),
            "end_silence_gap_ms": self.end_silence_gap_ms.snapshot(),
            "counts": {
                "partials": self.partial_transcripts,
                "finals": self.final_transcripts,
                "endpoints": self.endpoints,
            },
            "duplex_enabled": self.duplex_enabled,
        }
        
        # Add duplex metrics if available
        if self.duplex_metrics:
            base_snapshot["duplex"] = self.duplex_metrics.snapshot()
            # Update session duration for duplex metrics
            self.duplex_metrics.set_session_duration(time.time() - self.started_at)
        
        return base_snapshot


METRICS_LOG_ENABLED = os.getenv("VOICE_STREAMING_METRICS_JSON", "0") == "1"


def maybe_log_structured(logger, tag: str, payload: Dict[str, Any]) -> None:  # pragma: no cover
    if not METRICS_LOG_ENABLED:
        return
    try:
        import json
        logger.info("%s %s", tag, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.debug("Failed structured metrics log", exc_info=True)

__all__ = ["SessionMetrics", "DuplexMetrics", "maybe_log_structured"]
