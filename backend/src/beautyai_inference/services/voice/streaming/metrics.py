"""Streaming Voice Metrics Aggregation (Phase 10)

Lightweight per-session rolling statistics. Not a global store.

Environment flag VOICE_STREAMING_METRICS_JSON=1 enables structured JSON
logging of periodic snapshots (emitted on final transcripts or at
interval events).
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
class SessionMetrics:
    session_id: str
    started_at: float = field(default_factory=time.time)
    decode_ms: _RollingStat = field(default_factory=lambda: _RollingStat("decode_ms"))
    cycle_latency_ms: _RollingStat = field(default_factory=lambda: _RollingStat("cycle_latency_ms"))
    end_silence_gap_ms: _RollingStat = field(default_factory=lambda: _RollingStat("end_silence_gap_ms"))
    final_transcripts: int = 0
    partial_transcripts: int = 0
    endpoints: int = 0

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

    def snapshot(self) -> Dict[str, Any]:
        return {
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
        }


METRICS_LOG_ENABLED = os.getenv("VOICE_STREAMING_METRICS_JSON", "0") == "1"


def maybe_log_structured(logger, tag: str, payload: Dict[str, Any]) -> None:  # pragma: no cover
    if not METRICS_LOG_ENABLED:
        return
    try:
        import json
        logger.info("%s %s", tag, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.debug("Failed structured metrics log", exc_info=True)

__all__ = ["SessionMetrics", "maybe_log_structured"]
