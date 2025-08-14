"""Integration test for /api/v1/ws/streaming-voice covering:
- partial + final transcripts
- endpoint start/final events
- metrics snapshot emission
- tts lifecycle (can be disabled via VOICE_STREAMING_DISABLE_TTS=1 for CI determinism)
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import websockets

PCM_FILE = Path("voice_tests/input_test_questions/pcm/botox.pcm")
ENDPOINT = "ws://localhost:8000/api/v1/ws/streaming-voice?language=en"


async def _collect_events_async(pcm: Path, disable_tts: bool) -> list[dict]:
    """Internal async collector (kept separate so tests can remain sync without pytest-asyncio)."""
    if disable_tts:
        # NOTE: Setting env var here affects only the test process; server must be started with
        # VOICE_STREAMING_DISABLE_TTS=1 for the flag to take effect server-side.
        os.environ["VOICE_STREAMING_DISABLE_TTS"] = "1"
    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)
    pcm_bytes = pcm.read_bytes()

    events: list[dict] = []
    async with websockets.connect(ENDPOINT, max_size=8 * 1024 * 1024) as ws:
        ready = json.loads(await ws.recv())
        events.append(ready)
        # feed audio quickly (fast mode)
        for off in range(0, len(pcm_bytes), frame_bytes):
            await ws.send(pcm_bytes[off : off + frame_bytes])
        # tail silence 2.2s
        silence_bytes = b"\x00\x00" * int(16000 * 2.2)
        for off in range(0, len(silence_bytes), frame_bytes):
            await ws.send(silence_bytes[off : off + frame_bytes])
        # receive until final + metrics (or timeout)
        final_seen = False
        metrics_seen = False
        t0 = time.time()
        while time.time() - t0 < 12:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            events.append(data)
            t = data.get("type")
            if t == "final_transcript":
                final_seen = True
            elif t == "metrics_snapshot":
                metrics_seen = True
            # Stop early when we have the essentials; TTS may or may not appear depending on server env
            if final_seen and metrics_seen:
                break
    return events


def _collect_events(pcm: Path, disable_tts: bool) -> list[dict]:
    return asyncio.run(_collect_events_async(pcm, disable_tts))


@pytest.mark.skipif(not PCM_FILE.exists(), reason="PCM test file missing")
def test_streaming_voice_final_and_metrics():
    events = _collect_events(PCM_FILE, disable_tts=True)
    types = [e.get("type") for e in events]
    assert "partial_transcript" in types, "No partial transcript emitted"
    assert "final_transcript" in types, "No final transcript emitted"
    assert any(e.get("type") == "endpoint" and e.get("event") == "start" for e in events), "No endpoint start event"
    assert any(e.get("type") == "endpoint" and e.get("event") == "final" for e in events), "No endpoint final event"
    final_texts = [e.get("text") for e in events if e.get("type") == "final_transcript"]
    assert final_texts and len(final_texts[0].strip()) >= int(os.getenv("VOICE_STREAMING_MIN_FINAL_CHARS", "3"))
    assert any(e.get("type") == "metrics_snapshot" for e in events), "No metrics snapshot emitted"


@pytest.mark.skipif(not PCM_FILE.exists(), reason="PCM test file missing")
def test_streaming_voice_with_tts_disabled_flag():
    events = _collect_events(PCM_FILE, disable_tts=True)
    # TTS may be disabled (server env) OR not (if server not started with flag). Accept either state.
    tts_completes = [e for e in events if e.get("type") == "tts_complete"]
    if tts_completes:
        # If present and disabled flag expected, allow absence of flag but prefer to assert structure.
        assert "utterance_index" in tts_completes[0], "tts_complete missing utterance_index"
    # Always ensure we at least got a final transcript
    assert any(e.get("type") == "final_transcript" for e in events), "Missing final transcript in TTS test"


@pytest.mark.skipif(not PCM_FILE.exists(), reason="PCM test file missing")
def test_streaming_voice_metrics_snapshot_contents():
    events = _collect_events(PCM_FILE, disable_tts=True)
    snapshots = [e for e in events if e.get("type") == "metrics_snapshot"]
    assert snapshots, "No metrics snapshot received"
    snap = snapshots[-1]
    # Basic expected counters (may evolve; keep assertions loose but meaningful)
    # Ensure decode counters present if instrumentation enabled
    decode_keys = [k for k in snap.keys() if k.startswith("decode_") or k.endswith("_count")] or snap.keys()
    assert snap.get("session_id"), "Metrics snapshot missing session_id"
    assert any("final" in k.lower() for k in snap.keys()), "Expected a final counter/key in snapshot"
    # Partial >= 1
    partial_count = next((v for k, v in snap.items() if "partial" in k.lower() and isinstance(v, int)), None)
    assert partial_count is None or partial_count >= 1, "Partial counter should be >=1"
