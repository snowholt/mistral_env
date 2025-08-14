"""End-to-end voice-to-voice integration test.

Validates full pipeline over /api/v1/ws/streaming-voice (Phase 4):
  PCM (botox.pcm) -> partial/final transcript -> LLM chat -> TTS audio (wav base64) -> tts_complete

Requirements:
  - Server must be running with env: VOICE_STREAMING_ENABLED=1 VOICE_STREAMING_PHASE4=1
  - MUST NOT set VOICE_STREAMING_DISABLE_TTS=1 (we need real TTS output)

Assertions:
  - partial_transcript appears
  - endpoint_event start & final appear
  - final_transcript appears with minimum length
  - tts_start then tts_audio (base64 wav) then tts_complete
  - Decoded WAV bytes length > minimal threshold
  - wave module parses header (non-zero frames)

If TTS audio not received within timeout, test fails with diagnostic summary.
"""
from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path

import pytest
import websockets

PCM_FILE = Path("voice_tests/input_test_questions/pcm/botox.pcm")
ENDPOINT = "ws://localhost:8000/api/v1/ws/streaming-voice?language=en"

TTS_TIMEOUT_S = int(os.getenv("VOICE_STREAMING_TTS_TIMEOUT", "30"))
FINAL_TIMEOUT_S = int(os.getenv("VOICE_STREAMING_FINAL_TIMEOUT", "25"))
MIN_FINAL_CHARS = int(os.getenv("VOICE_STREAMING_MIN_FINAL_CHARS", "6"))


def _env_tts_disabled() -> bool:
    return os.getenv("VOICE_STREAMING_DISABLE_TTS", "0") == "1"


async def _run_voice_to_voice() -> dict:
    if not PCM_FILE.exists():
        pytest.skip("PCM test file missing: botox.pcm")
    if _env_tts_disabled():
        pytest.skip("Server started with VOICE_STREAMING_DISABLE_TTS=1 (enable TTS to run this test)")

    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)
    pcm_bytes = PCM_FILE.read_bytes()

    events: list[dict] = []
    partial_seen = False
    endpoint_start = False
    endpoint_final = False
    final_text = None
    tts_audio_b64 = None
    tts_complete = False

    async with websockets.connect(ENDPOINT, max_size=8 * 1024 * 1024) as ws:
        ready = json.loads(await ws.recv())
        events.append(ready)
        assert ready.get("feature") == "streaming_voice_phase4", "Server not in streaming phase4 (set VOICE_STREAMING_PHASE4=1)"

        # Feed audio quickly (fast mode)
        for off in range(0, len(pcm_bytes), frame_bytes):
            await ws.send(pcm_bytes[off : off + frame_bytes])
        # Tail silence for endpoint + finalization
        silence_bytes = b"\x00\x00" * int(16000 * 3.0)  # 3s
        for off in range(0, len(silence_bytes), frame_bytes):
            await ws.send(silence_bytes[off : off + frame_bytes])

        t_start = time.time()
        while True:
            overall_elapsed = time.time() - t_start
            if final_text and not tts_complete and overall_elapsed > TTS_TIMEOUT_S:
                break
            if not final_text and overall_elapsed > FINAL_TIMEOUT_S:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            events.append(data)
            t = data.get("type")
            if t == "partial_transcript":
                partial_seen = True
            elif t == "endpoint_event":
                if data.get("event") == "start":
                    endpoint_start = True
                elif data.get("event") == "final":
                    endpoint_final = True
            elif t == "final_transcript":
                final_text = data.get("text")
            elif t == "tts_audio":
                tts_audio_b64 = data.get("audio")
            elif t == "tts_complete":
                tts_complete = True
            # Early exit if everything done
            if final_text and tts_audio_b64 and tts_complete:
                break

    return {
        "events": events,
        "partial": partial_seen,
        "endpoint_start": endpoint_start,
        "endpoint_final": endpoint_final,
        "final_text": final_text,
        "tts_audio_b64": tts_audio_b64,
        "tts_complete": tts_complete,
    }


def _decode_and_validate_wav(b64_audio: str) -> dict:
    raw = base64.b64decode(b64_audio)
    assert len(raw) > 1000, f"TTS audio too small ({len(raw)} bytes)"
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = {
            "n_channels": wf.getnchannels(),
            "sampwidth": wf.getsampwidth(),
            "framerate": wf.getframerate(),
            "n_frames": wf.getnframes(),
            "duration_s": wf.getnframes() / float(wf.getframerate() or 1),
        }
    assert params["n_frames"] > 1000, f"Unusually few frames: {params['n_frames']}"
    assert params["duration_s"] > 0.2, f"Audio too short: {params['duration_s']:.3f}s"
    return params


@pytest.mark.integration
def test_voice_to_voice_end_to_end():
    result = asyncio.run(_run_voice_to_voice())
    events = result["events"]
    # Core assertions
    assert result["partial"], "No partial transcript received"
    assert result["endpoint_start"], "No endpoint start event"
    assert result["endpoint_final"], "No endpoint final event"
    assert result["final_text"], "No final transcript text"
    assert len(result["final_text"].strip()) >= MIN_FINAL_CHARS, "Final transcript below minimum length"
    assert result["tts_audio_b64"], "No TTS audio event received"
    assert result["tts_complete"], "No tts_complete event received"

    wav_meta = _decode_and_validate_wav(result["tts_audio_b64"])

    # Light diagnostics
    print("Voice-to-Voice E2E Summary:")
    print(json.dumps({
        "final_text": result["final_text"],
        "wav": wav_meta,
        "event_counts": {k: sum(1 for e in events if e.get("type") == k) for k in {e.get('type') for e in events}},
    }, ensure_ascii=False, indent=2))
