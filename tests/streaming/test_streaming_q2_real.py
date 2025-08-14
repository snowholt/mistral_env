"""Integration test for /api/v1/ws/streaming-voice using real audio (q2.wav).

Requirements / Intent:
 - Connect to a *running* backend server at localhost:8000 (not an in-process TestClient)
 - Send actual audio derived from `voice_tests/input_test_questions/q2.wav`
 - Ensure we are exercising REAL decoding (phase4) not mock loop; if mock detected -> xfail
 - Collect latency metrics: time to first partial, first final, tts start/audio/complete (if present)
 - Assert that at least one final transcript is produced and it's non‑trivial
 - Save rich JSON diagnostics for manual inspection

Notes:
 - Test is marked as "integration" and will SKIP if the server or feature isn't available
 - WAV must be 16kHz mono s16le. If not, test will be skipped (we do not resample here to avoid deps)
 - PCM frames of 20ms (320 samples / 640 bytes) are sent paced in *real time* by default
 - Set FAST_MODE=1 env var to disable pacing (burst send) during debugging
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None  # type: ignore


BACKEND_WS_URL = "ws://127.0.0.1:8000/api/v1/ws/streaming-voice?language=ar"
WAV_FILE = Path("/home/lumi/beautyai/voice_tests/input_test_questions/q2.wav")
RESULTS_PATH = Path(__file__).parent / "q2_streaming_results.json"


def test_q2_streaming_end_to_end() -> None:
    """Synchronous wrapper invoking async logic (no pytest-asyncio required)."""
    asyncio.run(_run_q2_streaming_test())


async def _run_q2_streaming_test() -> None:
    if websockets is None:
        pytest.skip("websockets library not installed in environment")

    if not WAV_FILE.exists():
        pytest.skip(f"Audio file not found: {WAV_FILE}")

    # Validate WAV format (expect 16kHz mono 16-bit)
    with wave.open(str(WAV_FILE), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        duration_s = nframes / float(sr)
        frames_bytes = wf.readframes(nframes)
    if not (nch == 1 and sampwidth == 2):
        pytest.skip(f"Unsupported WAV channel/width (need mono 16-bit) got channels={nch} sampwidth={sampwidth}")
    pcm_bytes = frames_bytes
    if sr != 16000:
        try:
            import audioop
            # Use ratecv to convert sample rate
            converted, _ = audioop.ratecv(pcm_bytes, sampwidth, nch, sr, 16000, None)
            pcm_bytes = converted
            print(f"[DEBUG] Resampled audio from {sr} Hz to 16000 Hz; duration_s≈{len(pcm_bytes)/2/16000:.2f}")
        except Exception as e:
            pytest.skip(f"Sample rate {sr} unsupported and resample failed: {e}")

    assert pcm_bytes, "Empty WAV payload"

    # Test will attempt up to 5 connection retries (server might still be starting)
    connect_ex: Optional[Exception] = None
    ws = None
    print(f"[DEBUG] Attempting connection to {BACKEND_WS_URL}")
    for attempt in range(5):
        try:
            ws = await websockets.connect(BACKEND_WS_URL, ping_interval=None)
            print(f"[DEBUG] Connected to {BACKEND_WS_URL} attempt={attempt+1}")
            break
        except Exception as e:  # pragma: no cover - network timing
            connect_ex = e
            await asyncio.sleep(1.0)
    if ws is None:
        print(f"[DEBUG] Failed all connection attempts: {connect_ex}")
        pytest.skip(f"Could not connect to backend websocket ({connect_ex})")

    # Runtime metrics containers
    ready_event: Dict[str, Any] | None = None
    partials: List[Dict[str, Any]] = []
    finals: List[Dict[str, Any]] = []
    tts_audio_events: List[Dict[str, Any]] = []
    tts_complete_events: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    first_partial_ts: Optional[float] = None
    first_final_ts: Optional[float] = None
    tts_start_ts: Optional[float] = None
    tts_audio_ts: Optional[float] = None
    tts_complete_ts: Optional[float] = None
    started = time.time()

    FRAME_MS = 20
    samples_per_frame = int(16000 * FRAME_MS / 1000)
    bytes_per_frame = samples_per_frame * 2
    fast_mode = os.getenv("FAST_MODE", "0") == "1"

    async def sender():
        # Ensure ready event observed before sending audio
        while ready_event is None:
            await asyncio.sleep(0.05)
        cursor = 0
        total_bytes = len(pcm_bytes)
        # Append 1s silence (16000 samples * 2 bytes)
        silence = b"\x00\x00" * 16000
        augmented = pcm_bytes + silence
        while cursor < len(augmented):
            chunk = augmented[cursor : cursor + bytes_per_frame]
            cursor += bytes_per_frame
            await ws.send(chunk)
            if not fast_mode:
                await asyncio.sleep(FRAME_MS / 1000)
        # Allow backend decode cycles to catch up
        await asyncio.sleep(2.0)
        # Close client side politely
        await ws.close()

    async def receiver():
        nonlocal ready_event, first_partial_ts, first_final_ts
        nonlocal tts_start_ts, tts_audio_ts, tts_complete_ts
        try:
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    break
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                etype = data.get("type")
                if etype == "ready":
                    ready_event = data
                elif etype == "partial_transcript":
                    partials.append(data)
                    if first_partial_ts is None:
                        first_partial_ts = time.time()
                elif etype == "final_transcript":
                    finals.append(data)
                    if first_final_ts is None:
                        first_final_ts = time.time()
                elif etype == "tts_start":
                    tts_start_ts = time.time()
                elif etype == "tts_audio":
                    tts_audio_events.append(data)
                    if tts_audio_ts is None:
                        tts_audio_ts = time.time()
                elif etype == "tts_complete":
                    tts_complete_events.append(data)
                    if tts_complete_ts is None:
                        tts_complete_ts = time.time()
                elif etype == "error":
                    errors.append(data)
        finally:
            pass

    # Start receiver first to capture ready
    recv_task = asyncio.create_task(receiver())
    send_task = asyncio.create_task(sender())
    await asyncio.wait({recv_task, send_task}, return_when=asyncio.ALL_COMPLETED)

    elapsed = time.time() - started

    # Persist diagnostics
    diagnostics = {
        "wav_file": str(WAV_FILE),
        "wav_duration_s": round(len(pcm_bytes) / 2 / 16000, 3),
        "ready": ready_event,
        "num_partials": len(partials),
        "num_finals": len(finals),
        "tts_audio_events": len(tts_audio_events),
        "tts_complete_events": len(tts_complete_events),
        "errors": errors,
        "timing_ms": {
            "elapsed_total": int(elapsed * 1000),
            "first_partial": int((first_partial_ts - started) * 1000) if first_partial_ts else None,
            "first_final": int((first_final_ts - started) * 1000) if first_final_ts else None,
            "tts_start": int((tts_start_ts - started) * 1000) if tts_start_ts else None,
            "tts_audio": int((tts_audio_ts - started) * 1000) if tts_audio_ts else None,
            "tts_complete": int((tts_complete_ts - started) * 1000) if tts_complete_ts else None,
        },
        "final_texts": [f.get("text") for f in finals],
        "phase": (ready_event or {}).get("feature"),
    }
    try:
        RESULTS_PATH.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    # ---- Assertions / Expectations ----
    if ready_event is None:
        print("[DEBUG] No 'ready' event received; diagnostics written to", RESULTS_PATH)
        # Attempt fetch of status endpoint for further context
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/api/v1/ws/streaming-voice/status", timeout=1) as resp:
                status_body = resp.read().decode('utf-8')
                print("[DEBUG] Status endpoint response:", status_body[:200])
        except Exception as e:  # pragma: no cover
            print("[DEBUG] Failed to query status endpoint:", e)
        pytest.skip("Did not receive ready event (server likely not phase4 / feature disabled)")

    feature = ready_event.get("feature", "")
    if feature.endswith("phase2"):
        pytest.xfail("Endpoint running in mock (phase2) mode; real decode unavailable")

    # Real path expectations
    assert finals, "No final_transcript events received"
    # Ensure at least one final transcript has some textual content (len>=4)
    assert any((f.get("text") or "").strip() and len((f.get("text") or "").strip()) >= 4 for f in finals), (
        "Final transcript(s) empty or too short: " + json.dumps([f.get("text") for f in finals], ensure_ascii=False)
    )

    # If TTS is enabled there should be at least one audio event (best-effort, do not hard fail if absent)
    if ready_event.get("feature", "").endswith("phase4") and not tts_audio_events:
        # Provide a soft assertion message for reporting later
        print("WARNING: No tts_audio events observed – TTS pipeline may not be initialized.")

    # Report any error events (fail hard if present)
    assert not errors, f"Received error events: {errors}"
