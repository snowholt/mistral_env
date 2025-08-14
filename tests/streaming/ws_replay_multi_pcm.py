#!/usr/bin/env python
"""Multi-utterance PCM replay harness for /api/v1/ws/streaming-voice.

Streams two PCM files (or the same one twice) separated by configurable tail silence
and inter-utterance silence. Collects events and prints a structured summary including
per-utterance finals, endpoint events, and basic timing metrics.

Usage:
  python tests/streaming/ws_replay_multi_pcm.py \
    --files voice_tests/input_test_questions/pcm/botox.pcm voice_tests/input_test_questions/pcm/botox.pcm \
    --language en --inter-silence-ms 1800 --tail-silence-ms 2500 --auto-close-seconds 18
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import websockets

DEFAULT_ENDPOINT = "ws://localhost:8000/api/v1/ws/streaming-voice"


def _read_pcm(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) % 2 != 0:
        print(f"[warn] PCM file {path} has odd length {len(data)} – truncating last byte", file=sys.stderr)
        data = data[:-1]
    return data

async def replay(files: list[Path], language: str, endpoint: str, frame_ms: int, inter_silence_ms: int, tail_silence_ms: int, auto_close_seconds: float, fast: bool):
    uri = f"{endpoint}?language={language}"
    events = []
    start_ts = time.time()
    partials = 0
    finals = 0
    endpoints = []

    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        ready = json.loads(await ws.recv())
        events.append(ready)
        decode_interval_ms = ready.get("decode_interval_ms", 480)
        frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)  # 16kHz * seconds * 2 bytes

        async def feeder():
            nonlocal finals
            for i, f in enumerate(files):
                pcm = _read_pcm(f)
                # chunk stream
                for off in range(0, len(pcm), frame_bytes):
                    chunk = pcm[off:off+frame_bytes]
                    if not chunk:
                        break
                    await ws.send(chunk)
                    await asyncio.sleep(0 if fast else frame_ms/1000.0)
                if i < len(files) - 1:
                    # inter-utterance silence
                    silence_bytes = b"\x00\x00" * int(16000 * (inter_silence_ms/1000.0))
                    for off in range(0, len(silence_bytes), frame_bytes):
                        await ws.send(silence_bytes[off:off+frame_bytes])
                        await asyncio.sleep(0 if fast else frame_ms/1000.0)
            # tail silence after last file
            silence_bytes = b"\x00\x00" * int(16000 * (tail_silence_ms/1000.0))
            for off in range(0, len(silence_bytes), frame_bytes):
                await ws.send(silence_bytes[off:off+frame_bytes])
                await asyncio.sleep(0 if fast else frame_ms/1000.0)

        async def receiver():
            nonlocal partials, finals
            last_final_time = None
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=auto_close_seconds)
                except asyncio.TimeoutError:
                    break
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                events.append(data)
                t = data.get("type")
                if t == "partial_transcript":
                    partials += 1
                elif t == "final_transcript":
                    finals += 1
                    last_final_time = time.time()
                elif t == "endpoint":
                    endpoints.append(data)
                # Auto close if both utterances finalized and some TTS completion (optional)
                if finals >= len(files) and (time.time() - start_ts) > 2:
                    # Give small grace for metrics snapshot
                    if last_final_time and (time.time() - last_final_time) > 1.0:
                        break

        feeder_task = asyncio.create_task(feeder())
        recv_task = asyncio.create_task(receiver())
        await asyncio.gather(feeder_task, recv_task)

    duration = time.time() - start_ts
    utterance_finals = [e for e in events if e.get("type") == "final_transcript"]
    summary = {
        "files": [str(f) for f in files],
        "language": language,
        "partials": partials,
        "finals": finals,
        "endpoints": len([e for e in events if e.get("type") == "endpoint" and e.get("event") == "final"]),
        "duration_s": round(duration, 3),
        "final_texts": [e.get("text") for e in utterance_finals],
    }
    print(json.dumps({"event": "multi_summary", **summary}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="List of PCM files (16kHz LE int16) to stream sequentially")
    ap.add_argument("--language", default="en")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--frame-ms", type=int, default=30)
    ap.add_argument("--inter-silence-ms", type=int, default=1500)
    ap.add_argument("--tail-silence-ms", type=int, default=2500)
    ap.add_argument("--auto-close-seconds", type=float, default=18.0)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            ap.error(f"File not found: {f}")

    asyncio.run(replay(files, args.language, args.endpoint, args.frame_ms, args.inter_silence_ms, args.tail_silence_ms, args.auto_close_seconds, args.fast))

if __name__ == "__main__":
    main()
