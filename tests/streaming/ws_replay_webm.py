#!/usr/bin/env python3
"""Replay a .webm (Opus) file to the streaming voice WebSocket endpoint.

This utility mirrors ws_replay_pcm.py but first decodes the provided WebM file
to 16kHz mono signed 16-bit PCM frames using ffmpeg (must be installed and on PATH).

Key features:
  - Decodes entire WebM to raw PCM bytes in-memory (simple & robust baseline).
  - Splits into frame-sized chunks (default 20ms) and streams as binary messages.
  - Supports real-time pacing or burst mode (--fast).
  - Appends trailing silence to encourage endpoint detection.
  - Captures server JSON events (stdout) and optionally writes them to a JSONL file.
  - Provides summary metrics (first partial latency, final latency, decode perf).

Usage:
  python ws_replay_webm.py \
      --file voice_tests/input_test_questions/webm/q6.webm \
      --language ar \
      --save-events reports/logs/webm_q6_events.jsonl

Prerequisites:
  - Backend server running with VOICE_STREAMING_ENABLED=1.
  - ffmpeg installed (`ffmpeg -version`).
  - Python package 'websockets' installed in the active environment.

Notes / Future Enhancements:
  - For very large files, consider streaming decode (ffmpeg -f s16le pipe) while sending.
  - Could add on-the-fly VAD segmentation before sending for more realistic incremental tests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
import contextlib
from pathlib import Path
from statistics import mean
from typing import Optional

try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover
    print("Please install websockets in backend venv: pip install websockets", file=sys.stderr)
    sys.exit(1)


def decode_webm_to_pcm(path: Path, sample_rate: int = 16000) -> bytes:
    """Decode a WebM (likely Opus) file to raw s16le mono PCM using ffmpeg.

    Returns raw PCM bytes. Raises RuntimeError if ffmpeg is missing or fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v","error",  # suppress non-error logs
        "-i", str(path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:  # ffmpeg missing
        raise RuntimeError("ffmpeg not found on PATH; please install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg decode failed: {e.stderr.decode(errors='ignore')}") from e
    return proc.stdout


async def replay_webm(
    file: Path,
    language: str,
    frame_ms: int,
    host: str,
    fast: bool,
    loop_count: int,
    auto_close_seconds: float,
    tail_silence_ms: int,
    save_events: Optional[Path],
    reference: Optional[str],
) -> None:
    # Decode
    t0 = time.time()
    pcm = decode_webm_to_pcm(file)
    decode_elapsed = time.time() - t0
    samples = len(pcm) // 2
    sample_rate = 16000
    audio_duration_s = samples / sample_rate
    samples_per_frame = int(sample_rate * frame_ms / 1000)
    total_frames = math.ceil(samples / samples_per_frame)
    url = f"ws://{host}/api/v1/ws/streaming-voice?language={language}"

    if save_events:
        save_events.parent.mkdir(parents=True, exist_ok=True)
        events_fp = open(save_events, "w", encoding="utf-8")
    else:
        events_fp = None

    def emit(obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        print(line)
        if events_fp:
            events_fp.write(line + "\n")
            events_fp.flush()

    emit({
        "event": "info",
        "message": "connecting",
        "url": url,
        "file": str(file),
        "decoded_seconds": round(audio_duration_s, 3),
        "decode_ms": int(decode_elapsed * 1000),
        "frames": total_frames,
        "frame_ms": frame_ms,
    })

    first_partial_ts = None
    final_ts = None
    start = time.time()
    decode_ms_values = []
    final_text = None

    async with websockets.connect(url, ping_interval=None) as ws:
        final_event = asyncio.Event()
        tts_complete_event = asyncio.Event()

        async def sender():
            for loop_i in range(loop_count):
                cursor = 0
                while cursor < len(pcm):
                    chunk = pcm[cursor: cursor + samples_per_frame * 2]
                    cursor += samples_per_frame * 2
                    if not chunk:
                        break
                    await ws.send(chunk)
                    if not fast:
                        await asyncio.sleep(frame_ms / 1000)
                if loop_i + 1 < loop_count:
                    await asyncio.sleep(0.4)
            # trailing silence
            if tail_silence_ms > 0:
                silence_frames = max(1, int(tail_silence_ms / frame_ms))
                silence_chunk = b"\x00\x00" * samples_per_frame
                for _ in range(silence_frames):
                    await ws.send(silence_chunk)
                    if not fast:
                        await asyncio.sleep(frame_ms / 1000)
            if auto_close_seconds > 0:
                try:
                    await asyncio.wait_for(tts_complete_event.wait(), timeout=auto_close_seconds)
                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(0.25)
                if not getattr(ws, 'closed', False):  # best-effort close
                    with contextlib.suppress(Exception):
                        await ws.close()

        async def receiver():
            nonlocal first_partial_ts, final_ts, final_text
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    break
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                emit(data)
                etype = data.get("type")
                if etype == "partial_transcript" and first_partial_ts is None:
                    first_partial_ts = time.time()
                if etype == "final_transcript":
                    final_ts = time.time()
                    final_text = data.get("text")
                    final_event.set()
                if etype == "tts_complete":
                    tts_complete_event.set()
                if etype == "perf_cycle" and "decode_ms" in data:
                    decode_ms_values.append(data["decode_ms"])
                if etype == "error":  # abort on error
                    break

        await asyncio.gather(sender(), receiver())

    end = time.time()
    summary = {
        "event": "summary",
        "file": file.name,
        "audio_seconds": round(audio_duration_s, 3),
        "duration_s": round(end - start, 3),
        "first_partial_ms": int(1000 * (first_partial_ts - start)) if first_partial_ts else None,
        "final_ms": int(1000 * (final_ts - start)) if final_ts else None,
        "avg_decode_ms": int(mean(decode_ms_values)) if decode_ms_values else None,
        "max_decode_ms": max(decode_ms_values) if decode_ms_values else None,
        "final_text": final_text,
    }
    emit(summary)
    if events_fp:
        events_fp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to .webm audio file')
    ap.add_argument('--language', default='ar')
    ap.add_argument('--frame-ms', type=int, default=20)
    ap.add_argument('--host', default='localhost:8000')
    ap.add_argument('--fast', action='store_true', help='Disable pacing (burst send)')
    ap.add_argument('--loop', type=int, default=1)
    ap.add_argument('--auto-close-seconds', type=float, default=6.0)
    ap.add_argument('--tail-silence-ms', type=int, default=1200)
    ap.add_argument('--save-events', help='Path to JSONL file to append all events')
    ap.add_argument('--reference', help='(Reserved) reference transcript for future WER calc')
    args = ap.parse_args()

    file = Path(args.file)
    if not file.exists():
        print(f"File not found: {file}", file=sys.stderr)
        sys.exit(1)

    save_path = Path(args.save_events) if args.save_events else None
    try:
        asyncio.run(replay_webm(
            file=file,
            language=args.language,
            frame_ms=args.frame_ms,
            host=args.host,
            fast=args.fast,
            loop_count=args.loop,
            auto_close_seconds=args.auto_close_seconds,
            tail_silence_ms=args.tail_silence_ms,
            save_events=save_path,
            reference=args.reference,
        ))
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
