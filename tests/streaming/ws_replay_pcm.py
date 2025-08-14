#!/usr/bin/env python3
"""Replay a 16kHz mono PCM (s16le) file to the streaming voice WebSocket endpoint.

Usage:
  python ws_replay_pcm.py --file q1.pcm --language ar --frame-ms 20 --loop 1

The script:
  - Reads raw PCM bytes.
  - Splits into Int16 frames of (frame_ms * 16000 / 1000) samples.
  - Sends each frame.buffer as binary WS message pacing real-time unless --fast specified.
  - Captures server events and prints JSON lines.
  - At end, prints summary metrics: first_partial_ms, final_ms, decode cycles, avg decode_ms.

Prereq: VOICE_STREAMING_ENABLED=1 and backend running.

Optional: Provide reference transcript via --reference "text" to compute a naive WER (whitespace token level).
"""
from __future__ import annotations
import argparse, asyncio, json, math, sys, time
from pathlib import Path
from statistics import mean

try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover
    print("Please install websockets in backend venv: pip install websockets", file=sys.stderr)
    sys.exit(1)


def compute_wer(ref: str, hyp: str) -> float:
    ref_tokens = ref.strip().split()
    hyp_tokens = hyp.strip().split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    # Simple Levenshtein distance
    dp = [[0]*(len(hyp_tokens)+1) for _ in range(len(ref_tokens)+1)]
    for i in range(len(ref_tokens)+1):
        dp[i][0] = i
    for j in range(len(hyp_tokens)+1):
        dp[0][j] = j
    for i in range(1, len(ref_tokens)+1):
        for j in range(1, len(hyp_tokens)+1):
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost,
            )
    dist = dp[-1][-1]
    return dist / len(ref_tokens)


async def replay_pcm(
    file: Path,
    language: str,
    frame_ms: int,
    host: str,
    fast: bool,
    reference: str | None,
    loop_count: int,
    auto_close_seconds: float,
    tail_silence_ms: int,
) -> None:
    pcm = file.read_bytes()
    samples = len(pcm) // 2
    samples_per_frame = int(16000 * frame_ms / 1000)
    total_frames = math.ceil(samples / samples_per_frame)
    url = f"ws://{host}/api/v1/ws/streaming-voice?language={language}"

    print(json.dumps({"event":"info","message":"connecting","url":url}))
    first_partial_ts = None
    final_ts = None
    start = time.time()
    decode_ms_values = []
    final_text = None

    async with websockets.connect(url, ping_interval=None) as ws:
        # Send frames concurrently with receive loop
        final_event = asyncio.Event()
        tts_complete_event = asyncio.Event()
        close_requested = asyncio.Event()

        async def sender():
            nonlocal first_partial_ts
            for loop_i in range(loop_count):
                cursor = 0
                while cursor < len(pcm):
                    chunk = pcm[cursor: cursor + samples_per_frame*2]
                    cursor += samples_per_frame*2
                    await ws.send(chunk)
                    if not fast:
                        await asyncio.sleep(frame_ms / 1000)
                # small gap between loops
                if loop_i + 1 < loop_count:
                    await asyncio.sleep(0.4)
            # Inject trailing silence to trigger endpointing (if configured)
            if tail_silence_ms > 0:
                silence_frames = max(1, int(tail_silence_ms / frame_ms))
                silence_chunk = b"\x00\x00" * samples_per_frame
                for _ in range(silence_frames):
                    await ws.send(silence_chunk)
                    if not fast:
                        await asyncio.sleep(frame_ms / 1000)
            # After all audio sent, optionally trigger auto-close routine
            if auto_close_seconds > 0:
                # Wait for TTS complete if it arrives, else fall back to final transcript, else timeout
                try:
                    # Wrap coroutines into tasks (asyncio.wait requires tasks, not bare coroutines)
                    tts_task = asyncio.create_task(tts_complete_event.wait())
                    final_task = asyncio.create_task(final_event.wait())
                    done, pending = await asyncio.wait(
                        {tts_task, final_task},
                        timeout=auto_close_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Cancel the other one to avoid leaks
                    for p in pending:
                        p.cancel()
                        with contextlib.suppress(Exception):
                            await p
                except Exception:
                    # Non-fatal
                    pass
                # Grace period (allow final events to flush)
                await asyncio.sleep(0.25)
                # websockets 12+ uses 'closed' property? fallback to close_code check
                already_closed = getattr(ws, 'closed', False) or getattr(ws, 'close_code', None) is not None
                if not already_closed:
                    try:
                        await ws.close()
                    finally:
                        close_requested.set()

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
                etype = data.get("type")
                print(json.dumps(data, ensure_ascii=False))
                if etype == "partial_transcript" and first_partial_ts is None:
                    first_partial_ts = time.time()
                if etype == "final_transcript":
                    final_ts = time.time()
                    final_text = data.get("text")
                    final_event.set()
                if etype == "tts_complete":
                    tts_complete_event.set()
                if etype == "perf_cycle":
                    if "decode_ms" in data:
                        decode_ms_values.append(data["decode_ms"])
                if etype == "error":
                    break
                if etype == "close_ack":  # (future use) server-initiated close acknowledgement
                    break

        await asyncio.gather(sender(), receiver())

    end = time.time()
    summary = {
        "event": "summary",
        "file": str(file.name),
        "duration_s": round(end - start, 3),
        "first_partial_ms": int(1000*(first_partial_ts - start)) if first_partial_ts else None,
        "final_ms": int(1000*(final_ts - start)) if final_ts else None,
        "avg_decode_ms": int(mean(decode_ms_values)) if decode_ms_values else None,
        "max_decode_ms": max(decode_ms_values) if decode_ms_values else None,
        "final_text": final_text,
    }
    if reference and final_text:
        summary["wer"] = round(compute_wer(reference, final_text), 3)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to PCM s16le 16k file')
    ap.add_argument('--language', default='ar')
    ap.add_argument('--frame-ms', type=int, default=20)
    ap.add_argument('--host', default='localhost:8000')
    ap.add_argument('--fast', action='store_true', help='Disable real-time pacing (send burst)')
    ap.add_argument('--reference', help='Reference transcript for WER')
    ap.add_argument('--loop', type=int, default=1)
    ap.add_argument('--auto-close-seconds', type=float, default=6.0,
                    help='If >0, automatically close websocket this many seconds after final transcript / TTS completion (whichever first).')
    ap.add_argument('--tail-silence-ms', type=int, default=1200, help='Amount of silence (ms) to append after audio to help endpoint detection.')
    args = ap.parse_args()
    file = Path(args.file)
    if not file.exists():
        print(f"File not found: {file}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(replay_pcm(
        file=file,
        language=args.language,
        frame_ms=args.frame_ms,
        host=args.host,
        fast=args.fast,
        reference=args.reference,
        loop_count=args.loop,
        auto_close_seconds=args.auto_close_seconds,
        tail_silence_ms=args.tail_silence_ms,
    ))

if __name__ == '__main__':
    main()
