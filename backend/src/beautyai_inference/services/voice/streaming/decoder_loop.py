"""Incremental Decoder Loop (Phase 4)

Implements a periodic decode over the tail of the PCM ring buffer using
the modern Whisper engine factory system. This is a *windowed* decode
approach (re-decode last N seconds) rather than true stateful streaming,
which is acceptable for Whisper-style models and keeps complexity low.

Strategy:
 - Every `decode_interval_ms` collect last `window_seconds` of PCM.
 - Run transcription (greedy, no condition_on_previous_text) on that window.
 - Tokenize text into a simple whitespace split surrogate for stability diff
   (real tokenization could leverage model tokenizer, deferred for now).
 - Emit partial transcript when new stable prefix extends.
 - Maintain `stable_prefix_tokens` (tokens that have not changed across cycles).

NOTE: This module does not send WS events directly; it yields events through
an async generator pattern so the endpoint can serialize and send.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
import os
from typing import AsyncGenerator, List, Optional, Dict, Any

from .streaming_session import StreamingSession
from .endpointing import (
    EndpointState,
    EndpointConfig,
    update_endpoint,
)
from ..transcription.transcription_factory import TranscriptionServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class DecoderConfig:
    window_seconds: float = 8.0
    decode_interval_ms: int = 480
    min_emit_chars: int = 3
    language: str = "ar"
    frame_rms_window: float = 0.2  # seconds for RMS sample to feed endpoint


@dataclass
class DecoderState:
    stable_prefix_tokens: List[str] = field(default_factory=list)
    last_transcript: str = ""
    last_emit_time: float = field(default_factory=time.time)
    running: bool = True
    last_final_utterance_index: int = -1  # guard against double final emits
    last_final_text: str = ""  # track last finalized text to suppress duplicate finals/partials when buffer window re-decodes trailing silence
    last_total_written_at_final: int = -1  # ring buffer total_written snapshot when last final emitted
    last_ring_total_written: int = -1      # updated each cycle for start-guard logic
    suppress_until_audio_growth: bool = False  # block decoding/token activation until new audio arrives

    def reset_after_final(self) -> None:
        """Reset token stability tracking for next utterance.

        We intentionally DO NOT clear last_transcript so that if the decode
        window continues to include the prior utterance (common when audio
        has stopped and we keep re-decoding an 8s tail), we don't emit the
        identical partial transcript again. Emission logic now also consults
        last_final_text to avoid duplicates.
        """
        self.stable_prefix_tokens.clear()
        # Preserve last_transcript by default (prevents duplicate partial re-emits
        # when decode window still contains prior utterance). Allow override via env.
        if os.getenv("VOICE_STREAMING_CLEAR_LAST_TRANSCRIPT", "0") == "1":
            self.last_transcript = ""
        # Record the final text for duplicate suppression.
        self.last_final_text = self.last_transcript


async def incremental_decode_loop(
    session: StreamingSession,
    fw_service: TranscriptionServiceProtocol,
    endpoint_state: EndpointState,
    config: Optional[DecoderConfig] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator performing periodic windowed decodes.

    Yields event dicts:
      {"type": "partial_transcript", ...}
      {"type": "final_transcript", ...}
      {"type": "endpoint_event", ...}
    """
    if config is None:
        config = DecoderConfig(language=session.language)
    # Allow runtime tuning of emission thresholds via env
    try:
        config.min_emit_chars = int(os.getenv("VOICE_STREAMING_MIN_EMIT_CHARS", str(config.min_emit_chars)))
    except Exception:
        pass
    lenient_final = os.getenv("VOICE_STREAMING_LENIENT_FINAL", "0") == "1"
    lenient_final_delay = float(os.getenv("VOICE_STREAMING_LENIENT_FINAL_DELAY_SEC", "1.1"))

    state = DecoderState()

    # Align endpoint frame duration with actual decode cadence to ensure
    # silence/token stability timers advance at real time instead of the
    # static 20ms frame size defined in EndpointConfig (which would make
    # a 600ms silence require 30 decode cycles at 480ms each ~14s).
    try:
        endpoint_state.config.frame_ms = config.decode_interval_ms  # type: ignore[attr-defined]
    except Exception:
        pass

    # Ensure whisper model loaded (new engines handle their own fallbacks internally)
    if not fw_service.is_model_loaded():
        try:
            logger.info(f"[decode] Loading Whisper model for session {session.session_id}")
            loaded = fw_service.load_whisper_model()
            if not loaded:
                logger.error(
                    "Failed to load transcription model for streaming session %s – aborting decode loop",
                    session.session_id,
                )
                # Emit error event before returning
                yield {
                    "type": "decoder_error",
                    "error": "Failed to load Whisper model",
                    "session_id": session.session_id,
                    "timestamp": time.time()
                }
                return
            else:
                logger.info(f"[decode] Successfully loaded Whisper model for session {session.session_id}")
        except Exception as e:
            logger.error(
                "Exception while loading transcription model for session %s: %s – aborting decode loop",
                session.session_id, e
            )
            yield {
                "type": "decoder_error",
                "error": f"Exception loading Whisper model: {str(e)}",
                "session_id": session.session_id,
                "timestamp": time.time()
            }
            return

    interval = config.decode_interval_ms / 1000.0
    logger.info(f"[decode] Starting incremental decode loop (interval=%.3fs window=%.1fs) with BUFFER RESET FIXES - session={session.session_id}" % (interval, config.window_seconds))

    try:
        while state.running and not session.closed:
            cycle_start = time.time()

            # 1. Obtain audio window (raw PCM bytes) and compute RMS for this interval
            # Track ring buffer growth to suppress phantom re-decodes after finalization
            try:
                ring_total = session.pcm_buffer.total_written  # type: ignore[attr-defined]
            except Exception:
                ring_total = -1
            state.last_ring_total_written = ring_total

            pcm_window = await session.pcm_buffer.read_last_window(config.window_seconds)
            if not pcm_window:
                # Nothing buffered yet; sleep next interval
                await asyncio.sleep(interval)
                continue

            # Derive a frame RMS sample from a short tail slice to feed endpoint
            tail_seconds = min(config.frame_rms_window, config.window_seconds)
            tail_pcm = await session.pcm_buffer.read_last_window(tail_seconds)
            frame_rms = 0.0
            if tail_pcm:
                import struct, math
                count = len(tail_pcm) // 2
                if count:
                    samples = struct.unpack('<' + 'h' * count, tail_pcm)
                    acc = 0.0
                    for s in samples:
                        f = s / 32768.0
                        acc += f * f
                    frame_rms = math.sqrt(acc / count) if count else 0.0

            # 2. Decode the current window (windowed re-decode approach)
            # Phantom suppression: if we finalized and ring buffer has NOT grown, skip decode to avoid
            # regenerating identical tokens from stale window (which we cleared) causing duplicate utterances.
            phantom_guard_enabled = os.getenv("VOICE_STREAMING_PHANTOM_GUARD", "1") == "1"
            skip_decode = False
            if (
                phantom_guard_enabled
                and state.last_final_utterance_index >= 0
                and state.last_total_written_at_final >= 0
                and ring_total >= 0
                and ring_total <= state.last_total_written_at_final
            ):
                skip_decode = True

            if skip_decode:
                transcription = ""
                decode_ms = 0
            else:
                decode_start = time.time()
                try:
                    transcription = fw_service.transcribe_audio_bytes(
                        pcm_window, audio_format="wav", language=config.language
                    )
                    if transcription is None:
                        logger.warning(f"[decode] Transcription service returned None for session {session.session_id}")
                        transcription = ""
                except Exception as e:
                    logger.error(f"[decode] Transcription failed for session {session.session_id}: {e}")
                    transcription = ""
                    # Emit error event for debugging
                    yield {
                        "type": "transcription_error",
                        "error": str(e),
                        "session_id": session.session_id,
                        "timestamp": time.time(),
                        "window_seconds": config.window_seconds
                    }
                decode_ms = int((time.time() - decode_start) * 1000)
            tokens: List[str] = []
            if transcription:
                tokens = transcription.strip().split()
                # --------------------------------------------------
                # Repetition Mitigation (client observed loop cases)
                # Detect large consecutive n-gram repetition and collapse to single cycle.
                # --------------------------------------------------
                try:
                    rep_threshold = int(os.getenv("VOICE_STREAMING_REPETITION_SCORE_THRESHOLD", "60"))
                    if len(tokens) >= 12 and rep_threshold > 0:
                        collapsed, rep_info = _collapse_repetition(tokens)
                        if rep_info and rep_info["score"] >= rep_threshold and collapsed != tokens:
                            yield {
                                "type": "decode_sanitized",
                                "reason": "repetition_collapse",
                                "removed_cycles": rep_info.get("repeats", 0) - 1,
                                "ngram": rep_info.get("phrase"),
                                "score": rep_info.get("score"),
                                "original_tokens": len(tokens),
                                "collapsed_tokens": len(collapsed),
                                "timestamp": time.time(),
                            }
                            tokens = collapsed
                except Exception:  # pragma: no cover - defensive
                    pass
            logger.debug(
                "[decode] session=%s chars=%d tokens=%d decode_ms=%d window=%.1fs",  # fine-grained debug
                session.session_id,
                len(transcription) if transcription else 0,
                len(tokens),
                decode_ms,
                config.window_seconds,
            )

            # 3. Update token stability (simple prefix diff)
            if tokens:
                prefix_len = 0
                for a, b in zip(state.stable_prefix_tokens, tokens):
                    if a == b:
                        prefix_len += 1
                    else:
                        break
                state.stable_prefix_tokens = tokens[:prefix_len]

            full_text = " ".join(tokens) if tokens else ""
            stable = tokens and (len(tokens) == len(state.stable_prefix_tokens))

            # Emit partial before endpoint events (client sees freshest text first)
            # Suppress duplicate partial emissions for the same finalized text
            if (
                full_text
                and full_text != state.last_transcript
                and full_text != state.last_final_text
                and len(full_text) >= config.min_emit_chars
            ):
                state.last_transcript = full_text
                state.last_emit_time = time.time()
                yield {
                    "type": "partial_transcript",
                    "text": full_text,
                    "stable": bool(stable),
                    "timestamp": state.last_emit_time,
                    "stable_tokens": len(state.stable_prefix_tokens),
                    "total_tokens": len(tokens),
                    "window_seconds": config.window_seconds,
                    "decode_ms": decode_ms,
                }
            elif not full_text:
                # Optional debug event when nothing decoded (only if debug env set)
                if os.getenv("VOICE_STREAMING_DEBUG_EMPTY", "0") == "1":
                    yield {
                        "type": "decode_debug",
                        "reason": "empty_transcription",
                        "timestamp": time.time(),
                        "decode_ms": decode_ms,
                        "window_seconds": config.window_seconds,
                        "rms_frame": frame_rms,
                    }

            # 4. Advance endpoint state with current tokens (Phase 5 integration)
            # If we skipped decode we must not feed old tokens to endpoint detector
            endpoint_events = update_endpoint(endpoint_state, frame_rms, current_tokens=(tokens or None) if not skip_decode else None)
            final_event = None
            start_event = None
            for ev in endpoint_events:
                yield {
                    "type": "endpoint_event",
                    "event": ev.type,
                    "utterance_index": ev.utterance_index,
                    "reason": ev.reason,
                    "voiced_ms": ev.voiced_ms,
                    "silence_ms": ev.silence_ms,
                    "utterance_ms": ev.utterance_ms,
                    "end_silence_gap_ms": getattr(ev, "end_silence_gap_ms", None),
                    "timestamp": time.time(),
                }
                if ev.type == "final":
                    final_event = ev
                elif ev.type == "start":
                    start_event = ev

            # Reset buffer at start of new utterance to prevent conversation bleeding
            if start_event and start_event.utterance_index > 0:  # Skip reset for very first utterance
                reset_buffer_at_start = os.getenv("VOICE_STREAMING_RESET_BUFFER_AT_START", "1") == "1"
                cleared_on_final_enabled = os.getenv("VOICE_STREAMING_CLEAR_BUFFER_ON_FINAL", "1") == "1"
                if reset_buffer_at_start and not cleared_on_final_enabled:
                    logger.info(
                        "[decode] Resetting ring buffer at start of new utterance (utterance_index=%d)",
                        start_event.utterance_index
                    )
                    await session.pcm_buffer.reset_for_new_utterance()
                    # Clear previous text to prevent bleed-through
                    state.last_transcript = ""
                    state.last_final_text = ""
                    state.stable_prefix_tokens.clear()

            # 5. Emit final transcript exactly once per utterance
            final_emitted = False
            if final_event and final_event.utterance_index > state.last_final_utterance_index:
                # Prefer newest non-empty full_text; fallback to last_transcript
                final_text = full_text or state.last_transcript
                min_final_chars = int(
                    __import__("os").getenv("VOICE_STREAMING_MIN_FINAL_CHARS", "3")
                )
                if final_text and final_text.strip() and len(final_text.strip()) >= min_final_chars:
                    if final_text != state.last_final_text:
                        yield {
                            "type": "final_transcript",
                            "text": final_text,
                            "utterance_index": final_event.utterance_index,
                            "reason": final_event.reason,
                            "timestamp": time.time(),
                            "voiced_ms": final_event.voiced_ms,
                            "silence_ms": final_event.silence_ms,
                            "utterance_ms": final_event.utterance_ms,
                            "stable_tokens": len(state.stable_prefix_tokens),
                            "total_tokens": len(tokens),
                            "decode_ms": decode_ms,
                        }
                        final_emitted = True
                else:
                    logger.debug(
                        "[decode] Suppressing empty/short final transcript (chars=%d utterance_index=%d)",
                        len(final_text.strip()) if final_text else 0,
                        final_event.utterance_index,
                    )
                    # Emit explicit suppression event for client diagnostics
                    yield {
                        "type": "final_suppressed",
                        "utterance_index": final_event.utterance_index,
                        "reason": "empty_or_short",
                        "chars": len(final_text.strip()) if final_text else 0,
                        "min_required": min_final_chars,
                        "timestamp": time.time(),
                    }
                # Always advance utterance index tracking to avoid re-trigger loops
                state.last_final_utterance_index = final_event.utterance_index
                # Record last_final_text only if non-empty to further suppress duplicates
                if final_text and final_text.strip():
                    state.reset_after_final()
                    # Snapshot current ring buffer total_written so we can detect if new audio arrives
                    try:
                        state.last_total_written_at_final = session.pcm_buffer.total_written  # type: ignore[attr-defined]
                    except Exception:
                        state.last_total_written_at_final = -1
                    # Optional: Clear ring buffer immediately to prevent stale token window re-decodes
                    if os.getenv("VOICE_STREAMING_CLEAR_BUFFER_ON_FINAL", "1") == "1":
                        try:
                            await session.pcm_buffer.reset_for_new_utterance()
                            logger.info("[decode] Cleared ring buffer on final (utterance_index=%d)", final_event.utterance_index)
                            # Activate phantom suppression until new audio arrives
                            state.last_total_written_at_final = session.pcm_buffer.total_written  # should now be 0
                            state.suppress_until_audio_growth = True
                            # Clear all transcript state to prevent bleeding
                            state.last_transcript = ""
                            state.stable_prefix_tokens.clear()
                        except Exception:
                            logger.warning("[decode] Failed to clear ring buffer on final")

            # Lenient fallback finalization (if no endpoint driven final)
            if (
                not final_emitted
                and lenient_final
                and full_text
                and full_text != state.last_final_text
                and state.last_emit_time
                and (time.time() - state.last_emit_time) >= lenient_final_delay
                and len(full_text) >= int(os.getenv("VOICE_STREAMING_MIN_FINAL_CHARS", "3"))
            ):
                yield {
                    "type": "final_transcript",
                    "text": full_text,
                    "utterance_index": state.last_final_utterance_index + 1,
                    "reason": "lenient_timeout",
                    "timestamp": time.time(),
                    "voiced_ms": None,
                    "silence_ms": None,
                    "utterance_ms": None,
                    "stable_tokens": len(state.stable_prefix_tokens),
                    "total_tokens": len(full_text.split()),
                    "decode_ms": decode_ms,
                }
                state.last_final_utterance_index += 1
                state.reset_after_final()

            # Sleep remaining interval
            elapsed = time.time() - cycle_start
            cycle_latency_ms = int(elapsed * 1000)
            # Emit performance heartbeat event (Phase 9 instrumentation)
            yield {
                "type": "perf_cycle",
                "timestamp": time.time(),
                "decode_ms": decode_ms,
                "cycle_latency_ms": cycle_latency_ms,
                "window_seconds": config.window_seconds,
                "interval_ms": config.decode_interval_ms,
                "tokens": len(tokens),
                "phantom_guard": skip_decode,
            }
            to_sleep = interval - elapsed
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
    except asyncio.CancelledError:  # graceful shutdown
        logger.debug("Decoder loop cancelled (%s)", session.session_id)
    except Exception as e:  # pragma: no cover
        logger.exception("Decoder loop error (%s): %s", session.session_id, e)
    finally:
        logger.debug("Decoder loop terminating (%s)", session.session_id)


def _collapse_repetition(tokens: List[str]) -> tuple[List[str], Optional[Dict[str, Any]]]:
    """Detect repeated n-gram loops and collapse to a single cycle.

    Returns (possibly_collapsed_tokens, repetition_info|None)

    Heuristic: search n-grams size 3..12 consuming >=60% of sequence via >=2 repeats.
    Score = (repeated_span / total_tokens) * 120 capped 100 (mirrors frontend logic).
    """
    best = None
    total = len(tokens)
    for n in range(12, 2, -1):  # try larger n-grams first for maximal collapse
        if n * 2 > total:
            continue
        for i in range(0, total - n * 2 + 1):
            phrase = tokens[i : i + n]
            repeats = 1
            j = i + n
            while j + n <= total and tokens[j : j + n] == phrase:
                repeats += 1
                j += n
            if repeats >= 2:
                span = repeats * n
                coverage = span / total
                score = min(100, int(coverage * 120))
                if not best or span > best["span"]:
                    best = {
                        "phrase": " ".join(phrase),
                        "repeats": repeats,
                        "span": span,
                        "coverage": coverage,
                        "score": score,
                        "start": i,
                        "end": j,
                    }
        if best:  # early exit once we found largest n producing repetition
            break
    if not best:
        return tokens, None
    # Collapse to single phrase + tail remainder beyond repeated span
    collapsed = tokens[: best["start"]] + tokens[best["start"] : best["start"] + (best["end"] - best["start"]) // best["repeats"]] + tokens[best["end"] :]
    return collapsed, best


__all__ = [
    "DecoderConfig",
    "DecoderState",
    "incremental_decode_loop",
]
