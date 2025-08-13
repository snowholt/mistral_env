"""Incremental Decoder Loop (Phase 4)

Implements a periodic decode over the tail of the PCM ring buffer using
the existing FasterWhisperTranscriptionService. This is a *windowed* decode
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
from typing import AsyncGenerator, List, Optional, Dict, Any

from .streaming_session import StreamingSession
from .endpointing import (
    EndpointState,
    EndpointConfig,
    update_endpoint,
)
from ..transcription.faster_whisper_service import FasterWhisperTranscriptionService

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

    def reset_after_final(self) -> None:
        """Reset token stability tracking for next utterance."""
        self.stable_prefix_tokens.clear()
        self.last_transcript = ""


async def incremental_decode_loop(
    session: StreamingSession,
    fw_service: FasterWhisperTranscriptionService,
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

    state = DecoderState()

    # Ensure whisper model loaded
    if not fw_service.is_model_loaded():
        loaded = fw_service.load_whisper_model()
        if not loaded:
            logger.error("Failed to load FasterWhisper model for streaming session %s", session.session_id)
            return

    interval = config.decode_interval_ms / 1000.0
    logger.debug("Starting incremental decode loop (interval=%.3fs window=%.1fs)", interval, config.window_seconds)

    try:
        while state.running and not session.closed:
            cycle_start = time.time()

            # 1. Obtain audio window (raw PCM bytes) and compute RMS for this interval
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
            transcription = fw_service.transcribe_audio_bytes(
                pcm_window, audio_format="wav", language=config.language
            )
            tokens: List[str] = []
            if transcription:
                tokens = transcription.strip().split()

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
            if full_text and full_text != state.last_transcript and len(full_text) >= config.min_emit_chars:
                state.last_transcript = full_text
                yield {
                    "type": "partial_transcript",
                    "text": full_text,
                    "stable": bool(stable),
                    "timestamp": time.time(),
                    "stable_tokens": len(state.stable_prefix_tokens),
                    "total_tokens": len(tokens),
                    "window_seconds": config.window_seconds,
                }

            # 4. Advance endpoint state with current tokens (Phase 5 integration)
            endpoint_events = update_endpoint(endpoint_state, frame_rms, current_tokens=tokens or None)
            final_event = None
            for ev in endpoint_events:
                yield {
                    "type": "endpoint_event",
                    "event": ev.type,
                    "utterance_index": ev.utterance_index,
                    "reason": ev.reason,
                    "voiced_ms": ev.voiced_ms,
                    "silence_ms": ev.silence_ms,
                    "utterance_ms": ev.utterance_ms,
                    "timestamp": time.time(),
                }
                if ev.type == "final":
                    final_event = ev

            # 5. Emit final transcript exactly once per utterance
            if final_event and final_event.utterance_index > state.last_final_utterance_index:
                final_text = full_text or state.last_transcript
                if final_text:
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
                    }
                state.last_final_utterance_index = final_event.utterance_index
                state.reset_after_final()

            # Sleep remaining interval
            elapsed = time.time() - cycle_start
            to_sleep = interval - elapsed
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
    except asyncio.CancelledError:  # graceful shutdown
        logger.debug("Decoder loop cancelled (%s)", session.session_id)
    except Exception as e:  # pragma: no cover
        logger.exception("Decoder loop error (%s): %s", session.session_id, e)
    finally:
        logger.debug("Decoder loop terminating (%s)", session.session_id)


__all__ = [
    "DecoderConfig",
    "DecoderState",
    "incremental_decode_loop",
]
