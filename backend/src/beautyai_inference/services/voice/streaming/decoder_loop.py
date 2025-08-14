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

    def reset_after_final(self) -> None:
        """Reset token stability tracking for next utterance.

        We intentionally DO NOT clear last_transcript so that if the decode
        window continues to include the prior utterance (common when audio
        has stopped and we keep re-decoding an 8s tail), we don't emit the
        identical partial transcript again. Emission logic now also consults
        last_final_text to avoid duplicates.
        """
        self.stable_prefix_tokens.clear()
        # Preserve last_transcript; we only change it when genuinely new
        # tokens appear. Record the final text for duplicate suppression.
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

    state = DecoderState()

    # Align endpoint frame duration with actual decode cadence to ensure
    # silence/token stability timers advance at real time instead of the
    # static 20ms frame size defined in EndpointConfig (which would make
    # a 600ms silence require 30 decode cycles at 480ms each ~14s).
    try:
        endpoint_state.config.frame_ms = config.decode_interval_ms  # type: ignore[attr-defined]
    except Exception:
        pass

    # Ensure whisper model loaded (with transparent fallback if Faster-Whisper fails)
    if not fw_service.is_model_loaded():
        loaded = fw_service.load_whisper_model()
        if not loaded:
            # Attempt fallback ONLY if original service was Faster-Whisper implementation
            try:
                from ..transcription.faster_whisper_service import FasterWhisperTranscriptionService  # type: ignore
                from ..transcription.transformers_whisper_service import TransformersWhisperService  # type: ignore
            except Exception:  # pragma: no cover - import safety
                FasterWhisperTranscriptionService = object  # type: ignore
                TransformersWhisperService = None  # type: ignore

            if 'FasterWhisperTranscriptionService' in locals() and isinstance(fw_service, FasterWhisperTranscriptionService):  # type: ignore[arg-type]
                logger.warning(
                    "Primary Faster-Whisper load failed for session %s – attempting fallback to Transformers backend",
                    session.session_id,
                )
                try:
                    fallback_service = TransformersWhisperService() if TransformersWhisperService else None  # type: ignore
                    if fallback_service and fallback_service.load_whisper_model():
                        fw_service = fallback_service  # swap reference used below
                        logger.info("Fallback Transformers Whisper model loaded successfully for session %s", session.session_id)
                    else:
                        logger.error(
                            "Fallback Transformers Whisper model also failed to load for session %s – aborting decode loop",
                            session.session_id,
                        )
                        return
                except Exception as e:  # pragma: no cover
                    logger.exception(
                        "Exception during fallback model load for session %s: %s", session.session_id, e
                    )
                    return
            else:
                logger.error(
                    "Failed to load transcription model (non-fallback path) for streaming session %s", session.session_id
                )
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
            decode_start = time.time()
            transcription = fw_service.transcribe_audio_bytes(
                pcm_window, audio_format="wav", language=config.language
            )
            decode_ms = int((time.time() - decode_start) * 1000)
            tokens: List[str] = []
            if transcription:
                tokens = transcription.strip().split()
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
                yield {
                    "type": "partial_transcript",
                    "text": full_text,
                    "stable": bool(stable),
                    "timestamp": time.time(),
                    "stable_tokens": len(state.stable_prefix_tokens),
                    "total_tokens": len(tokens),
                    "window_seconds": config.window_seconds,
                    "decode_ms": decode_ms,
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
                    "end_silence_gap_ms": getattr(ev, "end_silence_gap_ms", None),
                    "timestamp": time.time(),
                }
                if ev.type == "final":
                    final_event = ev

            # 5. Emit final transcript exactly once per utterance
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
                else:
                    logger.debug(
                        "[decode] Suppressing empty/short final transcript (chars=%d utterance_index=%d)",
                        len(final_text.strip()) if final_text else 0,
                        final_event.utterance_index,
                    )
                # Always advance utterance index tracking to avoid re-trigger loops
                state.last_final_utterance_index = final_event.utterance_index
                # Record last_final_text only if non-empty to further suppress duplicates
                if final_text and final_text.strip():
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


__all__ = [
    "DecoderConfig",
    "DecoderState",
    "incremental_decode_loop",
]
