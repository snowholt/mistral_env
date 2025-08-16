"""Streaming Voice WebSocket Endpoint (Phases 1 & 2)

Phase 1 (complete):
 - Accept WebSocket connections at /api/v1/ws/streaming-voice
 - Simple handshake + mock decode loop emitting synthetic transcript events
 - Lightweight in-module session registry

Phase 2 additions:
 - Integrate high-capacity PCM Int16 ring buffer per connection
 - Ingest binary WebSocket frames (assumed 16 kHz mono little-endian int16)
 - Track audio metrics (bytes, frames, buffer usage, drops)
 - Provide early consumption correlation in mock decode loop (exposes audio_level metric)

Still mock decoding (no Whisper yet). Frames are buffered & basic stats exposed.
Feature flag: VOICE_STREAMING_ENABLED=1 must be set to expose this router.

Upcoming phases:
 - Endpoint detection (Phase 3/5)
 - Whisper incremental decoding (Phase 4)
 - LLM + TTS integration (Phase 6)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from beautyai_inference.services.voice.streaming.streaming_session import StreamingSession
from beautyai_inference.services.voice.streaming.decoder_loop import (
    incremental_decode_loop,
    DecoderConfig,
)
from beautyai_inference.services.voice.streaming.endpointing import EndpointState, EndpointConfig
from beautyai_inference.services.voice.streaming.metrics import SessionMetrics, maybe_log_structured
from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service
from beautyai_inference.services.voice.utils.text_cleaning import sanitize_tts_text

logger = logging.getLogger(__name__)

def streaming_feature_enabled() -> bool:
    """Return True if streaming feature enabled (evaluated at call time).

    Defaults to enabled if the router was imported but env var not explicitly set.
    This prevents 403s in development where systemd env propagation or reload
    timing may omit the variable in child processes.
    Set VOICE_STREAMING_ENABLED=0 to force disable.
    """
    flag = os.getenv("VOICE_STREAMING_ENABLED")
    if flag is None:
        return True  # router already imported -> enable by default
    return flag == "1"

streaming_voice_router = APIRouter(prefix="/ws", tags=["streaming-voice"])  # included conditionally in app


@dataclass
class SessionState:
    connection_id: str
    session_id: str
    websocket: WebSocket
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    mock_token_cursor: int = 0
    mock_running: bool = False
    mock_task: Optional[asyncio.Task] = None
    disconnected: bool = False
    bytes_received: int = 0
    pcm_frames_received: int = 0
    audio_session: Optional[StreamingSession] = None  # Phase 2
    conversation: List[Dict[str, str]] = field(default_factory=list)  # Phase 6 conversation history
    llm_tts_task: Optional[asyncio.Task] = None  # background processing task for final transcript
    metrics: Optional[SessionMetrics] = None  # Phase 10 metrics aggregator
    # Pending finals queue (utterance_index, text) to ensure no final_transcript is dropped
    pending_finals: List[tuple[int, str]] = field(default_factory=list)
    # Deduplication tracking for finals (avoid echo / duplicate triggering)
    processed_utterance_indices: set[int] = field(default_factory=set)
    last_final_text: Optional[str] = None
    last_final_time: float = 0.0
    # Debug / diagnostics
    last_pipeline_start: Optional[float] = None
    pipeline_active_for: Optional[int] = None  # utterance index currently in pipeline
    emitted_assistant_for: set[int] = field(default_factory=set)
    assistant_turns: int = 0
    # Hybrid ingest additions (Aug 16 2025)
    compressed_mode: Optional[str] = None  # e.g. 'webm-opus'
    ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
    ffmpeg_reader_task: Optional[asyncio.Task] = None
    ffmpeg_writer_open: bool = False
    pending_pcm_fragment: bytearray = field(default_factory=bytearray)
    heartbeat_task: Optional[asyncio.Task] = None
    last_heartbeat_sent: float = 0.0
    last_client_msg: float = field(default_factory=time.time)
    silence_frame_count: int = 0
    voiced_frame_count: int = 0
    low_energy_mode: bool = True  # assume silent until energy detected

    def touch(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = time.time()


class StreamingSessionRegistry:
    """In-memory session registry for Phase 1 (simple dict)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def add(self, state: SessionState) -> None:
        self._sessions[state.connection_id] = state

    def get(self, connection_id: str) -> Optional[SessionState]:
        return self._sessions.get(connection_id)

    def remove(self, connection_id: str) -> Optional[SessionState]:
        return self._sessions.pop(connection_id, None)

    def active_count(self) -> int:
        return len(self._sessions)


session_registry = StreamingSessionRegistry()

MOCK_PHRASES_AR = ["مرحباً", "مرحباً بك", "مرحباً بك في وضع البث", "مرحبا بالتجربة"]
MOCK_PHRASES_EN = ["hello", "hello there", "hello there streaming", "streaming mock"]


async def _send_json(ws: WebSocket, payload: Dict[str, Any]) -> None:
    """Safely send JSON if socket still connected."""
    if ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as e:  # pragma: no cover - network layer
            logger.debug("Failed sending JSON payload: %s", e)


async def _mock_decode_loop(state: SessionState, language: str) -> None:
    """Emit synthetic partial_transcript events until disconnect.

    Simulates incremental token growth; resets every few iterations.
    Interval chosen (0.8s) to reflect later decode cadence for early UI dev.
    """
    phrases = MOCK_PHRASES_AR if language == "ar" else MOCK_PHRASES_EN
    try:
        state.mock_running = True
        while not state.disconnected:
            await asyncio.sleep(0.8)
            state.mock_token_cursor += 1
            idx = state.mock_token_cursor % len(phrases)
            text = " ".join(phrases[: idx + 1])
            # Derive an approximate instantaneous audio level (rough) from buffer occupancy
            audio_level = None
            buffer_usage = None
            drops = None
            if state.audio_session:
                rb = state.audio_session.pcm_buffer
                buffer_usage = rb.usage_ratio()
                # Corrected stats field name (was total_dropped)
                drops = rb.stats.total_dropped_events
                # crude level proxy: normalized buffer usage * 0.8 + random jitter (omitted jitter for determinism)
                audio_level = round(min(1.0, buffer_usage * 0.8), 3)
            await _send_json(state.websocket, {
                "type": "partial_transcript",
                "text": text,
                "stable": False if idx + 1 < len(phrases) else True,
                "mock": True,
                "timestamp": time.time(),
                "cursor": state.mock_token_cursor,
                "audio_level": audio_level,
                "buffer_usage": buffer_usage,
                "dropped": drops,
            })
            if idx + 1 == len(phrases):  # phrase cycle complete → emit final
                await _send_json(state.websocket, {
                    "type": "final_transcript",
                    "text": text,
                    "utterance_id": f"mock_{state.mock_token_cursor}",
                    "duration_ms": 1000 + (idx * 150),
                    "mock": True,
                    "timestamp": time.time(),
                    "total_pcm_frames": state.pcm_frames_received,
                })
                state.mock_token_cursor = 0
    except asyncio.CancelledError:  # graceful cancellation
        logger.debug("Mock decode loop cancelled for %s", state.connection_id)
    except Exception as e:  # pragma: no cover
        logger.warning("Mock decode loop error (%s): %s", state.connection_id, e)
    finally:
        state.mock_running = False


@streaming_voice_router.websocket("/streaming-voice")
async def streaming_voice_endpoint(
    websocket: WebSocket,
    language: str = Query("ar", description="Language code (ar|en)"),
) -> None:
    """Phase 1+ streaming voice endpoint (now with incremental decode & LLM/TTS).

    Patch (Aug 15 2025): Accept the WebSocket *before* feature flag / param validation so that
    browser clients see a normal 101 Switching Protocols instead of an HTTP 403 when the
    feature is disabled or parameters are invalid. We then emit a structured `error` event
    and close with an application close code. This prevents frontend libraries from
    classifying the failure as a hard permission error and allows us to implement client-side
    retry / diagnostics without triggering legacy fallback prematurely.
    """
    # Accept early to avoid nginx / ASGI surfacing 403 on close-before-accept.
    try:
        await websocket.accept()
    except Exception as e:  # pragma: no cover - accept failures are rare
        logger.error("[streaming_voice] failed to accept websocket: %s", e)
        return
    # Early debug trace of raw requested language prior to validation / feature flag checks
    try:
        logger.debug("[streaming_voice] accepted ws raw_query_language=%s", language)
    except Exception:
        pass

    flag_enabled = streaming_feature_enabled()
    logger.debug(
        "[streaming_voice] connection attempt language=%s env_flag_enabled=%s raw_env=%s",
        language,
        flag_enabled,
        os.getenv("VOICE_STREAMING_ENABLED"),
    )
    if not flag_enabled:
        await _send_json(websocket, {
            "type": "error",
            "stage": "init",
            "message": "Streaming voice disabled (set VOICE_STREAMING_ENABLED=1)",
        })
        # Use normal close (1000) so client can decide to retry later if flag toggles.
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="streaming_disabled")
        return
    if language not in ("ar", "en"):
        await _send_json(websocket, {
            "type": "error",
            "stage": "init",
            "message": f"Unsupported language '{language}' (allowed: ar,en)",
        })
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="unsupported_language")
        return

    connection_id = str(uuid.uuid4())
    session_id = f"stream_{connection_id}"

    state = SessionState(connection_id=connection_id, session_id=session_id, websocket=websocket)
    state.metrics = SessionMetrics(session_id=session_id)
    # Phase 2: attach audio streaming session (ring buffer size ~40s)
    state.audio_session = StreamingSession(connection_id=connection_id, session_id=session_id, language=language)
    session_registry.add(state)
    logger.info("🌐 Streaming voice connection established: %s (lang=%s) phase4_env=%s force_real=%s disable_mock=%s", connection_id, language, os.getenv("VOICE_STREAMING_PHASE4"), os.getenv("VOICE_STREAMING_FORCE_REAL"), os.getenv("VOICE_STREAMING_DISABLE_MOCK"))

    # Phase control flags
    # VOICE_STREAMING_PHASE4=1 -> enable incremental real decoding path
    # VOICE_STREAMING_FORCE_REAL=1 -> force real decoding even if PHASE4 flag not set (helpful when disabling mock)
    # VOICE_STREAMING_DISABLE_MOCK=1 -> refuse connection if real decoding not active (fail fast instead of emitting synthetic tokens)
    force_real = os.getenv("VOICE_STREAMING_FORCE_REAL", "0") == "1"
    disable_mock = os.getenv("VOICE_STREAMING_DISABLE_MOCK", "0") == "1"
    run_phase4 = os.getenv("VOICE_STREAMING_PHASE4", "0") == "1" or force_real

    if disable_mock and not run_phase4:
        # Client explicitly does not want mock output – close early with clear reason
        await _send_json(websocket, {
            "type": "error",
            "stage": "init",
            "message": "Mock decoding disabled (set VOICE_STREAMING_PHASE4=1 or VOICE_STREAMING_FORCE_REAL=1)",
        })
        await websocket.close(code=1011, reason="mock_disabled_no_real_decode")
        return

    await _send_json(websocket, {
        "type": "ready",
        "session_id": session_id,
        "connection_id": connection_id,
        "timestamp": time.time(),
        "feature": "streaming_voice_phase4" if run_phase4 else "streaming_voice_phase2",
        "message": "Streaming voice ready (incremental decode)" if run_phase4 else "Streaming voice ready (mock decode + PCM buffer)",
        "language": language,
        "pcm_sample_rate": state.audio_session.sample_rate if state.audio_session else 16000,
        "ring_buffer_seconds": 40.0,
        "decode_interval_ms": 480 if run_phase4 else None,
        "window_seconds": 8.0 if run_phase4 else None,
        "metrics": {"phase9_instrumented": bool(run_phase4)},
        "flags": {
            "forced_real": force_real,
            "mock_disabled": disable_mock,
        }
    })

    if run_phase4:
        try:
            fw_service = create_transcription_service()
        except Exception as e:
            logger.exception("Failed to initialize transcription service, falling back to mock decode: %s", e)
            await _send_json(websocket, {"type": "warning", "stage": "init", "message": f"transcription_init_failed: {e}"})
            run_phase4 = False
        # Endpoint / decoder dynamic tuning via env
        decode_interval_ms = int(os.getenv("VOICE_STREAMING_DECODE_INTERVAL_MS", "480"))
        window_seconds = float(os.getenv("VOICE_STREAMING_WINDOW_SECONDS", "8.0"))
        min_silence_ms = int(os.getenv("VOICE_STREAMING_MIN_SILENCE_MS", "600"))
        token_stable_ms = int(os.getenv("VOICE_STREAMING_TOKEN_STABLE_MS", "600"))
        max_utterance_ms = int(os.getenv("VOICE_STREAMING_MAX_UTTERANCE_MS", "12000"))
        ep_cfg = EndpointConfig(
            min_silence_ms=min_silence_ms,
            token_stable_ms=token_stable_ms,
            max_utterance_ms=max_utterance_ms,
        )
        ep_state = EndpointState(config=ep_cfg)
        # Provide aggressive fast-path if requested
        if os.getenv("VOICE_STREAMING_LOW_LATENCY_PRESET", "0") == "1":
            ep_state.config.min_silence_ms = min(ep_state.config.min_silence_ms, 480)
            ep_state.config.token_stable_ms = min(ep_state.config.token_stable_ms, 480)
        # Hard stop guard: force finalize if no final after grace window
        force_final_grace = float(os.getenv("VOICE_STREAMING_FORCE_FINAL_AFTER_SEC", "15"))
        session_start_time = time.time()

        # Lazy globals for LLM + TTS (Phase 6)
        chat_service_ref: Dict[str, Any] = {}
        voice_service_ref: Dict[str, Any] = {}

        async def _ensure_services() -> Dict[str, Any]:
            """Lazily initialize ChatService & SimpleVoiceService once (cached)."""
            if "chat" not in chat_service_ref:
                from beautyai_inference.services.inference.chat_service import ChatService
                chat = ChatService()
                # Attempt to load default model for fast responses
                with contextlib.suppress(Exception):
                    chat.load_default_model_from_config()
                chat_service_ref["chat"] = chat
            if "voice" not in voice_service_ref:
                from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
                voice = SimpleVoiceService()
                # Initialize Edge TTS etc.
                with contextlib.suppress(Exception):
                    await voice.initialize()
                voice_service_ref["voice"] = voice
            return {"chat": chat_service_ref["chat"], "voice": voice_service_ref["voice"]}

    async def _process_final_transcript(utterance_index: int, text: str, lang: str, state: SessionState) -> None:
            """Handle LLM + TTS pipeline for a finalized user utterance.

            Emits sequence: tts_start -> tts_audio (base64) -> tts_complete
            Failure paths emit error event.
            """
            started_ts = time.time()
            state.last_pipeline_start = started_ts
            state.pipeline_active_for = utterance_index
            debug_flag = os.getenv("VOICE_STREAMING_DEBUG_PIPELINE", "1") == "1"
            if debug_flag:
                await _send_json(state.websocket, {
                    "type": "assistant_pipeline_start",
                    "utterance_index": utterance_index,
                    "chars": len(text),
                    "timestamp": started_ts,
                })
            await _send_json(state.websocket, {
                "type": "tts_start",
                "utterance_index": utterance_index,
                "timestamp": started_ts,
            })
            try:
                if os.getenv("VOICE_STREAMING_DISABLE_TTS", "0") == "1":
                    # Skip synthesis entirely – still append conversation and emit synthetic completion
                    state.conversation.append({"role": "user", "content": text})
                    state.conversation.append({"role": "assistant", "content": "(TTS disabled)"})
                    await _send_json(state.websocket, {
                        "type": "tts_complete",
                        "utterance_index": utterance_index,
                        "processing_ms": int((time.time() - started_ts) * 1000),
                        "disabled": True,
                    })
                    return
                services = await _ensure_services()
                chat_service = services["chat"]
                voice_service = services["voice"]

                # NOTE: Do NOT append the current user message before calling chat().
                # chat_service.chat expects 'message' to be the *new* user turn and
                # 'conversation_history' to contain ONLY previous exchanges.
                # Previously we appended first which duplicated the user content in the prompt
                # inflating effective context tokens and causing context window overflow.

                prev_history = list(state.conversation)  # pass a shallow copy of prior turns

                # Generate LLM response (limit length for latency)
                # Offload potentially blocking LLM generation to thread pool to avoid stalling event loop
                loop = asyncio.get_running_loop()
                def _run_chat():
                    return chat_service.chat_fast(
                        message=text,
                        conversation_history=prev_history,
                        max_length=192,
                        language=lang or "auto",
                        temperature=0.3,
                    )
                chat_result = await loop.run_in_executor(None, _run_chat)
                if not chat_result.get("success"):
                    # Provide a graceful fallback response instead of raising to keep audio pipeline alive
                    logger.warning("Chat generation failed for utterance %s: %s", utterance_index, chat_result.get('error'))
                    response_text = (
                        "Hello! I'm here but couldn't generate a full answer just now. Please continue speaking or ask another question."
                        if (lang or "en").startswith("en") else
                        "مرحباً! لم أستطع توليد إجابة كاملة الآن، تابع أو اطرح سؤالاً آخر." if (lang or "ar").startswith("ar") else
                        "I'm here and listening. Could you repeat or ask differently?"
                    )
                else:
                    response_text = chat_result.get("response", "") or ""
                # Sanitize (remove thinking & emojis) defensively
                response_text = sanitize_tts_text(response_text)

                # Update conversation history (append user then assistant so history stays canonical)
                state.conversation.append({"role": "user", "content": text})
                if response_text:
                    state.conversation.append({"role": "assistant", "content": response_text})

                # Emit assistant_response event early (before synthesis) so UI can display text immediately
                await _send_json(state.websocket, {
                    "type": "assistant_response",
                    "utterance_index": utterance_index,
                    "text": response_text,
                    "chars": len(response_text),
                    "timestamp": time.time(),
                    "conversation_len": len(state.conversation),
                })
                state.emitted_assistant_for.add(utterance_index)
                # Increment assistant turn counter early (text surfaced)
                state.assistant_turns += 1

                # Synthesize speech (use language heuristic)
                from base64 import b64encode
                tts_path = await voice_service.text_to_speech(response_text, language=lang or chat_result.get("detected_language", "ar"))
                audio_bytes = tts_path.read_bytes()
                audio_b64 = b64encode(audio_bytes).decode("utf-8")

                await _send_json(state.websocket, {
                    "type": "tts_audio",
                    "utterance_index": utterance_index,
                    "mime_type": "audio/wav",  # current synthesis output
                    "encoding": "base64",
                    "audio": audio_b64,
                    "text": response_text,
                    "chars": len(response_text),
                    "timestamp": time.time(),
                })

                await _send_json(state.websocket, {
                    "type": "tts_complete",
                    "utterance_index": utterance_index,
                    "processing_ms": int((time.time() - started_ts) * 1000),
                })
                # Emit a concise assistant_turn summary event AFTER audio completion for export stability
                await _send_json(state.websocket, {
                    "type": "assistant_turn",
                    "utterance_index": utterance_index,
                    "chars": len(response_text),
                    "assistant_turns": state.assistant_turns,
                    "timestamp": time.time(),
                })
                if debug_flag:
                    await _send_json(state.websocket, {
                        "type": "assistant_pipeline_done",
                        "utterance_index": utterance_index,
                        "total_ms": int((time.time() - started_ts) * 1000),
                        "tts_ms": int((time.time() - started_ts) * 1000),
                    })
            except asyncio.CancelledError:  # pragma: no cover - client closed during synthesis
                logger.debug("TTS pipeline cancelled (client disconnect?) connection=%s", connection_id)
                return
            except Exception as e:  # pragma: no cover
                logger.exception("TTS pipeline failure (%s): %s", connection_id, e)
                await _send_json(state.websocket, {
                    "type": "error",
                    "stage": "tts_pipeline",
                    "message": str(e),
                    "utterance_index": utterance_index,
                })
            finally:
                # Mark current task slot free then schedule next pending final if any
                state.llm_tts_task = None
                if state.pending_finals:
                    next_utt, next_text = state.pending_finals.pop(0)
                    state.llm_tts_task = asyncio.create_task(
                        _process_final_transcript(next_utt, next_text, lang, state)
                    )
                else:
                    state.pipeline_active_for = None

    async def decoder_task() -> None:
        """Incremental decoder task wrapper.

        Restored after indentation corruption. Handles forwarding incremental decode
        events, metrics snapshots, endpoint events, finals, perf cycles, & forced finals.
        """
        try:
            async for event in incremental_decode_loop(
                state.audio_session,
                fw_service,
                ep_state,
                DecoderConfig(
                    language=language,
                    decode_interval_ms=decode_interval_ms,
                    window_seconds=window_seconds,
                ),
            ):
                # Forward events to client
                etype = event["type"]
                if etype == "partial_transcript":
                    if state.metrics and "decode_ms" in event:
                        state.metrics.inc_partial()
                    await _send_json(state.websocket, event)
                    # Emit metrics snapshot as a dedicated event (easier for tests)
                    await _send_json(state.websocket, {
                        "type": "metrics_snapshot",
                        **(state.metrics.snapshot() if state.metrics else {}),
                    })
                elif etype == "endpoint_event":
                    if state.metrics:
                        state.metrics.update_endpoint(event.get("end_silence_gap_ms"))
                    await _send_json(state.websocket, {"type": "endpoint", **event})
                elif etype == "final_transcript":
                    if state.metrics:
                        state.metrics.inc_final()
                    final_text_candidate = (event.get("text") or "").strip()
                    if not final_text_candidate:
                        logger.debug("[streaming_voice] Ignoring empty final_transcript event (utterance_index=%s)", event.get("utterance_index"))
                        continue
                    utterance_index = event.get("utterance_index", 0)
                    # Deduplicate by utterance index
                    if utterance_index in state.processed_utterance_indices:
                        logger.debug("[streaming_voice] Skipping already processed utterance_index=%s", utterance_index)
                        continue
                    # Window-based textual dedupe (handles echo loops producing identical finals rapidly)
                    dedupe_window = float(os.getenv("VOICE_STREAMING_DEDUPE_FINAL_WINDOW_SEC", "1.5"))
                    if (state.last_final_text and final_text_candidate == state.last_final_text and (time.time() - state.last_final_time) < dedupe_window):
                        logger.debug("[streaming_voice] Skipping duplicate final text within window: '%s'", final_text_candidate)
                        continue
                    state.processed_utterance_indices.add(utterance_index)
                    state.last_final_text = final_text_candidate
                    state.last_final_time = time.time()
                    await _send_json(state.websocket, event)
                    if state.metrics:
                        maybe_log_structured(logger, "voice_stream_metrics", state.metrics.snapshot())
                    final_text = final_text_candidate
                    min_chars = int(os.getenv("VOICE_STREAMING_MIN_FINAL_CHARS", "3"))
                    if len(final_text) < min_chars:
                        logger.debug(
                            "[streaming_voice] Final transcript below min chars (%d < %d) skipping LLM pipeline",
                            len(final_text),
                            min_chars,
                        )
                    else:
                        if state.llm_tts_task is None:
                            state.llm_tts_task = asyncio.create_task(
                                _process_final_transcript(
                                    utterance_index,
                                    final_text,
                                    language,
                                    state,
                                )
                            )
                        else:
                            # Queue this final for later processing to avoid dropping
                            state.pending_finals.append((utterance_index, final_text))
                            await _send_json(state.websocket, {
                                "type": "final_queued",
                                "utterance_index": utterance_index,
                                "queue_len": len(state.pending_finals),
                                "timestamp": time.time(),
                            })
                # If pipeline started (or queued) but assistant_response not yet emitted after timeout window, schedule watchdog
                if state.pipeline_active_for is not None and state.llm_tts_task is not None:
                    async def _watchdog(active_utt: int, start_time: float):
                        try:
                            await asyncio.sleep(float(os.getenv("VOICE_STREAMING_PIPELINE_WATCHDOG_SEC", "12")))
                            if state.pipeline_active_for == active_utt and active_utt not in state.emitted_assistant_for:
                                await _send_json(state.websocket, {
                                    "type": "assistant_pipeline_watchdog",
                                    "utterance_index": active_utt,
                                    "elapsed_ms": int((time.time() - start_time) * 1000),
                                    "message": "No assistant_response yet (watchdog)"
                                })
                        except Exception:
                            pass
                    asyncio.create_task(_watchdog(state.pipeline_active_for, state.last_pipeline_start or time.time()))
                elif etype == "perf_cycle":
                    if state.metrics:
                        state.metrics.update_perf_cycle(
                            decode_ms=event.get("decode_ms", 0),
                            cycle_latency_ms=event.get("cycle_latency_ms", 0),
                        )
                        if state.metrics.decode_ms.count % 10 == 0:
                            maybe_log_structured(logger, "voice_stream_perf_cycle", {
                                "session_id": session_id,
                                "decode_ms_last": event.get("decode_ms"),
                                "cycle_latency_ms_last": event.get("cycle_latency_ms"),
                            })
                    await _send_json(state.websocket, event)
                # Forced final injection logic
                if (
                    os.getenv("VOICE_STREAMING_LENIENT_FINAL", "0") == "1" and
                    (time.time() - session_start_time) > force_final_grace and
                    etype == "perf_cycle" and
                    state.metrics and
                    state.metrics.final_transcripts == 0 and
                    state.metrics.partial_transcripts > 0
                ):
                    await _send_json(state.websocket, {
                        "type": "final_transcript",
                        "text": "",
                        "utterance_index": 0,
                        "reason": "force_final_grace_expired",
                        "timestamp": time.time(),
                    })
        except Exception as e:  # pragma: no cover
            logger.error("Incremental decoder error (%s): %s", connection_id, e)

    # Schedule appropriate processing task based on phase selection
    if run_phase4:
        state.mock_task = asyncio.create_task(decoder_task())
        await _send_json(websocket, {"type": "decoder_started", "timestamp": time.time(), "mode": "incremental"})
    else:
        state.mock_task = asyncio.create_task(_mock_decode_loop(state, language))
        await _send_json(websocket, {"type": "decoder_started", "timestamp": time.time(), "mode": "mock"})

    try:
        # Heartbeat / liveness task
        async def _heartbeat_loop():
            while True:
                await asyncio.sleep(2.0)
                if state.disconnected:
                    break
                hb_payload = {
                    "type": "heartbeat",
                    "ts": time.time(),
                    "bytes_received": state.bytes_received,
                    "pcm_frames": state.pcm_frames_received,
                    "buffer_usage": (state.audio_session.pcm_buffer.usage_ratio() if state.audio_session else None),
                    "voiced_frames": state.voiced_frame_count,
                    "silence_frames": state.silence_frame_count,
                    "low_energy_mode": state.low_energy_mode,
                }
                await _send_json(websocket, hb_payload)
                state.last_heartbeat_sent = time.time()
        state.heartbeat_task = asyncio.create_task(_heartbeat_loop())

        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    try:
                        data = json.loads(msg["text"]) if msg["text"] else {}
                        if data.get("type") == "ping":
                            await _send_json(websocket, {"type": "pong", "ts": time.time()})
                        else:
                            await _send_json(websocket, {"type": "ack", "ts": time.time()})
                        state.last_client_msg = time.time()
                    except json.JSONDecodeError:
                        await _send_json(websocket, {"type": "error", "message": "invalid_json"})
                elif "bytes" in msg and msg["bytes"]:
                    payload = msg["bytes"]
                    state.bytes_received += len(payload)
                    state.touch()
                    state.last_client_msg = time.time()
                    # Hybrid ingest logic: detect compressed container on first frame
                    if state.compressed_mode is None and state.pcm_frames_received == 0 and state.bytes_received < 32768:
                        # Basic signature checks
                        if len(payload) >= 4 and payload[:4] == b"\x1a\x45\xdf\xa3":
                            state.compressed_mode = "webm-opus"
                        elif len(payload) >= 4 and payload[:4] == b"OggS":
                            state.compressed_mode = "ogg-opus"
                        if state.compressed_mode:
                            allow = os.getenv("VOICE_STREAMING_ALLOW_WEBM", "1") == "1"
                            if not allow:
                                await _send_json(websocket, {"type": "error", "stage": "ingest", "message": f"Compressed {state.compressed_mode} not allowed (set VOICE_STREAMING_ALLOW_WEBM=1)"})
                                await websocket.close(code=1003, reason="compressed_not_allowed")
                                break
                            # Spawn ffmpeg decoder process
                            try:
                                # Map mode to ffmpeg demux format parameters
                                demux_hint = []  # ffmpeg usually auto-detects
                                cmd = [
                                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                                    "-i", "pipe:0",
                                    "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
                                ]
                                state.ffmpeg_proc = await asyncio.create_subprocess_exec(
                                    *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
                                )
                                state.ffmpeg_writer_open = True
                                await _send_json(websocket, {"type": "info", "stage": "ingest", "message": f"Activated {state.compressed_mode} decoder"})
                                # Emit ingest_mode event so frontend can display mode early
                                await _send_json(websocket, {
                                    "type": "ingest_mode",
                                    "mode": state.compressed_mode,
                                    "timestamp": time.time(),
                                    "bytes_received": state.bytes_received,
                                    "frame_hint_bytes": 640,
                                })

                                async def _reader():
                                    assert state.ffmpeg_proc and state.ffmpeg_proc.stdout
                                    CHUNK = 640  # 20 ms @16k mono int16
                                    buf = bytearray()
                                    try:
                                        while True:
                                            data = await state.ffmpeg_proc.stdout.read(4096)
                                            if not data:
                                                break
                                            buf.extend(data)
                                            while len(buf) >= CHUNK:
                                                frame = bytes(buf[:CHUNK])
                                                del buf[:CHUNK]
                                                if state.audio_session:
                                                    await state.audio_session.ingest_pcm(frame)
                                                    state.pcm_frames_received += 1
                                                    if state.pcm_frames_received <= 3:
                                                        logger.debug("[%s] (dec) ingested PCM frame bytes=%d buf_usage=%.3f", connection_id, len(frame), state.audio_session.pcm_buffer.usage_ratio())
                                    except Exception as e:
                                        logger.warning("ffmpeg reader error %s: %s", connection_id, e)
                                state.ffmpeg_reader_task = asyncio.create_task(_reader())
                            except FileNotFoundError:
                                await _send_json(websocket, {"type": "error", "stage": "ingest", "message": "ffmpeg not installed on server"})
                                state.compressed_mode = None
                            except Exception as e:
                                logger.exception("Failed to start ffmpeg decoder: %s", e)
                                await _send_json(websocket, {"type": "error", "stage": "ingest", "message": f"decoder_start_failed: {e}"})
                                state.compressed_mode = None

                    if state.compressed_mode:
                        # Feed compressed payload into ffmpeg stdin
                        if state.ffmpeg_proc and state.ffmpeg_proc.stdin and state.ffmpeg_writer_open:
                            try:
                                state.ffmpeg_proc.stdin.write(payload)
                                await state.ffmpeg_proc.stdin.drain()
                            except Exception as e:
                                logger.warning("ffmpeg stdin write failed %s: %s", connection_id, e)
                                state.ffmpeg_writer_open = False
                    else:
                        # Raw PCM path (default)
                        if state.audio_session:
                            if len(payload) % 2 != 0:
                                logger.debug("[%s] odd-length PCM payload=%d", connection_id, len(payload))
                            # Simple energy gate (average absolute amplitude)
                            try:
                                # interpret payload as int16 little-endian
                                import array
                                arr = array.array('h')
                                arr.frombytes(payload)
                                if arr:
                                    avg_abs = sum(abs(s) for s in arr) / len(arr)
                                    # threshold ~ 220 (~0.0067 full scale) heuristically tuned
                                    if avg_abs > 220:
                                        state.voiced_frame_count += 1
                                        if state.low_energy_mode and state.voiced_frame_count >= 3:
                                            state.low_energy_mode = False
                                    else:
                                        state.silence_frame_count += 1
                            except Exception:
                                pass
                            await state.audio_session.ingest_pcm(payload)
                            state.pcm_frames_received += 1
                            if state.pcm_frames_received <= 3:  # log only first few for noise control
                                logger.debug("[%s] ingested PCM frame bytes=%d buffer_usage=%.3f", connection_id, len(payload), state.audio_session.pcm_buffer.usage_ratio())
                            # Emit ingest_mode once on first raw PCM frame (if not already sent)
                            if state.pcm_frames_received == 1 and state.compressed_mode is None:
                                await _send_json(websocket, {
                                    "type": "ingest_mode",
                                    "mode": "pcm16le",
                                    "timestamp": time.time(),
                                    "bytes_received": state.bytes_received,
                                    "frame_size_bytes": len(payload),
                                })
    except WebSocketDisconnect:
        logger.info("🔌 Streaming voice disconnect: %s", connection_id)
    except Exception as e:  # pragma: no cover
        logger.error("❌ Streaming voice error %s: %s", connection_id, e)
    finally:
        state.disconnected = True
        # Emit session_end early if still open
        with contextlib.suppress(Exception):
            await _send_json(websocket, {
                "type": "session_end",
                "timestamp": time.time(),
                "bytes_received": state.bytes_received,
                "pcm_frames": state.pcm_frames_received,
                "voiced_frames": state.voiced_frame_count,
                "silence_frames": state.silence_frame_count,
                "low_energy_mode": state.low_energy_mode,
            })
        if state.mock_task and not state.mock_task.done():
            state.mock_task.cancel()
            with contextlib.suppress(Exception):  # type: ignore[arg-type]
                await state.mock_task
        # Phase 8: ensure background LLM+TTS task cancelled
        if state.llm_tts_task and not state.llm_tts_task.done():
            state.llm_tts_task.cancel()
            with contextlib.suppress(Exception):
                await state.llm_tts_task
        # Heartbeat task
        if state.heartbeat_task and not state.heartbeat_task.done():
            state.heartbeat_task.cancel()
            with contextlib.suppress(Exception):
                await state.heartbeat_task
        # Close ring buffer
        if state.audio_session and not state.audio_session.pcm_buffer.closed:
            await state.audio_session.pcm_buffer.close()
        # Tear down decoder if active
        if state.ffmpeg_proc:
            with contextlib.suppress(Exception):
                if state.ffmpeg_proc.stdin and state.ffmpeg_writer_open:
                    state.ffmpeg_proc.stdin.close()
            if state.ffmpeg_reader_task and not state.ffmpeg_reader_task.done():
                state.ffmpeg_reader_task.cancel()
                with contextlib.suppress(Exception):
                    await state.ffmpeg_reader_task
            with contextlib.suppress(Exception):
                await asyncio.wait_for(state.ffmpeg_proc.wait(), timeout=2)
        session_registry.remove(connection_id)
        logger.info("🧹 Cleaned streaming session %s (active=%d)", connection_id, session_registry.active_count())
        # Emit final metrics summary if still connected (best-effort; socket may already be closed)
        with contextlib.suppress(Exception):
            await _send_json(websocket, {
                "type": "ingest_summary",
                "mode": state.compressed_mode or "pcm16le",
                "total_bytes": state.bytes_received,
                "pcm_frames": state.pcm_frames_received,
                "duration_sec": round(time.time() - state.created_at, 3),
            })


@streaming_voice_router.get("/streaming-voice/status")
async def streaming_voice_status() -> Dict[str, Any]:
    """Unified lightweight status endpoint for streaming voice feature.

    Consolidated duplicates introduced during phased development; provides a stable
    shape for monitoring & docs.
    """
    run_phase4 = os.getenv("VOICE_STREAMING_PHASE4", "0") == "1"
    return {
        "enabled": streaming_feature_enabled(),
        "active_sessions": session_registry.active_count(),
        "phase": 4 if run_phase4 else 2,
        "endpoint": "/api/v1/ws/streaming-voice",
        "description": (
            "Streaming voice operational (incremental windowed decode)" if run_phase4 else
            "Streaming voice operational (mock decode + PCM buffering)"
        ),
    }
