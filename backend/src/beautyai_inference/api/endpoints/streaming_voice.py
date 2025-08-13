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
from beautyai_inference.services.voice.transcription.faster_whisper_service import FasterWhisperTranscriptionService

logger = logging.getLogger(__name__)

STREAMING_FEATURE_ENABLED = os.getenv("VOICE_STREAMING_ENABLED", "0") == "1"

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
                drops = rb.stats.total_dropped
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
    """Phase 1 streaming voice endpoint (mock transcription only)."""
    if not STREAMING_FEATURE_ENABLED:
        await websocket.close(code=1003, reason="Streaming voice disabled")
        return
    if language not in ("ar", "en"):
        await websocket.close(code=1003, reason="Unsupported language")
        return

    connection_id = str(uuid.uuid4())
    session_id = f"stream_{connection_id}"
    await websocket.accept()

    state = SessionState(connection_id=connection_id, session_id=session_id, websocket=websocket)
    # Phase 2: attach audio streaming session (ring buffer size ~40s)
    state.audio_session = StreamingSession(connection_id=connection_id, session_id=session_id, language=language)
    session_registry.add(state)
    logger.info("🌐 Streaming voice connection established: %s (lang=%s)", connection_id, language)

    run_phase4 = os.getenv("VOICE_STREAMING_PHASE4", "0") == "1"

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
    })

    if run_phase4:
        fw_service = FasterWhisperTranscriptionService()
        ep_state = EndpointState(config=EndpointConfig())

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
            await _send_json(state.websocket, {
                "type": "tts_start",
                "utterance_index": utterance_index,
                "timestamp": started_ts,
            })
            try:
                services = await _ensure_services()
                chat_service = services["chat"]
                voice_service = services["voice"]

                # Append user message to conversation history
                state.conversation.append({"role": "user", "content": text})

                # Generate LLM response (limit length for latency)
                chat_result = chat_service.chat(
                    message=text,
                    conversation_history=state.conversation,
                    max_length=256,
                    language=lang or "auto",
                    temperature=0.3,
                )
                if not chat_result.get("success"):
                    raise RuntimeError(f"Chat generation failed: {chat_result.get('error')}")
                response_text = chat_result.get("response", "")
                if not response_text:
                    response_text = ""

                state.conversation.append({"role": "assistant", "content": response_text})

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
            except Exception as e:  # pragma: no cover
                logger.exception("TTS pipeline failure (%s): %s", connection_id, e)
                await _send_json(state.websocket, {
                    "type": "error",
                    "stage": "tts_pipeline",
                    "message": str(e),
                    "utterance_index": utterance_index,
                })
            finally:
                state.llm_tts_task = None

        async def decoder_task() -> None:
            try:
                async for event in incremental_decode_loop(
                    state.audio_session, fw_service, ep_state, DecoderConfig(language=language)
                ):
                    # Forward events to client
                    etype = event["type"]
                    if etype == "partial_transcript":
                        await _send_json(state.websocket, event)
                    elif etype == "endpoint_event":
                        await _send_json(state.websocket, {"type": "endpoint", **event})
                    elif etype == "final_transcript":
                        await _send_json(state.websocket, event)
                        # Kick off LLM + TTS if not already running for this utterance
                        if state.llm_tts_task is None:
                            state.llm_tts_task = asyncio.create_task(
                                _process_final_transcript(
                                    event.get("utterance_index", 0),
                                    event.get("text", ""),
                                    language,
                                    state,
                                )
                            )
            except Exception as e:  # pragma: no cover
                logger.error("Incremental decoder error (%s): %s", connection_id, e)

        state.mock_task = asyncio.create_task(decoder_task())
    else:
        state.mock_task = asyncio.create_task(_mock_decode_loop(state, language))

    try:
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
                    except json.JSONDecodeError:
                        await _send_json(websocket, {"type": "error", "message": "invalid_json"})
                elif "bytes" in msg and msg["bytes"]:
                    payload = msg["bytes"]
                    state.bytes_received += len(payload)
                    state.touch()
                    # Expect raw little-endian int16 PCM @ 16kHz
                    if state.audio_session:
                        if len(payload) % 2 != 0:
                            logger.debug("[%s] odd-length PCM payload=%d", connection_id, len(payload))
                        await state.audio_session.ingest_pcm(payload)
                        state.pcm_frames_received += 1
                        if state.pcm_frames_received <= 3:  # log only first few for noise control
                            logger.debug("[%s] ingested PCM frame bytes=%d buffer_usage=%.3f", connection_id, len(payload), state.audio_session.pcm_buffer.usage_ratio())
    except WebSocketDisconnect:
        logger.info("🔌 Streaming voice disconnect: %s", connection_id)
    except Exception as e:  # pragma: no cover
        logger.error("❌ Streaming voice error %s: %s", connection_id, e)
    finally:
        state.disconnected = True
        if state.mock_task and not state.mock_task.done():
            state.mock_task.cancel()
            with contextlib.suppress(Exception):  # type: ignore[arg-type]
                await state.mock_task
        # Close ring buffer
        if state.audio_session and not state.audio_session.pcm_buffer.closed:
            await state.audio_session.pcm_buffer.close()
        session_registry.remove(connection_id)
        logger.info("🧹 Cleaned streaming session %s (active=%d)", connection_id, session_registry.active_count())


@streaming_voice_router.get("/streaming-voice/status")
async def streaming_voice_status() -> Dict[str, Any]:
    """Lightweight status endpoint for monitoring."""
    # Provide aggregate ring buffer stats (lightweight – approximated)
    run_phase4 = os.getenv("VOICE_STREAMING_PHASE4", "0") == "1"
    return {
        "feature_enabled": STREAMING_FEATURE_ENABLED,
        "active_connections": session_registry.active_count(),
        "phase": 4 if run_phase4 else 2,
        "description": "Streaming voice operational (incremental windowed decode)" if run_phase4 else "Streaming voice operational (mock decode + PCM buffering)",
    }
