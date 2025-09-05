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
from pathlib import Path
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
from beautyai_inference.services.voice.echo_suppression import EchoSuppressor, EchoConfig
from beautyai_inference.logging.setup import session_id_ctx
from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service
from beautyai_inference.services.voice.utils.text_cleaning import sanitize_tts_text
from beautyai_inference.utils import create_realtime_decoder, WebMDecodingError

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
    bytes_received: int = 0  # Raw bytes received (includes WebM containers)
    pcm_bytes_received: int = 0  # Actual PCM bytes ingested into buffer
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
    webm_decoder: Optional[Any] = None  # WebMDecoder instance for compressed audio
    decoder_task: Optional[asyncio.Task] = None  # Background decoding task
    webm_chunk_queue: Optional[asyncio.Queue] = None  # Queue for WebM chunks
    heartbeat_task: Optional[asyncio.Task] = None
    last_heartbeat_sent: float = 0.0
    last_client_msg: float = field(default_factory=time.time)
    silence_frame_count: int = 0
    voiced_frame_count: int = 0
    low_energy_mode: bool = True  # assume silent until energy detected
    requested_language: Optional[str] = None  # original language param (may be 'auto')
    mismatch_notified: bool = False  # whether we already emitted a language mismatch advisory
    # Dedupe tracking additions (Aug 21 2025)
    emitted_pipeline_for: set[int] = field(default_factory=set)  # assistant_pipeline_start sent for these utterances
    last_endpoint_sig: Optional[str] = None
    last_endpoint_time: float = 0.0
    # Echo suppression for duplex streaming
    echo_suppressor: Optional[EchoSuppressor] = None
    tts_streaming_task: Optional[asyncio.Task] = None  # Task for streaming TTS chunks
    tts_streaming_active: bool = False

    def touch(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = time.time()

    async def reset_counters_for_new_utterance(self) -> None:
        """Reset per-utterance metrics after final transcript processing.
        
        This ensures heartbeat metrics reflect only current utterance activity
        and prevents counter inflation across multiple utterances.
        """
        # Reset PCM byte and frame counters for new utterance  
        self.pcm_bytes_received = 0
        self.pcm_frames_received = 0
        # Reset voice activity counters
        self.voiced_frame_count = 0
        self.silence_frame_count = 0
        # Reset to low energy mode for new utterance
        self.low_energy_mode = True
        
        # Also reset the audio session counters
        if self.audio_session:
            await self.audio_session.reset_counters_for_new_utterance()


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

# Binary frame protocol for duplex streaming
class MessageType:
    """Binary message types for duplex audio streaming."""
    MIC_CHUNK = 0x01
    TTS_CHUNK = 0x02  
    CONTROL = 0x03
    META = 0x04

class MessageFlags:
    """Binary message flags."""
    START = 0x01
    END = 0x02
    URGENT = 0x04
    COMPRESSED = 0x08

def pack_binary_frame(msg_type: int, seq_num: int, flags: int, payload: bytes) -> bytes:
    """Pack binary frame with header."""
    import struct
    timestamp = int(time.time() * 1000) & 0xFFFFFFFF  # 32-bit timestamp in ms
    header = struct.pack('<BBHBBBI', msg_type, seq_num & 0xFF, (seq_num >> 8) & 0xFFFF, flags, 0, 0, timestamp)
    return header + payload

def unpack_binary_frame(data: bytes) -> tuple[int, int, int, int, bytes]:
    """Unpack binary frame header."""
    import struct
    if len(data) < 8:
        raise ValueError("Frame too short")
    
    header = struct.unpack('<BBHBBBI', data[:8])
    msg_type, seq_low, seq_high, flags, _, _, timestamp = header
    seq_num = seq_low | (seq_high << 8)
    payload = data[8:]
    
    return msg_type, seq_num, flags, timestamp, payload

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
    # Accept 'auto' as a valid option (previously caused immediate close & silent failure on UI)
    original_language = language
    if language not in ("ar", "en", "auto"):
        await _send_json(websocket, {
            "type": "error",
            "stage": "init",
            "message": f"Unsupported language '{language}' (allowed: ar,en,auto)",
        })
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="unsupported_language")
        return
    # For now we do not have true automatic incremental language switching in the decoder loop yet.
    # When 'auto' is requested we pick a default decode language (configurable) but surface metadata so
    # the client can adjust UI. Later phases can integrate real detection & dynamic switching.
    if language == "auto":
        language = os.getenv("VOICE_STREAMING_AUTO_FALLBACK", "ar")

    connection_id = str(uuid.uuid4())
    session_id = f"stream_{connection_id}"

    state = SessionState(connection_id=connection_id, session_id=session_id, websocket=websocket)
    # Bind session id to logging context for structured correlation
    session_ctx_token = session_id_ctx.set(session_id)
    state.requested_language = original_language
    state.metrics = SessionMetrics(session_id=session_id)
    # Phase 2: attach audio streaming session (ring buffer size ~40s)
    state.audio_session = StreamingSession(connection_id=connection_id, session_id=session_id, language=language)
    
    # Initialize echo suppressor for duplex streaming
    echo_config = EchoConfig(
        energy_threshold=float(os.getenv("VOICE_STREAMING_VAD_ENERGY_THRESHOLD", "0.01")),
        min_speech_duration_ms=int(os.getenv("VOICE_STREAMING_MIN_SPEECH_DURATION", "300")),
        tts_duck_db=float(os.getenv("VOICE_STREAMING_TTS_DUCK_DB", "-12.0")),
        tts_pause_threshold_ms=int(os.getenv("VOICE_STREAMING_TTS_PAUSE_THRESHOLD", "500")),
        gate_mic_during_tts=os.getenv("VOICE_STREAMING_GATE_MIC_DURING_TTS", "1") == "1",
    )
    state.echo_suppressor = EchoSuppressor(config=echo_config)
    
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
        "requested_language": original_language,
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
    if original_language == "auto":
        await _send_json(websocket, {
            "type": "info",
            "stage": "init",
            "message": f"Auto language mode active (effective decode language='{language}')",
            "requested_language": original_language,
            "effective_language": language,
            "timestamp": time.time(),
        })

    if run_phase4:
        try:
            # UPDATED: Use ModelManager for persistent Whisper model
            from beautyai_inference.core.model_manager import ModelManager
            model_manager = ModelManager()
            fw_service = model_manager.get_streaming_whisper()
            
            if fw_service is None:
                logger.warning("Failed to get persistent Whisper model, falling back to factory")
                fw_service = create_transcription_service()
            else:
                logger.info("✅ Using persistent Whisper model from ModelManager")
                
        except Exception as e:
            logger.exception("Failed to initialize transcription service, falling back to mock decode: %s", e)
            await _send_json(websocket, {"type": "warning", "stage": "init", "message": f"transcription_init_failed: {e}"})
            run_phase4 = False
        # Endpoint / decoder dynamic tuning via env
        decode_interval_ms = int(os.getenv("VOICE_STREAMING_DECODE_INTERVAL_MS", "480"))
        window_seconds = float(os.getenv("VOICE_STREAMING_WINDOW_SECONDS", "8.0"))
        min_speech_ms = int(os.getenv("VOICE_STREAMING_MIN_SPEECH_MS", "480"))  # New configurable parameter
        min_silence_ms = int(os.getenv("VOICE_STREAMING_MIN_SILENCE_MS", "720"))  # Increased default
        token_stable_ms = int(os.getenv("VOICE_STREAMING_TOKEN_STABLE_MS", "600"))
        max_utterance_ms = int(os.getenv("VOICE_STREAMING_MAX_UTTERANCE_MS", "12000"))
        min_token_growth_cycles = int(os.getenv("VOICE_STREAMING_MIN_TOKEN_GROWTH_CYCLES", "3"))
        stability_buffer_ms = int(os.getenv("VOICE_STREAMING_STABILITY_BUFFER_MS", "240"))
        
        ep_cfg = EndpointConfig(
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            token_stable_ms=token_stable_ms,
            max_utterance_ms=max_utterance_ms,
            min_token_growth_cycles=min_token_growth_cycles,
            stability_buffer_ms=stability_buffer_ms,
        )
        ep_state = EndpointState(config=ep_cfg)
        # Provide aggressive fast-path if requested
        if os.getenv("VOICE_STREAMING_LOW_LATENCY_PRESET", "0") == "1":
            ep_state.config.min_speech_ms = min(ep_state.config.min_speech_ms, 240)
            ep_state.config.min_silence_ms = min(ep_state.config.min_silence_ms, 480)
            ep_state.config.token_stable_ms = min(ep_state.config.token_stable_ms, 480)
            ep_state.config.min_token_growth_cycles = min(ep_state.config.min_token_growth_cycles, 2)
            ep_state.config.stability_buffer_ms = min(ep_state.config.stability_buffer_ms, 120)
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
                from beautyai_inference.config.config_manager import AppConfig
                chat = ChatService()
                # Attempt to load default model for fast responses
                try:
                    success = chat.load_default_model_from_config()
                    if success:
                        logger.info("✅ Chat service initialized successfully")
                    else:
                        logger.warning("⚠️ Chat service initialized but default model loading failed")
                    # Load registry to capture model_config for required chat() signature
                    try:
                        config_dir = Path(__file__).parent.parent.parent / "config"
                        app_config = AppConfig()
                        app_config.models_file = str(config_dir / "model_registry.json")
                        app_config.load_model_registry()
                        default_name = "qwen3-unsloth-q4ks"
                        model_cfg = app_config.model_registry.get_model(default_name)
                        if model_cfg is None:
                            logger.error("❌ Default model '%s' not found in registry", default_name)
                        chat_service_ref["model_name"] = default_name
                        chat_service_ref["model_config"] = model_cfg
                    except Exception as reg_e:  # pragma: no cover - defensive
                        logger.error("Failed loading model registry for chat service: %s", reg_e)
                except Exception as e:
                    logger.error(f"❌ Chat service initialization failed: {e}")
                    # Don't suppress - we need to know about this
                    raise
                chat_service_ref["chat"] = chat
            if "voice" not in voice_service_ref:
                from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
                voice = SimpleVoiceService()
                # Initialize Edge TTS etc.
                try:
                    await voice.initialize()
                    logger.info("✅ Voice service initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Voice service initialization failed: {e}")
                    # Don't suppress - we need to know about this
                    raise
                voice_service_ref["voice"] = voice
            return {"chat": chat_service_ref["chat"], "voice": voice_service_ref["voice"]}

    async def _process_final_transcript(utterance_index: int, text: str, lang: str, state: SessionState) -> None:
            """Handle LLM + TTS pipeline for a finalized user utterance.

            Emits sequence: tts_start -> tts_audio (base64) -> tts_complete
            Failure paths emit error event.
            
            Note: The `text` parameter should already include '/no_think' suffix
            for voice conversations to disable thinking mode and improve response speed.
            """
            started_ts = time.time()
            state.last_pipeline_start = started_ts
            state.pipeline_active_for = utterance_index
            debug_flag = os.getenv("VOICE_STREAMING_DEBUG_PIPELINE", "1") == "1"
            if debug_flag:
                # Dedupe assistant_pipeline_start per utterance
                if utterance_index not in state.emitted_pipeline_for:
                    await _send_json(state.websocket, {
                        "type": "assistant_pipeline_start",
                        "utterance_index": utterance_index,
                        "chars": len(text),
                        "timestamp": started_ts,
                    })
                    state.emitted_pipeline_for.add(utterance_index)
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
                try:
                    services = await _ensure_services()
                    chat_service = services["chat"]
                    voice_service = services["voice"]
                except Exception as e:
                    logger.error(f"❌ Service initialization failed in LLM pipeline: {e}")
                    await _send_json(state.websocket, {
                        "type": "error",
                        "stage": "service_init",
                        "message": f"Failed to initialize services: {str(e)}",
                        "utterance_index": utterance_index,
                    })
                    return

                # NOTE: Do NOT append the current user message before calling chat().
                # chat_service.chat expects 'message' to be the *new* user turn and
                # 'conversation_history' to contain ONLY previous exchanges.
                # Previously we appended first which duplicated the user content in the prompt
                # inflating effective context tokens and causing context window overflow.

                # Apply conversation history management to prevent context confusion
                # For voice conversations, we typically want shorter context windows to avoid repetitive responses
                max_history_turns = int(os.getenv("VOICE_STREAMING_MAX_HISTORY_TURNS", "6"))  # Default: 3 user + 3 assistant turns
                
                # Get conversation history with length limitation
                if max_history_turns <= 0:
                    # If set to 0 or negative, disable conversation history (fresh context each time)
                    prev_history = []
                else:
                    # Take the most recent exchanges only
                    prev_history = list(state.conversation[-max_history_turns:])
                
                logger.debug(f"[streaming_voice] Using conversation history: {len(prev_history)} turns (max: {max_history_turns})")

                # Generate LLM response (limit length for latency)
                # Offload potentially blocking LLM generation to thread pool to avoid stalling event loop
                loop = asyncio.get_running_loop()
                def _run_chat():
                    # Always forward the originally requested language (state.requested_language)
                    # so enforcement logic can act even if 'auto' was converted earlier.
                    try:
                        model_name = chat_service_ref.get("model_name", "qwen3-unsloth-q4ks")
                        model_config = chat_service_ref.get("model_config")
                        if model_config is None:
                            raise RuntimeError("Model config not loaded for chat service")
                        generation_config = {
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "max_new_tokens": 192,
                            "enable_thinking": False,  # Disable thinking mode for voice conversations
                        }
                        response, detected_language, updated_history, _session = chat_service.chat(
                            message=text,
                            model_name=model_name,
                            model_config=model_config,
                            generation_config=generation_config,
                            conversation_history=prev_history,
                            response_language=state.requested_language or lang or "auto",
                            disable_content_filter=True,
                        )
                        return {
                            "success": True,
                            "response": response,
                            "detected_language": detected_language,
                            "updated_history": updated_history,
                        }
                    except Exception as e:
                        logger.error(f"❌ Chat service failed during generation: {e}")
                        return {"success": False, "error": f"Chat generation failed: {str(e)}"}
                
                try:
                    chat_result = await loop.run_in_executor(None, _run_chat)
                except Exception as e:
                    logger.error(f"❌ Chat execution failed in executor: {e}")
                    chat_result = {"success": False, "error": f"Chat execution failed: {str(e)}"}
                
                if not chat_result.get("success"):
                    # Provide a graceful fallback response instead of raising to keep audio pipeline alive
                    error_msg = chat_result.get('error', 'Unknown error')
                    logger.warning("Chat generation failed for utterance %s: %s", utterance_index, error_msg)
                    await _send_json(state.websocket, {
                        "type": "error",
                        "stage": "llm_generation",
                        "message": error_msg,
                        "utterance_index": utterance_index,
                    })
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
                    "requested_language": state.requested_language,
                    "effective_language": chat_result.get("detected_language"),
                })
                try:
                    maybe_log_structured(logger, "assistant_response", {
                        "session_id": state.session_id,
                        "utterance_index": utterance_index,
                        "requested_language": state.requested_language,
                        "effective_language": chat_result.get("detected_language"),
                        "chars": len(response_text),
                    })
                except Exception:
                    pass
                state.emitted_assistant_for.add(utterance_index)
                # Increment assistant turn counter early (text surfaced)
                state.assistant_turns += 1

                # Synthesize speech (use language heuristic)
                from base64 import b64encode
                # Prefer enforced/explicit requested language for synthesis if available
                synth_lang = state.requested_language if state.requested_language in ("ar","en") else chat_result.get("detected_language", lang or "ar")
                
                # Use duplex streaming if enabled
                duplex_streaming = os.getenv("VOICE_STREAMING_DUPLEX_ENABLED", "1") == "1"
                
                if duplex_streaming and state.echo_suppressor:
                    # Stream TTS chunks for duplex communication
                    await _stream_tts_response(response_text, synth_lang, utterance_index, state)
                else:
                    # Fallback to traditional base64 audio (for compatibility)
                    tts_path = await voice_service.text_to_speech(response_text, language=synth_lang)
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

    async def _stream_tts_response(text: str, language: str, utterance_index: int, state: SessionState) -> None:
        """Stream TTS response as binary audio chunks for duplex communication."""
        try:
            logger.debug(f"Starting TTS streaming for utterance {utterance_index}: {len(text)} chars")
            
            # Notify echo suppressor that TTS is starting
            if state.echo_suppressor:
                await state.echo_suppressor.on_tts_start()
            
            state.tts_streaming_active = True
            chunk_seq = 0
            
            # Get TTS engine from model manager or create new one
            from beautyai_inference.core.model_manager import ModelManager
            model_manager = ModelManager()
            
            # Try to get Edge TTS engine
            try:
                edge_tts_engine = model_manager.get_tts_engine("edge-tts")
                if edge_tts_engine is None:
                    # Create new Edge TTS engine
                    from beautyai_inference.inference_engines.edge_tts_engine import EdgeTTSEngine
                    from beautyai_inference.config.config_manager import ModelConfig
                    config = ModelConfig(name="edge-tts", model_id="edge-tts", engine_type="edge_tts")
                    edge_tts_engine = EdgeTTSEngine(config)
                    edge_tts_engine.load_model()
                
                # Stream TTS audio chunks
                async for audio_chunk in edge_tts_engine.stream_tts_chunks(
                    text=text,
                    language=language,
                    chunk_size_ms=40,  # 40ms chunks for low latency
                    target_sample_rate=16000
                ):
                    if not state.tts_streaming_active or state.disconnected:
                        logger.debug("TTS streaming cancelled")
                        break
                    
                    # Process chunk through echo suppressor
                    if state.echo_suppressor:
                        processed_chunk = await state.echo_suppressor.process_tts_audio(audio_chunk)
                    else:
                        processed_chunk = audio_chunk
                    
                    # Pack as binary frame
                    flags = MessageFlags.START if chunk_seq == 0 else 0
                    binary_frame = pack_binary_frame(
                        MessageType.TTS_CHUNK,
                        chunk_seq,
                        flags,
                        processed_chunk
                    )
                    
                    # Send binary frame to client
                    if state.websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await state.websocket.send_bytes(binary_frame)
                            chunk_seq += 1
                            
                            # Emit progress event
                            if chunk_seq % 10 == 0:  # Every ~400ms
                                await _send_json(state.websocket, {
                                    "type": "tts_progress",
                                    "utterance_index": utterance_index,
                                    "chunks_sent": chunk_seq,
                                    "timestamp": time.time(),
                                })
                        except Exception as e:
                            logger.warning(f"Failed to send TTS chunk: {e}")
                            break
                
                # Send end marker
                if state.tts_streaming_active and not state.disconnected:
                    end_frame = pack_binary_frame(
                        MessageType.TTS_CHUNK,
                        chunk_seq,
                        MessageFlags.END,
                        b""  # Empty payload for end marker
                    )
                    
                    if state.websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await state.websocket.send_bytes(end_frame)
                        except Exception as e:
                            logger.warning(f"Failed to send TTS end frame: {e}")
                
                logger.debug(f"TTS streaming completed: {chunk_seq} chunks sent")
                
            except Exception as e:
                logger.error(f"TTS streaming failed: {e}")
                # Emit error event
                await _send_json(state.websocket, {
                    "type": "error",
                    "stage": "tts_streaming",
                    "message": f"TTS streaming failed: {str(e)}",
                    "utterance_index": utterance_index,
                })
        
        finally:
            state.tts_streaming_active = False
            
            # Notify echo suppressor that TTS completed
            if state.echo_suppressor:
                await state.echo_suppressor.on_tts_complete()
            
            # Emit completion event
            await _send_json(state.websocket, {
                "type": "tts_streaming_complete",
                "utterance_index": utterance_index,
                "total_chunks": chunk_seq,
                "timestamp": time.time(),
            })

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
                    # Annotate decode language for client debugging
                    event.setdefault("decode_language", language)
                    await _send_json(state.websocket, event)
                    # Heuristic language mismatch advisory (simple character class ratio)
                    try:
                        if not state.mismatch_notified and state.requested_language in ("en", "ar"):
                            txt = event.get("text", "")
                            if txt:
                                arabic_chars = sum(1 for c in txt if '\u0600' <= c <= '\u06FF')
                                ratio = arabic_chars / max(1, len(txt))
                                if state.requested_language == "en" and ratio > 0.6:
                                    await _send_json(state.websocket, {
                                        "type": "language_mismatch_notice",
                                        "requested": state.requested_language,
                                        "observed_ratio_arabic": round(ratio,3),
                                        "sample": txt[:64],
                                        "message": "Observed dominant Arabic characters while English requested. Consider switching language parameter or enabling auto mode.",
                                        "timestamp": time.time(),
                                    })
                                    state.mismatch_notified = True
                                elif state.requested_language == "ar" and ratio < 0.2 and len(txt) > 8:
                                    await _send_json(state.websocket, {
                                        "type": "language_mismatch_notice",
                                        "requested": state.requested_language,
                                        "observed_ratio_arabic": round(ratio,3),
                                        "sample": txt[:64],
                                        "message": "Observed predominantly non-Arabic characters while Arabic requested.",
                                        "timestamp": time.time(),
                                    })
                                    state.mismatch_notified = True
                    except Exception:
                        pass
                    # Emit metrics snapshot as a dedicated event (easier for tests)
                    await _send_json(state.websocket, {
                        "type": "metrics_snapshot",
                        **(state.metrics.snapshot() if state.metrics else {}),
                    })
                elif etype == "endpoint_event":
                    if state.metrics:
                        state.metrics.update_endpoint(event.get("end_silence_gap_ms"))
                    # Dedupe endpoint events with identical signature in tight window
                    sig = f"{event.get('utterance_index')}:{event.get('end_silence_gap_ms')}"
                    now_ts = time.time()
                    dedupe_window = float(os.getenv("VOICE_STREAMING_ENDPOINT_DEDUPE_SEC", "0.75"))
                    if sig == state.last_endpoint_sig and (now_ts - state.last_endpoint_time) < dedupe_window:
                        logger.debug("[streaming_voice] Skipping duplicate endpoint event sig=%s", sig)
                    else:
                        state.last_endpoint_sig = sig
                        state.last_endpoint_time = now_ts
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
                    event.setdefault("decode_language", language)
                    await _send_json(state.websocket, event)
                    
                    # Reset counters for new utterance after final transcript
                    reset_counters_after_final = os.getenv("VOICE_STREAMING_RESET_COUNTERS_AFTER_FINAL", "1") == "1"
                    if reset_counters_after_final:
                        await state.reset_counters_for_new_utterance()
                        logger.debug(
                            "[streaming_voice] Reset metrics counters after final transcript (utterance_index=%d)",
                            utterance_index
                        )
                    
                    # Reset WebM decoder after final transcript to prevent audio bleeding
                    reset_webm_after_final = os.getenv("VOICE_STREAMING_RESET_WEBM_AFTER_FINAL", "1") == "1"
                    if reset_webm_after_final and state.webm_decoder:
                        logger.debug(
                            "[streaming_voice] Resetting WebM decoder after final transcript (utterance_index=%d)",
                            utterance_index
                        )
                        state.webm_decoder.reset_for_new_utterance()
                    
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
                        # Add /no_think to disable thinking mode for voice conversations (consistent with simple voice service)
                        final_text_with_no_think = final_text.strip() + " /no_think" if final_text.strip() else "unclear audio /no_think"
                        
                        if state.llm_tts_task is None:
                            state.llm_tts_task = asyncio.create_task(
                                _process_final_transcript(
                                    utterance_index,
                                    final_text_with_no_think,
                                    language,
                                    state,
                                )
                            )
                        else:
                            # Queue this final for later processing to avoid dropping (also add /no_think)
                            final_text_with_no_think = final_text.strip() + " /no_think" if final_text.strip() else "unclear audio /no_think"
                            state.pending_finals.append((utterance_index, final_text_with_no_think))
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
            try:
                while True:
                    await asyncio.sleep(2.0)
                    if state.disconnected:
                        break
                    hb_payload = {
                        "type": "heartbeat",
                        "ts": time.time(),
                        "bytes_received": state.pcm_bytes_received,  # Use actual PCM bytes, not raw input
                        "pcm_frames": state.pcm_frames_received,
                        "buffer_usage": (state.audio_session.pcm_buffer.usage_ratio() if state.audio_session else None),
                        "voiced_frames": state.voiced_frame_count,
                        "silence_frames": state.silence_frame_count,
                        "low_energy_mode": state.low_energy_mode,
                        "echo_suppression": (state.echo_suppressor.get_metrics() if state.echo_suppressor else None),
                        "tts_streaming_active": state.tts_streaming_active,
                    }
                    await _send_json(websocket, hb_payload)
                    state.last_heartbeat_sent = time.time()
            except asyncio.CancelledError:
                logger.debug("Heartbeat loop cancelled; closing cleanly")
                return
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
                        elif data.get("type") == "reset_conversation":
                            # Clear conversation history for fresh context
                            state.conversation.clear()
                            state.emitted_assistant_for.clear()
                            state.processed_utterance_indices.clear()
                            state.assistant_turns = 0
                            
                            # Also reset audio buffers to prevent bleeding from previous conversation
                            if state.audio_session and hasattr(state.audio_session, 'pcm_buffer'):
                                await state.audio_session.pcm_buffer.reset_for_new_utterance()
                                logger.debug(f"[streaming_voice] Reset audio buffer during conversation reset")
                            
                            if state.webm_decoder:
                                state.webm_decoder.reset_for_new_utterance()
                                logger.debug(f"[streaming_voice] Reset WebM decoder during conversation reset")
                                
                            logger.info(f"[streaming_voice] Conversation history reset for session {state.session_id}")
                            await _send_json(websocket, {
                                "type": "conversation_reset", 
                                "message": "Conversation history cleared",
                                "ts": time.time()
                            })
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
                            # Initialize WebM decoder using utility
                            try:
                                state.webm_decoder = create_realtime_decoder()
                                state.webm_chunk_queue = asyncio.Queue()
                                
                                # Reset PCM buffer when starting new audio file to prevent conversation bleeding
                                if state.audio_session and hasattr(state.audio_session, 'pcm_buffer'):
                                    await state.audio_session.pcm_buffer.reset_for_new_utterance()
                                    logger.info(f"[streaming_voice] Reset PCM buffer for new audio file (compressed mode: {state.compressed_mode})")
                                
                                # Start WebM decoding task
                                async def _webm_decoder_task():
                                    try:
                                        async def chunk_generator():
                                            while True:
                                                chunk = await state.webm_chunk_queue.get()
                                                if chunk is None:  # Sentinel to stop
                                                    break
                                                yield chunk
                                                
                                        # Process WebM chunks and emit PCM
                                        async for pcm_chunk in state.webm_decoder.stream_realtime_pcm(chunk_generator()):
                                            if state.audio_session:
                                                await state.audio_session.ingest_pcm(pcm_chunk)
                                                state.pcm_bytes_received += len(pcm_chunk)  # Track actual PCM bytes
                                                state.pcm_frames_received += 1
                                                if state.pcm_frames_received <= 3:
                                                    logger.debug("[%s] (dec) ingested PCM frame bytes=%d buf_usage=%.3f", 
                                                                connection_id, len(pcm_chunk), state.audio_session.pcm_buffer.usage_ratio())
                                    except WebMDecodingError as e:
                                        logger.warning("[%s] WebM decoding error: %s", connection_id, e)
                                        await _send_json(websocket, {"type": "error", "stage": "ingest", "message": f"webm_decode_error: {e}"})
                                    except Exception as e:
                                        logger.exception("[%s] WebM decoder task failed: %s", connection_id, e)
                                
                                state.decoder_task = asyncio.create_task(_webm_decoder_task())
                                await _send_json(websocket, {"type": "info", "stage": "ingest", "message": f"Activated {state.compressed_mode} decoder"})
                                
                                # Emit ingest_mode event so frontend can display mode early
                                await _send_json(websocket, {
                                    "type": "ingest_mode",
                                    "mode": state.compressed_mode,
                                    "timestamp": time.time(),
                                    "bytes_received": state.pcm_bytes_received,  # Use PCM bytes (initially 0 for WebM)
                                    "frame_hint_bytes": 640,
                                })
                                
                            except Exception as e:
                                logger.exception("Failed to start WebM decoder: %s", e)
                                await _send_json(websocket, {"type": "error", "stage": "ingest", "message": f"decoder_start_failed: {e}"})
                                state.compressed_mode = None
                    
                    # Detect new WebM file boundary even after compressed mode is set
                    elif state.compressed_mode == "webm-opus" and len(payload) >= 4 and payload[:4] == b"\x1a\x45\xdf\xa3":
                        logger.info(f"[streaming_voice] Detected new WebM file boundary - resetting for clean transcription")
                        # Reset decoder and buffer for new file
                        if state.webm_decoder:
                            state.webm_decoder.reset_for_new_utterance()
                        if state.audio_session and hasattr(state.audio_session, 'pcm_buffer'):
                            await state.audio_session.pcm_buffer.reset_for_new_utterance()
                            logger.info(f"[streaming_voice] Reset PCM buffer for new WebM file boundary")

                    if state.compressed_mode:
                        # Feed compressed payload into WebM decoder queue
                        if state.webm_chunk_queue:
                            try:
                                await state.webm_chunk_queue.put(payload)
                            except Exception as e:
                                logger.warning("[%s] WebM chunk queue error: %s", connection_id, e)
                    else:
                        # Raw PCM path (default)
                        if state.audio_session:
                            if len(payload) % 2 != 0:
                                logger.debug("[%s] odd-length PCM payload=%d", connection_id, len(payload))
                            
                            # Process through echo suppressor if available
                            if state.echo_suppressor:
                                echo_result = await state.echo_suppressor.process_mic_audio(payload)
                                processed_payload = echo_result.get("processed_audio", payload)
                            else:
                                processed_payload = payload
                            
                            # Simple energy gate (average absolute amplitude)
                            try:
                                # Ensure payload length is even before processing as int16
                                if len(processed_payload) % 2 != 0:
                                    logger.warning(f"[{connection_id}] Odd payload length {len(processed_payload)}, padding with zero")
                                    processed_payload = processed_payload + b'\x00'
                                    
                                # interpret payload as int16 little-endian
                                import array
                                arr = array.array('h')
                                arr.frombytes(processed_payload)
                                if arr:
                                    avg_abs = sum(abs(s) for s in arr) / len(arr)
                                    # threshold ~ 220 (~0.0067 full scale) heuristically tuned
                                    if avg_abs > 220:
                                        state.voiced_frame_count += 1
                                        if state.low_energy_mode and state.voiced_frame_count >= 3:
                                            state.low_energy_mode = False
                                    else:
                                        state.silence_frame_count += 1
                            except Exception as e:
                                logger.warning(f"[{connection_id}] Audio processing error: {e}")
                            await state.audio_session.ingest_pcm(processed_payload)
                            state.pcm_bytes_received += len(processed_payload)  # Track actual PCM bytes
                            state.pcm_frames_received += 1
                            if state.pcm_frames_received <= 3:  # log only first few for noise control
                                logger.debug("[%s] ingested PCM frame bytes=%d buffer_usage=%.3f", connection_id, len(payload), state.audio_session.pcm_buffer.usage_ratio())
                            # Emit ingest_mode once on first raw PCM frame (if not already sent)
                            if state.pcm_frames_received == 1 and state.compressed_mode is None:
                                await _send_json(websocket, {
                                    "type": "ingest_mode",
                                    "mode": "pcm16le",
                                    "timestamp": time.time(),
                                    "bytes_received": state.pcm_bytes_received,  # Use PCM bytes in event
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
                "bytes_received": state.pcm_bytes_received,  # Use actual PCM bytes
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
        
        # Cancel TTS streaming task
        if state.tts_streaming_task and not state.tts_streaming_task.done():
            state.tts_streaming_task.cancel()
            with contextlib.suppress(Exception):
                await state.tts_streaming_task
        
        # Stop TTS streaming
        state.tts_streaming_active = False
        # Heartbeat task
        if state.heartbeat_task and not state.heartbeat_task.done():
            state.heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.heartbeat_task
        # Close ring buffer
        if state.audio_session and not state.audio_session.pcm_buffer.closed:
            await state.audio_session.pcm_buffer.close()
        # Tear down WebM decoder if active
        if state.webm_decoder:
            # Stop chunk queue by sending sentinel
            if state.webm_chunk_queue:
                with contextlib.suppress(Exception):
                    await state.webm_chunk_queue.put(None)
            
            # Cancel decoder task
            if state.decoder_task and not state.decoder_task.done():
                state.decoder_task.cancel()
                with contextlib.suppress(Exception):
                    await state.decoder_task
            
            # Cleanup decoder resources
            with contextlib.suppress(Exception):
                state.webm_decoder.cleanup()
        session_registry.remove(connection_id)
        # Reset session id context var
        try:
            session_id_ctx.reset(session_ctx_token)
        except Exception:
            pass
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
