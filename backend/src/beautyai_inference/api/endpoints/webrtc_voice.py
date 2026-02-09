"""
WebRTC Voice Endpoint (Optimized Implementation)

Audio processing pipeline:
- Jitter Buffer: 128 packets (approx 2.5s) for network resilience
- Transient Suppressor: Optional @ 48kHz (disabled by default)
- Anti-aliasing: 6th-order Butterworth @ 8kHz
- Resampling: Single-stage 48kHz → 16kHz (with debug capture option)
- Noise Reduction: RNNoise (16→48→RNNoise→16 pipeline)
- VAD: Silero VAD with threshold 0.2
- STT: Whisper Large v3 Turbo (openai/whisper-large-v3-turbo, 809M params)
- LLM: Qwen3 14B Q4_K_S (via Llama.cpp, ~8GB VRAM)
- TTS: Genius XTTS v2 Arabic (Coqui TTS fine-tuned, voice cloning)
- Output: Text + Audio response via Data Channel

Pipeline Flow:
    48kHz Raw → [Transient Suppressor] → Float32 → Butterworth → Resample 16kHz
    → RNNoise (16→48→denoise→16) → VAD → Whisper → LLM → XTTS

Author: BeautyAI Framework
Date: December 2025
"""

import asyncio
import json
import logging
import os
import time
import uuid
import wave
from math import gcd
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, AliasChoices
from scipy.signal import butter, sosfiltfilt, resample_poly

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.jitterbuffer import JitterBuffer
import aiortc.rtcrtpreceiver

# ============================================================
# AIORTC JITTER BUFFER TUNING
# ============================================================
_original_RTCRtpReceiver_init = aiortc.rtcrtpreceiver.RTCRtpReceiver.__init__

AIORTC_AUDIO_JITTER_CAPACITY = int(os.getenv("AIORTC_AUDIO_JITTER_CAPACITY", "128"))
AIORTC_AUDIO_JITTER_PREFETCH = int(os.getenv("AIORTC_AUDIO_JITTER_PREFETCH", "50"))


def _patched_RTCRtpReceiver_init(self, kind, transport):
    """Patched RTCRtpReceiver.__init__ with increased audio jitter buffer."""
    _original_RTCRtpReceiver_init(self, kind, transport)
    if kind == "audio":
        self._RTCRtpReceiver__jitter_buffer = JitterBuffer(
            capacity=AIORTC_AUDIO_JITTER_CAPACITY,
            prefetch=AIORTC_AUDIO_JITTER_PREFETCH,
        )
        print(
            f"[VOICE] 🔧 Jitter Buffer: capacity={AIORTC_AUDIO_JITTER_CAPACITY}, "
            f"prefetch={AIORTC_AUDIO_JITTER_PREFETCH}",
            flush=True,
        )


aiortc.rtcrtpreceiver.RTCRtpReceiver.__init__ = _patched_RTCRtpReceiver_init

# ============================================================
# IMPORTS
# ============================================================
from ...core.persistent_model_manager import get_persistent_model_manager
from ...core.voice_session_manager import get_voice_session_manager, VoiceSessionManager
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState
from ...services.voice.turn_detection import EndOfTurnPredictor, EndOfTurnConfig
from ...services.voice.streaming.sentence_buffer import SentenceStreamBuffer, SentenceStreamConfig, StreamedSentence
from ...utils.rnnoise_wrapper import RNNoiseProcessor
from ...utils.transient_suppressor import TransientSuppressor
from ...utils.transcription_cleaner import filter_whisper_output, clean_llm_response_for_tts
from ...services.voice.tools import (
    VoiceToolExecutor,
    get_tools_for_openai,
    get_customer_service_system_prompt,
    tool_allows_interruption
)
import base64
import traceback as tb

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
# Environment variables for feature toggles
ENABLE_TRANSIENT_SUPPRESSOR = os.getenv("VOICE_TRANSIENT_SUPPRESSOR", "0") == "1"
ENABLE_SMART_TURN_DETECTION = os.getenv("VOICE_SMART_TURN_DETECTION", "1") == "1"
ENABLE_STREAMING_TTS = os.getenv("VOICE_STREAMING_TTS", "1") == "1"  # Progressive sentence-by-sentence TTS
ENABLE_DEBUG_CAPTURE = os.getenv("VOICE_DEBUG_CAPTURE", "0") == "1"
DEBUG_CAPTURE_DIR = Path(os.getenv("VOICE_DEBUG_CAPTURE_DIR", "/home/lumi/beautyai/reports/debug/voice"))
ENABLE_LANGGRAPH = os.getenv("VOICE_LANGGRAPH_ENABLED", "0") == "1"

# Ensure debug directory exists
if ENABLE_DEBUG_CAPTURE:
    DEBUG_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# LangGraph integration (optional)
_langgraph_integration = None
if ENABLE_LANGGRAPH:
    try:
        from ...services.voice.langgraph_integration import (
            LangGraphPipelineIntegration,
            get_or_create_integration,
            clear_integration,
        )
        print("[VOICE] ✅ LangGraph integration loaded", flush=True)
    except ImportError as e:
        print(f"[VOICE] ⚠️ LangGraph not available: {e}", flush=True)
        ENABLE_LANGGRAPH = False

# ============================================================
# TTS FALLBACK SINGLETON (prevents "cannot schedule new futures after shutdown")
# ============================================================
_edge_tts_fallback_instance = None
_edge_tts_fallback_lock = asyncio.Lock()


async def _get_edge_tts_fallback():
    """Get or create the singleton Edge TTS fallback instance (thread-safe)."""
    global _edge_tts_fallback_instance
    
    async with _edge_tts_fallback_lock:
        if _edge_tts_fallback_instance is None:
            from ...inference_engines.voice.tts import EdgeTTSEngine
            from ...config.config_manager import ModelConfig
            
            logger.info("[VOICE] 🔧 Creating persistent Edge TTS fallback instance")
            edge_config = ModelConfig(name="edge-tts-fallback", model_id="edge-tts", engine_type="edge_tts")
            _edge_tts_fallback_instance = EdgeTTSEngine(edge_config)
            _edge_tts_fallback_instance.load_model()
            logger.info("[VOICE] ✅ Edge TTS fallback instance ready")
        
        return _edge_tts_fallback_instance


async def _stream_tts_sentences(
    sentences: list,
    tts_engine,
    language: str,
    dc,
    context: Dict,
) -> float:
    """Stream TTS audio for multiple sentences progressively.
    
    Instead of synthesizing all text at once, this function:
    1. Synthesizes each sentence independently
    2. Sends audio for each sentence as soon as it's ready
    3. Allows interruption between sentences
    
    Args:
        sentences: List of sentence strings to synthesize
        tts_engine: TTS engine instance
        language: Language code
        dc: WebRTC data channel for sending audio
        context: Session context (for interruption detection)
    
    Returns:
        Total TTS time in milliseconds
    """
    import time
    
    total_tts_time = 0
    tts_type = type(tts_engine).__name__ if tts_engine else "None"
    
    for i, sentence_text in enumerate(sentences):
        # Check for interruption
        if context.get("interrupted", False):
            print(f"[VOICE] 🛑 TTS interrupted after sentence {i}", flush=True)
            break
        
        if not sentence_text.strip():
            continue
        
        try:
            sentence_start = time.time()
            
            # Generate TTS for this sentence
            tts_args = {"text": sentence_text, "language": language}
            
            if tts_type == "EdgeTTSEngine":
                tts_args["gender"] = "female"
            elif tts_type == "XTTSEngine":
                if language == "ar":
                    fallback_wav = Path("/home/lumi/beautyai/voice_tests/input_test_questions/q1.wav")
                else:
                    fallback_wav = Path("/home/lumi/beautyai/tests/webrtc/botox.wav")
                if fallback_wav.exists():
                    tts_args["speaker_wav"] = str(fallback_wav)
            elif tts_type == "SaudiXTTSEngine":
                if not getattr(tts_engine, "has_speaker_conditioning", lambda: False)():
                    fallback_wav = Path("/home/lumi/beautyai/backend/speakers/saudi-female/reference.wav")
                    if fallback_wav.exists():
                        tts_args["speaker_wav"] = str(fallback_wav)
            
            # Run TTS in executor
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(
                None,
                lambda: tts_engine.text_to_speech(**tts_args)
            )
            
            sentence_tts_time = (time.time() - sentence_start) * 1000
            total_tts_time += sentence_tts_time
            
            # Send audio chunk
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                
                if dc and dc.readyState == "open":
                    dc.send(json.dumps({
                        "type": "tts_audio",
                        "audio_base64": audio_b64,
                        "format": "wav",
                        "chunk_index": i,
                        "total_chunks": len(sentences),
                        "is_final": (i == len(sentences) - 1),
                        "tts_time_ms": sentence_tts_time,
                    }))
                    print(
                        f"[VOICE] 📤 Sent TTS chunk {i+1}/{len(sentences)} "
                        f"({len(audio_data)} bytes, {sentence_tts_time:.0f}ms)",
                        flush=True,
                    )
                
                # Clean up temp file
                try:
                    os.remove(audio_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"[VOICE] TTS error for sentence {i}: {e}")
            print(f"[VOICE] ❌ TTS chunk {i} error: {e}", flush=True)
    
    return total_tts_time


def _split_into_sentences(text: str, language: str = "en") -> list:
    """Split text into sentences for progressive TTS.
    
    Uses language-appropriate sentence boundaries.
    Merges short sentences to prevent TTS hallucination with tiny inputs.
    """
    import re
    
    if not text or not text.strip():
        return []
    
    # Minimum sentence length for stable TTS (prevents hallucination on short inputs)
    MIN_SENTENCE_LENGTH = 25  # Chars - "Hello!" alone causes TTS issues
    
    # Language-specific patterns
    if language.startswith("ar"):
        # Arabic: split on . ! ? ؟ and ، (Arabic comma sometimes ends sentences)
        pattern = r'(?<=[.!?؟])\s+'
    else:
        # English: split on . ! ?
        pattern = r'(?<=[.!?])\s+'
    
    raw_sentences = re.split(pattern, text.strip())
    
    # Filter empty sentences
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    # If no sentences found (no punctuation), return whole text
    if not raw_sentences:
        return [text.strip()]
    
    # Merge short sentences with the next one to prevent TTS hallucination
    merged_sentences = []
    buffer = ""
    
    for i, sentence in enumerate(raw_sentences):
        if buffer:
            buffer = buffer + " " + sentence
        else:
            buffer = sentence
        
        # Keep accumulating if current buffer is too short, unless it's the last sentence
        is_last = (i == len(raw_sentences) - 1)
        if len(buffer) >= MIN_SENTENCE_LENGTH or is_last:
            merged_sentences.append(buffer)
            buffer = ""
    
    # If all sentences were merged into one, return as single item
    if not merged_sentences and buffer:
        merged_sentences.append(buffer)
    
    return merged_sentences


webrtc_voice_router = APIRouter(
    prefix="/api/v1/webrtc/voice",
    tags=["webrtc-voice"],
)

# Store active connections
_active_connections: Dict[str, Dict[str, Any]] = {}


# ============================================================
# MODELS
# ============================================================
class OfferRequest(BaseModel):
    sdp: str = Field(..., min_length=10)
    type: str = Field(default="offer")
    language: Optional[str] = Field(default=None, description="Language code (ar, en)")
    customer_service_mode: Optional[bool] = Field(
        default=False,
        validation_alias=AliasChoices("customer_service_mode", "customerServiceMode", "demo_mode", "demoMode"),
        description="Enable customer service appointment booking tools"
    )


class OfferResponse(BaseModel):
    sdp: str
    type: str = "answer"
    session_id: str
    # Backward/forward compatibility: some clients use peer_id naming
    peer_id: str


class ICERequest(BaseModel):
    session_id: str = Field(
        ...,
        validation_alias=AliasChoices("sessionId", "session_id", "peer_id", "peerId"),
    )
    candidate: Union[str, Dict[str, Any]]
    sdp_mid: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("sdpMid", "sdp_mid"),
    )
    sdp_m_line_index: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("sdpMLineIndex", "sdp_m_line_index"),
    )

    class Config:
        populate_by_name = True

    @field_validator("candidate")
    @classmethod
    def parse_candidate(cls, v: Union[str, Dict[str, Any]]) -> str:
        if isinstance(v, dict):
            return v.get("candidate", "")
        return v


# ============================================================
# ENDPOINTS
# ============================================================
@webrtc_voice_router.post("/offer", response_model=OfferResponse)
async def handle_offer(request: OfferRequest):
    """Create WebRTC session with optimized audio pipeline."""
    try:
        session_id = str(uuid.uuid4())
        # Default to English if not specified (fixes language mixing issue)
        target_language = request.language or "en"
        customer_service_mode = bool(request.customer_service_mode)
        print(
            f"[VOICE] 🚀 Creating session {session_id} (language={target_language}, "
            f"customer_service={customer_service_mode})",
            flush=True
        )

        # RTC Configuration
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(
                    urls=["turn:dev.gmai.sa:15478"],
                    username="beautyai",
                    credential="beautyai2025",
                ),
            ]
        )
        pc = RTCPeerConnection(configuration=config)

        # Initialize Voice Session Manager for multi-turn context
        voice_session_mgr = get_voice_session_manager(persist_sessions=False)
        voice_session = await voice_session_mgr.create_session(
            connection_id=session_id,
            language=target_language,
            voice_type="female",
            session_id=session_id
        )
        print(f"[VOICE] 📋 Voice session created: {voice_session.session_id}", flush=True)

        # Session Context
        session_context = {
            "pc": pc,
            "session_id": session_id,
            "language": target_language,
            "start_time": time.time(),
            "audio_track": None,
            "data_channel": None,
            "processing_task": None,
            "vad_service": None,
            "whisper_model": None,
            "llm_model": None,
            "tts_model": None,
            "rnnoise_processor": None,
            "transient_suppressor": None,
            "transcript_buffer": [],
            "turn_timer_task": None,
            "is_speaking": False,  # Track if TTS is playing
            "loop": asyncio.get_event_loop(),
            "customer_service_mode": customer_service_mode,
            "tool_executor": None,
            "customer_context": {},
            # Voice session manager for multi-turn context
            "voice_session_manager": voice_session_mgr,
            "voice_session": voice_session,
            # Debug capture buffers
            "debug_capture_enabled": ENABLE_DEBUG_CAPTURE,
            "debug_16khz_buffer": [] if ENABLE_DEBUG_CAPTURE else None,
            "debug_rnnoise_buffer": [] if ENABLE_DEBUG_CAPTURE else None,
            # Smart turn detection
            "turn_predictor": None,
        }

        # Tool executor for customer service mode
        if customer_service_mode:
            tool_base_url = os.getenv("VOICE_TOOL_API_BASE_URL", "http://localhost:8000")
            session_context["tool_executor"] = VoiceToolExecutor(base_url=tool_base_url)

        # LangGraph integration (if enabled)
        if ENABLE_LANGGRAPH and customer_service_mode:
            try:
                session_context["langgraph_integration"] = get_or_create_integration(
                    session_id=session_id,
                    language=target_language,
                    llm_model=None  # Will be set later after model load
                )
                print(f"[VOICE] ✅ LangGraph integration created for session {session_id}", flush=True)
            except Exception as lg_err:
                logger.warning(f"[VOICE] LangGraph init failed: {lg_err}")
                session_context["langgraph_integration"] = None

        # Load Models (Persistent)
        try:
            model_manager = get_persistent_model_manager()
            session_context["whisper_model"] = model_manager.get_whisper_model()
            session_context["llm_model"] = model_manager.get_llm_model()
            
            # TTS Loading with Language-based routing: Arabic -> Saudi XTTS, English -> Edge TTS
            # Also load Edge TTS as fallback for Saudi XTTS failures
            from ...core.model_manager import get_model_manager
            base_model_manager = get_model_manager()
            
            # Get language-appropriate TTS engine
            tts_engine = base_model_manager.get_tts_engine(language=target_language)
            if tts_engine:
                tts_type = type(tts_engine).__name__
                session_context["tts_model"] = tts_engine
                print(f"[VOICE] ✅ TTS Model Loaded ({tts_type}) for language={target_language}", flush=True)
            else:
                # Primary TTS failed, try Edge TTS fallback
                print("[VOICE] ⚠️ Primary TTS not available, trying Edge TTS fallback...", flush=True)
                try:
                    edge_tts = base_model_manager.get_tts_engine(model_name="edge-tts")
                    if edge_tts:
                        session_context["tts_model"] = edge_tts
                        print("[VOICE] ✅ Edge TTS Fallback Loaded", flush=True)
                    else:
                        session_context["tts_model"] = None
                        print("[VOICE] ⚠️ TTS Model NOT Available (no fallback)", flush=True)
                except Exception as fallback_err:
                    logger.error(f"[VOICE] Edge TTS fallback failed: {fallback_err}")
                    session_context["tts_model"] = None
                    print(f"[VOICE] ⚠️ TTS Fallback failed: {fallback_err}", flush=True)

            if session_context["whisper_model"]:
                print("[VOICE] ✅ Whisper Model Loaded", flush=True)
            else:
                print("[VOICE] ⚠️ Whisper Model NOT Available", flush=True)

            if session_context["llm_model"]:
                print("[VOICE] ✅ LLM Model Loaded", flush=True)
            else:
                print("[VOICE] ⚠️ LLM Model NOT Available", flush=True)

        except Exception as e:
            logger.error(f"[VOICE] Model load error: {e}")
            print(f"[VOICE] ❌ Model load error: {e}", flush=True)

        # Initialize RNNoise
        try:
            session_context["rnnoise_processor"] = RNNoiseProcessor()
            print("[VOICE] ✅ RNNoise Initialized", flush=True)
        except Exception as e:
            logger.warning(f"[VOICE] RNNoise init failed (will skip): {e}")
            print(f"[VOICE] ⚠️ RNNoise unavailable: {e}", flush=True)

        # Initialize Transient Suppressor (optional)
        if ENABLE_TRANSIENT_SUPPRESSOR:
            try:
                session_context["transient_suppressor"] = TransientSuppressor(
                    sample_rate=48000,
                    kernel_size=5,
                    threshold=0.8,
                    energy_window=5,
                    frame_size=960,
                )
                print("[VOICE] ✅ Transient Suppressor ENABLED", flush=True)
            except Exception as e:
                logger.warning(f"[VOICE] Transient Suppressor init failed: {e}")
                print(f"[VOICE] ⚠️ Transient Suppressor unavailable: {e}", flush=True)
        else:
            print("[VOICE] ℹ️ Transient Suppressor DISABLED", flush=True)

        # Initialize VAD (matching debug_capture settings that work!)
        try:
            vad_config = WebRTCVADConfig()
            vad_config.silero_sensitivity = 0.3  # More sensitive (catches more speech)
            vad_config.webrtc_sensitivity = 2    # 0-3, where 3 = least sensitive
            vad_config.post_speech_silence_ms = 700  # Wait 700ms before ending speech
            vad_config.min_speech_duration_ms = 50   # Minimum 50ms speech to register
            vad_config.language_thresholds = {
                "ar": 0.1,
                "en": 0.1,
                "default": 0.1,
            }  # Lower threshold for better detection
            vad_config.warmup_filter_duration_ms = 200  # Filter initial 200ms noise
            vad_config.min_sustained_speech_frames = 2  # Need 2 consecutive frames (not 3!)
            vad_config.log_vad_decisions = True  # Enable VAD decision logging
            vad_config.enable_debug_dump = False  # Disable VAD internal dumping

            vad_service = WebRTCVADService(session_id, language=target_language, config=vad_config)
            if await vad_service.initialize():
                session_context["vad_service"] = vad_service
                print(
                    f"[VOICE] ✅ VAD Initialized (silero=0.3, thresh=0.1, sustained=2, lang={target_language})",
                    flush=True,
                )
            else:
                print("[VOICE] ❌ VAD Init Failed", flush=True)
        except Exception as e:
            logger.error(f"[VOICE] VAD error: {e}")

        # Initialize Smart Turn Detection (if enabled)
        if ENABLE_SMART_TURN_DETECTION:
            try:
                turn_config = EndOfTurnConfig.for_language(target_language)
                turn_predictor = EndOfTurnPredictor(config=turn_config, language=target_language)
                session_context["turn_predictor"] = turn_predictor
                print(
                    f"[VOICE] ✅ Smart Turn Detection ENABLED "
                    f"(min={turn_config.min_silence_ms}ms, max={turn_config.max_silence_ms}ms, "
                    f"threshold={turn_config.confidence_threshold})",
                    flush=True,
                )
            except Exception as e:
                logger.warning(f"[VOICE] Turn predictor init failed: {e}")
                print(f"[VOICE] ⚠️ Smart Turn Detection unavailable: {e}", flush=True)
        else:
            print("[VOICE] ℹ️ Smart Turn Detection DISABLED (using 2s timeout)", flush=True)

        # Handle Tracks
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                print("[VOICE] 🎤 Audio track received", flush=True)
                session_context["audio_track"] = track
                session_context["processing_task"] = asyncio.create_task(
                    _process_audio_track(session_id, track, session_context)
                )

        # Handle Data Channel
        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"[VOICE] 📡 Data channel received: {channel.label}", flush=True)
            session_context["data_channel"] = channel

            @channel.on("message")
            def on_message(message):
                print(f"[VOICE] 📨 Message from client: {message}", flush=True)

        # Connection State Monitoring
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[VOICE] 🔄 Connection state: {pc.connectionState}", flush=True)
            if pc.connectionState in ["failed", "closed"]:
                await _cleanup_session(session_id)

        # SDP Negotiation
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=request.sdp, type=request.type)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        _active_connections[session_id] = session_context

        return OfferResponse(
            sdp=pc.localDescription.sdp,
            session_id=session_id,
            peer_id=session_id,
        )

    except Exception as e:
        logger.error(f"[VOICE] Offer error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@webrtc_voice_router.post("/ice")
async def handle_ice(request: ICERequest):
    """Handle ICE candidates."""
    try:
        if request.session_id not in _active_connections:
            raise HTTPException(status_code=404, detail="Session not found")

        pc = _active_connections[request.session_id]["pc"]

        parts = request.candidate.split()
        if len(parts) < 8 or "typ" not in parts:
            return {"status": "ignored", "reason": "malformed"}

        candidate = RTCIceCandidate(
            component=int(parts[1]),
            foundation=parts[0].split(":")[1],
            ip=parts[4],
            port=int(parts[5]),
            priority=int(parts[3]),
            protocol=parts[2].upper(),
            type=parts[parts.index("typ") + 1],
            sdpMid=request.sdp_mid,
            sdpMLineIndex=request.sdp_m_line_index,
        )
        await pc.addIceCandidate(candidate)
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VOICE] ICE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# AUDIO PROCESSING
# ============================================================
async def _process_audio_track(
    session_id: str, track: MediaStreamTrack, context: Dict
):
    """Main audio processing loop with optimized pipeline."""
    print(f"[VOICE] ▶️ Starting audio processing for {session_id}", flush=True)

    frame_count = 0
    speech_buffer_16k = []
    
    # Pre-speech buffer (ring buffer) to capture audio BEFORE VAD triggers
    # Keeps ~500ms of audio (at 16kHz, 320 samples per 20ms frame = 25 frames)
    PRE_SPEECH_BUFFER_FRAMES = 25  # ~500ms at 20ms per frame
    pre_speech_buffer = []
    is_collecting_speech = False

    # Get processors
    rnnoise = context.get("rnnoise_processor")
    transient_suppressor = context.get("transient_suppressor")
    vad_service = context.get("vad_service")

    # Debug capture buffers
    debug_enabled = context.get("debug_capture_enabled", False)
    debug_16khz_buffer = context.get("debug_16khz_buffer")
    debug_rnnoise_buffer = context.get("debug_rnnoise_buffer")

    try:
        while True:
            try:
                # 1. Receive Frame
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
                frame_count += 1
                sample_rate = frame.sample_rate

                # Debug logging every 100 frames
                if frame_count % 100 == 0:
                    print(
                        f"[VOICE] Frame #{frame_count}: rate={sample_rate}, "
                        f"samples={frame.samples}",
                        flush=True,
                    )

                # 2. Convert to numpy array
                audio_array = frame.to_ndarray()

                # DIAGNOSTIC: Check amplitude
                if frame_count % 50 == 0:
                   max_amp = np.max(np.abs(audio_array))
                   print(f"[VOICE-DIAG] Frame #{frame_count} max_amp={max_amp}", flush=True)

                # 3. Stereo → Mono (int16)
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()

                if len(audio_array) == frame.samples * 2:
                    # Interleaved stereo
                    audio_array = (
                        audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    )

                # 4. Optional: Transient Suppressor @ 48kHz
                if transient_suppressor and sample_rate == 48000:
                    # Convert to float for processing
                    if np.issubdtype(audio_array.dtype, np.integer):
                        dtype_info = np.iinfo(audio_array.dtype)
                        scale = float(max(abs(dtype_info.min), dtype_info.max)) or 1.0
                        audio_float = audio_array.astype(np.float32) / scale
                    else:
                        audio_float = audio_array.astype(np.float32)

                    audio_float = transient_suppressor.process_frame(audio_float)

                    # Convert back to int16 for consistency
                    audio_array = (
                        np.clip(audio_float, -1.0, 1.0) * 32767
                    ).astype(np.int16)

                # 5. Convert to float32
                if np.issubdtype(audio_array.dtype, np.integer):
                    dtype_info = np.iinfo(audio_array.dtype)
                    scale = float(max(abs(dtype_info.min), dtype_info.max)) or 1.0
                    audio_float = audio_array.astype(np.float32) / scale
                else:
                    audio_float = audio_array.astype(np.float32)

                # 6. Anti-aliasing: 6th-order Butterworth @ 8kHz
                if sample_rate != 16000:
                    nyquist_freq = sample_rate / 2
                    cutoff_freq = 8000

                    if nyquist_freq > cutoff_freq:
                        normalized_cutoff = cutoff_freq / nyquist_freq
                        sos = butter(6, normalized_cutoff, btype="low", output="sos")
                        audio_float = sosfiltfilt(sos, audio_float)
                        audio_float = np.clip(audio_float, -1.0, 1.0)

                # 7. Single-stage Resample: 48kHz → 16kHz
                if sample_rate != 16000:
                    ratio_gcd = gcd(sample_rate, 16000)
                    up = 16000 // ratio_gcd
                    down = sample_rate // ratio_gcd
                    audio_16k = resample_poly(
                        audio_float, up, down, window=("kaiser", 8.0)
                    )
                    audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
                else:
                    audio_16k = audio_float.astype(np.float32)

                # Debug: Capture pre-RNNoise 16kHz
                if debug_enabled and debug_16khz_buffer is not None:
                    debug_16khz_buffer.append(audio_16k.copy())

                # 8. RNNoise: 16kHz → 48kHz → denoise → 16kHz
                if rnnoise:
                    try:
                        # Upsample to 48kHz for RNNoise
                        audio_48k_for_rnnoise = resample_poly(
                            audio_16k, up=3, down=1, window=("kaiser", 5.0)
                        ).astype(np.float32)

                        # Process with RNNoise
                        denoised_48k, _ = rnnoise.process_audio(audio_48k_for_rnnoise)

                        # Downsample back to 16kHz
                        audio_16k = resample_poly(
                            denoised_48k, up=1, down=3, window=("kaiser", 5.0)
                        ).astype(np.float32)
                        audio_16k = np.clip(audio_16k, -1.0, 1.0)

                    except Exception as e:
                        logger.warning(f"[VOICE] RNNoise error (using original): {e}")

                # Debug: Capture post-RNNoise 16kHz
                if debug_enabled and debug_rnnoise_buffer is not None:
                    debug_rnnoise_buffer.append(audio_16k.copy())

                # 9. VAD Processing
                if vad_service:
                    audio_int16 = (
                        np.clip(audio_16k, -1.0, 1.0) * 32767
                    ).astype(np.int16)

                    vad_result = await vad_service.process_audio_chunk(
                        audio_int16.tobytes(),
                        metadata={"sample_rate": 16000},
                    )

                    state = vad_result.get("voice_state")
                    silero_prob = vad_result.get("silero_probability", 0)

                    # Debug: Log state transitions
                    if frame_count % 50 == 0 or state in [VADState.VOICE_START, VADState.VOICE_END]:
                        print(
                            f"[VOICE-VAD] Frame #{frame_count}: state={state}, "
                            f"prob={silero_prob:.3f}, buffer={len(speech_buffer_16k)}",
                            flush=True,
                        )

                    # Check for voice interruption during TTS playback
                    if context.get("is_speaking") and state == VADState.VOICE_START:
                        dc = context.get("dc")
                        if dc and dc.readyState == "open":
                            print("[VOICE] 🛑 INTERRUPT detected - user speaking during TTS", flush=True)
                            dc.send(json.dumps({"type": "interrupt", "reason": "user_speaking"}))
                            context["is_speaking"] = False  # Mark TTS as interrupted
                            context["interrupted"] = True  # Signal to turn predictor

                    # Notify turn predictor of speech state changes
                    turn_predictor = context.get("turn_predictor")
                    if turn_predictor:
                        if state == VADState.VOICE_START:
                            turn_predictor.on_speech_detected()
                        elif state == VADState.VOICE_END:
                            turn_predictor.on_silence_detected()

                    # Accumulate Speech (with pre-speech buffer for capturing word beginnings)
                    if state == VADState.VOICE_START:
                        # Cancel pending turn timer
                        if context.get("turn_timer_task"):
                            context["turn_timer_task"].cancel()
                            context["turn_timer_task"] = None
                        
                        # CRITICAL: Prepend pre-speech buffer to capture word beginnings
                        if pre_speech_buffer and not is_collecting_speech:
                            speech_buffer_16k.extend(pre_speech_buffer)
                            print(f"[VOICE] 🔊 Prepended {len(pre_speech_buffer)} pre-speech frames (~{len(pre_speech_buffer)*20}ms)", flush=True)
                        
                        is_collecting_speech = True
                        speech_buffer_16k.append(audio_16k)
                        
                    elif state in [VADState.VOICE_ACTIVE, VADState.VOICE_END_PENDING]:
                        # Cancel pending turn timer
                        if context.get("turn_timer_task"):
                            context["turn_timer_task"].cancel()
                            context["turn_timer_task"] = None

                        speech_buffer_16k.append(audio_16k)

                    # End of Speech → Process
                    elif state == VADState.VOICE_END:
                        is_collecting_speech = False
                        if speech_buffer_16k:
                            full_audio = np.concatenate(speech_buffer_16k)
                            asyncio.create_task(
                                _process_speech_segment(
                                    session_id, full_audio, context
                                )
                            )
                            speech_buffer_16k = []
                    
                    else:
                        # Not speaking - maintain pre-speech ring buffer
                        pre_speech_buffer.append(audio_16k)
                        # Keep only last N frames (ring buffer behavior)
                        if len(pre_speech_buffer) > PRE_SPEECH_BUFFER_FRAMES:
                            pre_speech_buffer.pop(0)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "MediaStreamError" in str(type(e).__name__) or "End of file" in str(e):
                    print("[VOICE] ⏹️ Track ended", flush=True)
                    # Flush any remaining speech buffer if the track ends while speaking
                    if speech_buffer_16k and is_collecting_speech:
                        print(f"[VOICE] ⚠️ Track ended while speaking - flushing {len(speech_buffer_16k)} frames", flush=True)
                        full_audio = np.concatenate(speech_buffer_16k)
                        asyncio.create_task(
                            _process_speech_segment(
                                session_id, full_audio, context
                            )
                        )
                    break

                import traceback
                traceback.print_exc()
                print(f"[VOICE] ⚠️ Frame error: {e!r}", flush=True)

                if context["pc"].connectionState in ["closed", "failed"]:
                    break
                continue

    except Exception as e:
        logger.error(f"[VOICE] Loop error: {e}")
    finally:
        print(f"[VOICE] ⏹️ Audio loop ended for {session_id}", flush=True)

        # Save debug captures
        if debug_enabled:
            await _save_debug_captures(session_id, context)

        await _cleanup_session(session_id)


async def _save_debug_captures(session_id: str, context: Dict):
    """Save debug audio captures to WAV files."""
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save pre-RNNoise 16kHz
        debug_16khz = context.get("debug_16khz_buffer")
        if debug_16khz:
            audio = np.concatenate(debug_16khz)
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            path = DEBUG_CAPTURE_DIR / f"{timestamp}_{session_id[:8]}_16khz_pre_rnnoise.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_int16.tobytes())

            duration = len(audio) / 16000
            print(f"[VOICE] 💾 Saved pre-RNNoise: {path.name} ({duration:.2f}s)", flush=True)

        # Save post-RNNoise 16kHz
        debug_rnnoise = context.get("debug_rnnoise_buffer")
        if debug_rnnoise:
            audio = np.concatenate(debug_rnnoise)
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            path = DEBUG_CAPTURE_DIR / f"{timestamp}_{session_id[:8]}_16khz_post_rnnoise.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_int16.tobytes())

            duration = len(audio) / 16000
            print(f"[VOICE] 💾 Saved post-RNNoise: {path.name} ({duration:.2f}s)", flush=True)

    except Exception as e:
        logger.error(f"[VOICE] Debug capture save error: {e}")


async def _process_speech_segment(
    session_id: str, audio_data: np.ndarray, context: Dict
):
    """Handle STT and schedule LLM generation."""
    whisper = context.get("whisper_model")
    dc = context.get("data_channel")
    loop = context.get("loop")
    language = context.get("language", "ar")

    if not whisper:
        return

    try:
        # Convert to int16 bytes for Whisper
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        start_time = time.time()
        duration = len(audio_data) / 16000
        
        # Audio length validation: Whisper fails for audio > 30 seconds
        MAX_WHISPER_DURATION_SECONDS = 30.0
        if duration > MAX_WHISPER_DURATION_SECONDS:
            logger.warning(f"[VOICE] Audio too long ({duration:.1f}s > {MAX_WHISPER_DURATION_SECONDS}s), truncating to last 30s")
            print(f"[VOICE] ⚠️ Audio too long ({duration:.1f}s), truncating to last 30s", flush=True)
            # Keep the last 30 seconds (most recent speech)
            max_samples = int(MAX_WHISPER_DURATION_SECONDS * 16000)
            audio_data = audio_data[-max_samples:]
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            duration = MAX_WHISPER_DURATION_SECONDS
        
        print(f"[VOICE] 🗣️ Transcribing {duration:.2f}s (lang={language})...", flush=True)

        raw_text = await loop.run_in_executor(
            None,
            lambda: whisper.transcribe_audio_bytes(
                audio_bytes, audio_format="pcm_raw", language=language
            ),
        )
        whisper_time = (time.time() - start_time) * 1000

        if not raw_text or not raw_text.strip():
            return

        # Apply repetition filter to clean Whisper output
        text = filter_whisper_output(raw_text, language=language)
        if text != raw_text:
            print(f"[VOICE] 🧹 Cleaned: '{raw_text}' → '{text}'", flush=True)
        
        if not text or not text.strip():
            return

        print(f"[VOICE] 📝 User: {text}", flush=True)

        # Send Transcript to Client
        if dc and dc.readyState == "open":
            dc.send(
                json.dumps(
                    {
                        "type": "transcription",
                        "text": text,
                        "role": "user",
                        "metrics": {"whisper_ms": whisper_time},
                    }
                )
            )

        # Aggregate and Schedule Turn
        context["transcript_buffer"].append(text)

        # Cancel existing timer
        if context.get("turn_timer_task"):
            context["turn_timer_task"].cancel()

        # Schedule new timer (2 seconds silence)
        context["turn_timer_task"] = asyncio.create_task(
            _wait_for_silence_and_respond(session_id, context)
        )

    except Exception as e:
        logger.error(f"[VOICE] Processing error: {e}")
        print(f"[VOICE] ❌ Processing error: {e}", flush=True)


async def _wait_for_silence_and_respond(session_id: str, context: Dict):
    """Wait for turn end using smart detection or fallback timeout."""
    try:
        turn_predictor = context.get("turn_predictor")
        
        if ENABLE_SMART_TURN_DETECTION and turn_predictor:
            # Use smart turn detection with confidence scoring
            turn_predictor.on_silence_detected()
            
            # Get current transcript for linguistic analysis
            buffer = context.get("transcript_buffer", [])
            if buffer:
                turn_predictor.update_transcript(" ".join(buffer))
            
            # Wait for turn end with adaptive timeout
            breakdown = await turn_predictor.wait_for_turn_end(
                context=context,
            )
            
            # Log turn detection metrics
            print(
                f"[VOICE] 🎯 Turn detected: reason={breakdown.trigger_reason} "
                f"conf={breakdown.total_confidence:.2f} "
                f"silence={breakdown.silence_duration_ms:.0f}ms",
                flush=True,
            )
            
            # Only proceed if turn was confirmed (not interrupted)
            if breakdown.is_turn_complete and breakdown.trigger_reason != "interrupted":
                await _trigger_llm_response(session_id, context)
        else:
            # Fallback: Fixed 2-second timeout (legacy behavior)
            print("[VOICE] ⏱️ Using legacy 2s timeout", flush=True)
            await asyncio.sleep(2.0)
            await _trigger_llm_response(session_id, context)
            
    except asyncio.CancelledError:
        # Timer cancelled (user started speaking again)
        turn_predictor = context.get("turn_predictor")
        if turn_predictor:
            turn_predictor.on_speech_detected()
        pass
    except Exception as e:
        logger.error(f"[VOICE] Timer error: {e}")
        print(f"[VOICE] ❌ Turn timer error: {e}", flush=True)


async def _trigger_llm_response(session_id: str, context: Dict):
    """Generate LLM response and synthesize TTS audio."""
    llm = context.get("llm_model")
    tts = context.get("tts_model")
    dc = context.get("data_channel")
    loop = context.get("loop")
    buffer = context.get("transcript_buffer", [])
    language = context.get("language", "ar")

    if not buffer or not llm:
        return

    full_text = " ".join(buffer)
    context["transcript_buffer"] = []
    context["turn_timer_task"] = None

    print(f"[VOICE] 🤖 Generating response for: {full_text}", flush=True)

    # Notify client: Processing (keep mic enabled for interruption detection)
    if dc and dc.readyState == "open":
        dc.send(json.dumps({"type": "state", "state": "processing"}))
        # Keep mic enabled so we can detect interruption during TTS
        # dc.send(json.dumps({"type": "mic_control", "action": "mute"}))
    
    context["is_speaking"] = True

    try:
        customer_service_mode = context.get("customer_service_mode", False)
        tool_executor = context.get("tool_executor")
        customer_context = context.get("customer_context", {})
        last_intent = context.get("last_intent")
        last_requested_date = context.get("last_requested_date")
        
        # Initialize response variables
        full_response = ""
        metrics_payload = None
        
        # ============================================================
        # LANGGRAPH WORKFLOW (if enabled) - processes before manual code
        # Set VOICE_LANGGRAPH_ENABLED=1 to use graph-based intent routing
        # ============================================================
        langgraph_integration = context.get("langgraph_integration")
        if ENABLE_LANGGRAPH and langgraph_integration and customer_service_mode:
            try:
                print(f"[VOICE] 🔀 Using LangGraph workflow for: {full_text[:50]}...", flush=True)
                
                lg_result = await langgraph_integration.handle_speech(
                    transcript=full_text,
                    language=language,
                    interrupt_flag=context.get("is_interrupted", False),
                    send_callback=lambda msg: dc.send(msg) if dc and dc.readyState == "open" else None
                )
                
                lg_response = lg_result.get("response_text", "")
                intent_str = str(lg_result.get("intent", "general"))
                tool_results = lg_result.get("tool_results", [])
                
                print(f"[VOICE] 🔀 LangGraph: intent={intent_str}, tools={len(tool_results)}", flush=True)
                
                if lg_result.get("customer_info"):
                    customer_context["customer"] = lg_result["customer_info"]
                    context["customer_context"] = customer_context
                
                if lg_response and lg_response.strip():
                    full_response = lg_response
                    if dc and dc.readyState == "open":
                        dc.send(json.dumps({
                            "type": "response_chunk",
                            "text": full_response,
                            "role": "assistant",
                            "via": "langgraph"
                        }))
                    print(f"[VOICE] 🔀 LangGraph response: {full_response[:60]}...", flush=True)
                    
            except Exception as lg_err:
                logger.error(f"[VOICE] LangGraph error: {lg_err}")
                print(f"[VOICE] ❌ LangGraph error: {lg_err}", flush=True)

        def detect_intent(text: str) -> str:
            """Detect user intent from text. Returns: 'greeting', 'slots', 'booking', 'name_intro', 'general'."""
            lowered = text.lower()
            
            # Name introduction (HIGHEST PRIORITY - check BEFORE greeting)
            # This ensures "Hello my name is X" is detected as name_intro, not greeting
            if any(phrase in lowered for phrase in ["my name is", "i am ", "i'm ", "this is "]):
                return "name_intro"
            
            # Greeting (only if no name introduction)
            if any(word in lowered for word in ["hello", "hi", "hey", "good morning", "good afternoon", "مرحبا", "السلام"]):
                if len(lowered.split()) <= 5 and not any(w in lowered for w in ["appointment", "book", "slot"]):
                    return "greeting"
            
            # Asking for slots/availability
            if any(word in lowered for word in ["available", "availability", "slots", "show me", "what times", "when can"]):
                return "slots"
            
            # Booking intent
            if any(word in lowered for word in [
                "appointment", "book", "booking", "schedule", "set an appointment",
                "موعد", "حجز", "مواعيد"
            ]):
                return "booking"
            
            # Follow-up on previous booking intent
            if last_intent in ["booking", "slots"]:
                # Short responses likely continue the booking flow
                if len(lowered.split()) <= 5:
                    return "booking"
                if any(word in lowered for word in ["ok", "okay", "yes", "sure", "great", "fine", "that's", "28"]):
                    return "booking"
            
            return "general"

        def detect_appointment_intent(text: str) -> bool:
            """Legacy function - returns True if intent is booking-related."""
            intent = detect_intent(text)
            return intent in ["booking", "slots", "name_intro"]

        def extract_name(text: str) -> Optional[Dict[str, str]]:
            """Extract name from text, filtering out common false positives.
            Supports single names (first name only) or full names (first + last).
            """
            import re
            # Common words that shouldn't be names
            invalid_names = {
                "looking", "checking", "booking", "asking", "calling", "wanting",
                "trying", "going", "coming", "doing", "making", "taking",
                "here", "there", "just", "also", "very", "really",
                "a", "an", "the", "for", "and", "but", "not",
                "good", "fine", "okay", "well", "yes", "no",
                "it", "is", "be", "so", "as", "at", "to"  # Short filler words
            }
            match = re.search(r"\b(?:my name is|i am|i'm|this is)\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?", text, re.IGNORECASE)
            if not match:
                print(f"[VOICE] 🔍 extract_name: no pattern match in '{text}'", flush=True)
                return None
            first_name = match.group(1)
            last_name = match.group(2)
            print(f"[VOICE] 🔍 extract_name: matched first='{first_name}', last='{last_name}'", flush=True)
            
            # Filter out invalid first names
            if first_name.lower() in invalid_names:
                print(f"[VOICE] 🔍 extract_name: '{first_name}' is invalid word", flush=True)
                return None
            
            # Names should be at least 2 chars
            if len(first_name) < 2:
                print(f"[VOICE] 🔍 extract_name: '{first_name}' too short", flush=True)
                return None
            
            # Build result - last_name is optional
            result = {"first_name": first_name}
            if last_name and last_name.lower() not in invalid_names and len(last_name) >= 2:
                result["last_name"] = last_name
            else:
                # Use "Customer" as default last name for single-name registrations
                result["last_name"] = "Customer"
            
            print(f"[VOICE] 🔍 extract_name: returning {result}", flush=True)
            return result
        
        def extract_time_slot(text: str) -> Optional[str]:
            """Extract time from text like '2pm', '9:00 AM', 'two o'clock'."""
            import re
            lowered = text.lower()
            
            # Pattern for times like "2pm", "2 pm", "14:00"
            time_patterns = [
                r'(\d{1,2})\s*(?::|\.)?(\d{2})?\s*(am|pm)',  # 2pm, 2:00pm, 2:00 pm
                r'(\d{1,2}):(\d{2})',  # 14:00
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, lowered)
                if match:
                    groups = match.groups()
                    hour = int(groups[0])
                    minutes = groups[1] if len(groups) > 1 and groups[1] else "00"
                    if len(groups) > 2 and groups[2]:  # am/pm present
                        meridiem = groups[2]
                        if meridiem == "pm" and hour < 12:
                            hour += 12
                        elif meridiem == "am" and hour == 12:
                            hour = 0
                    return f"{hour:02d}:{minutes}"
            
            return None
        
        def detect_booking_confirmation(text: str) -> bool:
            """Detect if user is confirming a booking (e.g., 'yes 2pm', 'at 9am', '9am please')."""
            lowered = text.lower()
            # Check for confirmation words
            has_confirmation = any(word in lowered for word in [
                "yes", "yeah", "yep", "sure", "ok", "okay", "please", "book",
                "confirm", "that one", "sounds good", "perfect", "great",
                "at ", "for ", "i want", "i'll take", "let's do", "i choose"
            ])
            # Check for time reference
            has_time = extract_time_slot(text) is not None
            return has_confirmation and has_time

        def get_next_workday(from_date) -> 'date':
            """Get next workday (skipping Friday/Saturday - Saudi weekend)."""
            from datetime import timedelta
            check_date = from_date
            for _ in range(7):  # Look up to 7 days ahead
                if check_date.weekday() not in [4, 5]:  # Friday=4, Saturday=5
                    return check_date
                check_date = check_date + timedelta(days=1)
            return from_date  # Fallback to original if no workday found

        def extract_date(text: str, default_tomorrow: bool = True) -> Optional[str]:
            """Extract date from text. Returns ISO date string.
            Defaults to next workday (skipping Saudi weekend: Fri/Sat).
            """
            import re
            from datetime import datetime, timedelta
            
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
            }
            lowered = text.lower()
            today = datetime.utcnow().date()
            
            # Check for "today"
            if "today" in lowered:
                if today.weekday() in [4, 5]:  # Weekend
                    return get_next_workday(today).isoformat()
                return today.isoformat()
            
            # Check for "tomorrow"
            if "tomorrow" in lowered:
                tomorrow = today + timedelta(days=1)
                if tomorrow.weekday() in [4, 5]:  # Weekend
                    return get_next_workday(tomorrow).isoformat()
                return tomorrow.isoformat()
            
            # Look for day number
            day_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", lowered)
            if day_match:
                day = int(day_match.group(1))
                
                # Find month
                month = None
                for name, value in month_map.items():
                    if name in lowered:
                        month = value
                        break
                
                if month is None:
                    if "next month" in lowered:
                        month = today.month + 1 if today.month < 12 else 1
                        year = today.year if today.month < 12 else today.year + 1
                    else:
                        month = today.month
                        year = today.year
                else:
                    year = today.year
                
                try:
                    target = datetime(year, month, day).date()
                    # If date is in the past, assume next month/year
                    if target < today:
                        if month == 12:
                            target = datetime(year + 1, 1, day).date()
                        else:
                            target = datetime(year, month + 1, day).date()
                    return target.isoformat()
                except ValueError:
                    pass
            
            # Default to today (if workday) or next workday
            if default_tomorrow:
                # Use today if it's a workday, otherwise next workday
                if today.weekday() not in [4, 5]:  # Not weekend
                    return today.isoformat()
                return get_next_workday(today).isoformat()
            return None

        async def execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
            if not tool_executor:
                return {"success": False, "error": "Tool executor not configured"}

            if dc and dc.readyState == "open":
                dc.send(
                    json.dumps(
                        {
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": tool_args,
                            "status": "executing",
                            "allows_interruption": tool_allows_interruption(tool_name),
                        }
                    )
                )

            try:
                result = await tool_executor.execute(
                    tool_name,
                    tool_args,
                    session_id=session_id,
                )
            except Exception as tool_err:
                result = {"success": False, "error": str(tool_err)}

            if result.get("success") and result.get("customer"):
                customer_context["customer"] = result["customer"]
                context["customer_context"] = customer_context

            if dc and dc.readyState == "open":
                dc.send(
                    json.dumps(
                        {
                            "type": "tool_call",
                            "tool": tool_name,
                            "status": "complete",
                            "result": result,
                        }
                    )
                )

            return result

        def build_system_prompt() -> str:
            """Build a concise system prompt that fits within context window."""
            base_prompt = (
                "You are a helpful voice assistant for Kesay Beauty Clinic. "
                "Keep responses SHORT (1-2 sentences) and conversational. "
                "Avoid numbers - use 'first', 'next', 'also' instead."
            )
            if not customer_service_mode:
                return base_prompt

            # Concise customer service instructions - NO tools JSON (too large!)
            customer_name = customer_context.get("customer", {}).get("full_name", "")
            context_hint = f" Current customer: {customer_name}." if customer_name else ""
            
            return (
                f"{base_prompt}\n\n"
                f"Help customers with: checking appointments, booking new ones, registration.{context_hint}\n"
                "When a customer wants to book: First ask their name if unknown, then check available times, then confirm booking.\n"
                "Be friendly, professional, and brief."
            )
        
        def get_conversation_context() -> str:
            """Get recent conversation context from VoiceSessionManager."""
            voice_session = context.get("voice_session")
            if voice_session and voice_session.conversation_history:
                return voice_session.get_recent_context(max_turns=3)
            return ""

        def summarize_tool_result(tool_result: Dict[str, Any]) -> str:
            """Create a concise summary of tool results to fit in context window."""
            summary_parts = []
            
            # Customer info
            customer = tool_result.get("customer")
            if customer:
                name = customer.get("full_name", "Unknown")
                summary_parts.append(f"Customer found: {name}")
            elif tool_result.get("success") == False and "not found" in str(tool_result.get("error", "")).lower():
                summary_parts.append("Customer not found - needs registration")
            
            # Available slots (just the count and first few)
            slots = tool_result.get("available_slots", [])
            if slots:
                count = len(slots)
                first_slots = slots[:5]  # Show first 5 slots
                slot_str = ", ".join(f"{s.get('date', '')} {s.get('time', s.get('start_time', ''))}" for s in first_slots)
                summary_parts.append(f"{count} slots available: {slot_str}")
            elif "available_slots" in tool_result:
                # Explicitly checked slots but none found - prevent LLM hallucination
                summary_parts.append("NO SLOTS AVAILABLE - tell customer there are no available slots")
            elif tool_result.get("booking_failed"):
                # Explicit booking failure - don't let LLM hallucinate
                summary_parts.append("BOOKING FAILED - slot not available")
            
            # Booking result
            if tool_result.get("appointment"):
                appt = tool_result["appointment"]
                summary_parts.append(f"Booking confirmed: {appt.get('date', '')} {appt.get('time', '')}")
            
            # Error (includes booking failure messages)
            if tool_result.get("error"):
                summary_parts.append(f"Error: {tool_result['error']}")
            
            return ". ".join(summary_parts) if summary_parts else "Tool executed successfully."

        def build_prompt(user_text: str, tool_result: Optional[Dict[str, Any]] = None) -> str:
            system_prompt = build_system_prompt()
            
            # Get multi-turn conversation context
            conversation_context = get_conversation_context()
            context_section = ""
            if conversation_context:
                context_section = f"\n\nPrevious conversation:\n{conversation_context}\n"
            
            if tool_result is None:
                user_message = f"/no_think {user_text}"
            else:
                # Use concise summary instead of full JSON
                result_summary = summarize_tool_result(tool_result)
                user_message = (
                    f"/no_think User: {user_text}\n"
                    f"System info: {result_summary}\n"
                    "Respond helpfully based on this information."
                )
            return (
                f"<|im_start|>system\n{system_prompt}{context_section}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
            import re
            clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            clean_text = re.sub(r"<think>.*$", "", clean_text, flags=re.DOTALL)
            clean_text = clean_text.replace("</think>", "")
            match = re.search(r"<tool_call>(.*?)</tool_call>", clean_text, flags=re.DOTALL)
            payload = None
            if match:
                payload = match.group(1).strip()
            else:
                stripped = clean_text.strip()
                if stripped.startswith("{") and "\"tool\"" in stripped:
                    payload = stripped
            if not payload:
                return None
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except Exception as parse_err:
                logger.warning(f"[VOICE] Tool call parse failed: {parse_err}")
            return None

        async def generate_llm(prompt: str, allow_tool_detection: bool) -> Dict[str, Any]:
            start_time = time.time()
            token_count = 0
            queue = asyncio.Queue()

            def generate_and_enqueue():
                nonlocal token_count
                try:
                    # === DIAGNOSTIC: Check model state ===
                    print(f"[VOICE-LLM] 🔍 Model check: llm={llm}, llm.model={getattr(llm, 'model', 'N/A')}", flush=True)
                    print(f"[VOICE-LLM] 📝 Prompt length: {len(prompt)} chars", flush=True)
                    print(f"[VOICE-LLM] 📝 Prompt preview: {prompt[:500]}...", flush=True)
                    
                    if not llm.model:
                        print("[VOICE-LLM] ⚠️ Model not loaded, attempting load...", flush=True)
                        llm.load_model()
                        print(f"[VOICE-LLM] ✅ Model loaded: {llm.model}", flush=True)

                    first_token_time = None
                    print("[VOICE-LLM] 🚀 Starting create_completion...", flush=True)
                    
                    generator = llm.model.create_completion(
                        prompt,
                        max_tokens=512,
                        stop=["<|im_end|>"],
                        stream=True,
                    )
                    
                    for chunk in generator:
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = (first_token_time - start_time) * 1000
                            print(f"[VOICE-LLM] ⚡ First token in {ttft:.0f}ms", flush=True)
                        token_count += 1
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    
                    print(f"[VOICE-LLM] ✅ Generation complete: {token_count} tokens", flush=True)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                    
                except Exception as e:
                    print(f"[VOICE-LLM] ❌ Generation error: {e}", flush=True)
                    print(f"[VOICE-LLM] ❌ Traceback:\n{tb.format_exc()}", flush=True)
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, generate_and_enqueue)

            full_response = ""
            last_sent_length = 0
            stream_decision: Optional[bool] = None

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break

                delta = chunk["choices"][0]["text"]
                full_response += delta
                token_count += 1

                import re
                clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
                clean_response = re.sub(r"<think>.*$", "", clean_response, flags=re.DOTALL)
                clean_response = clean_response.replace("</think>", "")

                if allow_tool_detection and stream_decision is None:
                    stripped = clean_response.lstrip()
                    if stripped:
                        if stripped.startswith("<tool_call>") or stripped.startswith("{"):
                            stream_decision = False
                        else:
                            stream_decision = True

                if stream_decision is False:
                    continue

                # Default to streaming if no tool detection
                if stream_decision is None and not allow_tool_detection:
                    stream_decision = True

                if stream_decision:
                    if len(clean_response) > last_sent_length:
                        new_content = clean_response[last_sent_length:]
                        last_sent_length = len(clean_response)

                        if new_content.strip() and dc and dc.readyState == "open":
                            dc.send(
                                json.dumps(
                                    {
                                        "type": "response_chunk",
                                        "text": new_content,
                                        "role": "assistant",
                                    }
                                )
                            )

            llm_time = time.time() - start_time
            tps = token_count / llm_time if llm_time > 0 else 0

            return {
                "response": full_response.strip(),
                "tokens": token_count,
                "tps": tps,
                "llm_time_ms": llm_time * 1000,
            }

        # Detect intent and execute relevant tools (SKIP if LangGraph already produced response)
        tool_call_data = None
        
        # Only run manual processing if LangGraph didn't produce a response
        if not full_response:
            intent = detect_intent(full_text) if customer_service_mode else "general"
            print(f"[VOICE] 🎯 Manual intent detection: {intent}", flush=True)
            
            # Execute tools based on intent
            forced_tool_results = []
            if customer_service_mode and tool_executor and intent != "general" and intent != "greeting":
                context["last_intent"] = intent
                
                # Name introduction -> check/register customer
                if intent == "name_intro":
                    extracted_name = extract_name(full_text)
                    if extracted_name:
                        print(f"[VOICE] 👤 Extracted name: {extracted_name}", flush=True)
                        check_result = await execute_tool_call("check_customer", extracted_name)
                        forced_tool_results.append({"tool": "check_customer", "result": check_result})
                        
                        # If customer found, save to context
                        if check_result.get("success") and check_result.get("customer"):
                            customer_context["customer"] = check_result["customer"]
                            context["customer_context"] = customer_context
                            print(f"[VOICE] ✅ Customer found and saved to context: {check_result['customer']}", flush=True)
                        else:
                            # Customer not found, register them automatically
                            print(f"[VOICE] 📝 Customer not found, registering: {extracted_name}", flush=True)
                            register_args = {
                                **extracted_name,
                                "preferred_language": language
                            }
                            register_result = await execute_tool_call("register_customer", register_args)
                            forced_tool_results.append({"tool": "register_customer", "result": register_result})
                            
                            # Save newly registered customer to context
                            if register_result.get("success") and register_result.get("customer"):
                                customer_context["customer"] = register_result["customer"]
                                context["customer_context"] = customer_context
                                print(f"[VOICE] ✅ Customer registered and saved to context: {register_result['customer']}", flush=True)
                
                # Slots/booking intent -> get available slots
                if intent in ["slots", "booking"]:
                    lowered_text = full_text.lower()
                    time_slot = extract_time_slot(full_text)
                    has_confirm_words = any(word in lowered_text for word in [
                        "yes", "yeah", "yep", "sure", "ok", "okay", "please", "book",
                        "confirm", "that one", "sounds good", "perfect", "great",
                        "at ", "for ", "i want", "i'll take", "let's do", "i choose"
                    ])
                    has_previous_selection = context.get("last_selected_slot") is not None
                    is_confirmation = detect_booking_confirmation(full_text) or (has_confirm_words and not time_slot and has_previous_selection)
                    print(f"[VOICE] 🔍 Booking check: is_confirmation={is_confirmation}, time_slot={time_slot}, text='{full_text}'", flush=True)
                    
                    if is_confirmation:
                        requested_date = context.get("last_requested_date") or extract_date(full_text, default_tomorrow=True)
                        customer = customer_context.get("customer")
                        selected_slot = None

                        # If user said "yes/confirm" without repeating the time, use last selected slot
                        if not time_slot and has_previous_selection:
                            selected_slot = context.get("last_selected_slot")
                            if selected_slot:
                                time_slot = selected_slot.get("start_time") or selected_slot.get("time")
                                requested_date = selected_slot.get("date") or requested_date
                                print(f"[VOICE] 🔍 Using last_selected_slot: {selected_slot}", flush=True)

                        print(f"[VOICE] 🔍 Booking details: time_slot={time_slot}, date={requested_date}, customer={customer}", flush=True)
                        
                        if (time_slot or selected_slot) and customer:
                            print(f"[VOICE] 🎯 Booking confirmation detected: {requested_date} {time_slot}", flush=True)
                            # Find matching slot from available slots if not already selected
                            if not selected_slot:
                                available_slots = context.get("last_available_slots", [])
                                matching_slot = None
                                for slot in available_slots:
                                    slot_time = slot.get("start_time", "")
                                    if slot_time.startswith(time_slot) or time_slot in slot_time:
                                        matching_slot = slot
                                        break
                                selected_slot = matching_slot
                            
                            if selected_slot:
                                context["last_selected_slot"] = selected_slot
                                book_args = {
                                    "customer_id": customer.get("id"),
                                    "time_slot_id": selected_slot.get("id"),
                                    "service_type": "consultation"
                                }
                                print(f"[VOICE] 📅 Booking appointment: {book_args}", flush=True)
                                book_result = await execute_tool_call("book_appointment", book_args)
                                if not book_result.get("success"):
                                    book_result["booking_failed"] = True
                                    book_result.setdefault("error", "Booking failed. Please choose another time.")
                                forced_tool_results.append({"tool": "book_appointment", "result": book_result})
                            else:
                                print(f"[VOICE] ⚠️ No matching slot found for {time_slot}", flush=True)
                                # Add explicit failure to prevent LLM hallucination
                                forced_tool_results.append({
                                    "tool": "book_appointment",
                                    "result": {
                                        "success": False,
                                        "error": f"No available slot found for {time_slot} on {requested_date}. Please ask customer to choose from the available times.",
                                        "booking_failed": True
                                    }
                                })
                        else:
                            if not customer:
                                print(f"[VOICE] ⚠️ Cannot book - no customer in context (customer_context={customer_context})", flush=True)
                                forced_tool_results.append({
                                    "tool": "book_appointment",
                                    "result": {
                                        "success": False,
                                        "error": "Customer not registered. Please ask for the customer's name first.",
                                        "booking_failed": True
                                    }
                                })
                            if not time_slot and not selected_slot:
                                print(f"[VOICE] ⚠️ Cannot book - no time slot extracted from '{full_text}'", flush=True)
                    else:
                        # Just fetch available slots
                        requested_date = extract_date(full_text, default_tomorrow=True)
                        if requested_date:
                            context["last_requested_date"] = requested_date
                            print(f"[VOICE] 📅 Getting slots for: {requested_date}", flush=True)
                            slots_result = await execute_tool_call("list_available_slots", {"date": requested_date})
                            
                            # If no slots found for requested date, try next 7 days
                            if not slots_result.get("available_slots"):
                                from datetime import datetime, timedelta
                                print(f"[VOICE] ⚠️ No slots for {requested_date}, trying next 7 days...", flush=True)
                                slots_result = await execute_tool_call("list_available_slots", {"days_ahead": 7})
                                if slots_result.get("available_slots"):
                                    print(f"[VOICE] ✅ Found {len(slots_result['available_slots'])} slots in next 7 days", flush=True)
                                else:
                                    print(f"[VOICE] ⚠️ No slots available in next 7 days either", flush=True)
                            
                            forced_tool_results.append({"tool": "list_available_slots", "result": slots_result})
                            # Cache available slots for later booking
                            if slots_result.get("available_slots"):
                                context["last_available_slots"] = slots_result["available_slots"]
                                # Update last_requested_date to the first available slot's date
                                first_slot = slots_result["available_slots"][0]
                                context["last_requested_date"] = first_slot.get("date", requested_date)
                
                # Combine results
                if forced_tool_results:
                    tool_call_data = {
                        "success": True,
                        "customer": customer_context.get("customer"),
                    }
                    for entry in forced_tool_results:
                        if entry.get("tool") == "list_available_slots":
                            tool_call_data.update(entry.get("result", {}))
                        if entry.get("tool") == "check_customer":
                            tool_call_data.update(entry.get("result", {}))
                        if entry.get("tool") == "register_customer":
                            tool_call_data.update(entry.get("result", {}))
                        if entry.get("tool") == "book_appointment":
                            tool_call_data.update(entry.get("result", {}))

                # Generate LLM response (single call, tools already executed)
                prompt = build_prompt(full_text, tool_result=tool_call_data)
                result = await generate_llm(prompt, allow_tool_detection=False)  # No tool detection needed
                full_response = result["response"]
                metrics_payload = result

                if metrics_payload and dc and dc.readyState == "open":
                    dc.send(
                        json.dumps(
                            {
                                "type": "metrics",
                                "llm_time_ms": metrics_payload["llm_time_ms"],
                                "tokens_per_sec": metrics_payload["tps"],
                                "total_tokens": metrics_payload["tokens"],
                            }
                        )
                    )

                if metrics_payload:
                    print(
                        f"[VOICE] 🤖 AI ({metrics_payload['tps']:.1f} t/s): {full_response[:80]}...",
                        flush=True,
                    )
            
            # ============================================================
            # FALLBACK: Generate LLM response for general/greeting intents
            # This handles simple conversations without customer_service_mode
            # ============================================================
            if not full_response and llm:
                print(f"[VOICE] 🤖 Generating general response for: {full_text}", flush=True)
                prompt = build_prompt(full_text, tool_result=None)
                result = await generate_llm(prompt, allow_tool_detection=False)
                full_response = result["response"]
                metrics_payload = result

                if metrics_payload and dc and dc.readyState == "open":
                    dc.send(
                        json.dumps(
                            {
                                "type": "metrics",
                                "llm_time_ms": metrics_payload["llm_time_ms"],
                                "tokens_per_sec": metrics_payload["tps"],
                                "total_tokens": metrics_payload["tokens"],
                            }
                        )
                    )

                if metrics_payload:
                    print(
                        f"[VOICE] 🤖 AI ({metrics_payload['tps']:.1f} t/s): {full_response[:80]}...",
                        flush=True,
                    )
        else:
            print(f"[VOICE] 🔀 Skipping manual processing - LangGraph response available", flush=True)
        # END of manual processing block

        # ============================================================
        # SAVE CONVERSATION TURN for multi-turn context
        # ============================================================
        if full_response.strip():
            voice_session_mgr = context.get("voice_session_manager")
            if voice_session_mgr:
                processing_time = int((time.time() - context.get("start_time", time.time())) * 1000)
                try:
                    await voice_session_mgr.add_conversation_turn(
                        session_id=session_id,
                        user_input=full_text,
                        ai_response=full_response,
                        processing_time_ms=processing_time,
                        transcription_quality="ok"
                    )
                    print(f"[VOICE] 💾 Saved conversation turn (user: {len(full_text)} chars, ai: {len(full_response)} chars)", flush=True)
                except Exception as save_err:
                    logger.warning(f"[VOICE] Failed to save conversation turn: {save_err}")

        # ============================================================
        # TTS SYNTHESIS (with fallback to Edge TTS if primary fails)
        # Supports: EdgeTTSEngine, XTTSEngine, SaudiXTTSEngine
        # ============================================================
        if full_response.strip():
            async def _generate_tts(tts_engine, tts_text: str, language: str, loop) -> Optional[str]:
                """Generate TTS audio with given engine."""
                tts_args = {"text": tts_text, "language": language}
                tts_type = type(tts_engine).__name__
                
                if tts_type == "EdgeTTSEngine":
                    tts_args["gender"] = "female"
                elif tts_type == "XTTSEngine":
                    # Check for speaker reference
                    if not getattr(tts_engine, "speaker_embedding", None):
                        if language == "ar":
                            fallback_wav = Path("/home/lumi/beautyai/voice_tests/input_test_questions/q1.wav")
                        else:
                            fallback_wav = Path("/home/lumi/beautyai/tests/webrtc/botox.wav")
                        
                        if fallback_wav.exists():
                            tts_args["speaker_wav"] = str(fallback_wav)
                elif tts_type == "SaudiXTTSEngine":
                    # Saudi XTTS has pre-computed speaker embeddings at load time
                    # If not available, try fallback speaker reference
                    if not getattr(tts_engine, "has_speaker_conditioning", lambda: False)():
                        fallback_wav = Path("/home/lumi/beautyai/backend/speakers/saudi-female/reference.wav")
                        if fallback_wav.exists():
                            tts_args["speaker_wav"] = str(fallback_wav)
                        else:
                            logger.warning("[VOICE] SaudiXTTSEngine has no speaker conditioning and no fallback WAV")
                
                return await loop.run_in_executor(
                    None,
                    lambda: tts_engine.text_to_speech(**tts_args)
                )
            
            try:
                # Clean the response for TTS (remove markdown, convert bullet points)
                tts_text = clean_llm_response_for_tts(full_response, language=language)
                
                if tts_text:
                    tts_type_name = type(tts).__name__ if tts else "None"
                    print(f"[VOICE] 🔊 Synthesizing TTS ({len(tts_text)} chars, lang={language}, engine={tts_type_name})...", flush=True)
                    
                    # Notify client: TTS starting
                    if dc and dc.readyState == "open":
                        dc.send(json.dumps({"type": "state", "state": "speaking"}))
                    
                    tts_start = time.time()
                    tts_time = 0
                    
                    # ============================================================
                    # STREAMING TTS: Progressive sentence-by-sentence synthesis
                    # Sends audio for each sentence as soon as it's ready
                    # ============================================================
                    if ENABLE_STREAMING_TTS:
                        sentences = _split_into_sentences(tts_text, language)
                        
                        if len(sentences) > 1:
                            print(
                                f"[VOICE] 🔊 Streaming TTS: {len(sentences)} sentences, "
                                f"lang={language}, engine={tts_type_name}",
                                flush=True,
                            )
                            
                            # Get TTS engine (primary or fallback)
                            tts_engine_to_use = tts
                            if not tts_engine_to_use:
                                tts_engine_to_use = await _get_edge_tts_fallback()
                            
                            # Stream TTS for each sentence
                            tts_time = await _stream_tts_sentences(
                                sentences=sentences,
                                tts_engine=tts_engine_to_use,
                                language=language,
                                dc=dc,
                                context=context,
                            )
                            
                            print(f"[VOICE] 🔊 Streaming TTS complete in {tts_time:.0f}ms total", flush=True)
                        else:
                            # Single sentence - fall through to batch mode
                            ENABLE_STREAMING_TTS_LOCAL = False
                    else:
                        ENABLE_STREAMING_TTS_LOCAL = False
                    
                    # ============================================================
                    # BATCH TTS: Traditional full-response synthesis (fallback)
                    # ============================================================
                    if not ENABLE_STREAMING_TTS or (ENABLE_STREAMING_TTS and len(_split_into_sentences(tts_text, language)) <= 1):
                        audio_path = None
                        
                        # Try primary TTS engine first
                        if tts:
                            try:
                                audio_path = await _generate_tts(tts, tts_text, language, loop)
                            except Exception as primary_err:
                                logger.warning(f"[VOICE] Primary TTS ({tts_type_name}) failed, trying Edge TTS fallback: {primary_err}")
                                print(f"[VOICE] ⚠️ Primary TTS failed: {primary_err}, trying Edge TTS...", flush=True)
                        
                        # Fallback to Edge TTS if primary failed or not available
                        if not audio_path or not os.path.exists(str(audio_path) if audio_path else ""):
                            try:
                                print("[VOICE] 🔄 Using Edge TTS fallback (singleton)...", flush=True)
                                edge_tts_fallback = await _get_edge_tts_fallback()
                                audio_path = await _generate_tts(edge_tts_fallback, tts_text, language, loop)
                            except Exception as fallback_err:
                                logger.error(f"[VOICE] Edge TTS fallback also failed: {fallback_err}")
                                print(f"[VOICE] ❌ Edge TTS fallback failed: {fallback_err}", flush=True)
                        
                        tts_time = (time.time() - tts_start) * 1000
                        print(f"[VOICE] 🔊 TTS generated in {tts_time:.0f}ms: {audio_path}", flush=True)
                        
                        # Read the audio file and send as base64
                        if audio_path and os.path.exists(audio_path):
                            with open(audio_path, 'rb') as f:
                                audio_data = f.read()
                            
                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                            
                            # Send audio to client
                            if dc and dc.readyState == "open":
                                dc.send(
                                    json.dumps(
                                        {
                                            "type": "tts_audio",
                                            "audio_base64": audio_b64,
                                            "format": "wav",
                                            "language": language,
                                            "tts_time_ms": tts_time,
                                        }
                                    )
                                )
                                print(f"[VOICE] 📤 Sent TTS audio ({len(audio_data)} bytes)", flush=True)
                            
                            # Clean up temp file
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                        else:
                            print(f"[VOICE] ⚠️ TTS audio file not found: {audio_path}", flush=True)
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"[VOICE] TTS error: {e}")
                print(f"[VOICE] ❌ TTS error: {e}", flush=True)

        # Signal ready to listen again
        context["is_speaking"] = False
        if dc and dc.readyState == "open":
            dc.send(json.dumps({"type": "mic_control", "action": "unmute"}))
            dc.send(json.dumps({"type": "state", "state": "listening"}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[VOICE] LLM error: {e}")
        context["is_speaking"] = False
        if dc and dc.readyState == "open":
            dc.send(json.dumps({"type": "mic_control", "action": "unmute"}))
            dc.send(json.dumps({"type": "state", "state": "listening"}))


async def _cleanup_session(session_id: str):
    """Clean up session resources."""
    if session_id in _active_connections:
        ctx = _active_connections.pop(session_id)
        print(f"[VOICE] 🧹 Cleaning up session {session_id}", flush=True)
        try:
            if ctx.get("pc"):
                await ctx["pc"].close()
            if ctx.get("processing_task"):
                ctx["processing_task"].cancel()
            if ctx.get("turn_timer_task"):
                ctx["turn_timer_task"].cancel()
            if ctx.get("rnnoise_processor"):
                ctx["rnnoise_processor"].cleanup()
            # Clean up voice session for multi-turn context
            voice_session_mgr = ctx.get("voice_session_manager")
            if voice_session_mgr:
                await voice_session_mgr.close_session(session_id)
                print(f"[VOICE] 💾 Voice session closed", flush=True)
        except Exception as e:
            logger.error(f"[VOICE] Cleanup error: {e}")
