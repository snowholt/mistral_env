"""
WebRTC Voice Endpoint (Optimized Implementation)

Audio processing pipeline:
- Jitter Buffer: 128 packets (approx 2.5s) for network resilience
- Transient Suppressor: Optional @ 48kHz (disabled by default)
- Anti-aliasing: 6th-order Butterworth @ 8kHz
- Resampling: Single-stage 48kHz → 16kHz (with debug capture option)
- Noise Reduction: RNNoise (16→48→RNNoise→16 pipeline)
- VAD: Silero VAD with threshold 0.2
- STT: Faster-Whisper (Turbo)
- LLM: Qwen (via Llama.cpp)
- Output: Text response via Data Channel

Pipeline Flow:
    48kHz Raw → [Transient Suppressor] → Float32 → Butterworth → Resample 16kHz
    → RNNoise (16→48→denoise→16) → VAD → Whisper → LLM

Author: BeautyAI Framework
Date: November 2025
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
from typing import Dict, Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
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
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState
from ...utils.rnnoise_wrapper import RNNoiseProcessor
from ...utils.transient_suppressor import TransientSuppressor
from ...utils.transcription_cleaner import filter_whisper_output

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
# Environment variables for feature toggles
ENABLE_TRANSIENT_SUPPRESSOR = os.getenv("VOICE_TRANSIENT_SUPPRESSOR", "0") == "1"
ENABLE_DEBUG_CAPTURE = os.getenv("VOICE_DEBUG_CAPTURE", "0") == "1"
DEBUG_CAPTURE_DIR = Path(os.getenv("VOICE_DEBUG_CAPTURE_DIR", "/home/lumi/beautyai/reports/debug/voice"))

# Ensure debug directory exists
if ENABLE_DEBUG_CAPTURE:
    DEBUG_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

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
    language: str = Field(default="ar", description="Language code (ar, en)")


class OfferResponse(BaseModel):
    sdp: str
    type: str = "answer"
    session_id: str


class ICERequest(BaseModel):
    session_id: str
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_m_line_index: Optional[int] = None


# ============================================================
# ENDPOINTS
# ============================================================
@webrtc_voice_router.post("/offer", response_model=OfferResponse)
async def handle_offer(request: OfferRequest):
    """Create WebRTC session with optimized audio pipeline."""
    try:
        session_id = str(uuid.uuid4())
        print(f"[VOICE] 🚀 Creating session {session_id} (language={request.language})", flush=True)

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

        # Session Context
        session_context = {
            "pc": pc,
            "session_id": session_id,
            "language": request.language,
            "start_time": time.time(),
            "audio_track": None,
            "data_channel": None,
            "processing_task": None,
            "vad_service": None,
            "whisper_model": None,
            "llm_model": None,
            "rnnoise_processor": None,
            "transient_suppressor": None,
            "transcript_buffer": [],
            "turn_timer_task": None,
            "loop": asyncio.get_event_loop(),
            # Debug capture buffers
            "debug_capture_enabled": ENABLE_DEBUG_CAPTURE,
            "debug_16khz_buffer": [] if ENABLE_DEBUG_CAPTURE else None,
            "debug_rnnoise_buffer": [] if ENABLE_DEBUG_CAPTURE else None,
        }

        # Load Models (Persistent)
        try:
            model_manager = get_persistent_model_manager()
            session_context["whisper_model"] = model_manager.get_whisper_model()
            session_context["llm_model"] = model_manager.get_llm_model()

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

            vad_service = WebRTCVADService(session_id, language=request.language, config=vad_config)
            if await vad_service.initialize():
                session_context["vad_service"] = vad_service
                print(
                    f"[VOICE] ✅ VAD Initialized (silero=0.3, thresh=0.1, sustained=2, lang={request.language})",
                    flush=True,
                )
            else:
                print("[VOICE] ❌ VAD Init Failed", flush=True)
        except Exception as e:
            logger.error(f"[VOICE] VAD error: {e}")

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

                    # Accumulate Speech
                    if state in [
                        VADState.VOICE_START,
                        VADState.VOICE_ACTIVE,
                        VADState.VOICE_END_PENDING,
                    ]:
                        # Cancel pending turn timer
                        if context.get("turn_timer_task"):
                            context["turn_timer_task"].cancel()
                            context["turn_timer_task"] = None

                        speech_buffer_16k.append(audio_16k)

                    # End of Speech → Process
                    elif state == VADState.VOICE_END:
                        if speech_buffer_16k:
                            full_audio = np.concatenate(speech_buffer_16k)
                            asyncio.create_task(
                                _process_speech_segment(
                                    session_id, full_audio, context
                                )
                            )
                            speech_buffer_16k = []

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "MediaStreamError" in str(type(e).__name__) or "End of file" in str(e):
                    print("[VOICE] ⏹️ Track ended", flush=True)
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
    """Wait for 2 seconds of silence, then trigger LLM."""
    try:
        await asyncio.sleep(2.0)
        await _trigger_llm_response(session_id, context)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[VOICE] Timer error: {e}")


async def _trigger_llm_response(session_id: str, context: Dict):
    """Generate and stream LLM response."""
    llm = context.get("llm_model")
    dc = context.get("data_channel")
    loop = context.get("loop")
    buffer = context.get("transcript_buffer", [])

    if not buffer or not llm:
        return

    full_text = " ".join(buffer)
    context["transcript_buffer"] = []
    context["turn_timer_task"] = None

    print(f"[VOICE] 🤖 Generating response for: {full_text}", flush=True)

    # Notify client: Processing
    if dc and dc.readyState == "open":
        dc.send(json.dumps({"type": "state", "state": "processing"}))

    try:
        prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful AI assistant. /no_think<|im_end|>\n"
            f"<|im_start|>user\n{full_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        start_time = time.time()
        token_count = 0
        queue = asyncio.Queue()

        def generate_and_enqueue():
            try:
                if not llm.model:
                    llm.load_model()

                generator = llm.model.create_completion(
                    prompt,
                    max_tokens=512,
                    stop=["<|im_end|>"],
                    stream=True,
                )
                for chunk in generator:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                print(f"[VOICE] ❌ Generation error: {e}", flush=True)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, generate_and_enqueue)

        full_response = ""

        while True:
            chunk = await queue.get()
            if chunk is None:
                break

            delta = chunk["choices"][0]["text"]
            full_response += delta
            token_count += 1

            # Filter <think> tags
            clean_delta = delta.replace("<think>", "").replace("</think>", "")

            if clean_delta and dc and dc.readyState == "open":
                dc.send(
                    json.dumps(
                        {
                            "type": "response_chunk",
                            "text": clean_delta,
                            "role": "assistant",
                        }
                    )
                )

        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0

        print(f"[VOICE] 🤖 AI ({tps:.1f} t/s): {full_response[:50]}...", flush=True)

        # Send Final Metrics
        if dc and dc.readyState == "open":
            dc.send(
                json.dumps(
                    {
                        "type": "metrics",
                        "llm_time_ms": total_time * 1000,
                        "tokens_per_sec": tps,
                        "total_tokens": token_count,
                    }
                )
            )
            dc.send(json.dumps({"type": "state", "state": "listening"}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[VOICE] LLM error: {e}")
        if dc and dc.readyState == "open":
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
        except Exception as e:
            logger.error(f"[VOICE] Cleanup error: {e}")
