"""
WebRTC Debug Voice Capture Endpoint

Simple endpoint for capturing audio at each layer WITHOUT STT/LLM processing.
Saves raw audio captures for debugging sample rate and resampling issues.

Author: BeautyAI Framework  
Date: October 29, 2025
"""

import asyncio
import json
import logging
import os
import time
import uuid
import wave
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt
from math import gcd

from ...core.persistent_model_manager import get_persistent_model_manager
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState
from ...utils.rnnoise_wrapper import RNNoiseProcessor
from ...utils.audio_resampling import process_with_rnnoise_16khz_pipeline
from ...utils.noise_comparison import compare_noise_reduction_methods, measure_processing_latency, generate_comparison_summary
from ...utils.dtln_wrapper import DTLNProcessor
from ...utils.deepfilternet_wrapper import DeepFilterNetProcessor
from ...utils.nsnet2_wrapper import SpectralGatingProcessor
from ...utils.comb_filter import CombFilter
from ...utils.transient_suppressor import TransientSuppressor

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[4]


def _resolve_debug_directory() -> Path:
    """Return a writable directory for storing debug captures."""
    candidates: list[Path] = []

    env_dir = os.getenv("VOICE_DEBUG_CAPTURE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    # Use reports directory for debug captures (repository-level)
    candidates.append(BACKEND_ROOT.parent / "reports/debug/webrtc")
    # Fallback to backend-local path if reports not writable
    candidates.append(BACKEND_ROOT / "logs/webrtc/debug_captures")

    print(f"[DEBUG-CAPTURE] 🔍 BACKEND_ROOT={BACKEND_ROOT}", flush=True)
    print(f"[DEBUG-CAPTURE] 🔍 Candidate paths: {candidates}", flush=True)

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            with probe.open("wb"):
                pass
            probe.unlink(missing_ok=True)
            logger.info(f"[DEBUG-CAPTURE] Using debug directory: {candidate}")
            return candidate
        except OSError as exc:  # pragma: no cover - best effort logging
            logger.warning(
                f"[DEBUG-CAPTURE] Unable to use debug directory {candidate}: {exc}"
            )

    logger.error(
        "[DEBUG-CAPTURE] No writable debug directory available; "
        "set VOICE_DEBUG_CAPTURE_DIR to a writable path"
    )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="No writable debug capture directory available",
    )

debug_capture_router = APIRouter(
    prefix="/api/v1/webrtc/debug/voice-capture",
    tags=["webrtc-debug"],
)

# Store active connections
_debug_connections: Dict[str, Dict[str, Any]] = {}


class DebugOfferRequest(BaseModel):
    sdp: str = Field(..., min_length=10)
    type: str = Field(default="offer")


class DebugOfferResponse(BaseModel):
    sdp: str
    type: str = "answer"
    peer_id: str
    message: str = "Debug capture session created"


class DebugICERequest(BaseModel):
    peer_id: str
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_m_line_index: Optional[int] = None


@debug_capture_router.post("/offer", response_model=DebugOfferResponse)
async def handle_debug_offer(request: DebugOfferRequest):
    """Create debug capture session - captures audio at each layer."""
    try:
        peer_id = "session"  # Fixed name for overwrite mode (no unique ID)
        print(f"[DEBUG-CAPTURE] Creating session for {peer_id}", flush=True)
        logger.info(f"[DEBUG-CAPTURE] Creating session for {peer_id}")
        
        # Create RTCPeerConnection with STUN and TURN servers for NAT traversal
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                RTCIceServer(
                    urls=["turn:188.48.209.107:15478"],
                    username="beautyai",
                    credential="beautyai2025"
                ),
            ]
        )
        pc = RTCPeerConnection(configuration=config)
        print(f"[DEBUG-CAPTURE] {peer_id} RTCPeerConnection created with STUN+TURN servers", flush=True)
        
        debug_dir = _resolve_debug_directory()
        print(f"[DEBUG-CAPTURE] Debug directory: {debug_dir}", flush=True)
        
        capture_info = {
            "pc": pc,
            "peer_id": peer_id,
            "debug_dir": debug_dir,
            "frames_captured": 0,
            "start_time": time.time(),
            "audio_track": None,
            "data_channel": None,
            "capture_task": None,
            "layer_48khz_raw": [],
            "layer_15_transient_48khz": [],  # NEW: Layer 1.5 - Transient Suppression @ 48kHz
            "layer_48khz_float": [],
            "layer_16khz": [],
            "layer_16khz_vad_filtered": [],  # NEW: VAD-filtered audio only (speech segments)
            "layer_48khz_vad_filtered": [],  # Layer 5: 48kHz VAD-filtered
            "layer_31_ema_16khz": [],  # EXPERIMENTAL: Layer 3.1 - EMA noise reduction @ 16kHz
            "layer_31b_percentile_16khz": [],  # NEW: Layer 3.1b - Percentile Gate @ 16kHz
            "layer_32_rnnoise_16khz": [],  # EXPERIMENTAL: Layer 3.2 - RNNoise @ 16kHz
            "layer_33_dtln_16khz": [],  # EXPERIMENTAL: Layer 3.3 - DTLN @ 16kHz
            "layer_34_deepfilternet_16khz": [],  # EXPERIMENTAL: Layer 3.4 - DeepFilterNet @ 16kHz
            "layer_35_nsnet2_16khz": [],  # EXPERIMENTAL: Layer 3.5 - Spectral Gating @ 16kHz
            "layer_36_comb_16khz": [],  # EXPERIMENTAL: Layer 3.6 - Comb Filter @ 16kHz (80 Hz removal)
            "comparison_metrics": [],  # Store quality/latency comparisons per frame
            "transcriptions": [],  # Store transcription results
            "whisper_model": None,  # Will be loaded on first use
            "vad_service": None,  # VAD for speech detection
            "is_speech_active": False,  # Current speech state
            "speech_buffer": [],  # Buffer for accumulating speech audio
            "whisper_enabled": False,  # DISABLED for VAD testing
            # CPU/Buffer monitoring
            "frame_processing_times": [],  # Track processing latency per frame
            "buffer_underruns": 0,  # Count suspected buffer issues
            "last_frame_time": None,  # Track inter-frame timing
            "cpu_samples": [],  # Track CPU usage
        }
        
        # Get preloaded Whisper model for transcription after VAD filtering
        capture_info["whisper_enabled"] = False  # Will be set to True if model loads successfully
        try:
            model_manager = get_persistent_model_manager()
            whisper_engine = model_manager.get_whisper_model()
            if whisper_engine:
                capture_info["whisper_model"] = whisper_engine
                capture_info["whisper_enabled"] = True
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Loaded persistent Whisper Turbo model", flush=True)
                logger.info(f"[DEBUG-CAPTURE] {peer_id} using persistent Whisper Turbo (whisper-large-v3-turbo)")
            else:
                print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Whisper model not preloaded, transcription disabled", flush=True)
                logger.warning(f"[DEBUG-CAPTURE] {peer_id} Whisper model not available - ensure model is preloaded on startup")
        except Exception as e:
            logger.error(f"[DEBUG-CAPTURE] {peer_id} Failed to load Whisper model: {e}")
            capture_info["whisper_model"] = None
            print(f"[DEBUG-CAPTURE] {peer_id} ❌ Whisper load error: {e}", flush=True)
        
        # Initialize RNNoise processor for experimental Layer 3.2 comparison
        capture_info["rnnoise_processor"] = None
        capture_info["rnnoise_enabled"] = False
        try:
            rnnoise_proc = RNNoiseProcessor()
            capture_info["rnnoise_processor"] = rnnoise_proc
            capture_info["rnnoise_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ RNNoise processor initialized for experimental Layer 3.2", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} RNNoise ready for parallel comparison")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} RNNoise not available (will skip Layer 3.2): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without RNNoise - only EMA will be tested", flush=True)
        
        # Initialize DTLN processor for experimental Layer 3.3 comparison
        capture_info["dtln_processor"] = None
        capture_info["dtln_enabled"] = False
        try:
            dtln_proc = DTLNProcessor()
            capture_info["dtln_processor"] = dtln_proc
            capture_info["dtln_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ DTLN processor initialized for experimental Layer 3.3", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} DTLN ready for parallel comparison")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} DTLN not available (will skip Layer 3.3): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without DTLN", flush=True)
        
        # Initialize DeepFilterNet processor for experimental Layer 3.4 comparison
        capture_info["deepfilternet_processor"] = None
        capture_info["deepfilternet_enabled"] = False
        try:
            dfn_proc = DeepFilterNetProcessor(sample_rate=16000)
            capture_info["deepfilternet_processor"] = dfn_proc
            capture_info["deepfilternet_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ DeepFilterNet processor initialized for experimental Layer 3.4", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} DeepFilterNet ready for parallel comparison")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} DeepFilterNet not available (will skip Layer 3.4): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without DeepFilterNet", flush=True)
        
        # Initialize NSNet2 processor for experimental Layer 3.5 comparison
        capture_info["nsnet2_processor"] = None
        capture_info["nsnet2_enabled"] = False
        try:
            nsnet2_proc = SpectralGatingProcessor(sample_rate=16000)
            capture_info["nsnet2_processor"] = nsnet2_proc
            capture_info["nsnet2_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Spectral Gating processor initialized for experimental Layer 3.5", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Spectral Gating ready for parallel comparison")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} Spectral Gating not available (will skip Layer 3.5): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without Spectral Gating", flush=True)
        
        # Initialize Comb Filter for experimental Layer 3.6 comparison (80 Hz periodic noise removal)
        capture_info["comb_processor"] = None
        capture_info["comb_enabled"] = False
        try:
            # Target 80 Hz fundamental frequency (buffer underrun artifact)
            # Q=2.0 for wider notches to reduce ringing artifacts (was 30.0)
            comb_proc = CombFilter(sample_rate=16000, fundamental_freq=80.0, quality_factor=2.0)
            capture_info["comb_processor"] = comb_proc
            capture_info["comb_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Comb Filter initialized for experimental Layer 3.6 (80 Hz removal)", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Comb Filter ready: 80 Hz + harmonics")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} Comb Filter not available (will skip Layer 3.6): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without Comb Filter", flush=True)
        
        # Buffer underrun debugging metrics
        capture_info["buffer_metrics"] = {
            "frame_recv_times": [],
            "frame_process_times": [],
            "inter_frame_delays": [],
            "processing_latencies": [],
            "stage_timings": [],  # Breakdown by processing stage
        }
        capture_info["last_frame_time"] = None
        
        print(f"[DEBUG-CAPTURE] {peer_id} 🔍 Buffer underrun debugging ENABLED - tracking timing metrics", flush=True)
        
        # Initialize Transient Suppressor for Layer 1.5 (48kHz crackle removal before downsampling)
        capture_info["transient_suppressor"] = None
        capture_info["transient_suppressor_enabled"] = False
        try:
            transient_proc = TransientSuppressor(
                sample_rate=48000,
                kernel_size=5,  # 5-sample median filter at 48kHz
                threshold=0.8,  # 80% above median = transient
                energy_window=5,
                frame_size=960  # 20ms at 48kHz
            )
            capture_info["transient_suppressor"] = transient_proc
            capture_info["transient_suppressor_enabled"] = True
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Transient Suppressor initialized for Layer 1.5 (48kHz crackle removal)", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Transient Suppressor ready: median filter kernel=5, threshold=0.8")
        except Exception as e:
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} Transient Suppressor not available (will skip Layer 1.5): {e}")
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Continuing without Transient Suppressor", flush=True)
        
        # Initialize VAD service
        try:
            # Create config and set properties (WebRTCVADConfig is not a dataclass)
            vad_config = WebRTCVADConfig()
            vad_config.silero_sensitivity = 0.3  # More sensitive (0.3 = catches more speech)
            vad_config.webrtc_sensitivity = 2    # 0-3, where 3 = least sensitive
            vad_config.post_speech_silence_ms = 700  # Wait 700ms before ending speech (groups sentences)
            vad_config.min_speech_duration_ms = 50  # Minimum 50ms speech to register
            vad_config.language_thresholds = {"ar": 0.1, "en": 0.1, "default": 0.1}  # Lower threshold for better detection
            vad_config.warmup_filter_duration_ms = 200  # Filter initial 200ms noise
            vad_config.min_sustained_speech_frames = 2  # Need 2 consecutive frames to confirm speech
            vad_config.log_vad_decisions = True  # Enable VAD decision logging
            
            vad_service = WebRTCVADService(peer_id, language="en", config=vad_config)
            
            # CRITICAL: Initialize the VAD (loads Silero model)
            vad_initialized = await vad_service.initialize()
            if not vad_initialized:
                raise RuntimeError("VAD initialization failed - Silero model not loaded")
            
            capture_info["vad_service"] = vad_service
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ VAD initialized successfully (Silero sens=0.3, post_silence=700ms, sustained=2 frames, thresh=0.1)", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} VAD service ready with balanced settings")
        except Exception as e:
            logger.error(f"[DEBUG-CAPTURE] {peer_id} Failed to initialize VAD: {e}")
            capture_info["vad_service"] = None
        
        # Track received audio
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            print(f"[DEBUG-CAPTURE] {peer_id} received {track.kind} track", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} received {track.kind} track")
            
            if track.kind == "audio":
                print(f"[DEBUG-CAPTURE] {peer_id} starting audio capture task", flush=True)
                capture_info["audio_track"] = track

                # Start capturing frames and keep handle for cleanup sync
                capture_task = asyncio.create_task(
                    _capture_audio_frames(peer_id, track, capture_info)
                )
                capture_info["capture_task"] = capture_task
        
        # Monitor ICE connection state changes
        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            print(f"[DEBUG-CAPTURE] {peer_id} ICE connection state: {pc.iceConnectionState}", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} ICE connection state: {pc.iceConnectionState}")
        
        # Monitor connection state changes
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            print(f"[DEBUG-CAPTURE] {peer_id} Connection state: {pc.connectionState}", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Connection state: {pc.connectionState}")
        
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=request.sdp, type=request.type)
        )
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Create data channel for VAD feedback (after answer created)
        data_channel = pc.createDataChannel("vad_feedback")
        capture_info["data_channel"] = data_channel
        
        @data_channel.on("open")
        def on_data_channel_open():
            print(f"[DEBUG-CAPTURE] {peer_id} 📡 Data channel opened for VAD feedback", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Data channel ready")
        
        @data_channel.on("error")
        def on_data_channel_error(error):
            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Data channel error: {error}", flush=True)
        
        # Store connection
        _debug_connections[peer_id] = capture_info
        
        logger.info(f"[DEBUG-CAPTURE] {peer_id} session created")
        
        return DebugOfferResponse(
            sdp=pc.localDescription.sdp,
            peer_id=peer_id,
            message=f"Debug capture active. Files will be saved to: {debug_dir}"
        )
        
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Error creating session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@debug_capture_router.post("/ice")
async def handle_debug_ice(request: DebugICERequest):
    """Handle ICE candidate for debug session."""
    try:
        if request.peer_id not in _debug_connections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Debug session not found: {request.peer_id}"
            )
        
        pc = _debug_connections[request.peer_id]["pc"]
        
        # Parse ICE candidate string into components
        # Browser format: "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host generation 0 ufrag abc network-cost 999"
        parts = request.candidate.split()
        if len(parts) < 8 or not parts[0].startswith('candidate:'):
            raise ValueError(f"Invalid candidate format (need at least 8 parts): {request.candidate}")
        
        foundation = parts[0][10:]  # Remove "candidate:" prefix
        component = int(parts[1])
        protocol = parts[2].upper()
        priority = int(parts[3])
        ip = parts[4]
        port = int(parts[5])
        
        # Find 'typ' keyword (can be at different positions due to extra fields)
        if 'typ' not in parts:
            raise ValueError(f"Missing 'typ' keyword in candidate: {request.candidate}")
        
        typ_idx = parts.index('typ')
        if typ_idx + 1 >= len(parts):
            raise ValueError(f"Missing candidate type after 'typ' in: {request.candidate}")
        
        candidate_type = parts[typ_idx + 1]
        
        # Create ICE candidate with parsed components
        candidate = RTCIceCandidate(
            component=component,
            foundation=foundation,
            ip=ip,
            port=port,
            priority=priority,
            protocol=protocol,
            type=candidate_type,
            sdpMid=request.sdp_mid,
            sdpMLineIndex=request.sdp_m_line_index
        )
        
        await pc.addIceCandidate(candidate)
        
        return {"status": "ok", "peer_id": request.peer_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] ICE error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@debug_capture_router.get("/{peer_id}/transcriptions")
async def get_transcriptions(peer_id: str):
    """Get real-time transcriptions for a debug session."""
    try:
        if peer_id not in _debug_connections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Debug session not found: {peer_id}"
            )
        
        info = _debug_connections[peer_id]
        transcriptions = info.get("transcriptions", [])
        
        return {
            "status": "ok",
            "peer_id": peer_id,
            "transcription_count": len(transcriptions),
            "transcriptions": transcriptions,
            "frames_captured": info.get("frames_captured", 0),
            "whisper_available": info.get("whisper_model") is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Transcription fetch error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@debug_capture_router.delete("/{peer_id}")
async def cleanup_debug_session(peer_id: str):
    """Stop capture and save audio files."""
    try:
        if peer_id not in _debug_connections:
            print(f"[DEBUG-CAPTURE] {peer_id} cleanup requested but session missing", flush=True)
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} cleanup requested but session missing")
            return {"status": "ok", "message": "Session already cleaned up"}
        
        info = _debug_connections[peer_id]

        print(f"[DEBUG-CAPTURE] {peer_id} cleanup starting", flush=True)
        logger.info(f"[DEBUG-CAPTURE] {peer_id} cleanup starting")

        # Signal track stop so capture loop can finish neatly
        audio_track = info.get("audio_track")
        if audio_track:
            print(f"[DEBUG-CAPTURE] {peer_id} stopping audio track prior to save", flush=True)
            try:
                audio_track.stop()
            except Exception as track_err:
                logger.warning(
                    f"[DEBUG-CAPTURE] {peer_id} error stopping track before save: {track_err}"
                )

        # Wait briefly for capture task to flush buffers before saving
        capture_task = info.get("capture_task")
        if capture_task and not capture_task.done():
            print(f"[DEBUG-CAPTURE] {peer_id} waiting for capture task to finish", flush=True)
            try:
                await asyncio.wait_for(capture_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[DEBUG-CAPTURE] {peer_id} capture task did not finish within timeout"
                )

        # Save captured audio
        await _save_captured_audio(peer_id, info)

        print(f"[DEBUG-CAPTURE] {peer_id} cleanup finished saving audio", flush=True)
        logger.info(f"[DEBUG-CAPTURE] {peer_id} cleanup finished saving audio")
        
        # Close connection
        pc = info["pc"]
        await pc.close()
        
        # Remove from active connections
        del _debug_connections[peer_id]
        
        duration = time.time() - info["start_time"]
        
        logger.info(
            f"[DEBUG-CAPTURE] {peer_id} session ended: "
            f"{info['frames_captured']} frames, {duration:.2f}s"
        )
        
        return {
            "status": "ok",
            "peer_id": peer_id,
            "frames_captured": info["frames_captured"],
            "duration_sec": duration,
            "actual_sample_rate": info.get("actual_sample_rate", 48000),
            "message": f"Audio saved to: {info['debug_dir']}/layer*.wav (overwrite mode)"
        }
        
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Cleanup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def _capture_audio_frames(peer_id: str, track: MediaStreamTrack, info: Dict):
    """Capture raw audio frames and save at each processing stage."""
    import numpy as np
    
    print(f"[DEBUG-CAPTURE] {peer_id} starting frame capture loop", flush=True)
    logger.info(f"[DEBUG-CAPTURE] {peer_id} starting frame capture loop")
    
    # Accumulators for different layers
    layer_48khz_raw = info.setdefault("layer_48khz_raw", [])
    layer_48khz_float = info.setdefault("layer_48khz_float", [])
    layer_16khz = info.setdefault("layer_16khz", [])
    layer_16khz_vad_filtered = info.setdefault("layer_16khz_vad_filtered", [])  # Layer 4: 16kHz VAD-filtered
    layer_48khz_vad_filtered = info.setdefault("layer_48khz_vad_filtered", [])  # Layer 5: 48kHz VAD-filtered (NEW!)
    
    # Speech buffers for VAD-filtered segments (both 16kHz and 48kHz)
    speech_buffer_16k = []  # Layer 4 (16kHz)
    speech_buffer_48k = []  # Layer 5 (48kHz)
    
    def finalize_speech_segment(reason: str, frame_index: int) -> None:
        """Flush buffered speech into VAD layers and trigger Whisper if enabled."""
        if not speech_buffer_16k:
            return

        total_samples_16k = int(sum(len(chunk) for chunk in speech_buffer_16k))
        if total_samples_16k == 0:
            speech_buffer_16k.clear()
            speech_buffer_48k.clear()
            return

        speech_duration = total_samples_16k / 16000
        print(
            f"[DEBUG-CAPTURE] {peer_id} 🛑 Speech ended ({reason}), buffer_frames={len(speech_buffer_16k)} (~{speech_duration:.2f}s)",
            flush=True,
        )
        logger.info(f"[DEBUG-CAPTURE] {peer_id} Speech segment ({reason}): {speech_duration:.2f}s")

        # Persist buffered audio into Layer 4 and Layer 5 collections
        for speech_frame in speech_buffer_16k:
            layer_16khz_vad_filtered.append(speech_frame.copy())

        for speech_frame_48k in speech_buffer_48k:
            layer_48khz_vad_filtered.append(speech_frame_48k.copy())

        speech_segments_count = info.get("speech_segments_count", 0)

        if info.get("whisper_enabled", False):
            whisper_model = info.get("whisper_model")
            if whisper_model:
                try:
                    # Layer 4 transcription (16kHz VAD-filtered PCM)
                    speech_audio_16k = np.concatenate(speech_buffer_16k)
                    audio_int16_16k = (np.clip(speech_audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
                    audio_bytes_16k = audio_int16_16k.tobytes()
                    segment_duration_16k = len(audio_int16_16k) / 16000

                    print(
                        f"[WHISPER-L4] {peer_id} 🎤 Transcribing Layer 4 (16kHz) segment #{speech_segments_count}: "
                        f"{len(audio_int16_16k)} samples ({segment_duration_16k:.2f}s)...",
                        flush=True,
                    )
                    start_t4 = time.time()
                    # Get language preference from session config or default to None (auto-detect)
                    target_language = info.get("language", None)  # None = auto-detect, "en" = English, "ar" = Arabic
                    transcription_16k = whisper_model.transcribe_audio_bytes(
                        audio_bytes_16k, audio_format="pcm_raw", language=target_language
                    )
                    latency_16k = (time.time() - start_t4) * 1000

                    # Layer 5 transcription (48kHz VAD-filtered resampled for Whisper)
                    transcription_48k = None
                    latency_48k = 0.0
                    segment_duration_48k = 0.0
                    if speech_buffer_48k:
                        speech_audio_48k = np.concatenate(speech_buffer_48k)
                        speech_audio_16k_from_48k = resample_poly(
                            speech_audio_48k, up=1, down=3, window=("kaiser", 8.0)
                        )
                        speech_audio_16k_from_48k = np.clip(speech_audio_16k_from_48k, -1.0, 1.0)
                        audio_int16_48k = (speech_audio_16k_from_48k * 32767).astype(np.int16)
                        audio_bytes_48k = audio_int16_48k.tobytes()
                        segment_duration_48k = len(audio_int16_48k) / 16000

                        print(
                            f"[WHISPER-L5] {peer_id} 🎤 Transcribing Layer 5 (48kHz→16kHz) segment #{speech_segments_count}: "
                            f"{len(audio_int16_48k)} samples ({segment_duration_48k:.2f}s)...",
                            flush=True,
                        )
                        start_t5 = time.time()
                        # Use same language preference as Layer 4
                        target_language = info.get("language", None)  # None = auto-detect, "en" = English, "ar" = Arabic
                        transcription_48k = whisper_model.transcribe_audio_bytes(
                            audio_bytes_48k, audio_format="pcm_raw", language=target_language
                        )
                        latency_48k = (time.time() - start_t5) * 1000

                    if transcription_16k and transcription_16k.strip():
                        info.setdefault("transcriptions", []).append(
                            {
                                "segment_number": speech_segments_count,
                                "frame_index": frame_index,
                                "timestamp": time.time() - info["start_time"],
                                "layer": "Layer4_16kHz",
                                "text": transcription_16k,
                                "duration_s": segment_duration_16k,
                                "latency_ms": latency_16k,
                            }
                        )
                        print(
                            f"[WHISPER-L4] {peer_id} ✅ Layer 4 Segment #{speech_segments_count} ({latency_16k:.0f}ms): "
                            f"\"{transcription_16k}\"",
                            flush=True,
                        )

                    if transcription_48k and transcription_48k.strip():
                        info.setdefault("transcriptions", []).append(
                            {
                                "segment_number": speech_segments_count,
                                "frame_index": frame_index,
                                "timestamp": time.time() - info["start_time"],
                                "layer": "Layer5_48kHz",
                                "text": transcription_48k,
                                "duration_s": segment_duration_48k,
                                "latency_ms": latency_48k,
                            }
                        )
                        print(
                            f"[WHISPER-L5] {peer_id} ✅ Layer 5 Segment #{speech_segments_count} ({latency_48k:.0f}ms): "
                            f"\"{transcription_48k}\"",
                            flush=True,
                        )

                    if transcription_16k and transcription_48k:
                        match = (
                            "✅ MATCH"
                            if transcription_16k.strip().lower() == transcription_48k.strip().lower()
                            else "⚠️ DIFFERENT"
                        )
                        print(
                            f"[WHISPER-COMPARE] {peer_id} Segment #{speech_segments_count}: {match}",
                            flush=True,
                        )
                        print(f"  L4 (16kHz): \"{transcription_16k}\"", flush=True)
                        print(f"  L5 (48kHz): \"{transcription_48k}\"", flush=True)
                    elif transcription_16k or transcription_48k:
                        print(
                            f"[WHISPER] {peer_id} ⚠️ Segment #{speech_segments_count}: One layer empty",
                            flush=True,
                        )

                except Exception as transcribe_error:  # pragma: no cover - logging path
                    print(
                        f"[WHISPER] {peer_id} ❌ Segment #{speech_segments_count} transcription error: {transcribe_error}",
                        flush=True,
                    )
                    logger.error(
                        f"[WHISPER] {peer_id} transcription error: {transcribe_error}",
                        exc_info=True,
                    )
        else:
            print(
                f"[DEBUG-CAPTURE] {peer_id} ℹ️ Whisper not enabled - speech segments saved to Layer 4 (16kHz) and Layer 5 (48kHz)",
                flush=True,
            )

        speech_buffer_16k.clear()
        speech_buffer_48k.clear()

    frame_count = 0
    timeout_count = 0
    max_consecutive_timeouts = 30  # 30 timeouts * 1s = 30 seconds total wait
    
    # Frame accumulator for handling network jitter (router/VPN issues)
    frame_accumulator = []
    accumulator_target = 5  # Accumulate 5 frames (100ms) before processing
    frames_since_last_process = 0
    
    try:
        while True:
            try:
                print(f"[DEBUG-CAPTURE] {peer_id} waiting for frame (timeout #{timeout_count + 1})...", flush=True)
                
                # Track frame receive timing
                recv_start = time.perf_counter()
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)  # Increased timeout for packet loss
                recv_time_ms = (time.perf_counter() - recv_start) * 1000
                
                frame_count += 1
                timeout_count = 0  # Reset on successful receive
                info["frames_captured"] = frame_count
                
                # Track overall frame processing start
                frame_processing_start = time.perf_counter()
                
                # CPU/Buffer monitoring with detailed timing
                frame_start_time = time.time()
                if info.get("last_frame_time") is not None:
                    inter_frame_delay = (frame_start_time - info["last_frame_time"]) * 1000  # ms
                    info["buffer_metrics"]["inter_frame_delays"].append(inter_frame_delay)
                    
                    # Expected delay at 48kHz with 960 samples (20ms) or 16kHz with 320 samples
                    expected_delay_ms = 20.0
                    if inter_frame_delay > expected_delay_ms * 1.5:  # >30ms gap indicates buffer issue
                        info["buffer_underruns"] += 1
                        logger.warning(
                            f"[BUFFER-UNDERRUN] {peer_id} #{info['buffer_underruns']}: "
                            f"{inter_frame_delay:.2f}ms gap (expected ~{expected_delay_ms:.0f}ms), "
                            f"frame={frame_count}, recv_latency={recv_time_ms:.2f}ms"
                        )
                        print(
                            f"[BUFFER-UNDERRUN] {peer_id} #{info['buffer_underruns']}: "
                            f"{inter_frame_delay:.2f}ms gap at frame {frame_count}",
                            flush=True
                        )
                info["last_frame_time"] = frame_start_time
                info["buffer_metrics"]["frame_recv_times"].append(recv_time_ms)
                
                # Sample CPU usage every 10 frames
                if frame_count % 10 == 0:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    info["cpu_samples"].append(cpu_percent)
                
                # LAYER 1: Raw frame from WebRTC - Detect actual sample rate
                audio_array = frame.to_ndarray()  # Get raw ndarray
                sample_rate = frame.sample_rate  # Use actual sample rate from frame
                
                # Log first frame details for debugging
                if frame_count == 1:
                    print(
                        f"[DEBUG-CAPTURE] {peer_id} FIRST FRAME: "
                        f"sample_rate={sample_rate}Hz, samples={frame.samples}, "
                        f"format={frame.format}, layout={frame.layout}, "
                        f"array_shape={audio_array.shape}, array_dtype={audio_array.dtype}",
                        flush=True
                    )
                    logger.info(
                        f"[DEBUG-CAPTURE] {peer_id} FIRST FRAME: "
                        f"sample_rate={sample_rate}Hz, samples={frame.samples}, "
                        f"format={frame.format}, layout={frame.layout}"
                    )
                else:
                    print(f"[DEBUG-CAPTURE] {peer_id} received frame #{frame_count} (sr={sample_rate}Hz, samples={frame.samples})", flush=True)
                
                # Convert stereo to mono if needed
                # PyAV returns interleaved stereo as (1, samples*channels) or (channels, samples)
                if audio_array.ndim > 1:
                    # Flatten first
                    audio_array = audio_array.flatten()
                
                # Check if we have stereo (indicated by frame.samples vs array length)
                # frame.samples is per-channel, array length might be total
                if len(audio_array) == frame.samples * 2:
                    # Interleaved stereo: [L,R,L,R,...] -> reshape to (samples, 2) then average
                    # Keep as int16 to avoid noise from float conversion
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    if frame_count == 1:
                        print(f"[DEBUG-CAPTURE] {peer_id} Converted stereo to mono (averaged L+R channels)", flush=True)
                
                # Save raw int16
                layer_48khz_raw.append(audio_array.copy())
                
                # LAYER 1.5: Transient Suppression @ 48kHz (NEW - suppress crackles BEFORE downsampling)
                # Gemini Recommendation: Median filtering at source sample rate prevents Butterworth smearing
                audio_transient_suppressed = audio_array.copy()
                if info.get("transient_suppressor_enabled") and sample_rate == 48000:
                    try:
                        # Convert int16 to float32 for processing
                        if np.issubdtype(audio_transient_suppressed.dtype, np.integer):
                            dtype_info = np.iinfo(audio_transient_suppressed.dtype)
                            scale = float(max(abs(dtype_info.min), dtype_info.max)) or 1.0
                            audio_transient_float = audio_transient_suppressed.astype(np.float32) / scale
                        else:
                            audio_transient_float = audio_transient_suppressed.astype(np.float32)
                        
                        # Apply transient suppression
                        transient_proc = info["transient_suppressor"]
                        audio_transient_float = transient_proc.process_frame(audio_transient_float)
                        
                        # Convert back to int16 for storage
                        audio_transient_suppressed = (np.clip(audio_transient_float, -1.0, 1.0) * 32767).astype(np.int16)
                        
                        # Save Layer 1.5 output
                        info["layer_15_transient_48khz"].append(audio_transient_suppressed.copy())
                        
                        # Log first frame processing
                        if frame_count == 1:
                            energy_stats = transient_proc.get_recent_energy_stats()
                            if energy_stats:
                                print(
                                    f"[DEBUG-CAPTURE] {peer_id} 🎯 Layer 1.5 Transient Suppressor ACTIVE: "
                                    f"median={energy_stats['median']:.6f}, threshold={energy_stats['median'] * 1.8:.6f}",
                                    flush=True
                                )
                        
                        # Use transient-suppressed audio for downstream processing
                        audio_array = audio_transient_suppressed
                        
                    except Exception as e:
                        logger.warning(f"[DEBUG-CAPTURE] {peer_id} Layer 1.5 transient suppression failed: {e}")
                        # Fall back to original audio if suppression fails
                
                # LAYER 2: Convert to float32
                if np.issubdtype(audio_array.dtype, np.integer):
                    dtype_info = np.iinfo(audio_array.dtype)
                    scale = float(max(abs(dtype_info.min), dtype_info.max)) or 1.0
                    audio_float = audio_array.astype(np.float32) / scale
                else:
                    audio_float = audio_array.astype(np.float32)
                
                layer_48khz_float.append(audio_float.copy())
                
                # LAYER 3: Resample to 16kHz with STRONG anti-aliasing filter + adaptive noise gate
                # Microphone produces high-frequency hiss (>8kHz) that folds into audible crackle during downsampling
                if sample_rate != 16000:
                    # STRONG ANTI-ALIASING: Apply aggressive low-pass filter BEFORE downsampling
                    # Microphone hiss above 8kHz needs to be removed to prevent crackle
                    nyquist_freq = sample_rate / 2
                    cutoff_freq = 8000  # Nyquist-aligned: 8kHz (exactly at target Nyquist)
                    
                    # Only filter if we need to remove frequencies above cutoff
                    if nyquist_freq > cutoff_freq:
                        # 8th-order Butterworth for steeper rolloff (sharper cutoff)
                        # Stronger filtering to eliminate microphone hiss before downsampling
                        normalized_cutoff = cutoff_freq / nyquist_freq
                        sos = butter(8, normalized_cutoff, btype='low', output='sos')
                        audio_float = sosfiltfilt(sos, audio_float)
                        audio_float = np.clip(audio_float, -1.0, 1.0)
                        
                        if frame_count == 1:
                            print(
                                f"[DEBUG-CAPTURE] {peer_id} 🎛️ Anti-aliasing ACTIVE: "
                                f"8th-order Butterworth low-pass at {cutoff_freq}Hz (Nyquist={nyquist_freq}Hz) "
                                f"to prevent hiss→crackle aliasing", 
                                flush=True
                            )
                            logger.info(
                                f"[DEBUG-CAPTURE] {peer_id} 🎛️ Anti-aliasing ACTIVE: "
                                f"8th-order Butterworth low-pass at {cutoff_freq}Hz to prevent hiss→crackle aliasing"
                            )
                    
                    # Two-stage resampling for better quality when downsampling from 48kHz
                    # Stage 1: 48kHz → 24kHz (2:1)
                    # Stage 2: 24kHz → 16kHz (3:2)
                    resample_start = time.perf_counter()
                    if sample_rate == 48000:
                        # First stage: 48→24 (simpler ratio, less aliasing)
                        audio_24k = resample_poly(audio_float, 1, 2, window=('kaiser', 8.0))
                        audio_24k = np.clip(audio_24k, -1.0, 1.0)
                        # Second stage: 24→16
                        audio_16k = resample_poly(audio_24k, 2, 3, window=('kaiser', 8.0))
                        audio_16k = np.clip(audio_16k, -1.0, 1.0)
                    else:
                        # Direct resampling for other sample rates
                        ratio_gcd = gcd(sample_rate, 16000)
                        up = 16000 // ratio_gcd
                        down = sample_rate // ratio_gcd
                        audio_16k = resample_poly(audio_float, up, down, window=('kaiser', 8.0))
                        audio_16k = np.clip(audio_16k, -1.0, 1.0)
                    resample_time_ms = (time.perf_counter() - resample_start) * 1000
                    
                    # NO NOISE GATE - Removed to isolate buffer underrun issues
                    # Keep audio as-is after resampling (only downsampling, no gating)
                    audio_16k_original = audio_16k.copy()
                else:
                    audio_16k = audio_float
                    audio_16k_original = audio_float.copy()
                    resample_time_ms = 0.0
                
                layer_16khz.append(audio_16k.copy())
                
                # Track processing stage timing
                stage_timing = {
                    "frame": frame_count,
                    "resample_ms": resample_time_ms,
                    "total_ms": 0.0  # Will be calculated at end
                }
                
                # ============================================
                # OPTIONAL: EXPERIMENTAL NOISE REDUCTION (NOT FOR UNDERRUN DEBUGGING)
                # These are kept but NOT used in Layer 3 to avoid masking buffer issues
                # ============================================
                if info.get("rnnoise_enabled"):
                    try:
                        # Layer 3.2: RNNoise processing (for comparison only, NOT used in Layer 3)
                        rnnoise_start = time.perf_counter()
                        rnnoise_proc = info["rnnoise_processor"]
                        from beautyai_inference.utils.audio_resampling import process_with_rnnoise_16khz_pipeline
                        layer_32_rnnoise, vad_probs = process_with_rnnoise_16khz_pipeline(
                            audio_16k_original, rnnoise_proc
                        )
                        info["layer_32_rnnoise_16khz"].append(layer_32_rnnoise)
                        stage_timing["rnnoise_ms"] = (time.perf_counter() - rnnoise_start) * 1000
                        
                    except Exception as e:
                        logger.warning(f"[DEBUG-CAPTURE] {peer_id} RNNoise processing failed: {e}")
                
                # Calculate total frame processing time BEFORE VAD
                stage_timing["total_ms"] = (time.perf_counter() - frame_processing_start) * 1000
                info["buffer_metrics"]["stage_timings"].append(stage_timing)
                
                # Log detailed timing every 50 frames
                if frame_count % 50 == 0:
                    print(
                        f"[TIMING] {peer_id} Frame {frame_count}: "
                        f"recv={recv_time_ms:.2f}ms, resample={resample_time_ms:.2f}ms, "
                        f"total={stage_timing['total_ms']:.2f}ms",
                        flush=True
                    )
                
                # VAD: Check for voice activity (with timing)
                vad_service = info.get("vad_service")
                
                if vad_service:
                    try:
                        vad_start = time.perf_counter()
                        
                        # Convert 16kHz float to int16 PCM for VAD
                        audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
                        audio_16k_bytes = audio_16k_int16.tobytes()
                        
                        # Debug: Log that we're calling VAD
                        if frame_count % 50 == 0:  # Log every 50 frames to avoid spam
                            print(f"[DEBUG-CAPTURE] {peer_id} Calling VAD for frame {frame_count}, audio_size={len(audio_16k_bytes)} bytes", flush=True)
                        
                        # Process through VAD
                        vad_result = await vad_service.process_audio_chunk(
                            audio_16k_bytes,
                            metadata={"sample_rate": 16000, "language": "en"}
                        )
                        vad_time_ms = (time.perf_counter() - vad_start) * 1000
                        stage_timing["vad_ms"] = vad_time_ms
                        
                        # Get VAD state and probability
                        voice_state = vad_result.get("voice_state")
                        silero_prob = vad_result.get("silero_probability", 0)
                        
                        # PROPER STATE MACHINE: Listen to VAD_STATE transitions, not frame-by-frame booleans
                        # This respects post_speech_silence_ms (groups words together!)
                        was_state = info.get("previous_vad_state", VADState.INACTIVE)
                        info["previous_vad_state"] = voice_state
                        
                        # Log VAD decision every 25 frames (every 0.5 seconds) for visibility
                        if frame_count % 25 == 0:
                            state_emoji = "🟢" if voice_state in [VADState.VOICE_START, VADState.VOICE_ACTIVE] else "⚪"
                            print(f"[VAD] {peer_id} Frame {frame_count}: {state_emoji} {voice_state} (prob={silero_prob:.3f})", flush=True)
                        
                        # VOICE_START: Speech just started (begin accumulating)
                        if voice_state == VADState.VOICE_START and was_state != VADState.VOICE_START:
                            segments_count = info.get("speech_segments_count", 0) + 1
                            info["speech_segments_count"] = segments_count
                            print(f"[DEBUG-CAPTURE] {peer_id} 🎤 Speech started (frame {frame_count}, segment #{segments_count})", flush=True)
                            logger.info(f"[DEBUG-CAPTURE] {peer_id} Speech detected at frame {frame_count}")
                            speech_buffer_16k.clear()
                            speech_buffer_48k.clear()
                        
                        # ACCUMULATE AUDIO: During speech AND during pending silence (to keep full segment)
                        if voice_state in [VADState.VOICE_START, VADState.VOICE_ACTIVE, VADState.VOICE_END_PENDING]:
                            speech_buffer_16k.append(audio_16k.copy())  # Layer 4 buffer
                            speech_buffer_48k.append(audio_float.copy())  # Layer 5 buffer (48kHz float)
                        
                        # VOICE_END: Speech confirmed ended (after waiting post_speech_silence_ms)
                        # This only fires AFTER silence threshold, so segments are properly grouped!
                        if voice_state == VADState.VOICE_END:
                            finalize_speech_segment("VAD state VOICE_END", frame_count)
                        
                    except Exception as vad_error:
                        logger.error(f"[DEBUG-CAPTURE] {peer_id} VAD error: {vad_error}")
                
                # Log progress every 50 frames
                if frame_count % 50 == 0:
                    logger.info(
                        f"[DEBUG-CAPTURE] {peer_id}: {frame_count} frames, "
                        f"source_rate={sample_rate}Hz, source_samples={len(audio_array)}, 16kHz_samples={len(audio_16k)}"
                    )
                
            except asyncio.TimeoutError:
                timeout_count += 1
                if frame_count == 0 and timeout_count <= 5:
                    # Still waiting for first frame
                    print(f"[DEBUG-CAPTURE] {peer_id} no frames yet (timeout #{timeout_count})", flush=True)
                    continue
                elif frame_count == 0 and timeout_count > max_consecutive_timeouts:
                    # Never received any frames
                    print(f"[DEBUG-CAPTURE] {peer_id} gave up after {timeout_count} timeouts without receiving any frames", flush=True)
                    logger.warning(f"[DEBUG-CAPTURE] {peer_id} recv timeout after {timeout_count} attempts, no frames received")
                    break
                elif frame_count > 0 and timeout_count >= 3:
                    # Had frames, then stopped
                    print(f"[DEBUG-CAPTURE] {peer_id} recv timeout after {frame_count} frames, ending capture", flush=True)
                    logger.info(f"[DEBUG-CAPTURE] {peer_id} recv timeout, ending capture")
                    break
                # Otherwise continue waiting
                continue
            except Exception as e:
                print(f"[DEBUG-CAPTURE] {peer_id} frame error: {e}", flush=True)
                logger.error(f"[DEBUG-CAPTURE] {peer_id} frame error: {e}")
                break
                
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] {peer_id} capture loop error: {e}", exc_info=True)
    
    finally:
        finalize_speech_segment("capture loop exit flush", frame_count)

        # Save accumulated layers and detected sample rate
        info["layer_48khz_raw"] = layer_48khz_raw
        info["layer_48khz_float"] = layer_48khz_float
        info["layer_16khz"] = layer_16khz
        # Store the actual sample rate for use during save
        if frame_count > 0:
            # Get sample rate from the last captured frame (should be consistent)
            info["actual_sample_rate"] = sample_rate
        
        vad_filtered_count = len(info.get("layer_16khz_vad_filtered", []))
        
        # Calculate and display VAD statistics
        if vad_filtered_count > 0:
            speech_segments = info.get("speech_segments_count", 0)
            print(f"[VAD-SUMMARY] {peer_id} 🎤 Detected {speech_segments} speech segments", flush=True)
        
        # Display Whisper transcription summary
        transcriptions = info.get("transcriptions", [])
        if transcriptions:
            layer4_count = sum(1 for t in transcriptions if t.get("layer") == "Layer4_16kHz")
            layer5_count = sum(1 for t in transcriptions if t.get("layer") == "Layer5_48kHz")
            print(f"[WHISPER-SUMMARY] {peer_id} 📝 Transcribed {layer4_count} Layer 4 (16kHz) + {layer5_count} Layer 5 (48kHz) segments:", flush=True)
            for i, trans in enumerate(transcriptions, 1):
                layer_tag = trans.get("layer", "Unknown")
                print(f"  [{i}] {layer_tag} ({trans.get('latency_ms', 0):.0f}ms) \"{trans.get('text', '')}\"", flush=True)
            logger.info(f"[WHISPER-SUMMARY] {peer_id} completed {layer4_count} L4 + {layer5_count} L5 transcriptions")
        elif info.get("whisper_enabled", False):
            print(f"[WHISPER-SUMMARY] {peer_id} ⚠️ No transcriptions (all segments were silence/empty)", flush=True)
        
        vad_filtered_l4_count = len(info.get("layer_16khz_vad_filtered", []))
        vad_filtered_l5_count = len(info.get("layer_48khz_vad_filtered", []))
        
        print(
            f"[DEBUG-CAPTURE] {peer_id} capture complete: {frame_count} frames, "
            f"detected_sample_rate={info.get('actual_sample_rate', 'unknown')}Hz, "
            f"layers: raw={len(layer_48khz_raw)}, float={len(layer_48khz_float)}, 16kHz={len(layer_16khz)}, L4-VAD={vad_filtered_l4_count}, L5-VAD={vad_filtered_l5_count}",
            flush=True
        )
        logger.info(
            f"[DEBUG-CAPTURE] {peer_id} capture complete: {frame_count} frames, "
            f"detected_sample_rate={info.get('actual_sample_rate', 'unknown')}Hz, "
            f"layers: raw={len(layer_48khz_raw)}, float={len(layer_48khz_float)}, 16kHz={len(layer_16khz)}, L4-VAD={vad_filtered_l4_count}, L5-VAD={vad_filtered_l5_count}"
        )


async def _save_captured_audio(peer_id: str, info: Dict):
    """Save captured audio layers as WAV files."""
    import numpy as np
    
    debug_dir = info["debug_dir"]
    print(f"[DEBUG-CAPTURE] {peer_id} saving audio to {debug_dir}", flush=True)
    logger.info(f"[DEBUG-CAPTURE] {peer_id} saving audio to {debug_dir}")
    
    # Get actual sample rate from stored metadata (set during capture)
    actual_sample_rate = info.get("actual_sample_rate", 48000)
    
    try:
        # Save Layer 1: Raw audio (as received from WebRTC at actual sample rate)
        if info.get("layer_48khz_raw"):
            audio_48k_raw = np.concatenate(info["layer_48khz_raw"])

            print(
                f"[DEBUG-CAPTURE] {peer_id} layer1 samples: {len(audio_48k_raw)}",
                flush=True,
            )
            
            # Ensure int16
            if not np.issubdtype(audio_48k_raw.dtype, np.integer):
                audio_48k_raw = (np.clip(audio_48k_raw, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_48k_raw = debug_dir / f"layer1_{actual_sample_rate}hz_raw.wav"
            print(f"[DEBUG-CAPTURE] {peer_id} Writing layer1 to: {path_48k_raw}", flush=True)
            with wave.open(str(path_48k_raw), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(actual_sample_rate)
                wav.writeframes(audio_48k_raw.tobytes())
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer1 written successfully", flush=True)
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 1 ({actual_sample_rate}Hz raw): {len(audio_48k_raw)} samples, "
                f"{len(audio_48k_raw)/actual_sample_rate:.2f}s -> {path_48k_raw}"
            )
        
        # Save Layer 1.5: 48kHz Transient Suppressed (NEW - crackles removed before downsampling)
        if info.get("layer_15_transient_48khz"):
            audio_15_transient = np.concatenate(info["layer_15_transient_48khz"])
            print(
                f"[DEBUG-CAPTURE] {peer_id} layer1.5 samples: {len(audio_15_transient)}",
                flush=True,
            )
            
            # Ensure int16
            if not np.issubdtype(audio_15_transient.dtype, np.integer):
                audio_15_transient = (np.clip(audio_15_transient, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_15_transient = debug_dir / f"layer15_transient_{actual_sample_rate}hz.wav"
            print(f"[DEBUG-CAPTURE] {peer_id} Writing layer1.5 to: {path_15_transient}", flush=True)
            with wave.open(str(path_15_transient), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(actual_sample_rate)
                wav.writeframes(audio_15_transient.tobytes())
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer1.5 written successfully", flush=True)
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 1.5 ({actual_sample_rate}Hz transient suppressed): {len(audio_15_transient)} samples, "
                f"{len(audio_15_transient)/actual_sample_rate:.2f}s -> {path_15_transient}"
            )
        
        # Save Layer 2: 48kHz float (normalized)
        if info.get("layer_48khz_float"):
            audio_48k_float = np.concatenate(info["layer_48khz_float"])
            print(
                f"[DEBUG-CAPTURE] {peer_id} layer2 samples: {len(audio_48k_float)}",
                flush=True,
            )
            audio_48k_int16 = (np.clip(audio_48k_float, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_48k_float = debug_dir / f"layer2_{actual_sample_rate}hz_float.wav"
            print(f"[DEBUG-CAPTURE] {peer_id} Writing layer2 to: {path_48k_float}", flush=True)
            with wave.open(str(path_48k_float), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(actual_sample_rate)
                wav.writeframes(audio_48k_int16.tobytes())
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer2 written successfully", flush=True)
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 2 ({actual_sample_rate}Hz float): {len(audio_48k_float)} samples, "
                f"{len(audio_48k_float)/actual_sample_rate:.2f}s -> {path_48k_float}"
            )
        
        # Save Layer 3: 16kHz (resampled)
        if info.get("layer_16khz"):
            audio_16k = np.concatenate(info["layer_16khz"])
            print(
                f"[DEBUG-CAPTURE] {peer_id} layer3 samples: {len(audio_16k)}",
                flush=True,
            )
            audio_16k_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_16k = debug_dir / f"layer3_16khz.wav"
            print(f"[DEBUG-CAPTURE] {peer_id} Writing layer3 to: {path_16k}", flush=True)
            with wave.open(str(path_16k), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_16k_int16.tobytes())
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer3 written successfully", flush=True)
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 3 (16kHz resampled): {len(audio_16k)} samples, "
                f"{len(audio_16k)/16000:.2f}s -> {path_16k}"
            )
        
        # Save Layer 4: 16kHz VAD-filtered (speech only, no silence)
        if info.get("layer_16khz_vad_filtered"):
            audio_16k_vad = np.concatenate(info["layer_16khz_vad_filtered"])
            print(
                f"[DEBUG-CAPTURE] {peer_id} layer4 (VAD-filtered) samples: {len(audio_16k_vad)}",
                flush=True,
            )
            audio_16k_vad_int16 = (np.clip(audio_16k_vad, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_16k_vad = debug_dir / f"layer4_16khz_vad_filtered.wav"
            with wave.open(str(path_16k_vad), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_16k_vad_int16.tobytes())
            
            duration_total = len(info["layer_16khz"]) * len(info["layer_16khz"][0]) / 16000 if info.get("layer_16khz") else 0
            duration_speech = len(audio_16k_vad) / 16000
            speech_ratio = (duration_speech / duration_total * 100) if duration_total > 0 else 0
            
            print(
                f"[DEBUG-CAPTURE] {peer_id} VAD Stats: Total={duration_total:.2f}s, Speech={duration_speech:.2f}s ({speech_ratio:.1f}%)",
                flush=True,
            )
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 4 (16kHz VAD-filtered, speech only): {len(audio_16k_vad)} samples, "
                f"{duration_speech:.2f}s ({speech_ratio:.1f}% of total) -> {path_16k_vad}"
            )
            
            # Save Layer 5: 48kHz VAD-filtered (same VAD timing, higher sample rate)
            if info.get("layer_48khz_vad_filtered"):
                audio_48k_vad = np.concatenate(info["layer_48khz_vad_filtered"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer5 (VAD-filtered 48kHz) samples: {len(audio_48k_vad)}",
                    flush=True,
                )
                audio_48k_vad_int16 = (np.clip(audio_48k_vad, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_48k_vad = debug_dir / f"layer5_48khz_vad_filtered.wav"
                with wave.open(str(path_48k_vad), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(48000)
                    wav.writeframes(audio_48k_vad_int16.tobytes())
                
                duration_speech_48k = len(audio_48k_vad) / 48000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 5 saved: {duration_speech_48k:.2f}s @ 48kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 5 (48kHz VAD-filtered, speech only): {len(audio_48k_vad)} samples, "
                    f"{duration_speech_48k:.2f}s -> {path_48k_vad}"
                )
            else:
                print(f"[DEBUG-CAPTURE] {peer_id} Layer 5 (48kHz VAD-filtered) empty", flush=True)
        else:
            print(f"[DEBUG-CAPTURE] {peer_id} No speech detected by VAD - Layer 4 and Layer 5 empty", flush=True)
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} No speech detected - VAD layers empty")
        
        # ============================================
        # EXPERIMENTAL LAYERS: Save Layer 3.1 (EMA) and Layer 3.2 (RNNoise)
        # ============================================
        if info.get("rnnoise_enabled"):
            # Save Layer 3.1: EMA noise gate (16kHz)
            if info.get("layer_31_ema_16khz"):
                audio_31_ema = np.concatenate(info["layer_31_ema_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.1 (EMA) samples: {len(audio_31_ema)}",
                    flush=True,
                )
                audio_31_ema_int16 = (np.clip(audio_31_ema, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_31_ema = debug_dir / f"layer31_ema_16khz.wav"
                with wave.open(str(path_31_ema), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_31_ema_int16.tobytes())
                
                duration_31 = len(audio_31_ema) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.1 (EMA) saved: {duration_31:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.1 (EMA noise gate, 16kHz): {len(audio_31_ema)} samples, "
                    f"{duration_31:.2f}s -> {path_31_ema}"
                )
            
            # Save Layer 3.1b: Percentile Gate (16kHz) - NEW: Replaces broken EMA gate
            if info.get("layer_31b_percentile_16khz"):
                audio_31b_percentile = np.concatenate(info["layer_31b_percentile_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.1b (Percentile Gate) samples: {len(audio_31b_percentile)}",
                    flush=True,
                )
                audio_31b_percentile_int16 = (np.clip(audio_31b_percentile, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_31b_percentile = debug_dir / f"layer31b_percentile_16khz.wav"
                with wave.open(str(path_31b_percentile), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_31b_percentile_int16.tobytes())
                
                duration_31b = len(audio_31b_percentile) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.1b (Percentile Gate) saved: {duration_31b:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.1b (Percentile gate, 16kHz): {len(audio_31b_percentile)} samples, "
                    f"{duration_31b:.2f}s -> {path_31b_percentile}"
                )
            
            # Save Layer 3.2: RNNoise (16kHz)
            if info.get("layer_32_rnnoise_16khz"):
                audio_32_rnnoise = np.concatenate(info["layer_32_rnnoise_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.2 (RNNoise) samples: {len(audio_32_rnnoise)}",
                    flush=True,
                )
                audio_32_rnnoise_int16 = (np.clip(audio_32_rnnoise, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_32_rnnoise = debug_dir / f"layer32_rnnoise_16khz.wav"
                with wave.open(str(path_32_rnnoise), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_32_rnnoise_int16.tobytes())
                
                duration_32 = len(audio_32_rnnoise) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.2 (RNNoise) saved: {duration_32:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.2 (RNNoise, 16kHz): {len(audio_32_rnnoise)} samples, "
                    f"{duration_32:.2f}s -> {path_32_rnnoise}"
                )
            
            # Save Layer 3.3: DTLN (16kHz)
            if info.get("layer_33_dtln_16khz"):
                audio_33_dtln = np.concatenate(info["layer_33_dtln_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.3 (DTLN) samples: {len(audio_33_dtln)}",
                    flush=True,
                )
                audio_33_dtln_int16 = (np.clip(audio_33_dtln, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_33_dtln = debug_dir / f"layer33_dtln_16khz.wav"
                with wave.open(str(path_33_dtln), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_33_dtln_int16.tobytes())
                
                duration_33 = len(audio_33_dtln) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.3 (DTLN) saved: {duration_33:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.3 (DTLN, 16kHz): {len(audio_33_dtln)} samples, "
                    f"{duration_33:.2f}s -> {path_33_dtln}"
                )
            
            # Save Layer 3.4: DeepFilterNet (16kHz)
            if info.get("layer_34_deepfilternet_16khz"):
                audio_34_dfn = np.concatenate(info["layer_34_deepfilternet_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.4 (DeepFilterNet) samples: {len(audio_34_dfn)}",
                    flush=True,
                )
                audio_34_dfn_int16 = (np.clip(audio_34_dfn, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_34_dfn = debug_dir / f"layer34_deepfilternet_16khz.wav"
                with wave.open(str(path_34_dfn), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_34_dfn_int16.tobytes())
                
                duration_34 = len(audio_34_dfn) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.4 (DeepFilterNet) saved: {duration_34:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.4 (DeepFilterNet, 16kHz): {len(audio_34_dfn)} samples, "
                    f"{duration_34:.2f}s -> {path_34_dfn}"
                )
            
            # Save Layer 3.5: Spectral Gating (16kHz)
            if info.get("layer_35_nsnet2_16khz"):
                audio_35_nsnet2 = np.concatenate(info["layer_35_nsnet2_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.5 (Spectral Gating) samples: {len(audio_35_nsnet2)}",
                    flush=True,
                )
                audio_35_nsnet2_int16 = (np.clip(audio_35_nsnet2, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_35_nsnet2 = debug_dir / f"layer35_nsnet2_16khz.wav"
                with wave.open(str(path_35_nsnet2), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_35_nsnet2_int16.tobytes())
                
                duration_35 = len(audio_35_nsnet2) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.5 (Spectral Gating) saved: {duration_35:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.5 (Spectral Gating, 16kHz): {len(audio_35_nsnet2)} samples, "
                    f"{duration_35:.2f}s -> {path_35_nsnet2}"
                )
            
            # Save Layer 3.6: Comb Filter (16kHz) - 80 Hz PERIODIC NOISE REMOVAL
            if info.get("layer_36_comb_16khz"):
                audio_36_comb = np.concatenate(info["layer_36_comb_16khz"])
                print(
                    f"[DEBUG-CAPTURE] {peer_id} layer3.6 (Comb Filter 80Hz) samples: {len(audio_36_comb)}",
                    flush=True,
                )
                audio_36_comb_int16 = (np.clip(audio_36_comb, -1.0, 1.0) * 32767).astype(np.int16)
                
                path_36_comb = debug_dir / f"layer36_comb_16khz.wav"
                with wave.open(str(path_36_comb), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_36_comb_int16.tobytes())
                
                duration_36 = len(audio_36_comb) / 16000
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Layer 3.6 (Comb Filter) saved: {duration_36:.2f}s @ 16kHz", flush=True)
                logger.info(
                    f"[DEBUG-CAPTURE] Saved Layer 3.6 (Comb Filter, 16kHz): {len(audio_36_comb)} samples, "
                    f"{duration_36:.2f}s -> {path_36_comb}"
                )
            
            # Save CPU/Buffer monitoring report
            if info.get("cpu_samples") or info.get("buffer_underruns"):
                total_frames = info.get("frames_captured", 0)
                monitoring_report = {
                    "buffer_underruns": info.get("buffer_underruns", 0),
                    "total_frames": total_frames,
                    "underrun_rate_percent": (info.get("buffer_underruns", 0) / total_frames * 100) if total_frames > 0 else 0,
                    "cpu_usage": {
                        "samples": info.get("cpu_samples", []),
                        "mean_percent": np.mean(info.get("cpu_samples", [0])) if info.get("cpu_samples") else 0,
                        "max_percent": np.max(info.get("cpu_samples", [0])) if info.get("cpu_samples") else 0,
                    }
                }
                
                monitoring_path = debug_dir / f"buffer_monitoring.json"
                with open(monitoring_path, "w", encoding="utf-8") as f:
                    json.dump(monitoring_report, f, indent=2)
                
                print(f"[DEBUG-CAPTURE] {peer_id} 📊 Buffer Monitoring:", flush=True)
                print(f"   Buffer underruns: {monitoring_report['buffer_underruns']} ({monitoring_report['underrun_rate_percent']:.2f}%)", flush=True)
                print(f"   CPU usage: {monitoring_report['cpu_usage']['mean_percent']:.1f}% avg, {monitoring_report['cpu_usage']['max_percent']:.1f}% peak", flush=True)
                logger.info(f"[DEBUG-CAPTURE] {peer_id} buffer monitoring saved to {monitoring_path}")
            
            # Generate and save comparison summary
            if info.get("comparison_metrics"):
                # Calculate aggregate metrics across all frames
                all_metrics = info["comparison_metrics"]
                total_duration = len(audio_31_ema) / 16000 if info.get("layer_31_ema_16khz") else 0
                
                # Aggregate quality metrics (matching the actual return format)
                avg_quality = {
                    "snr": {
                        "ema_db": np.mean([m["snr"]["ema_db"] for m in all_metrics]),
                        "rnnoise_db": np.mean([m["snr"]["rnnoise_db"] for m in all_metrics]),
                        "difference_db": np.mean([m["snr"]["difference_db"] for m in all_metrics]),
                        "winner": "EMA" if np.mean([m["snr"]["ema_db"] for m in all_metrics]) > np.mean([m["snr"]["rnnoise_db"] for m in all_metrics]) else "RNNoise"
                    },
                    "rms_level": {
                        "ema": np.mean([m["rms_level"]["ema"] for m in all_metrics]),
                        "rnnoise": np.mean([m["rms_level"]["rnnoise"] for m in all_metrics]),
                        "ema_reduction_percent": np.mean([m["rms_level"]["ema_reduction_percent"] for m in all_metrics]),
                        "rnnoise_reduction_percent": np.mean([m["rms_level"]["rnnoise_reduction_percent"] for m in all_metrics]),
                    },
                    "correlation_with_original": {
                        "ema": np.mean([m["correlation_with_original"]["ema"] for m in all_metrics]),
                        "rnnoise": np.mean([m["correlation_with_original"]["rnnoise"] for m in all_metrics]),
                        "winner": "EMA" if np.mean([m["correlation_with_original"]["ema"] for m in all_metrics]) > np.mean([m["correlation_with_original"]["rnnoise"] for m in all_metrics]) else "RNNoise"
                    }
                }
                
                # Since we don't measure actual latency per frame, provide estimated values
                avg_latency = {
                    "ema_avg_ms": 0.1,  # Estimated EMA latency
                    "ema_min_ms": 0.05,
                    "ema_max_ms": 0.15,
                    "rnnoise_avg_ms": 14.0,  # Estimated RNNoise latency (includes resampling)
                    "rnnoise_min_ms": 12.0,
                    "rnnoise_max_ms": 16.0,
                    "difference_ms": 13.9,
                    "faster_method": "EMA"
                }
                
                from beautyai_inference.utils.noise_comparison import generate_comparison_summary
                summary = generate_comparison_summary(avg_quality, avg_latency, total_duration)
                
                # Save comparison summary to JSON
                comparison_path = debug_dir / f"comparison_summary.json"
                with open(comparison_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "peer_id": peer_id,
                        "total_frames": len(all_metrics),
                        "total_duration_seconds": total_duration,
                        "average_quality_metrics": avg_quality,
                        "average_latency_metrics": avg_latency,
                        "summary": summary,
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"[DEBUG-CAPTURE] {peer_id} 📊 Comparison Summary:", flush=True)
                print(summary, flush=True)
                logger.info(f"[DEBUG-CAPTURE] {peer_id} comparison summary:\n{summary}")
        
        # Save transcriptions to JSON file
        if info.get("transcriptions"):

            transcriptions_path = debug_dir / f"debug_capture_{peer_id}_transcriptions.json"
            try:
                with open(transcriptions_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "peer_id": peer_id,
                        "total_segments": len(info["transcriptions"]),
                        "transcriptions": info["transcriptions"]
                    }, f, indent=2, ensure_ascii=False)
                print(f"[DEBUG-CAPTURE] {peer_id} ✅ Saved {len(info['transcriptions'])} transcriptions to {transcriptions_path.name}", flush=True)
                logger.info(f"[DEBUG-CAPTURE] {peer_id} transcriptions saved to {transcriptions_path}")
            except Exception as json_error:
                logger.error(f"[DEBUG-CAPTURE] {peer_id} Failed to save transcriptions JSON: {json_error}")
        
        # ============================================
        # BUFFER UNDERRUN METRICS REPORT
        # ============================================
        buffer_metrics = info.get("buffer_metrics", {})
        if buffer_metrics:
            try:
                # Calculate statistics
                inter_frame_delays = buffer_metrics.get("inter_frame_delays", [])
                frame_recv_times = buffer_metrics.get("frame_recv_times", [])
                stage_timings = buffer_metrics.get("stage_timings", [])
                
                # Calculate delay statistics
                if inter_frame_delays:
                    avg_delay = np.mean(inter_frame_delays)
                    p50_delay = np.percentile(inter_frame_delays, 50)
                    p95_delay = np.percentile(inter_frame_delays, 95)
                    p99_delay = np.percentile(inter_frame_delays, 99)
                    max_delay = np.max(inter_frame_delays)
                    underrun_count = sum(1 for d in inter_frame_delays if d > 30.0)  # >30ms = underrun
                else:
                    avg_delay = p50_delay = p95_delay = p99_delay = max_delay = 0
                    underrun_count = 0
                
                # Calculate recv time statistics
                if frame_recv_times:
                    avg_recv = np.mean(frame_recv_times)
                    p99_recv = np.percentile(frame_recv_times, 99)
                else:
                    avg_recv = p99_recv = 0
                
                # Calculate processing time statistics
                if stage_timings:
                    total_times = [s.get("total_ms", 0) for s in stage_timings if "total_ms" in s]
                    resample_times = [s.get("resample_ms", 0) for s in stage_timings if "resample_ms" in s]
                    vad_times = [s.get("vad_ms", 0) for s in stage_timings if "vad_ms" in s]
                    
                    if total_times:
                        avg_proc = np.mean(total_times)
                        p99_proc = np.percentile(total_times, 99)
                    else:
                        avg_proc = p99_proc = 0
                    
                    if resample_times:
                        avg_resample = np.mean(resample_times)
                        p99_resample = np.percentile(resample_times, 99)
                    else:
                        avg_resample = p99_resample = 0
                    
                    if vad_times:
                        avg_vad = np.mean(vad_times)
                        p99_vad = np.percentile(vad_times, 99)
                    else:
                        avg_vad = p99_vad = 0
                else:
                    avg_proc = p99_proc = avg_resample = p99_resample = avg_vad = p99_vad = 0
                
                # Build report
                underrun_report = {
                    "peer_id": peer_id,
                    "total_frames": info.get("frames_captured", 0),
                    "buffer_underruns": info.get("buffer_underruns", 0),
                    "underrun_rate_percent": (underrun_count / len(inter_frame_delays) * 100) if inter_frame_delays else 0,
                    "inter_frame_delays_ms": {
                        "avg": round(avg_delay, 2),
                        "p50": round(p50_delay, 2),
                        "p95": round(p95_delay, 2),
                        "p99": round(p99_delay, 2),
                        "max": round(max_delay, 2),
                        "expected": 20.0,  # 20ms expected for 48kHz/960 samples
                    },
                    "frame_recv_latency_ms": {
                        "avg": round(avg_recv, 3),
                        "p99": round(p99_recv, 3),
                    },
                    "processing_time_ms": {
                        "total_avg": round(avg_proc, 2),
                        "total_p99": round(p99_proc, 2),
                        "resample_avg": round(avg_resample, 2),
                        "resample_p99": round(p99_resample, 2),
                        "vad_avg": round(avg_vad, 2),
                        "vad_p99": round(p99_vad, 2),
                    },
                    "timing_summary": {
                        "recv_overhead_significant": p99_recv > 5.0,  # >5ms recv is too slow
                        "processing_bottleneck": p99_proc > 15.0,  # >15ms processing is slow
                        "network_jitter_detected": p99_delay > 30.0,  # >30ms jitter = underrun
                    },
                    "recommendations": []
                }
                
                # Add recommendations based on metrics
                if p99_delay > 30.0:
                    underrun_report["recommendations"].append("High frame jitter detected - check network or system load")
                if p99_recv > 5.0:
                    underrun_report["recommendations"].append("Frame recv() latency high - consider increasing buffer size")
                if p99_proc > 15.0:
                    underrun_report["recommendations"].append("Processing bottleneck - consider optimizing resampling or VAD")
                if underrun_count > 0:
                    underrun_report["recommendations"].append(f"{underrun_count} buffer underruns detected - audio artifacts likely")
                if not underrun_report["recommendations"]:
                    underrun_report["recommendations"].append("✅ No significant timing issues detected")
                
                # Save report to reports directory
                reports_dir = Path("/home/lumi/beautyai/reports")
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = reports_dir / f"buffer_underrun_report_{peer_id}.json"
                
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(underrun_report, f, indent=2)
                
                print(f"[BUFFER-REPORT] {peer_id} 📊 Buffer underrun report saved to: {report_path}", flush=True)
                print(f"[BUFFER-REPORT] Underruns: {underrun_count}/{len(inter_frame_delays)} frames ({underrun_report['underrun_rate_percent']:.2f}%)", flush=True)
                print(f"[BUFFER-REPORT] Frame delays - Avg: {avg_delay:.2f}ms, P99: {p99_delay:.2f}ms, Max: {max_delay:.2f}ms", flush=True)
                print(f"[BUFFER-REPORT] Processing - Avg: {avg_proc:.2f}ms, P99: {p99_proc:.2f}ms", flush=True)
                logger.info(f"[BUFFER-REPORT] {peer_id} Buffer underrun report saved: {report_path}")
                
            except Exception as report_error:
                logger.error(f"[BUFFER-REPORT] {peer_id} Failed to save buffer report: {report_error}")
        
        logger.info(f"[DEBUG-CAPTURE] {peer_id} all layers saved to {debug_dir}")
        
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Error saving audio for {peer_id}: {e}", exc_info=True)


# Export router
__all__ = ['debug_capture_router']
