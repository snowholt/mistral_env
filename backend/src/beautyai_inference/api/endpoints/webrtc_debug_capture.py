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
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder
import numpy as np

from ...core.persistent_model_manager import get_persistent_model_manager
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[4]


def _resolve_debug_directory() -> Path:
    """Return a writable directory for storing debug captures."""
    candidates: list[Path] = []

    env_dir = os.getenv("VOICE_DEBUG_CAPTURE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    # Prefer backend-local path to satisfy systemd ProtectSystem restrictions.
    candidates.append(BACKEND_ROOT / "logs/webrtc/debug_captures")
    # Fallback to repository-level logs directory for manual runs.
    candidates.append(BACKEND_ROOT.parent / "logs/webrtc/debug_captures")

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
        peer_id = f"debug_{uuid.uuid4().hex[:8]}"
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
            "layer_48khz_float": [],
            "layer_16khz": [],
            "layer_16khz_vad_filtered": [],  # NEW: VAD-filtered audio only (speech segments)
            "transcriptions": [],  # Store transcription results
            "whisper_model": None,  # Will be loaded on first use
            "vad_service": None,  # VAD for speech detection
            "is_speech_active": False,  # Current speech state
            "speech_buffer": [],  # Buffer for accumulating speech audio
            "whisper_enabled": False,  # DISABLED for VAD testing
        }
        
        # Try to get preloaded Whisper model (DISABLED for VAD testing)
        capture_info["whisper_enabled"] = False  # Set to True to enable Whisper
        if False:  # Disabled - change to True to enable Whisper
            try:
                model_manager = get_persistent_model_manager()
                whisper_engine = model_manager.get_whisper_model()
                if whisper_engine:
                    capture_info["whisper_model"] = whisper_engine
                    capture_info["whisper_enabled"] = True
                    print(f"[DEBUG-CAPTURE] {peer_id} loaded preloaded Whisper model", flush=True)
                    logger.info(f"[DEBUG-CAPTURE] {peer_id} using preloaded Whisper Turbo model")
                else:
                    print(f"[DEBUG-CAPTURE] {peer_id} Whisper model not preloaded, transcription disabled", flush=True)
                    logger.warning(f"[DEBUG-CAPTURE] {peer_id} Whisper model not available")
            except Exception as e:
                logger.error(f"[DEBUG-CAPTURE] {peer_id} Failed to load Whisper model: {e}")
                capture_info["whisper_model"] = None
        else:
            print(f"[DEBUG-CAPTURE] {peer_id} Whisper DISABLED for VAD testing", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} Whisper disabled - VAD testing mode")
        
        # Initialize VAD service
        try:
            # Create config and set properties (WebRTCVADConfig is not a dataclass)
            vad_config = WebRTCVADConfig()
            vad_config.silero_sensitivity = 0.3  # Lower = more sensitive (default 0.5)
            vad_config.webrtc_sensitivity = 2    # 0-3, where 3 = least sensitive
            vad_config.post_speech_silence_ms = 1000
            vad_config.min_speech_duration_ms = 30
            vad_config.language_thresholds = {"ar": 0.0001, "en": 0.0001, "default": 0.0001}  # ULTRA-LOW for testing
            vad_config.warmup_filter_duration_ms = 0  # DISABLE warmup for testing
            vad_config.min_sustained_speech_frames = 1  # Only need 1 frame (disable sustained check for testing)
            vad_config.log_vad_decisions = True  # Enable VAD decision logging
            
            vad_service = WebRTCVADService(peer_id, language="en", config=vad_config)
            
            # CRITICAL: Initialize the VAD (loads Silero model)
            vad_initialized = await vad_service.initialize()
            if not vad_initialized:
                raise RuntimeError("VAD initialization failed - Silero model not loaded")
            
            capture_info["vad_service"] = vad_service
            print(f"[DEBUG-CAPTURE] {peer_id} ✅ VAD initialized successfully (Silero loaded, silero_sens=0.3, log_decisions=True)", flush=True)
            logger.info(f"[DEBUG-CAPTURE] {peer_id} VAD service ready with Silero model loaded")
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
            "message": f"Audio saved to: {info['debug_dir']}/debug_capture_{peer_id}_*.wav"
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
    from scipy.signal import resample_poly
    from math import gcd
    
    print(f"[DEBUG-CAPTURE] {peer_id} starting frame capture loop", flush=True)
    logger.info(f"[DEBUG-CAPTURE] {peer_id} starting frame capture loop")
    
    # Accumulators for different layers
    layer_48khz_raw = info.setdefault("layer_48khz_raw", [])
    layer_48khz_float = info.setdefault("layer_48khz_float", [])
    layer_16khz = info.setdefault("layer_16khz", [])
    
    frame_count = 0
    timeout_count = 0
    max_consecutive_timeouts = 30  # 30 timeouts * 1s = 30 seconds total wait
    
    try:
        while True:
            try:
                print(f"[DEBUG-CAPTURE] {peer_id} waiting for frame (timeout #{timeout_count + 1})...", flush=True)
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                frame_count += 1
                timeout_count = 0  # Reset on successful receive
                info["frames_captured"] = frame_count
                
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
                
                # LAYER 2: Convert to float32
                if np.issubdtype(audio_array.dtype, np.integer):
                    dtype_info = np.iinfo(audio_array.dtype)
                    scale = float(max(abs(dtype_info.min), dtype_info.max)) or 1.0
                    audio_float = audio_array.astype(np.float32) / scale
                else:
                    audio_float = audio_array.astype(np.float32)
                
                layer_48khz_float.append(audio_float.copy())
                
                # LAYER 3: Resample to 16kHz with improved quality
                if sample_rate != 16000:
                    # Two-stage resampling for better quality when downsampling from 48kHz
                    # Stage 1: 48kHz → 24kHz (2:1)
                    # Stage 2: 24kHz → 16kHz (3:2)
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
                else:
                    audio_16k = audio_float
                
                layer_16khz.append(audio_16k.copy())
                
                # VAD: Check for voice activity
                vad_service = info.get("vad_service")
                speech_buffer = info.get("speech_buffer", [])
                layer_16khz_vad_filtered = info.get("layer_16khz_vad_filtered", [])
                
                if vad_service:
                    try:
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
                        
                        # Get VAD state and probability
                        voice_state = vad_result.get("voice_state")
                        silero_prob = vad_result.get("silero_probability", 0)
                        is_speech = voice_state in [VADState.VOICE_START, VADState.VOICE_ACTIVE]
                        
                        # Log VAD decision every 25 frames (every 0.5 seconds) for visibility
                        if frame_count % 25 == 0:
                            state_emoji = "🟢" if is_speech else "⚪"
                            print(f"[VAD] {peer_id} Frame {frame_count}: {state_emoji} {voice_state} (prob={silero_prob:.3f})", flush=True)
                        
                        # Update speech state
                        was_speech = info.get("is_speech_active", False)
                        info["is_speech_active"] = is_speech
                        
                        # Speech started
                        if is_speech and not was_speech:
                            segments_count = info.get("speech_segments_count", 0) + 1
                            info["speech_segments_count"] = segments_count
                            print(f"[DEBUG-CAPTURE] {peer_id} 🎤 Speech started (frame {frame_count}, segment #{segments_count})", flush=True)
                            logger.info(f"[DEBUG-CAPTURE] {peer_id} Speech detected at frame {frame_count}")
                            speech_buffer.clear()
                        
                        # Accumulate speech audio (for both VAD layer and potential transcription)
                        if is_speech:
                            speech_buffer.append(audio_16k.copy())
                        
                        # Speech ended - save to VAD-filtered layer
                        if was_speech and not is_speech and len(speech_buffer) > 0:
                            speech_duration = len(speech_buffer) * len(audio_16k) / 16000
                            print(f"[DEBUG-CAPTURE] {peer_id} 🛑 Speech ended, buffer: {len(speech_buffer)} frames (~{speech_duration:.2f}s)", flush=True)
                            logger.info(f"[DEBUG-CAPTURE] {peer_id} Speech segment: {speech_duration:.2f}s")
                            
                            # Save speech segment to VAD-filtered layer
                            for speech_frame in speech_buffer:
                                layer_16khz_vad_filtered.append(speech_frame.copy())
                            
                            # Only transcribe if Whisper is enabled
                            if info.get("whisper_enabled", False):
                                whisper_model = info.get("whisper_model")
                                if whisper_model:
                                    try:
                                        # Concatenate all speech frames
                                        speech_audio = np.concatenate(speech_buffer)
                                        
                                        # Convert to int16 PCM for Whisper
                                        audio_int16 = (speech_audio * 32767).astype(np.int16)
                                        audio_bytes = audio_int16.tobytes()
                                        
                                        # Transcribe
                                        print(f"[DEBUG-CAPTURE] {peer_id} transcribing {len(audio_int16)} samples (~{len(audio_int16)/16000:.1f}s)...", flush=True)
                                        transcription = whisper_model.transcribe_audio_bytes(
                                            audio_bytes,
                                            audio_format="pcm_raw",
                                            language="en"
                                        )
                                        
                                        if transcription:
                                            info["transcriptions"].append({
                                                "frame_index": frame_count,
                                                "timestamp": time.time() - info["start_time"],
                                                "text": transcription,
                                                "duration_s": len(audio_int16) / 16000
                                            })
                                            print(f"[DEBUG-CAPTURE] {peer_id} ✅ transcription: '{transcription[:100]}...'", flush=True)
                                            logger.info(f"[DEBUG-CAPTURE] {peer_id} transcribed: {transcription[:50]}...")
                                        else:
                                            print(f"[DEBUG-CAPTURE] {peer_id} ⚠️ Empty transcription", flush=True)
                                            
                                    except Exception as transcribe_error:
                                        logger.error(f"[DEBUG-CAPTURE] {peer_id} transcription error: {transcribe_error}")
                            else:
                                print(f"[DEBUG-CAPTURE] {peer_id} ℹ️ Whisper disabled - speech segment saved to VAD layer only", flush=True)
                            
                            # Clear buffer after processing
                            speech_buffer.clear()
                        
                        info["speech_buffer"] = speech_buffer
                        info["layer_16khz_vad_filtered"] = layer_16khz_vad_filtered
                        
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
        
        print(
            f"[DEBUG-CAPTURE] {peer_id} capture complete: {frame_count} frames, "
            f"detected_sample_rate={info.get('actual_sample_rate', 'unknown')}Hz, "
            f"layers: raw={len(layer_48khz_raw)}, float={len(layer_48khz_float)}, 16kHz={len(layer_16khz)}, VAD-filtered={vad_filtered_count}",
            flush=True
        )
        logger.info(
            f"[DEBUG-CAPTURE] {peer_id} capture complete: {frame_count} frames, "
            f"detected_sample_rate={info.get('actual_sample_rate', 'unknown')}Hz, "
            f"layers: raw={len(layer_48khz_raw)}, float={len(layer_48khz_float)}, 16kHz={len(layer_16khz)}, VAD-filtered={vad_filtered_count}"
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
            
            path_48k_raw = debug_dir / f"debug_capture_{peer_id}_layer1_{actual_sample_rate}hz_raw.wav"
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
        
        # Save Layer 2: 48kHz float (normalized)
        if info.get("layer_48khz_float"):
            audio_48k_float = np.concatenate(info["layer_48khz_float"])
            print(
                f"[DEBUG-CAPTURE] {peer_id} layer2 samples: {len(audio_48k_float)}",
                flush=True,
            )
            audio_48k_int16 = (np.clip(audio_48k_float, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_48k_float = debug_dir / f"debug_capture_{peer_id}_layer2_{actual_sample_rate}hz_float.wav"
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
            
            path_16k = debug_dir / f"debug_capture_{peer_id}_layer3_16khz.wav"
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
            
            path_16k_vad = debug_dir / f"debug_capture_{peer_id}_layer4_16khz_vad_filtered.wav"
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
        else:
            print(f"[DEBUG-CAPTURE] {peer_id} No speech detected by VAD - Layer 4 empty", flush=True)
            logger.warning(f"[DEBUG-CAPTURE] {peer_id} No speech detected - VAD layer empty")
        
        logger.info(f"[DEBUG-CAPTURE] {peer_id} all layers saved to {debug_dir}")
        
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Error saving audio for {peer_id}: {e}", exc_info=True)


# Export router
__all__ = ['debug_capture_router']
