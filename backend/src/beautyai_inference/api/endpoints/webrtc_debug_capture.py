"""
WebRTC Debug Voice Capture Endpoint

Simple endpoint for capturing audio at each layer WITHOUT STT/LLM processing.
Saves raw audio captures for debugging sample rate and resampling issues.

Author: BeautyAI Framework  
Date: October 29, 2025
"""

import asyncio
import logging
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder

logger = logging.getLogger(__name__)

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
        logger.info(f"[DEBUG-CAPTURE] Creating session for {peer_id}")
        
        # Create RTCPeerConnection
        pc = RTCPeerConnection()
        
        # Setup paths for saving audio
        try:
            backend_root = Path(__file__).resolve().parents[3]
        except IndexError:
            backend_root = Path.cwd()
        
        debug_dir = backend_root / "logs" / "webrtc" / "debug_captures"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        capture_info = {
            "pc": pc,
            "peer_id": peer_id,
            "debug_dir": debug_dir,
            "frames_captured": 0,
            "start_time": time.time(),
            "audio_track": None,
            "data_channel": None,
            "raw_frames": [],  # Store raw frames
        }
        
        # Track received audio
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            logger.info(f"[DEBUG-CAPTURE] {peer_id} received {track.kind} track")
            
            if track.kind == "audio":
                capture_info["audio_track"] = track
                
                # Start capturing frames
                asyncio.create_task(_capture_audio_frames(peer_id, track, capture_info))
        
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=request.sdp, type=request.type)
        )
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
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
        
        candidate = RTCIceCandidate(
            sdpMid=request.sdp_mid,
            sdpMLineIndex=request.sdp_m_line_index,
            candidate=request.candidate
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


@debug_capture_router.delete("/{peer_id}")
async def cleanup_debug_session(peer_id: str):
    """Stop capture and save audio files."""
    try:
        if peer_id not in _debug_connections:
            return {"status": "ok", "message": "Session already cleaned up"}
        
        info = _debug_connections[peer_id]
        
        # Save captured audio
        await _save_captured_audio(peer_id, info)
        
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
    
    logger.info(f"[DEBUG-CAPTURE] {peer_id} starting frame capture loop")
    
    # Accumulators for different layers
    layer_48khz_raw = []      # Layer 1: Raw 48kHz from WebRTC
    layer_48khz_float = []    # Layer 2: Converted to float32
    layer_16khz = []          # Layer 3: Downsampled to 16kHz
    
    frame_count = 0
    
    try:
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                frame_count += 1
                info["frames_captured"] = frame_count
                
                # LAYER 1: Raw frame from WebRTC (48kHz int16)
                audio_array = frame.to_ndarray()  # Get raw ndarray
                sample_rate = getattr(frame, "sample_rate", 48000)
                
                # Flatten to 1D if needed
                if audio_array.ndim > 1:
                    if audio_array.shape[0] <= 2 and audio_array.shape[1] > audio_array.shape[0]:
                        audio_array = audio_array[0, :]  # (channels, samples) -> select LEFT
                    else:
                        audio_array = audio_array[:, 0]  # (samples, channels) -> select LEFT
                
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
                
                # LAYER 3: Resample to 16kHz
                if sample_rate != 16000:
                    ratio_gcd = gcd(sample_rate, 16000)
                    up = 16000 // ratio_gcd
                    down = sample_rate // ratio_gcd
                    audio_16k = resample_poly(audio_float, up, down)
                else:
                    audio_16k = audio_float
                
                layer_16khz.append(audio_16k.copy())
                
                # Log progress every 50 frames
                if frame_count % 50 == 0:
                    logger.info(
                        f"[DEBUG-CAPTURE] {peer_id}: {frame_count} frames, "
                        f"48kHz={len(audio_array)} samples, 16kHz={len(audio_16k)} samples"
                    )
                
            except asyncio.TimeoutError:
                logger.info(f"[DEBUG-CAPTURE] {peer_id} recv timeout, ending capture")
                break
            except Exception as e:
                logger.error(f"[DEBUG-CAPTURE] {peer_id} frame error: {e}")
                break
                
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] {peer_id} capture loop error: {e}", exc_info=True)
    
    finally:
        # Save accumulated layers
        info["layer_48khz_raw"] = layer_48khz_raw
        info["layer_48khz_float"] = layer_48khz_float
        info["layer_16khz"] = layer_16khz
        
        logger.info(
            f"[DEBUG-CAPTURE] {peer_id} capture complete: {frame_count} frames, "
            f"layers: 48kHz_raw={len(layer_48khz_raw)}, "
            f"48kHz_float={len(layer_48khz_float)}, 16kHz={len(layer_16khz)}"
        )


async def _save_captured_audio(peer_id: str, info: Dict):
    """Save captured audio layers as WAV files."""
    import numpy as np
    
    debug_dir = info["debug_dir"]
    
    try:
        # Save Layer 1: 48kHz raw (as received from WebRTC)
        if info.get("layer_48khz_raw"):
            audio_48k_raw = np.concatenate(info["layer_48khz_raw"])
            
            # Ensure int16
            if not np.issubdtype(audio_48k_raw.dtype, np.integer):
                audio_48k_raw = (np.clip(audio_48k_raw, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_48k_raw = debug_dir / f"debug_capture_{peer_id}_layer1_48khz_raw.wav"
            with wave.open(str(path_48k_raw), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(audio_48k_raw.tobytes())
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 1 (48kHz raw): {len(audio_48k_raw)} samples, "
                f"{len(audio_48k_raw)/48000:.2f}s -> {path_48k_raw}"
            )
        
        # Save Layer 2: 48kHz float (normalized)
        if info.get("layer_48khz_float"):
            audio_48k_float = np.concatenate(info["layer_48khz_float"])
            audio_48k_int16 = (np.clip(audio_48k_float, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_48k_float = debug_dir / f"debug_capture_{peer_id}_layer2_48khz_float.wav"
            with wave.open(str(path_48k_float), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(audio_48k_int16.tobytes())
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 2 (48kHz float): {len(audio_48k_float)} samples, "
                f"{len(audio_48k_float)/48000:.2f}s -> {path_48k_float}"
            )
        
        # Save Layer 3: 16kHz (resampled)
        if info.get("layer_16khz"):
            audio_16k = np.concatenate(info["layer_16khz"])
            audio_16k_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
            
            path_16k = debug_dir / f"debug_capture_{peer_id}_layer3_16khz.wav"
            with wave.open(str(path_16k), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_16k_int16.tobytes())
            
            logger.info(
                f"[DEBUG-CAPTURE] Saved Layer 3 (16kHz resampled): {len(audio_16k)} samples, "
                f"{len(audio_16k)/16000:.2f}s -> {path_16k}"
            )
        
        logger.info(f"[DEBUG-CAPTURE] {peer_id} all layers saved to {debug_dir}")
        
    except Exception as e:
        logger.error(f"[DEBUG-CAPTURE] Error saving audio for {peer_id}: {e}", exc_info=True)


# Export router
__all__ = ['debug_capture_router']
