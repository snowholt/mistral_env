"""
WebRTC Lean Audio Capture Endpoint (Hardened for Production)

Optimized audio capture with bounded queues, minimal recv loop, and lean processing pipeline.
Designed to eliminate buffer underruns (<1% target) without sacrificing latency.

Architecture:
- Hot-path recv loop: <2ms (raw capture + LPF + resample only)
- Bounded queue with drop-oldest policy (never block recv)
- Fixed worker pool for CPU-bound processing
- Frame reordering for deterministic output
- Lean pipeline: Limiter → Resample → Single Denoiser → Adaptive Comb → Gate

Author: BeautyAI Framework
Date: November 13, 2025
"""

import asyncio
import json
import logging
import os
import time
import wave
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt
from math import gcd

from ...utils.frame_queue import BoundedFrameQueue, FramePacket
from ...utils.lean_pipeline import LeanPipeline
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[4]

# Configuration presets
PIPELINE_PRESETS = {
    "lean_rnnoise": {
        "denoiser_type": "rnnoise",
        "enable_limiter": True,
        "enable_adaptive_comb": True,
        "enable_gate": True,
    },
    "lean_dtln": {
        "denoiser_type": "dtln",
        "enable_limiter": True,
        "enable_adaptive_comb": True,
        "enable_gate": True,
    },
    "minimal": {
        "denoiser_type": "none",
        "enable_limiter": True,
        "enable_adaptive_comb": False,
        "enable_gate": True,
    },
}

# Environment configuration
DEBUG_VERBOSE = os.getenv("WEBRTC_DEBUG_VERBOSE", "0") == "1"
PIPELINE_PRESET = os.getenv("WEBRTC_PIPELINE_PRESET", "lean_rnnoise")


def _resolve_debug_directory() -> Path:
    """Return a writable directory for storing debug captures."""
    candidates: list[Path] = []

    env_dir = os.getenv("VOICE_DEBUG_CAPTURE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.append(BACKEND_ROOT.parent / "reports/debug/webrtc")
    candidates.append(BACKEND_ROOT / "logs/webrtc/debug_captures")

    if DEBUG_VERBOSE:
        print(f"[LEAN-CAPTURE] 🔍 BACKEND_ROOT={BACKEND_ROOT}", flush=True)
        print(f"[LEAN-CAPTURE] 🔍 Candidate paths: {candidates}", flush=True)

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            with probe.open("wb"):
                pass
            probe.unlink(missing_ok=True)
            logger.info(f"[LEAN-CAPTURE] Using debug directory: {candidate}")
            return candidate
        except OSError as exc:
            logger.warning(f"[LEAN-CAPTURE] Unable to use debug directory {candidate}: {exc}")

    logger.error("[LEAN-CAPTURE] No writable debug directory available")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="No writable debug capture directory available",
    )


lean_capture_router = APIRouter(
    prefix="/api/v1/webrtc/lean/voice-capture",
    tags=["webrtc-lean"],
)

_lean_connections: Dict[str, Dict[str, Any]] = {}


class LeanOfferRequest(BaseModel):
    sdp: str = Field(..., min_length=10)
    type: str = Field(default="offer")
    preset: Optional[str] = Field(default=None, description="Pipeline preset: lean_rnnoise, lean_dtln, minimal")


class LeanOfferResponse(BaseModel):
    sdp: str
    type: str = "answer"
    peer_id: str
    preset: str
    message: str = "Lean capture session created"


async def _minimal_recv_loop(track, info: Dict[str, Any], queue: BoundedFrameQueue, pipeline: LeanPipeline):
    """
    ULTRA-MINIMAL recv loop: ONLY frame reception and stereo→mono.
    
    Hot path - target <0.5ms p99:
    1. await track.recv()
    2. Stereo → Mono conversion
    3. queue.enqueue() (raw 48kHz audio)
    
    ALL heavy processing (LPF, resample, denoise) moved to worker thread!
    NO disk I/O, NO filters, NO resampling, NO per-frame prints!
    """
    peer_id = info["peer_id"]
    frame_count = 0
    timeout_count = 0
    max_consecutive_timeouts = 30
    
    try:
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                frame_count += 1
                timeout_count = 0
                info["frames_captured"] = frame_count
                
                recv_start = time.monotonic()
                
                # ===== STAGE 1: Raw audio extraction (ONLY!) =====
                audio_array = frame.to_ndarray()
                sample_rate = frame.sample_rate
                
                # Log first frame only
                if frame_count == 1:
                    print(
                        f"[LEAN-CAPTURE] {peer_id} FIRST FRAME: sr={sample_rate}Hz, "
                        f"samples={frame.samples}, dtype={audio_array.dtype}",
                        flush=True
                    )
                    logger.info(f"[LEAN-CAPTURE] {peer_id} started: {sample_rate}Hz")
                
                # Flatten and convert stereo to mono (FAST operation)
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                if len(audio_array) == frame.samples * 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # Store raw 48kHz int16 (no processing!)
                audio_48k_int16 = audio_array.copy()
                
                # ===== STAGE 2: Enqueue packet (non-blocking) =====
                # Worker will do ALL processing: LPF, resample, denoise, comb, gate
                packet = FramePacket(
                    frame_index=frame_count,
                    timestamp_mono=time.monotonic(),
                    sample_rate=sample_rate,
                    audio_48k_int16=audio_48k_int16,
                    audio_16k_float32=None,  # Will be computed by worker
                    samples_48k=len(audio_48k_int16),
                    samples_16k=0,  # Unknown until worker resamples
                )
                
                queue.enqueue(packet)
                
                recv_end = time.monotonic()
                recv_duration_ms = (recv_end - recv_start) * 1000
                
                # Throttled logging (every 100 frames)
                if DEBUG_VERBOSE and frame_count % 100 == 0:
                    print(f"[LEAN-CAPTURE] {peer_id} recv_p99={recv_duration_ms:.2f}ms, queue_depth={queue.stats.current_depth}")
                
            except asyncio.TimeoutError:
                timeout_count += 1
                if DEBUG_VERBOSE:
                    print(f"[LEAN-CAPTURE] {peer_id} timeout #{timeout_count}", flush=True)
                
                if timeout_count >= max_consecutive_timeouts:
                    print(f"[LEAN-CAPTURE] {peer_id} max timeouts reached, stopping", flush=True)
                    break
    
    except Exception as e:
        logger.error(f"[LEAN-CAPTURE] {peer_id} recv loop error: {e}", exc_info=True)
    
    finally:
        print(f"[LEAN-CAPTURE] {peer_id} recv loop stopped: {frame_count} frames", flush=True)
        info["recv_loop_complete"] = True


async def _async_worker(queue: BoundedFrameQueue, pipeline: LeanPipeline, info: Dict[str, Any]):
    """
    Async worker: drain queue and process frames via pipeline.
    
    Runs CPU-bound processing in executor (non-blocking).
    Commits results to reordering dict for frame-order preservation.
    """
    peer_id = info["peer_id"]
    processed_count = 0
    
    try:
        while True:
            # Check if recv loop finished and queue empty
            if info.get("recv_loop_complete") and queue.stats.current_depth == 0:
                print(f"[LEAN-CAPTURE] {peer_id} worker: recv complete, queue empty, exiting", flush=True)
                break
            
            # Dequeue packet (wait up to 100ms)
            packet = queue.dequeue(timeout=0.1)
            
            if packet is None:
                # Queue empty, brief wait
                await asyncio.sleep(0.01)
                continue
            
            # Process via lean pipeline (offloaded to executor)
            # Worker now does: LPF + resample + limiter + denoise + comb + gate
            result = await pipeline.process_frame_async(
                packet.audio_48k_int16,
                None,  # Will be computed by pipeline (moved from recv loop!)
            )
            
            # Store the resampled audio in the packet for disk writer
            if result and "audio_16k_float32" in result:
                packet.audio_16k_float32 = result["audio_16k_float32"]
                packet.samples_16k = len(packet.audio_16k_float32)
            
            packet.result = result
            
            # Commit result for reordering
            queue.commit_result(packet)
            processed_count += 1
            
            # Throttled logging
            if DEBUG_VERBOSE and processed_count % 100 == 0:
                stats = queue.get_stats_snapshot()
                print(
                    f"[LEAN-CAPTURE] {peer_id} worker: processed={processed_count}, "
                    f"pending={stats['pending_results']}, service_p90={stats['worker_service_p90_ms']:.1f}ms"
                )
    
    except Exception as e:
        logger.error(f"[LEAN-CAPTURE] {peer_id} worker error: {e}", exc_info=True)
    
    finally:
        print(f"[LEAN-CAPTURE] {peer_id} worker stopped: processed {processed_count} frames", flush=True)


async def _batch_disk_writer(queue: BoundedFrameQueue, info: Dict[str, Any], flush_interval_ms: float = 500.0):
    """
    Batch disk writer: flush contiguous results every N ms.
    
    Maintains frame order via commit cursor.
    Accumulates layers in memory, flushes periodically.
    """
    peer_id = info["peer_id"]
    last_flush_time = time.monotonic()
    total_flushed = 0
    
    try:
        while True:
            now = time.monotonic()
            elapsed_ms = (now - last_flush_time) * 1000
            
            # Check flush condition
            if elapsed_ms >= flush_interval_ms or info.get("recv_loop_complete"):
                # Get contiguous results
                ready = queue.get_contiguous_results()
                
                if ready:
                    # Append to layer buffers
                    for packet in ready:
                        result = packet.result
                        
                        # Layer 1: Raw 48kHz
                        info["layer_48khz_raw"].append(packet.audio_48k_int16)
                        
                        # Layer 3: Baseline 16kHz
                        info["layer_16khz"].append(packet.audio_16k_float32)
                        
                        # Lean pipeline layers
                        if result:
                            info["layer_15_limited_48k"].append(result.get("layer_15_limited_48k"))
                            info["layer_32_denoised_16k"].append(result.get("layer_32_denoised_16k"))
                            info["layer_36_comb_16k"].append(result.get("layer_36_comb_16k"))
                            info["layer_31b_gated_16k"].append(result.get("layer_31b_gated_16k"))
                            
                            # Timing stats
                            timing = result.get("timing", {})
                            info["timing_stats"].append(timing)
                    
                    total_flushed += len(ready)
                    last_flush_time = now
                    
                    if DEBUG_VERBOSE:
                        print(f"[LEAN-CAPTURE] {peer_id} flushed {len(ready)} frames (total={total_flushed})")
                
                # If recv complete and no more ready frames, exit
                if info.get("recv_loop_complete") and not ready:
                    break
            
            await asyncio.sleep(0.1)  # Check every 100ms
    
    except Exception as e:
        logger.error(f"[LEAN-CAPTURE] {peer_id} disk writer error: {e}", exc_info=True)
    
    finally:
        print(f"[LEAN-CAPTURE] {peer_id} disk writer stopped: flushed {total_flushed} frames", flush=True)


async def _save_capture_results(info: Dict[str, Any]):
    """Save all captured layers to WAV files and generate reports."""
    peer_id = info["peer_id"]
    debug_dir = info["debug_dir"]
    
    print(f"[LEAN-CAPTURE] {peer_id} Saving capture results...", flush=True)
    
    # Helper: save audio layer
    def save_layer(audio_list, filename, sample_rate, dtype="float32"):
        if not audio_list or not any(x is not None for x in audio_list):
            return
        
        # Filter None values
        audio_list = [x for x in audio_list if x is not None]
        audio = np.concatenate(audio_list)
        
        # Convert to int16
        if dtype == "float32":
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        path = debug_dir / filename
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        duration = len(audio) / sample_rate
        print(f"[LEAN-CAPTURE] {peer_id} ✅ {filename}: {duration:.2f}s @ {sample_rate}Hz", flush=True)
    
    # Save layers
    save_layer(info["layer_48khz_raw"], "layer1_raw_48khz.wav", 48000, dtype="int16")
    save_layer(info["layer_15_limited_48k"], "layer15_limited_48khz.wav", 48000, dtype="int16")
    save_layer(info["layer_16khz"], "layer3_baseline_16khz.wav", 16000, dtype="float32")
    save_layer(info["layer_32_denoised_16k"], "layer32_denoised_16khz.wav", 16000, dtype="float32")
    save_layer(info["layer_36_comb_16k"], "layer36_comb_16khz.wav", 16000, dtype="float32")
    save_layer(info["layer_31b_gated_16k"], "layer31b_gated_16khz.wav", 16000, dtype="float32")
    
    # Save queue statistics
    queue_stats = info["queue"].get_stats_snapshot()
    stats_path = debug_dir / "queue_stats.json"
    with open(stats_path, "w") as f:
        json.dump(queue_stats, f, indent=2)
    
    print(f"[LEAN-CAPTURE] {peer_id} 📊 Queue Stats:")
    print(f"   Underruns: {queue_stats['underruns']} ({queue_stats['underruns'] / queue_stats['enqueued'] * 100:.2f}%)")
    print(f"   Recv delta p99: {queue_stats['recv_delta_p99_ms']:.2f}ms")
    print(f"   Worker service p99: {queue_stats['worker_service_p99_ms']:.2f}ms")
    print(f"   Queue peak depth: {queue_stats['peak_depth']}")
    
    # Save pipeline statistics
    pipeline_stats = info["pipeline"].get_stats()
    pipeline_stats_path = debug_dir / "pipeline_stats.json"
    with open(pipeline_stats_path, "w") as f:
        json.dump(pipeline_stats, f, indent=2)
    
    print(f"[LEAN-CAPTURE] {peer_id} 📊 Pipeline Stats:")
    print(f"   Limiter activations: {pipeline_stats['limiter_activations']}")
    print(f"   Comb active frames: {pipeline_stats['comb_active_frames']}")
    print(f"   Gate closed frames: {pipeline_stats['gate_closed_frames']}")
    
    logger.info(f"[LEAN-CAPTURE] {peer_id} capture complete: {info['frames_captured']} frames")


@lean_capture_router.post("/offer", response_model=LeanOfferResponse)
async def handle_lean_offer(request: LeanOfferRequest):
    """Create lean capture session with hardened architecture."""
    try:
        peer_id = "session"
        preset = request.preset or PIPELINE_PRESET
        
        if preset not in PIPELINE_PRESETS:
            raise HTTPException(status_code=400, detail=f"Invalid preset: {preset}")
        
        preset_config = PIPELINE_PRESETS[preset]
        
        print(f"[LEAN-CAPTURE] Creating session with preset: {preset}", flush=True)
        logger.info(f"[LEAN-CAPTURE] Creating session: preset={preset}")
        
        # Create RTCPeerConnection
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
        
        debug_dir = _resolve_debug_directory()
        
        # Initialize bounded queue (5 frames = 100ms buffer)
        queue = BoundedFrameQueue(max_size=5)
        
        # Initialize lean pipeline
        pipeline = LeanPipeline(
            sample_rate_in=48000,
            sample_rate_out=16000,
            **preset_config
        )
        
        capture_info = {
            "pc": pc,
            "peer_id": peer_id,
            "preset": preset,
            "debug_dir": debug_dir,
            "frames_captured": 0,
            "start_time": time.time(),
            "queue": queue,
            "pipeline": pipeline,
            "recv_loop_complete": False,
            # Layer buffers
            "layer_48khz_raw": [],
            "layer_15_limited_48k": [],
            "layer_16khz": [],
            "layer_32_denoised_16k": [],
            "layer_36_comb_16k": [],
            "layer_31b_gated_16k": [],
            "timing_stats": [],
        }
        
        @pc.on("track")
        async def on_track(track):
            print(f"[LEAN-CAPTURE] {peer_id} Track received: {track.kind}", flush=True)
            
            if track.kind == "audio":
                capture_info["audio_track"] = track
                
                # Spawn concurrent tasks
                recv_task = asyncio.create_task(_minimal_recv_loop(track, capture_info, queue, pipeline))
                
                # Spawn multiple parallel worker tasks to handle frame bursts
                # 4 workers can process frames concurrently, reducing drops during bursts
                num_workers = 4
                worker_tasks = [
                    asyncio.create_task(_async_worker(queue, pipeline, capture_info))
                    for _ in range(num_workers)
                ]
                
                writer_task = asyncio.create_task(_batch_disk_writer(queue, capture_info))
                
                # Wait for all tasks (recv, all workers, writer)
                await asyncio.gather(recv_task, *worker_tasks, writer_task)
                
                # Save results
                await _save_capture_results(capture_info)
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(sdp=request.sdp, type=request.type))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        _lean_connections[peer_id] = capture_info
        
        return LeanOfferResponse(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
            peer_id=peer_id,
            preset=preset,
            message=f"Lean capture session created with preset: {preset}",
        )
    
    except Exception as e:
        logger.error(f"[LEAN-CAPTURE] Offer handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@lean_capture_router.get("/stats/{peer_id}")
async def get_lean_stats(peer_id: str):
    """Get real-time statistics for active session."""
    if peer_id not in _lean_connections:
        raise HTTPException(status_code=404, detail="Session not found")
    
    info = _lean_connections[peer_id]
    queue_stats = info["queue"].get_stats_snapshot()
    pipeline_stats = info["pipeline"].get_stats()
    
    return {
        "peer_id": peer_id,
        "frames_captured": info["frames_captured"],
        "preset": info["preset"],
        "queue": queue_stats,
        "pipeline": pipeline_stats,
    }
