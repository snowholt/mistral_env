"""
WebRTC Voice Endpoint (Lean Implementation)

Optimized for low CPU usage and high network resilience.
- Jitter Buffer: 128 packets (approx 2.5s) to handle network jitter.
- VAD: Silero VAD for accurate speech detection.
- STT: Faster-Whisper (Turbo) for transcription.
- LLM: Qwen (via Llama.cpp) for response generation.
- Output: Text response via Data Channel (TTS disabled for performance).

Author: BeautyAI Framework
Date: November 2025
"""

import asyncio
import json
import logging
import os
import time
import uuid
import re
import psutil
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder
import numpy as np
from scipy.signal import resample_poly

# ============================================================
# AIORTC JITTER BUFFER TUNING (CRITICAL FIX)
# ============================================================
import aiortc.rtcrtpreceiver
from aiortc.jitterbuffer import JitterBuffer

# Store original __init__ for potential restoration
_original_RTCRtpReceiver_init = aiortc.rtcrtpreceiver.RTCRtpReceiver.__init__

# Environment variable configuration
AIORTC_AUDIO_JITTER_CAPACITY = int(os.getenv("AIORTC_AUDIO_JITTER_CAPACITY", "128"))
AIORTC_AUDIO_JITTER_PREFETCH = int(os.getenv("AIORTC_AUDIO_JITTER_PREFETCH", "32"))

def _patched_RTCRtpReceiver_init(self, kind, transport):
    """Patched RTCRtpReceiver.__init__ with increased audio jitter buffer."""
    _original_RTCRtpReceiver_init(self, kind, transport)
    if kind == "audio":
        self._RTCRtpReceiver__jitter_buffer = JitterBuffer(
            capacity=AIORTC_AUDIO_JITTER_CAPACITY, 
            prefetch=AIORTC_AUDIO_JITTER_PREFETCH
        )
        print(f"[LEAN-VOICE] 🔧 Jitter Buffer: capacity={AIORTC_AUDIO_JITTER_CAPACITY}, prefetch={AIORTC_AUDIO_JITTER_PREFETCH}", flush=True)

# Apply the monkey-patch
aiortc.rtcrtpreceiver.RTCRtpReceiver.__init__ = _patched_RTCRtpReceiver_init

# ============================================================
# IMPORTS
# ============================================================
from ...core.persistent_model_manager import get_persistent_model_manager
from ...services.voice.vad import WebRTCVADService, WebRTCVADConfig, VADState

logger = logging.getLogger(__name__)

webrtc_voice_router = APIRouter(
    prefix="/api/v1/webrtc/voice",
    tags=["webrtc-voice-lean"],
)

# Store active connections
_active_connections: Dict[str, Dict[str, Any]] = {}

class OfferRequest(BaseModel):
    sdp: str = Field(..., min_length=10)
    type: str = Field(default="offer")

class OfferResponse(BaseModel):
    sdp: str
    type: str = "answer"
    session_id: str

class ICERequest(BaseModel):
    session_id: str
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_m_line_index: Optional[int] = None

@webrtc_voice_router.post("/offer", response_model=OfferResponse)
async def handle_offer(request: OfferRequest):
    """Create WebRTC session (Lean Mode)."""
    try:
        session_id = str(uuid.uuid4())
        print(f"[LEAN-VOICE] 🚀 Creating session {session_id}", flush=True)
        
        # RTC Configuration
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(
                    urls=["turn:dev.gmai.sa:15478"],
                    username="beautyai",
                    credential="beautyai2025"
                ),
            ]
        )
        pc = RTCPeerConnection(configuration=config)
        
        # Session Context
        session_context = {
            "pc": pc,
            "session_id": session_id,
            "start_time": time.time(),
            "audio_track": None,
            "data_channel": None,
            "processing_task": None,
            "vad_service": None,
            "whisper_model": None,
            "llm_model": None,
            "speech_buffer": [],
            "transcript_buffer": [],  # Buffer for aggregated transcripts
            "turn_timer_task": None,  # Task for 2s silence timer
            "loop": asyncio.get_event_loop()
        }
        
        # Load Models (Persistent)
        try:
            model_manager = get_persistent_model_manager()
            session_context["whisper_model"] = model_manager.get_whisper_model()
            session_context["llm_model"] = model_manager.get_llm_model()
            
            if session_context["whisper_model"]:
                print(f"[LEAN-VOICE] ✅ Whisper Model Loaded", flush=True)
            else:
                print(f"[LEAN-VOICE] ⚠️ Whisper Model NOT Available", flush=True)
                
            if session_context["llm_model"]:
                print(f"[LEAN-VOICE] ✅ LLM Model Loaded", flush=True)
            else:
                print(f"[LEAN-VOICE] ⚠️ LLM Model NOT Available", flush=True)
                
        except Exception as e:
            logger.error(f"[LEAN-VOICE] Model load error: {e}")
            print(f"[LEAN-VOICE] ❌ Model load error: {e}", flush=True)

        # Initialize VAD
        try:
            vad_config = WebRTCVADConfig()
            vad_config.silero_sensitivity = 0.3
            vad_config.post_speech_silence_ms = 700
            vad_config.min_speech_duration_ms = 50
            
            vad_service = WebRTCVADService(session_id, language="en", config=vad_config)
            if await vad_service.initialize():
                session_context["vad_service"] = vad_service
                print(f"[LEAN-VOICE] ✅ VAD Initialized", flush=True)
            else:
                print(f"[LEAN-VOICE] ❌ VAD Init Failed", flush=True)
        except Exception as e:
            logger.error(f"[LEAN-VOICE] VAD error: {e}")

        # Handle Tracks
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                print(f"[LEAN-VOICE] 🎤 Audio track received", flush=True)
                session_context["audio_track"] = track
                session_context["processing_task"] = asyncio.create_task(
                    _process_audio_track(session_id, track, session_context)
                )

        # Handle Data Channel
        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"[LEAN-VOICE] 📡 Data channel received: {channel.label}", flush=True)
            session_context["data_channel"] = channel
            
            @channel.on("message")
            def on_message(message):
                print(f"[LEAN-VOICE] 📨 Message from client: {message}", flush=True)

        # Connection State Monitoring
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[LEAN-VOICE] 🔄 Connection state: {pc.connectionState}", flush=True)
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
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"[LEAN-VOICE] Offer error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@webrtc_voice_router.post("/ice")
async def handle_ice(request: ICERequest):
    """Handle ICE candidates."""
    try:
        if request.session_id not in _active_connections:
            raise HTTPException(status_code=404, detail="Session not found")
        
        pc = _active_connections[request.session_id]["pc"]
        
        # Parse candidate (simplified)
        parts = request.candidate.split()
        if len(parts) < 8 or 'typ' not in parts:
            return {"status": "ignored", "reason": "malformed"}
            
        candidate = RTCIceCandidate(
            component=int(parts[1]),
            foundation=parts[0].split(':')[1],
            ip=parts[4],
            port=int(parts[5]),
            priority=int(parts[3]),
            protocol=parts[2].upper(),
            type=parts[parts.index('typ') + 1],
            sdpMid=request.sdp_mid,
            sdpMLineIndex=request.sdp_m_line_index
        )
        await pc.addIceCandidate(candidate)
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"[LEAN-VOICE] ICE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_audio_track(session_id: str, track: MediaStreamTrack, context: Dict):
    """Main audio processing loop."""
    print(f"[LEAN-VOICE] ▶️ Starting audio processing loop for {session_id}", flush=True)
    
    frame_count = 0
    speech_buffer_16k = []
    
    try:
        while True:
            try:
                # 1. Receive Frame (with timeout for network issues)
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
                frame_count += 1
                
                # 2. Convert to 16kHz Mono (Standard for VAD/Whisper)
                audio_data = frame.to_ndarray()
                
                # Stereo to Mono
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Resample if needed
                if frame.sample_rate != 16000:
                    # Simple resampling (fast)
                    num_samples = int(len(audio_data) * 16000 / frame.sample_rate)
                    audio_16k = resample_poly(audio_data, 16000, frame.sample_rate)
                else:
                    audio_16k = audio_data

                # Ensure float32 for VAD/Whisper
                audio_16k = audio_16k.astype(np.float32)
                if np.abs(audio_16k).max() > 1.0:
                    audio_16k /= 32768.0 # Normalize int16 to float

                # 3. VAD Processing
                vad_service = context.get("vad_service")
                if vad_service:
                    # Convert to int16 bytes for VAD service
                    audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
                    
                    vad_result = await vad_service.process_audio_chunk(
                        audio_int16.tobytes(),
                        metadata={"sample_rate": 16000}
                    )
                    
                    state = vad_result.get("voice_state")
                    
                    # Accumulate Speech
                    if state in [VADState.VOICE_START, VADState.VOICE_ACTIVE, VADState.VOICE_END_PENDING]:
                        # Cancel any pending turn timer if user starts speaking again
                        if context.get("turn_timer_task"):
                            print(f"[LEAN-VOICE] 🛑 User interrupted silence, cancelling turn timer", flush=True)
                            context["turn_timer_task"].cancel()
                            context["turn_timer_task"] = None
                            
                        speech_buffer_16k.append(audio_16k)
                        
                    # End of Speech -> Process
                    elif state == VADState.VOICE_END:
                        if speech_buffer_16k:
                            # Process in background to not block audio loop
                            full_audio = np.concatenate(speech_buffer_16k)
                            asyncio.create_task(_process_speech_segment(session_id, full_audio, context))
                            speech_buffer_16k = []
                            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "End of file" in str(e):
                    break
                print(f"[LEAN-VOICE] Frame error: {e}", flush=True)
                break
                
    except Exception as e:
        logger.error(f"[LEAN-VOICE] Loop error: {e}")
    finally:
        print(f"[LEAN-VOICE] ⏹️ Audio loop ended for {session_id}", flush=True)
        await _cleanup_session(session_id)

async def _process_speech_segment(session_id: str, audio_data: np.ndarray, context: Dict):
    """Handle STT and schedule LLM generation."""
    whisper = context.get("whisper_model")
    dc = context.get("data_channel")
    loop = context.get("loop")
    
    if not whisper:
        return

    try:
        # 1. Transcribe (Run in executor to avoid blocking)
        # Convert float32 array to int16 bytes for Whisper
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        start_time = time.time()
        print(f"[LEAN-VOICE] 🗣️ Transcribing {len(audio_data)/16000:.2f}s...", flush=True)
        
        text = await loop.run_in_executor(
            None, 
            lambda: whisper.transcribe_audio_bytes(audio_bytes, audio_format="pcm_raw", language="en")
        )
        whisper_time = (time.time() - start_time) * 1000
        
        if not text or not text.strip():
            return
            
        print(f"[LEAN-VOICE] 📝 User (Partial): {text}", flush=True)
        
        # Send Partial Transcript to Client
        if dc and dc.readyState == "open":
            dc.send(json.dumps({
                "type": "transcription", 
                "text": text, 
                "role": "user",
                "metrics": {"whisper_ms": whisper_time}
            }))
            
        # 2. Aggregate and Schedule Turn
        context["transcript_buffer"].append(text)
        
        # Cancel existing timer if any
        if context.get("turn_timer_task"):
            context["turn_timer_task"].cancel()
            
        # Schedule new timer (2 seconds silence)
        context["turn_timer_task"] = asyncio.create_task(_wait_for_silence_and_respond(session_id, context))
                
    except Exception as e:
        logger.error(f"[LEAN-VOICE] Processing error: {e}")
        print(f"[LEAN-VOICE] ❌ Processing error: {e}", flush=True)

async def _wait_for_silence_and_respond(session_id: str, context: Dict):
    """Wait for 2 seconds of silence, then trigger LLM."""
    try:
        await asyncio.sleep(2.0)
        
        # If we get here, no new speech interrupted us
        await _trigger_llm_response(session_id, context)
        
    except asyncio.CancelledError:
        # Timer was cancelled by new speech
        pass
    except Exception as e:
        logger.error(f"[LEAN-VOICE] Timer error: {e}")

async def _trigger_llm_response(session_id: str, context: Dict):
    """Generate and stream LLM response."""
    llm = context.get("llm_model")
    dc = context.get("data_channel")
    loop = context.get("loop")
    buffer = context.get("transcript_buffer", [])
    
    if not buffer or not llm:
        return
        
    full_text = " ".join(buffer)
    context["transcript_buffer"] = [] # Clear buffer
    context["turn_timer_task"] = None
    
    print(f"[LEAN-VOICE] 🤖 Generating response for: {full_text}", flush=True)
    
    # Notify client: Processing (Disable Mic)
    if dc and dc.readyState == "open":
        dc.send(json.dumps({"type": "state", "state": "processing"}))
        
    try:
        # Construct Prompt with /no_think to disable thinking mode
        prompt = f"<|im_start|>system\nYou are a helpful AI assistant. /no_think<|im_end|>\n<|im_start|>user\n{full_text}<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.time()
        token_count = 0
        
        # Run generation in executor (Streaming)
        # Note: Llama.cpp create_completion with stream=True returns a generator
        # We need to iterate it in a way that doesn't block the event loop
        
        def generate_stream():
            return llm.create_completion(
                prompt,
                max_tokens=512,
                stop=["<|im_end|>"],
                stream=True
            )
            
        stream_generator = await loop.run_in_executor(None, generate_stream)
        
        full_response = ""
        
        for chunk in stream_generator:
            delta = chunk["choices"][0]["text"]
            full_response += delta
            token_count += 1
            
            # Filter <think> tags if they leak through
            clean_delta = delta
            if "<think>" in full_response or "</think>" in full_response:
                 # Simple suppression: don't send if inside think block
                 # This is a naive implementation; for robust streaming filtering we'd need a state machine
                 # For now, let's just strip the tags from the final output if needed, 
                 # but for streaming, we send raw delta unless it contains the tag itself
                 clean_delta = delta.replace("<think>", "").replace("</think>", "")
            
            if clean_delta and dc and dc.readyState == "open":
                dc.send(json.dumps({
                    "type": "response_chunk", 
                    "text": clean_delta, 
                    "role": "assistant"
                }))
                # Small yield to let event loop breathe
                await asyncio.sleep(0)
                
        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        
        print(f"[LEAN-VOICE] 🤖 AI ({tps:.1f} t/s): {full_response[:50]}...", flush=True)
        
        # Send Final Metrics and State
        if dc and dc.readyState == "open":
            dc.send(json.dumps({
                "type": "metrics",
                "llm_time_ms": total_time * 1000,
                "tokens_per_sec": tps,
                "total_tokens": token_count
            }))
            dc.send(json.dumps({"type": "state", "state": "listening"}))
            
    except Exception as e:
        logger.error(f"[LEAN-VOICE] LLM error: {e}")
        print(f"[LEAN-VOICE] ❌ LLM error: {e}", flush=True)
        if dc and dc.readyState == "open":
            dc.send(json.dumps({"type": "state", "state": "listening"})) # Reset state on error

async def _cleanup_session(session_id: str):
    """Clean up session resources."""
    if session_id in _active_connections:
        ctx = _active_connections.pop(session_id)
        print(f"[LEAN-VOICE] 🧹 Cleaning up session {session_id}", flush=True)
        try:
            if ctx["pc"]:
                await ctx["pc"].close()
            if ctx["processing_task"]:
                ctx["processing_task"].cancel()
            if ctx.get("turn_timer_task"):
                ctx["turn_timer_task"].cancel()
        except Exception as e:
            logger.error(f"[LEAN-VOICE] Cleanup error: {e}")
