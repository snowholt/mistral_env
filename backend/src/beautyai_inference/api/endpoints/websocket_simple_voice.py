"""
Simple Voice WebSocket Endpoint for Real-time Voice Chat.

This endpoint provides a streamlined, ultra-fast voice conversation interface
using Edge TTS and the SimpleVoiceService for maximum speed (<2 seconds).

Features:
- Simplified parameter set (language, voice_type only)
- Direct Edge TTS integration via SimpleVoiceService
- No complex configuration or session management
- Target response time: <2 seconds
- Arabic and English support only

Author: BeautyAI Framework
Date: 2025-07-23
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import base64
import tempfile
import os
import subprocess
import shlex

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from ...services.voice.conversation.simple_voice_service import SimpleVoiceService
from ...services.voice.vad_service import RealTimeVADService, VADConfig, get_vad_service, initialize_vad_service
from ...utils import create_batch_decoder, WebMDecodingError
from ...core.websocket_connection_pool import get_websocket_pool, WebSocketConnectionData

logger = logging.getLogger(__name__)

websocket_simple_voice_router = APIRouter(prefix="/ws", tags=["simple-voice"])

# Initialize connection pool (will be created on first use)
_connection_pool = None


class SimpleVoiceWebSocketManager:
    """
    Enhanced WebSocket manager for simple voice conversations with VAD.
    
    Features:
    - Real-time VAD-driven turn-taking
    - Audio chunk buffering and processing
    - Server-side silence detection
    - Automatic turn completion
    - Gemini Live / GPT Voice style interaction
    - Connection pool management for scalability
    """
    
    def __init__(self):
        self.voice_service = SimpleVoiceService()
        self._service_initialized = False
        
        # VAD service for real-time processing
        self.vad_service = None
        self._vad_initialized = False
        
        # Connection pool for scalable connection management
        self.connection_pool = None
    
    async def _get_connection_pool(self):
        """Get or initialize the connection pool."""
        global _connection_pool
        if _connection_pool is None:
            _connection_pool = get_websocket_pool()
            if not _connection_pool._health_check_task:
                await _connection_pool.start()
        return _connection_pool
    # (imports moved to module scope)

    # --------------------------------------------------
    # Debug utilities for audio capture & inspection
    # --------------------------------------------------
    def _get_debug_audio_dir(self) -> Path:
        """Resolve stable absolute directory for saving debug audio.
        Priority:
          1. BEAUTYAI_AUDIO_DEBUG_DIR env var
          2. Writable backend directory  / 'vad_testting_temp' (works with ProtectSystem=strict)
          3. Repo root (parent containing 'backend') / 'vad_testting_temp' (legacy)
          4. /tmp/beautyai_vad_debug (may be private when PrivateTmp=yes)
        Ensures directory exists and is writable; falls through until a writable path is found.
        """
        candidates: list[Path] = []

        # 1. Environment variable override
        env_dir = os.environ.get("BEAUTYAI_AUDIO_DEBUG_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())

        # 2. Backend local writable directory (inside systemd ReadWritePaths)
        # Locate backend directory containing this file (…/backend/src/…)
        file_path = Path(__file__).resolve()
        backend_dir = None
        for parent in file_path.parents:
            if parent.name == "backend":
                backend_dir = parent
                break
        if backend_dir:
            candidates.append(backend_dir / "vad_testting_temp")

        # 3. Legacy repo root/vad_testting_temp (may be read-only under ProtectSystem=strict)
        repo_root = None
        for parent in file_path.parents:
            if (parent / "backend").exists():
                repo_root = parent
                break
        if repo_root:
            candidates.append(repo_root / "vad_testting_temp")

        # 4. Fallback /tmp (note: PrivateTmp=yes isolates this path for the service)
        candidates.append(Path("/tmp/beautyai_vad_debug"))

        for cand in candidates:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                test_file = cand / ".write_test"
                with open(test_file, "w") as f:
                    f.write("ok")
                test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                if not getattr(self, "_debug_dir_reported", False):
                    logger.info("🔍 Using debug audio directory: %s", cand)
                    self._debug_dir_reported = True
                return cand
            except Exception as e:  # pragma: no cover - best effort
                logger.debug("Debug dir candidate not writable (%s): %s", cand, e)

        # Last resort: current working directory
        cwd_fallback = Path.cwd() / "vad_testting_temp"
        cwd_fallback.mkdir(parents=True, exist_ok=True)
        logger.warning("⚠️ Falling back to CWD debug directory: %s", cwd_fallback)
        return cwd_fallback

    def _maybe_convert_and_probe(self, webm_path: Path) -> Optional[Path]:  # type: ignore[name-defined]
        """If BEAUTYAI_DEBUG_VOICE=1 convert WebM to 16k mono WAV and log metadata."""
        try:
            if os.environ.get("BEAUTYAI_DEBUG_VOICE") != "1":
                return None
            
            # Use WebMDecoder utility for conversion
            decoder = create_batch_decoder()
            wav_path = webm_path.with_suffix(".wav")
            
            # Convert WebM to PCM using utility
            pcm_data = asyncio.run(decoder.decode_file_to_numpy(webm_path, normalize=False))
            
            # Convert numpy array back to WAV for debugging
            import wave
            with wave.open(str(wav_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes((pcm_data * 32767).astype('int16').tobytes())
            
            if wav_path.exists():
                # Use ffprobe for metadata (if available)
                try:
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_name,sample_rate,channels",
                        "-of", "json", str(wav_path)
                    ]
                    proc = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        logger.info(f"🔍 Segment probe ({webm_path.name}): {proc.stdout.strip()}")
                    else:
                        logger.warning(f"⚠️ ffprobe failed on {wav_path.name}: {proc.stderr.strip()}")
                except FileNotFoundError:
                    logger.info(f"🔍 Segment converted to WAV: {wav_path.name} (ffprobe not available)")
                
                return wav_path
        except Exception as e:
            logger.warning(f"⚠️ WAV conversion/probe failed for {webm_path.name}: {e}")
        return None
        
    async def _ensure_service_initialized(self):
        """Ensure the voice service and VAD service are initialized."""
        if not self._service_initialized:
            logger.info("Initializing SimpleVoiceService...")
            await self.voice_service.initialize()
            self._service_initialized = True
            logger.info("✅ SimpleVoiceService initialized successfully")
        
        if not self._vad_initialized:
            logger.info("Initializing VAD service for real-time processing...")
            
            # Create VAD configuration optimized for real-time processing
            vad_config = VADConfig(
                chunk_size_ms=30,  # 30ms chunks for low latency
                silence_threshold_ms=500,  # 500ms silence to trigger turn end
                sampling_rate=16000,  # Standard sampling rate
                speech_threshold=0.5,  # Speech detection threshold
                buffer_max_duration_ms=30000  # 30 second max buffer
            )
            
            # Initialize global VAD service
            success = await initialize_vad_service(vad_config)
            if success:
                self.vad_service = get_vad_service()
                self._vad_initialized = True
                logger.info("✅ VAD service initialized successfully")
            else:
                logger.error("❌ Failed to initialize VAD service")
                # Continue without VAD for backward compatibility
                self.vad_service = None
    
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        language: str,
        voice_type: str,
        session_id: str = None
    ) -> bool:
        """Accept connection with minimal setup using connection pool."""
        try:
            await websocket.accept()
            
            # Ensure voice service is ready
            await self._ensure_service_initialized()
            
            # Get connection pool
            pool = await self._get_connection_pool()
            
            # Register WebSocket with pool (includes client and session info)
            pool_connection_id = await pool.register_websocket(
                websocket=websocket,
                user_id=None,  # Could be added from auth context later
                session_id=session_id or f"simple_{connection_id}",
                client_info={
                    "connection_type": "simple_voice",
                    "language": language,
                    "voice_type": voice_type,
                    "connected_at": time.time()
                },
                streaming_config={
                    "vad_enabled": self._vad_initialized,
                    "chunk_size_ms": 30,
                    "silence_threshold_ms": 500,
                    "max_buffer_duration_ms": 30000
                }
            )
            
            # Store additional voice-specific data in connection metadata
            connection_data = pool.get_connection(pool_connection_id)
            if connection_data:
                connection_data.voice_session_data = {
                    "language": language,
                    "voice_type": voice_type,
                    "message_count": 0,
                    "vad_enabled": self._vad_initialized,
                    "audio_buffer": [],
                    "processing_turn": False,
                    "capture_file_path": None,
                    "chunk_buffer": [],
                    "first_chunk_received": False,
                    "total_buffered_bytes": 0,
                    "webm_header_chunk": None,
                    "_segments_processed": 0,
                    "_last_silence_guidance_ts": 0
                }
            
            logger.info(f"✅ Simple voice WebSocket connected (pool ID: {pool_connection_id}): {connection_id} (lang: {language}, voice: {voice_type})")
            
            # Send connection confirmation directly to websocket
            welcome_msg = {
                "type": "connection_established",
                "success": True,
                "connection_id": connection_id,
                "pool_connection_id": pool_connection_id,
                "session_id": session_id or f"simple_{connection_id}",
                "timestamp": time.time(),
                "message": "Simple voice chat WebSocket connected successfully",
                "vad_enabled": self._vad_initialized,
                "actual_language": language,
                "config": {
                    "language": language,
                    "voice_type": voice_type,
                    "target_response_time": "< 2 seconds",
                    "vad_config": {
                        "chunk_size_ms": 30,
                        "silence_threshold_ms": 500
                    } if self._vad_initialized else None
                }
            }
            await pool.send_to_connection(pool_connection_id, welcome_msg, "json")
            
            # Store mapping from original connection_id to pool connection_id
            setattr(self, f"_pool_mapping_{connection_id}", pool_connection_id)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to establish simple voice WebSocket connection {connection_id}: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Quick cleanup on disconnect."""
        try:
            # Get pool connection ID from mapping
            pool_connection_id = getattr(self, f"_pool_mapping_{connection_id}", None)
            
            if pool_connection_id:
                # Get connection pool and unregister
                pool = await self._get_connection_pool()
                connection_data = pool.get_connection(pool_connection_id)
                
                if connection_data and connection_data.voice_session_data:
                    session_duration = time.time() - connection_data.client_info.get("connected_at", time.time())
                    message_count = connection_data.voice_session_data.get("message_count", 0)
                    
                    logger.info(f"🔌 Simple voice WebSocket disconnected: {connection_id} (pool ID: {pool_connection_id}, duration: {session_duration:.1f}s, messages: {message_count})")
                
                # Unregister from pool
                await pool.unregister_websocket(pool_connection_id)
                
                # Clean up mapping
                delattr(self, f"_pool_mapping_{connection_id}")
            else:
                logger.warning(f"⚠️ No pool mapping found for connection {connection_id}")
            
            # Clean up voice service if no active connections
            pool = await self._get_connection_pool()
            pool_metrics = pool.get_metrics()
            if pool_metrics["pool"]["active_connections"] == 0:
                logger.info("No active simple voice connections, cleaning up...")
                await self._cleanup_service()
                
        except Exception as e:
            logger.error(f"❌ Error during disconnect cleanup for {connection_id}: {e}")
        
    async def _cleanup_service(self):
        """Clean up voice service when no active connections."""
        try:
            # Wait a bit before cleanup in case of quick reconnections
            await asyncio.sleep(1)  # Reduced wait time for testing
            
            pool = await self._get_connection_pool()
            pool_metrics = pool.get_metrics()
            
            if pool_metrics["pool"]["active_connections"] == 0 and self._service_initialized:
                await self.voice_service.cleanup()
                self._service_initialized = False
                logger.info("✅ SimpleVoiceService cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during simple voice service cleanup: {e}")
    
    def _get_connection_data_by_original_id(self, connection_id: str) -> Optional[WebSocketConnectionData]:
        """Helper method to get connection data by original connection ID."""
        try:
            pool_connection_id = getattr(self, f"_pool_mapping_{connection_id}", None)
            if pool_connection_id:
                pool = asyncio.run(self._get_connection_pool())
                return pool.get_connection(pool_connection_id)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting connection data for {connection_id}: {e}")
            return None
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """
        Detect audio format from binary data using magic bytes.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Detected format as string (webm, wav, mp3, or webm as default)
        """
        if len(audio_data) < 12:
            return "webm"  # Default for short data
        
        # Check for common audio format signatures
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
            return "wav"
        elif audio_data.startswith(b'\x1aEG') or audio_data.startswith(b'\x1a\x45\xdf\xa3'):
            return "webm"  # WebM/Matroska header
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xf3'):
            return "mp3"
        elif audio_data.startswith(b'OggS'):
            return "ogg"
        else:
            # Default to webm for WebSocket streaming (most common for real-time)
            logger.debug(f"Unknown audio format, defaulting to webm. First 16 bytes: {audio_data[:16].hex()}")
            return "webm"
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific WebSocket connection."""
        try:
            pool_connection_id = getattr(self, f"_pool_mapping_{connection_id}", None)
            if not pool_connection_id:
                logger.warning(f"⚠️ No pool mapping found for connection: {connection_id}")
                return False
            
            pool = await self._get_connection_pool()
            return await pool.send_to_connection(pool_connection_id, message, "json")
            
        except Exception as e:
            logger.error(f"❌ Failed to send message to connection {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def process_audio_message(
        self,
        connection_id: str,
        audio_data: bytes
    ) -> Dict[str, Any]:
        """Process audio with SimpleVoiceService for maximum speed."""
        connection_data = self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        language = voice_session["language"]
        voice_type = voice_session["voice_type"]
        session_id = connection_data.session_id
        
        start_time = time.time()
        
        try:
            # Update message count
            voice_session["message_count"] += 1
            
            logger.info(f"🎤 Processing audio message {voice_session['message_count']} for {connection_id} (lang: {language}, voice: {voice_type})")
            
            # Send processing started message
            await self.send_message(connection_id, {
                "type": "processing_started",
                "success": True,
                "timestamp": time.time(),
                "message": "Processing your audio..."
            })
            
            # Use SimpleVoiceService for fast processing
            # Note: For now we'll use a simplified flow since the service has placeholders
            # In a real implementation, this would call the full voice processing pipeline
            
            # Detect audio format from the binary data
            audio_format = self._detect_audio_format(audio_data)
            logger.info(f"🎵 Detected audio format: {audio_format}")
            
            # Save audio to temporary file for processing
            file_extension = audio_format if audio_format in ["webm", "wav", "mp3"] else "webm"
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # Process with SimpleVoiceService
                # Use the real voice processing pipeline
                
                # Process using SimpleVoiceService with correct audio format
                result = await self.voice_service.process_voice_message(
                    audio_data=audio_data,
                    audio_format=audio_format,  # Pass the detected format
                    language=language,
                    gender=voice_type
                )
                
                processing_time = time.time() - start_time
                
                # Read audio file and encode to base64
                audio_base64 = None
                if result.get("audio_file_path"):
                    audio_path = Path(result["audio_file_path"])
                    if audio_path.exists():
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                        # Clean up the audio file
                        audio_path.unlink()
                
                # Prepare response
                transcribed_text = result.get("transcribed_text", "") or ""
                unclear = any(marker in transcribed_text.lower() for marker in ["unclear audio", "صوت غير واضح"])
                transcription_quality = "unclear" if unclear else "ok"

                if unclear and audio_base64 is None:
                    # Send guidance-only message (prevents loops & auto-restart)
                    guidance = result.get("response_text") or ("أعد المحاولة وتحدث بوضوح لمدة ثانيتين" if language == "ar" else "Please try again and speak clearly for 2 seconds")
                    await self.send_message(connection_id, {
                        "type": "guidance",
                        "success": False,
                        "guidance": guidance,
                        "transcription": transcribed_text,
                        "transcription_quality": transcription_quality,
                        "language": result.get("language_detected", language),
                        "voice_type": voice_type,
                        "response_time_ms": int(processing_time * 1000),
                        "session_id": session_id,
                        "message_count": voice_session["message_count"],
                        "timestamp": time.time()
                    })
                else:
                    response_data = {
                        "type": "voice_response",
                        "success": True,
                        "audio_base64": audio_base64,
                        "transcription": transcribed_text,
                        "response_text": result.get("response_text", ""),
                        "language": result.get("language_detected", language),  # Use actual detected language
                        "voice_type": voice_type,
                        "response_time_ms": int(processing_time * 1000),
                        "session_id": session_id,
                        "message_count": voice_session["message_count"],
                        "transcription_quality": transcription_quality,
                        "timestamp": time.time()
                    }
                    await self.send_message(connection_id, response_data)
                
                logger.info(f"✅ Simple voice processing completed in {processing_time:.2f}s for {connection_id}")
                return {"success": True, "processing_time": processing_time}
            
            finally:
                # Clean up temporary file
                try:
                    Path(temp_audio_path).unlink()
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to cleanup temp file {temp_audio_path}: {cleanup_error}")
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Error processing simple voice message for {connection_id}: {e}")
            
            await self.send_message(connection_id, {
                "type": "error",
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": f"Processing error: {str(e)}",
                "retry_suggested": True,
                "response_time_ms": int(processing_time * 1000),
                "timestamp": time.time()
            })
            return {"success": False, "error": str(e)}
    
    async def process_realtime_audio_chunk(
        self,
        connection_id: str,
        audio_data: bytes
    ) -> Dict[str, Any]:
        """
        Process real-time audio chunk with proper accumulation for WebM streams.
        
        CRITICAL FIX: MediaRecorder WebM chunks are NOT standalone files!
        - Only the first chunk contains WebM headers/metadata
        - Subsequent chunks are raw data fragments
        - Individual chunks CANNOT be decoded by ffmpeg
        - Must accumulate chunks into complete audio segments
        
        This method implements proper chunk accumulation:
        - Buffers WebM chunks until complete segment is ready
        - Uses VAD to detect speech boundaries for segmentation
        - Only processes complete, decodable audio segments
        - Prevents infinite loop from failed individual chunk decoding
        """
        connection_data = self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        
        # 🛡️ CRITICAL: Ignore chunks during turn processing
        if voice_session.get("processing_turn", False):
            logger.debug(f"🚫 BLOCKED: Ignoring chunk during processing for {connection_id}")
            return {"success": True, "status": "ignored_during_processing"}
        
        # Initialize chunk buffer if not exists
        if "chunk_buffer" not in voice_session:
            voice_session["chunk_buffer"] = []
            voice_session["first_chunk_received"] = False
            voice_session["total_buffered_bytes"] = 0
        
        # Add chunk to buffer
        voice_session["chunk_buffer"].append(audio_data)
        voice_session["total_buffered_bytes"] += len(audio_data)
        
        # Mark if this is the first chunk (contains WebM header)
        if not voice_session["first_chunk_received"]:
            voice_session["first_chunk_received"] = True
            # Preserve the initial WebM header chunk for re-use (subsequent buffered segments lack header)
            voice_session["webm_header_chunk"] = audio_data
            logger.info(f"📦 First chunk received for {connection_id}: {len(audio_data)} bytes")
        
        logger.debug(f"📦 Buffered chunk {len(voice_session['chunk_buffer'])} for {connection_id}: {len(audio_data)} bytes (total: {voice_session['total_buffered_bytes']})")
        
        # 🚨 CRITICAL FIX: Detect audio format from first chunk
        audio_format = "webm"  # Default assumption for MediaRecorder
        if voice_session.get("first_chunk_received", False) and len(voice_session["chunk_buffer"]) > 0:
            first_chunk = voice_session["chunk_buffer"][0]
            audio_format = self._detect_audio_format(first_chunk)
        
        # 🛡️ CRITICAL: VAD Service cannot process WebM format properly!  
        # VAD expects PCM/WAV data, not WebM container format.
        # For WebM streams, we MUST use time-based accumulation instead of VAD.
        if audio_format == "webm":
            logger.info(f"🎯 WebM format detected - using time-based accumulation (VAD disabled for WebM)")
            
            # Use time-based segmentation for WebM (no VAD)
            # Process when we have accumulated sufficient chunks for a meaningful segment
            if len(voice_session["chunk_buffer"]) >= 30:  # ~3 seconds worth of data
                logger.info(f"🕒 WebM time-based segment complete for {connection_id} ({len(voice_session['chunk_buffer'])} chunks) - processing buffered audio")
                return await self._process_buffered_chunks(connection_id)
            else:
                logger.debug(f"📦 WebM accumulating: {len(voice_session['chunk_buffer'])}/30 chunks for {connection_id}")
                return {"success": True, "status": "buffering", "chunks_buffered": len(voice_session["chunk_buffer"])}
        
        # For non-WebM formats, we can try VAD if available
        if not voice_session.get("vad_enabled", False) or not self.vad_service:
            # Without VAD: Use time-based segmentation (fallback)
            if len(voice_session["chunk_buffer"]) >= 50:  # ~5 seconds at 100ms chunks
                logger.info(f"🕒 Time-based segment complete for {connection_id} - processing buffered audio")
                return await self._process_buffered_chunks(connection_id)
            else:
                return {"success": True, "status": "buffering", "chunks_buffered": len(voice_session["chunk_buffer"])}
        
        try:
            # Setup VAD callbacks if needed (only for non-WebM formats)
            if not hasattr(self, f'_vad_setup_{connection_id}'):
                await self._setup_vad_callbacks(connection_id)
                setattr(self, f'_vad_setup_{connection_id}', True)
            
            # Try to process accumulated chunks with VAD
            # Note: Only process if we have enough data to attempt decoding
            if len(voice_session["chunk_buffer"]) >= 5:  # Minimum viable segment
                concatenated_audio = b''.join(voice_session["chunk_buffer"])
                
                # Try VAD processing on accumulated data
                vad_result = await self.vad_service.process_audio_chunk(concatenated_audio, audio_format)
                
                if vad_result.get("success", False):
                    # Send VAD feedback
                    current_state = vad_result.get("current_state", {})
                    
                    await self.send_message(connection_id, {
                        "type": "vad_update",
                        "success": True,
                        "timestamp": time.time(),
                        "state": {
                            "is_speaking": current_state.get("is_speaking", False),
                            "silence_duration_ms": current_state.get("silence_duration_ms", 0),
                            "buffered_chunks": len(voice_session["chunk_buffer"])
                        },
                        "processing_time_ms": vad_result.get("processing_time_ms", 0)
                    })
                else:
                    logger.warning(f"⚠️ VAD failed on accumulated chunks for {connection_id}, falling back to time-based processing")
                    # Fallback to time-based processing when VAD fails
                    if len(voice_session["chunk_buffer"]) >= 10:
                        return await self._process_buffered_chunks(connection_id)
            
            return {
                "success": True,
                "processing_mode": "chunk_accumulation",
                "chunks_buffered": len(voice_session["chunk_buffer"]),
                "total_bytes": voice_session["total_buffered_bytes"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing chunk accumulation for {connection_id}: {e}")
            
            # Emergency fallback: try to process buffered chunks
            if len(voice_session["chunk_buffer"]) >= 10:
                logger.info(f"🆘 Emergency processing of buffered chunks for {connection_id}")
                return await self._process_buffered_chunks(connection_id)
            
            return {"success": False, "error": f"Chunk processing failed: {str(e)}"}
    
    async def _process_buffered_chunks(self, connection_id: str) -> Dict[str, Any]:
        """
        Process accumulated audio chunks as a complete WebM stream.
        
        This method handles the conversion of accumulated WebM chunks into
        a complete, decodable audio file for speech processing.
        """
        connection_data = self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        
        if not voice_session.get("chunk_buffer") or len(voice_session["chunk_buffer"]) == 0:
            logger.warning(f"⚠️ No chunks to process for {connection_id}")
            return {"success": False, "error": "No buffered chunks"}
        
        # Mark as processing to prevent new chunks
        voice_session["processing_turn"] = True
        last_turn_id = f"turn_{voice_session['message_count'] + 1}_{int(time.time() * 1000)}"
        voice_session["last_turn_id"] = last_turn_id
        
        logger.info(f"🎯 Processing {len(voice_session['chunk_buffer'])} buffered chunks for {connection_id} (turn: {last_turn_id})")
        
        try:
            # Concatenate all buffered chunks; ensure header present for decodability
            raw_buffer = b''.join(voice_session["chunk_buffer"])
            if voice_session.get("webm_header_chunk") and voice_session.get("_segments_processed", 0) > 0:
                concatenated_audio = voice_session["webm_header_chunk"] + raw_buffer
            else:
                concatenated_audio = raw_buffer
            
            # Clear the buffer
            voice_session["chunk_buffer"] = []
            voice_session["total_buffered_bytes"] = 0
            
            logger.info(f"📦 Concatenated {len(concatenated_audio)} bytes for processing")

            # ------------------------------------------------------------
            # SILENCE / ENERGY CHECK (Prevent repeated unclear guidance)
            # ------------------------------------------------------------
            def _compute_rms_and_dbfs(webm_bytes: bytes) -> tuple[float, float]:  # type: ignore
                """Compute RMS and dBFS of a WebM audio buffer using WebMDecoder utility.
                Returns (rms, dbfs). On failure returns (0.0, -100.0).
                """
                try:
                    import tempfile, math
                    
                    # Use WebMDecoder utility instead of FFmpeg subprocess
                    decoder = create_batch_decoder()
                    
                    # Create temporary WebM file
                    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src_f:
                        src_f.write(webm_bytes)
                        src_path = src_f.name
                    
                    try:
                        # Decode WebM to numpy array
                        data = asyncio.run(decoder.decode_file_to_numpy(Path(src_path), normalize=True))
                        
                        if data.size == 0:
                            return 0.0, -100.0
                        
                        # Compute RMS and dBFS
                        import numpy as np
                        rms = float(np.sqrt(np.mean(np.square(data))))
                        if rms <= 0:
                            dbfs = -100.0
                        else:
                            dbfs = 20 * math.log10(rms)
                        
                        return rms, dbfs
                        
                    finally:
                        # Cleanup temporary file
                        try:
                            os.unlink(src_path)
                        except Exception:
                            pass
                            
                except Exception as e:  # pragma: no cover
                    logger.debug(f"RMS computation failed: {e}")
                    return 0.0, -100.0

            # Only attempt energy check for webm (streaming) format and if sizable data
            if len(concatenated_audio) > 500:  # avoid tiny segments
                rms, dbfs = _compute_rms_and_dbfs(concatenated_audio)
                logger.info(f"🔊 Segment energy for {connection_id}: RMS={rms:.5f}, dBFS={dbfs:.1f}")
                # Thresholds (tunable): treat as silence if below -45 dBFS OR rms < 0.008
                silence = (dbfs < -45.0) or (rms < 0.008)
                if silence:
                    # Send guidance only once until a non-silent segment arrives
                    last_guidance = voice_session.get("_last_silence_guidance_ts", 0)
                    now_ts = time.time()
                    # 3 second spacing between silence guidance messages
                    if now_ts - last_guidance > 3:
                        guidance_text = "أعد المحاولة وتحدث بوضوح خلال ثانيتين" if voice_session.get("language") == "ar" else "Please try again and speak clearly for 2 seconds"
                        await self.send_message(connection_id, {
                            "type": "guidance",
                            "success": False,
                            "guidance": guidance_text,
                            "transcription": "",
                            "transcription_quality": "unclear",
                            "language": voice_session.get("language", "ar"),
                            "voice_type": voice_session.get("voice_type", "female"),
                            "response_time_ms": 0,
                            "session_id": connection_data.session_id,
                            "message_count": voice_session.get("message_count", 0),
                            "timestamp": time.time(),
                            "note": "silence_suppressed_segment"
                        })
                        voice_session["_last_silence_guidance_ts"] = now_ts
                    else:
                        logger.debug(f"Silence guidance suppressed (cooldown) for {connection_id}")
                    # Mark not processing turn since we skipped
                    voice_session["processing_turn"] = False
                    return {"success": True, "status": "silence_segment_skipped"}
                else:
                    # Reset silence guidance timer on voiced segment
                    voice_session.pop("_last_silence_guidance_ts", None)
            
            # Process the complete audio with traditional flow
            # This now has a complete WebM file that should be decodable
            result = await self.process_audio_message(connection_id, concatenated_audio)
            voice_session["_segments_processed"] = voice_session.get("_segments_processed", 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing buffered chunks for {connection_id}: {e}")
            return {"success": False, "error": f"Buffered chunk processing failed: {str(e)}"}
        finally:
            # Always reset processing state
            if connection_data and connection_data.voice_session_data:
                connection_data.voice_session_data["processing_turn"] = False
    
    async def _setup_vad_callbacks(self, connection_id: str):
        """Setup VAD callbacks for a specific connection.
        Renamed from _legacy_setup_vad_callbacks to match call sites.
        """
        if not self.vad_service:
            return
        
        async def on_speech_start():
            """Called when speech is detected."""
            await self.send_message(connection_id, {
                "type": "speech_start",
                "success": True,
                "timestamp": time.time(),
                "message": "Speech detected - recording..."
            })
        
        async def on_speech_end():
            """Called when speech ends."""
            await self.send_message(connection_id, {
                "type": "speech_end",
                "success": True,
                "timestamp": time.time(),
                "message": "Speech ended - processing..."
            })
        
        async def on_turn_complete(audio_file_path: str, audio_data):
            """Called when a complete turn is ready for processing."""
            logger.info(f"🔄 Turn complete for {connection_id}, processing complete audio...")
            
            # Check if connection still exists
            connection_data = self._get_connection_data_by_original_id(connection_id)
            if not connection_data or not connection_data.voice_session_data:
                logger.warning(f"⚠️ Turn complete for disconnected connection {connection_id} - skipping")
                try:
                    Path(audio_file_path).unlink()
                except Exception:
                    pass
                return
            
            voice_session = connection_data.voice_session_data
            
            # 🛡️ Prevent duplicate turn processing
            if voice_session.get("processing_turn", False):
                logger.warning(f"🚫 Turn already being processed for {connection_id} - ignoring duplicate turn complete")
                try:
                    Path(audio_file_path).unlink()
                except Exception:
                    pass
                return
            
            # Mark connection as processing a turn immediately
            voice_session["processing_turn"] = True
            last_turn_id = f"turn_{voice_session['message_count'] + 1}_{int(time.time() * 1000)}"
            voice_session["last_turn_id"] = last_turn_id
            
            logger.info(f"🎯 Processing turn {last_turn_id} for {connection_id}")
            
            try:
                # Read the audio file created by VAD
                with open(audio_file_path, "rb") as f:
                    complete_audio_data = f.read()
                
                # Process the complete audio with voice service
                language = voice_session["language"]
                voice_type = voice_session["voice_type"]
                session_id = connection_data.session_id
                
                # Update message count
                voice_session["message_count"] += 1
                
                # Send processing started message
                await self.send_message(connection_id, {
                    "type": "turn_processing_started",
                    "success": True,
                    "timestamp": time.time(),
                    "turn_id": last_turn_id,
                    "message": "Processing complete turn..."
                })
                
                # Process with voice service
                result = await self.voice_service.process_voice_message(
                    audio_data=complete_audio_data,
                    audio_format="wav",  # VAD service saves as WAV
                    language=language,
                    gender=voice_type
                )
                
                # Double-check we're still the active turn (prevent race conditions)
                current_connection_data = self._get_connection_data_by_original_id(connection_id)
                if not current_connection_data or not current_connection_data.voice_session_data:
                    logger.warning(f"⚠️ Connection {connection_id} disconnected during processing - aborting")
                    return
                
                current_voice_session = current_connection_data.voice_session_data
                if current_voice_session.get("last_turn_id") != last_turn_id:
                    logger.warning(f"🚫 Turn ID mismatch for {connection_id} - aborting response (expected: {last_turn_id}, current: {current_voice_session.get('last_turn_id')})")
                    return
                
                # Prepare and send response
                audio_base64 = None
                if result.get("audio_file_path"):
                    audio_path = Path(result["audio_file_path"])
                    if audio_path.exists():
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                        audio_path.unlink()
                
                response_data = {
                    "type": "voice_response",
                    "success": True,
                    "audio_base64": audio_base64,
                    "transcription": result.get("transcribed_text", ""),
                    "response_text": result.get("response_text", ""),
                    "language": result.get("language_detected", language),
                    "voice_type": voice_type,
                    "response_time_ms": result.get("processing_time", 0) * 1000,
                    "session_id": session_id,
                    "message_count": voice_session["message_count"],
                    "timestamp": time.time(),
                    "turn_id": last_turn_id,
                    "processing_mode": "vad_driven"
                }
                
                await self.send_message(connection_id, response_data)
                
                logger.info(f"✅ VAD-driven turn processing completed for {connection_id} (turn: {last_turn_id})")
                
            except Exception as e:
                logger.error(f"❌ Error processing turn complete for {connection_id}: {e}")
                
                await self.send_message(connection_id, {
                    "type": "error",
                    "success": False,
                    "error_code": "TURN_PROCESSING_ERROR",
                    "message": f"Failed to process complete turn: {str(e)}",
                    "timestamp": time.time(),
                    "turn_id": last_turn_id
                })
            
            finally:
                # Clean up VAD audio file
                try:
                    Path(audio_file_path).unlink()
                except Exception:
                    pass
                
                # Mark turn processing as complete and reset VAD state
                connection_data = self._get_connection_data_by_original_id(connection_id)
                if connection_data and connection_data.voice_session_data:
                    connection_data.voice_session_data["processing_turn"] = False
                    # Reset VAD service turn processing state to allow new turns
                    if self.vad_service:
                        self.vad_service.reset_turn_processing()
                    logger.info(f"🏁 Turn processing completed for {connection_id} (turn: {last_turn_id})")
        
        # Set the callbacks (ensure consistent indentation / spaces only)
        if self.vad_service:
            self.vad_service.set_callbacks(
                on_speech_start=on_speech_start,
                on_speech_end=on_speech_end,
                on_turn_complete=on_turn_complete
            )
            logger.debug("✅ VAD callbacks registered for connection %s", connection_id)
        else:
            logger.warning("VAD service missing while setting callbacks for %s", connection_id)


# Global WebSocket manager instance
simple_ws_manager = SimpleVoiceWebSocketManager()


@websocket_simple_voice_router.websocket("/simple-voice-chat")
async def websocket_simple_voice_chat(
    websocket: WebSocket,
    # Simplified parameters only
    language: str = Query("ar", description="Language: ar (Arabic) or en (English)"),
    voice_type: str = Query("female", description="Voice type: male or female"),
    session_id: Optional[str] = Query(None, description="Optional session ID"),
):
    """
    🚀 Ultra-Fast Voice Chat WebSocket
    
    Simplified real-time voice conversation optimized for speed:
    - Arabic and English support only
    - No auto-detection or complex parameters
    - Target <2 second response time
    - Direct Edge TTS integration
    
    🎯 **Performance Goals:**
    - End-to-End Response: <2 seconds (audio in → audio out)
    - Connection Overhead: Minimal parameters and setup
    - Simplicity: No complex configuration or session management
    - Languages: Arabic and English only
    
    📡 **Message Flow:**
    
    **Input (Client → Server):**
    - Binary messages: Audio data (WebM, WAV, MP3)
    
    **Output (Server → Client):**
    ```json
    {
        "type": "voice_response",
        "success": true,
        "audio_base64": "UklGRi4gAABXQVZFZm10...",
        "transcription": "مرحبا، كيف حالك؟",
        "response_text": "أهلا وسهلا! أنا بخير، شكرا لك",
        "language": "ar",
        "voice_type": "female",
        "response_time_ms": 1250,
        "session_id": "simple_session_123"
    }
    ```
    
    🚀 **Usage Example:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/simple-voice-chat?language=ar&voice_type=female');
    
    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.type === 'voice_response' && response.audio_base64) {
            // Play received audio
            const audioBlob = base64ToBlob(response.audio_base64, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            new Audio(audioUrl).play();
        }
    };
    
    // Send audio
    ws.send(audioData); // Send binary audio data
    ```
    """
    
    # Validate parameters
    if language not in ["ar", "en"]:
        await websocket.close(code=1003, reason="Invalid language. Supported: ar, en")
        return
    
    if voice_type not in ["male", "female"]:
        await websocket.close(code=1003, reason="Invalid voice_type. Supported: male, female")
        return
    
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    
    logger.info(f"🎤 New simple voice WebSocket connection: {connection_id} (lang: {language}, voice: {voice_type})")
    
    # Establish connection
    connected = await simple_ws_manager.connect(
        websocket, 
        connection_id, 
        language, 
        voice_type, 
        session_id
    )
    
    if not connected:
        await websocket.close(code=1000, reason="Failed to establish connection")
        return
    
    try:
        while True:
            # Wait for message from client
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            elif message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Audio data received
                    audio_data = message["bytes"]
                    
                    if len(audio_data) > 0:
                        logger.info(f"🎵 Received audio data: {len(audio_data)} bytes from {connection_id}")

                        # ================= TEST AUDIO CAPTURE (Raw WebM stream) =================
                        # Unified capture directory using helper for stability
                        try:
                            # Get connection data from pool
                            pool_connection_id = getattr(simple_ws_manager, f"_pool_mapping_{connection_id}", None)
                            if pool_connection_id:
                                pool = await simple_ws_manager._get_connection_pool()
                                connection_data = pool.get_connection(pool_connection_id)
                                
                                if connection_data and connection_data.voice_session_data:
                                    voice_session = connection_data.voice_session_data
                                    
                                    # Lazy-init capture path
                                    if not voice_session.get("capture_file_path"):
                                        debug_dir = simple_ws_manager._get_debug_audio_dir()
                                        capture_filename = f"ws_capture_{connection_id}_{int(time.time())}.webm"
                                        capture_path = debug_dir / capture_filename
                                        voice_session["capture_file_path"] = str(capture_path)
                                        logger.info(
                                            "💾 Initiated raw audio capture for %s -> %s", connection_id, capture_path
                                        )
                                    
                                    try:
                                        with open(voice_session["capture_file_path"], "ab") as cap_f:
                                            cap_f.write(audio_data)
                                    except OSError as primary_err:
                                        # Handle read-only filesystem (errno 30) or permission errors gracefully
                                        if getattr(primary_err, "errno", None) in (30, 13):  # 30=Read-only FS, 13=Permission
                                            fallback_dir = Path("/tmp/beautyai_vad_debug")
                                            try:
                                                fallback_dir.mkdir(parents=True, exist_ok=True)
                                                if not voice_session.get("_capture_fallback_switched"):
                                                    logger.warning(
                                                        "⚠️ Primary capture path unwritable (%s). Switching to fallback %s", 
                                                        primary_err, fallback_dir
                                                    )
                                                fallback_path = fallback_dir / Path(voice_session["capture_file_path"]).name
                                                voice_session["capture_file_path"] = str(fallback_path)
                                                voice_session["_capture_fallback_switched"] = True
                                                with open(fallback_path, "ab") as cap_f2:
                                                    cap_f2.write(audio_data)
                                            except Exception as fb_err:
                                                logger.warning(
                                                    "⚠️ Fallback capture also failed for %s: %s", connection_id, fb_err
                                                )
                                        else:
                                            raise
                        except Exception as cap_err:
                            logger.warning("⚠️ Failed to capture raw audio chunk for %s: %s", connection_id, cap_err)
                        # ========================================================================
                        
                        # Check if we should use real-time VAD processing
                        pool_connection_id = getattr(simple_ws_manager, f"_pool_mapping_{connection_id}", None)
                        if pool_connection_id:
                            pool = await simple_ws_manager._get_connection_pool()
                            connection_data = pool.get_connection(pool_connection_id)
                            
                            if connection_data and connection_data.voice_session_data:
                                voice_session = connection_data.voice_session_data
                                
                                if voice_session.get("vad_enabled", False) and not voice_session.get("processing_turn", False):
                                    # Use real-time VAD processing for smoother interaction
                                    await simple_ws_manager.process_realtime_audio_chunk(
                                        connection_id,
                                        audio_data
                                    )
                                else:
                                    # Use traditional processing (complete audio at once)
                                    await simple_ws_manager.process_audio_message(
                                        connection_id,
                                        audio_data
                                    )
                            else:
                                logger.warning(f"⚠️ No voice session data found for {connection_id}")
                        else:
                            logger.warning(f"⚠️ No pool mapping found for {connection_id}")
                    else:
                        logger.warning(f"⚠️ Received empty audio data from {connection_id}")
                
                elif "text" in message:
                    # Control message received
                    try:
                        control_msg = json.loads(message["text"])
                        
                        if control_msg.get("type") == "ping":
                            # Respond to ping
                            await simple_ws_manager.send_message(connection_id, {
                                "type": "pong",
                                "timestamp": time.time()
                            })
                        
                        else:
                            logger.warning(f"⚠️ Unknown control message type: {control_msg.get('type')} from {connection_id}")
                    
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON in text message from {connection_id}")
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Simple voice WebSocket disconnected: {connection_id}")
    
    except Exception as e:
        logger.error(f"❌ Simple voice WebSocket error for {connection_id}: {e}")
    
    finally:
        # Clean up connection
        await simple_ws_manager.disconnect(connection_id)


@websocket_simple_voice_router.get("/simple-voice-chat/status")
async def get_simple_voice_status():
    """Get simple voice service status."""
    try:
        # Get connection pool
        pool = get_websocket_pool()
        if not pool._health_check_task:
            await pool.start()
        
        pool_metrics = pool.get_metrics()
        
        # Get active connections info from pool
        active_connections_info = []
        for conn_data in pool_metrics["connections"]:
            if conn_data.get("state") == "active":
                # Try to extract voice session data
                connection_obj = pool.get_connection(conn_data["connection_id"])
                if connection_obj and hasattr(connection_obj, 'voice_session_data'):
                    voice_data = connection_obj.voice_session_data
                    active_connections_info.append({
                        "connection_id": conn_data["connection_id"],
                        "session_id": connection_obj.session_id,
                        "language": voice_data.get("language", "ar"),
                        "voice_type": voice_data.get("voice_type", "female"),
                        "connected_at": connection_obj.client_info.get("connected_at", time.time()),
                        "duration_seconds": conn_data.get("age_seconds", 0),
                        "message_count": voice_data.get("message_count", 0)
                    })
        
        return {
            "service": "simple_voice_chat",
            "status": "available",
            "active_connections": len(active_connections_info),
            "connections": active_connections_info,
            "pool_metrics": {
                "total_connections": pool_metrics["pool"]["total_connections"],
                "active_connections": pool_metrics["pool"]["active_connections"],
                "idle_connections": pool_metrics["pool"]["idle_connections"],
                "pool_utilization": pool_metrics["pool"]["pool_utilization"],
                "peak_usage": pool_metrics["pool"]["peak_usage"],
                "avg_acquisition_time_ms": pool_metrics["pool"]["avg_acquisition_time_ms"]
            },
            "performance": {
                "target_response_time": "< 2 seconds",
                "supported_languages": ["ar", "en"],
                "voice_types": ["male", "female"],
                "engine": "Edge TTS via SimpleVoiceService"
            },
            "features": [
                "Ultra-fast voice responses",
                "Simplified parameter set",
                "Connection pooling for scalability",
                "No complex configuration",
                "Direct Edge TTS integration",
                "Arabic and English only",
                "Real-time VAD processing",
                "Server-side turn-taking",
                "Gemini Live / GPT Voice style interaction"
            ],
            "audio_formats": {
                "input": ["webm", "wav", "mp3"],
                "output": ["webm"]
            },
            "comparison_with_advanced": {
                "simple_endpoint": {
                    "route": "/ws/simple-voice-chat",
                    "parameters": 3,
                    "response_time": "< 2 seconds",
                    "memory_usage": "< 50MB",
                    "languages": "ar, en only",
                    "features": "Voice chat only",
                    "connection_management": "Pooled"
                },
                "advanced_endpoint": {
                    "route": "/ws/voice-conversation", 
                    "parameters": "20+",
                    "response_time": "5-8 seconds",
                    "memory_usage": "3GB+",
                    "languages": "Auto-detection + many",
                    "features": "Voice cloning, filtering, etc.",
                    "connection_management": "Legacy"
                }
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting simple voice status: {e}")
        return {
            "service": "simple_voice_chat",
            "status": "error",
            "error": str(e),
            "active_connections": 0
        }

