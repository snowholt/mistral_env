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
from ...core.buffer_integration import BufferIntegrationHelper, WebSocketBufferWrapper, get_buffer_manager
from ..performance_integration import get_performance_monitoring_service
from ...core.persistent_model_manager import PersistentModelManager
from ...core.voice_session_manager import get_voice_session_manager, VoiceSessionManager
from ...api.schemas.debug_schemas import (
    WebSocketDebugMessage, PipelineDebugSummary, DebugEvent, PipelineStage, DebugLevel
)

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
    - Persistent model preloading for faster responses
    """
    
    def __init__(self, debug_mode: bool = False):
        self.voice_service = SimpleVoiceService(debug_mode=debug_mode)
        self._service_initialized = False
        self.debug_mode = debug_mode
        
        # VAD service for real-time processing
        self.vad_service = None
        self._vad_initialized = False
        
        # Connection pool for scalable connection management
        self.connection_pool = None
        
        # Buffer optimization system
        self.buffer_manager = None
        self._buffer_wrappers: Dict[str, WebSocketBufferWrapper] = {}
        
        # Persistent model manager for preloaded models
        self.persistent_model_manager = None
        self._persistent_models_initialized = False
        
        # Voice session manager for conversation context  
        # Use backend directory for session files to avoid duplication
        # Find the backend directory more robustly
        current_file = Path(__file__).resolve()
        backend_dir = None
        for parent in current_file.parents:
            if parent.name == "backend":
                backend_dir = parent
                break
        
        if backend_dir:
            backend_session_dir = backend_dir / "sessions" / "voice"
        else:
            # Fallback to the correct number of parents if backend dir not found
            backend_session_dir = Path(__file__).parent.parent.parent.parent.parent / "sessions" / "voice"
        
        self.session_manager = get_voice_session_manager(
            persist_sessions=True, 
            session_dir=backend_session_dir
        )
        
        # Background task for session cleanup
        self._cleanup_task = None
        self._cleanup_interval = 300  # 5 minutes
    
    async def _get_connection_pool(self):
        """Get or initialize the connection pool."""
        global _connection_pool
        if _connection_pool is None:
            _connection_pool = get_websocket_pool()
            if not _connection_pool._health_check_task:
                await _connection_pool.start()
                
        # Initialize buffer manager if available
        if self.buffer_manager is None:
            self.buffer_manager = get_buffer_manager()
            if self.buffer_manager:
                logger.info("Buffer optimization enabled for WebSocket connections")
                
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

    async def _maybe_convert_and_probe(self, webm_path: Path) -> Optional[Path]:  # type: ignore[name-defined]
        """If BEAUTYAI_DEBUG_VOICE=1 convert WebM to 16k mono WAV and log metadata."""
        try:
            if os.environ.get("BEAUTYAI_DEBUG_VOICE") != "1":
                return None
            
            # Use WebMDecoder utility for conversion
            decoder = create_batch_decoder()
            wav_path = webm_path.with_suffix(".wav")
            
            # Convert WebM to PCM using utility
            pcm_data = await decoder.decode_file_to_numpy(webm_path, normalize=False)
            
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
        """Ensure the voice service, VAD service, and persistent models are initialized."""
        
        # Initialize persistent model manager first (if not already done)
        if not self._persistent_models_initialized:
            logger.info("Initializing persistent model manager...")
            try:
                self.persistent_model_manager = PersistentModelManager()
                
                # Start preloading models in background
                logger.info("Starting model preloading...")
                await self.persistent_model_manager.preload_models()
                
                self._persistent_models_initialized = True
                logger.info("✅ Persistent model manager initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize persistent model manager: {e}")
                # Continue without persistent models
                self.persistent_model_manager = None
        
        # Initialize voice service with access to persistent models
        if not self._service_initialized:
            logger.info("Initializing SimpleVoiceService...")
            
            # Pass persistent model manager to voice service if available
            if self.persistent_model_manager:
                # Check if voice service can use persistent models
                if hasattr(self.voice_service, 'set_persistent_model_manager'):
                    self.voice_service.set_persistent_model_manager(self.persistent_model_manager)
                    logger.info("✅ Connected voice service to persistent model manager")
            
            # Set up debug callback if in debug mode
            if self.debug_mode:
                self.voice_service.set_debug_callback(self._on_debug_event)
                logger.info("🔍 Debug callback configured for real-time debug events")
            
            await self.voice_service.initialize()
            self._service_initialized = True
            logger.info("✅ SimpleVoiceService initialized successfully")
        
        # Initialize VAD service with adaptive configuration
        if not self._vad_initialized:
            logger.info("Initializing adaptive VAD service for real-time processing...")
            
            # Create VAD configuration optimized for real-time processing
            vad_config = VADConfig(
                chunk_size_ms=30,  # 30ms chunks for low latency
                silence_threshold_ms=500,  # 500ms silence to trigger turn end
                sampling_rate=16000,  # Standard sampling rate
                speech_threshold=0.5,  # Speech detection threshold
                buffer_max_duration_ms=30000,  # 30 second max buffer
                # Enable adaptive features
                adaptive_threshold=True,
                language_specific_thresholds={
                    'ar': 0.45,
                    'en': 0.5,
                    'auto': 0.5
                }
            )
            
            # Initialize global VAD service
            success = await initialize_vad_service(vad_config)
            if success:
                self.vad_service = get_vad_service()
                self._vad_initialized = True
                logger.info("✅ Adaptive VAD service initialized successfully")
            else:
                logger.error("❌ Failed to initialize VAD service")
                # Continue without VAD for backward compatibility
                self.vad_service = None
        
        # Start background session cleanup if not already running
        await self._start_session_cleanup_task()
    
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        language: str,
        voice_type: str,
        session_id: str = None,
        debug_mode: bool = False
    ) -> bool:
        """Accept connection with minimal setup using connection pool."""
        connect_start_time = time.time()
        
        try:
            await websocket.accept()
            
            # Add performance metric for connection
            perf_service = get_performance_monitoring_service()
            if perf_service.is_enabled():
                await perf_service.add_custom_metric(
                    "websocket_simple_voice_connections_total", 
                    1, 
                    {"language": language, "voice_type": voice_type}
                )
            
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
                    "debug_mode": debug_mode or self.debug_mode,
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
                
                # Create voice session for context management
                voice_session = await self.session_manager.create_session(
                    connection_id=connection_id,
                    language=language,
                    voice_type=voice_type,
                    session_id=session_id or f"simple_{connection_id}"
                )
                
                # Store session reference in connection data
                connection_data.voice_session_data["session_object"] = voice_session
                
                # Initialize optimized buffer wrapper if buffer manager is available
                if self.buffer_manager:
                    try:
                        # Register buffers for this session
                        await BufferIntegrationHelper.register_websocket_buffers(
                            self.buffer_manager, connection_id
                        )
                        
                        # Create optimized buffer wrapper
                        buffer_wrapper = WebSocketBufferWrapper(connection_id, initial_size=65536)
                        self._buffer_wrappers[connection_id] = buffer_wrapper
                        
                        logger.info(f"📊 Initialized optimized buffer for connection {connection_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to initialize optimized buffer for {connection_id}: {e}")
                        # Continue without optimized buffers
            
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
                "debug_mode": debug_mode or self.debug_mode,
                "actual_language": language,
                "config": {
                    "language": language,
                    "voice_type": voice_type,
                    "target_response_time": "< 2 seconds",
                    "debug_enabled": debug_mode or self.debug_mode,
                    "vad_config": {
                        "chunk_size_ms": 30,
                        "silence_threshold_ms": 500
                    } if self._vad_initialized else None
                }
            }
            await pool.send_to_connection(pool_connection_id, welcome_msg, "json")
            
            # Store mapping from original connection_id to pool connection_id
            setattr(self, f"_pool_mapping_{connection_id}", pool_connection_id)
            
            # Add connection time performance metric
            if perf_service.is_enabled():
                connection_time_ms = (time.time() - connect_start_time) * 1000
                await perf_service.add_custom_metric(
                    "websocket_simple_voice_connection_time_ms", 
                    connection_time_ms, 
                    {"language": language, "voice_type": voice_type}
                )
            
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
                    
                    # Close voice session
                    voice_session = connection_data.voice_session_data.get("session_object")
                    if voice_session:
                        await self.session_manager.close_session(voice_session.session_id)
                        logger.info(f"Voice session closed: {voice_session.session_id}")
                    
                    # Add performance metrics for disconnection
                    perf_service = get_performance_monitoring_service()
                    if perf_service.is_enabled():
                        language = connection_data.voice_session_data.get("language", "unknown")
                        voice_type = connection_data.voice_session_data.get("voice_type", "unknown")
                        labels = {"language": language, "voice_type": voice_type}
                        
                        await perf_service.add_custom_metric(
                            "websocket_simple_voice_disconnections_total", 1, labels
                        )
                        await perf_service.add_custom_metric(
                            "websocket_simple_voice_session_duration_seconds", session_duration, labels
                        )
                        await perf_service.add_custom_metric(
                            "websocket_simple_voice_messages_per_session", message_count, labels
                        )
                    
                    logger.info(f"🔌 Simple voice WebSocket disconnected: {connection_id} (pool ID: {pool_connection_id}, duration: {session_duration:.1f}s, messages: {message_count})")
                
                # Unregister from pool
                await pool.unregister_websocket(pool_connection_id)
                
                # Clean up optimized buffer wrapper if exists
                if connection_id in self._buffer_wrappers:
                    try:
                        buffer_wrapper = self._buffer_wrappers[connection_id]
                        await buffer_wrapper.cleanup()
                        del self._buffer_wrappers[connection_id]
                        logger.info(f"📊 Cleaned up optimized buffer for connection {connection_id}")
                    except Exception as buffer_error:
                        logger.warning(f"⚠️ Failed to cleanup optimized buffer for {connection_id}: {buffer_error}")
                
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
                
                # Note: We don't cleanup persistent models as they should stay loaded
                # for faster subsequent connections
                if self.persistent_model_manager:
                    logger.info("ℹ️ Persistent models kept loaded for faster reconnections")
        except Exception as e:
            logger.error(f"❌ Error during simple voice service cleanup: {e}")
    
    async def _start_session_cleanup_task(self):
        """Start the background session cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
            logger.info("🧹 Started background session cleanup task")
    
    async def _session_cleanup_loop(self):
        """Background loop for cleaning up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                # Clean up expired sessions
                cleaned_count = await self.session_manager.cleanup_expired_sessions()
                if cleaned_count > 0:
                    logger.info(f"🧹 Cleaned up {cleaned_count} expired voice sessions")
                
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                # Continue the loop even if there's an error
    
    async def _on_debug_event(self, debug_event: DebugEvent) -> None:
        """Handle debug events from the voice service."""
        if not self.debug_mode:
            return
            
        try:
            # Create debug message for WebSocket clients
            debug_msg = WebSocketDebugMessage(
                type="debug_event",
                debug_mode=True,
                timestamp=debug_event.timestamp,
                connection_id="",  # Will be set per connection
                events=[debug_event]
            )
            
            # Send to all active debug-enabled connections
            pool = await self._get_connection_pool()
            for connection_data in pool.get_all_connections():
                if (connection_data.voice_session_data and 
                    connection_data.voice_session_data.get("debug_mode", False)):
                    
                    # Update connection ID in debug message
                    debug_msg.connection_id = connection_data.connection_id
                    
                    # Send debug event to this connection
                    await pool.send_to_connection(
                        connection_data.connection_id, 
                        debug_msg.dict(), 
                        "json"
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to send debug event: {e}")
    
    def _get_connection_data_by_original_id(self, connection_id: str) -> Optional[WebSocketConnectionData]:
        """Helper method to get connection data by original connection ID."""
        try:
            pool_connection_id = getattr(self, f"_pool_mapping_{connection_id}", None)
            if pool_connection_id:
                pool = await self._get_connection_pool()
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
        connection_data = await self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        language = voice_session["language"]
        voice_type = voice_session["voice_type"]
        session_id = connection_data.session_id
        
        start_time = time.time()
        
        # Add performance metric for audio message processing
        perf_service = get_performance_monitoring_service()
        if perf_service.is_enabled():
            await perf_service.add_custom_metric(
                "websocket_simple_voice_audio_messages_total", 
                1, 
                {"language": language, "voice_type": voice_type}
            )
        
        try:
            # Update message count
            voice_session["message_count"] += 1
            
            # Get conversation context for better responses
            voice_session_object = voice_session.get("session_object")
            conversation_context = ""
            if voice_session_object:
                conversation_context = await self.session_manager.get_conversation_context(
                    voice_session_object.session_id
                )
            
            logger.info(f"🎤 Processing audio message {voice_session['message_count']} for {connection_id} (lang: {language}, voice: {voice_type})")
            if conversation_context:
                logger.info(f"📖 Using conversation context: {conversation_context[:100]}...")
            
            # Send processing started message
            await self.send_message(connection_id, {
                "type": "processing_started",
                "success": True,
                "timestamp": time.time(),
                "message": "Processing your audio..."
            })
            
            # Prepare debug context
            debug_context = {
                "session_id": session_id,
                "connection_id": connection_id,
                "turn_id": f"turn_{voice_session['message_count']}_{int(time.time() * 1000)}"
            } if voice_session.get("debug_mode", False) else None
            
            # Process with SimpleVoiceService
            # Use the real voice processing pipeline
            
            # Process using SimpleVoiceService with correct audio format
            result = await self.voice_service.process_voice_message(
                audio_data=audio_data,
                audio_format=audio_format,  # Pass the detected format
                language=language,
                gender=voice_type,
                conversation_context=conversation_context,  # Add context for better responses
                debug_context=debug_context  # Add debug context
            )
            
            processing_time = time.time() - start_time
            
            # Get debug summary if available
            debug_summary = result.get("debug_summary")
            
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
                
                # Add debug data if available
                if voice_session.get("debug_mode", False) and debug_summary:
                    response_data["debug_summary"] = debug_summary.dict() if hasattr(debug_summary, 'dict') else debug_summary
                    
                    # Send detailed pipeline summary as separate message
                    debug_msg = WebSocketDebugMessage(
                        type="pipeline_complete",
                        debug_mode=True,
                        connection_id=connection_id,
                        pipeline_summary=debug_summary,
                        stage_update={
                            "stage": "complete",
                            "success": True,
                            "total_time_ms": int(processing_time * 1000),
                            "stage_timings": debug_summary.stage_timings if hasattr(debug_summary, 'stage_timings') else {}
                        }
                    )
                    
                    await self.send_message(connection_id, debug_msg.dict())
                
                await self.send_message(connection_id, response_data)
                
                # Record conversation turn in session
                if voice_session_object:
                    transcribed_text = result.get("transcribed_text", "") or ""
                    response_text = result.get("response_text", "") or ""
                    
                    await self.session_manager.add_conversation_turn(
                        session_id=voice_session_object.session_id,
                        user_input=transcribed_text,
                        ai_response=response_text,
                        processing_time_ms=int(processing_time * 1000),
                        transcription_quality=transcription_quality
                    )
                
                logger.info(f"✅ Simple voice processing completed in {processing_time:.2f}s for {connection_id}")
                
                # Add performance metrics
                if perf_service.is_enabled():
                    labels = {"language": language, "voice_type": voice_type, "quality": transcription_quality}
                    await perf_service.add_custom_metric(
                        "websocket_simple_voice_processing_time_ms", 
                        processing_time * 1000, 
                        labels
                    )
                    await perf_service.add_custom_metric(
                        "websocket_simple_voice_processing_success_total", 
                        1, 
                        labels
                    )
                
                return {"success": True, "processing_time": processing_time}
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Error processing simple voice message for {connection_id}: {e}")
            
            # Add error performance metric
            if perf_service.is_enabled():
                await perf_service.add_custom_metric(
                    "websocket_simple_voice_processing_errors_total", 
                    1, 
                    {"language": language, "voice_type": voice_type, "error_type": "processing_error"}
                )
            
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
        connection_data = await self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        
        # 🛡️ CRITICAL: Ignore chunks during turn processing
        if voice_session.get("processing_turn", False):
            logger.debug(f"🚫 BLOCKED: Ignoring chunk during processing for {connection_id}")
            return {"success": True, "status": "ignored_during_processing"}

        # Try to use optimized buffer wrapper if available
        optimized_buffer = self._buffer_wrappers.get(connection_id)
        if optimized_buffer:
            # Use optimized buffer wrapper
            start_time = time.time()
            success = await optimized_buffer.write(audio_data)
            processing_time_ms = (time.time() - start_time) * 1000
            
            if not success:
                logger.warning(f"⚠️ Optimized buffer overflow for {connection_id}, falling back to legacy buffer")
                # Fall through to legacy buffer handling
            else:
                # Update buffer metrics
                await BufferIntegrationHelper.update_buffer_metrics_from_websocket(
                    f"websocket_audio_{connection_id}",
                    bytes_processed=len(audio_data),
                    processing_time_ms=processing_time_ms,
                    queue_size=optimized_buffer.get_available_data(),
                    overflows=optimized_buffer._overflows,
                    underruns=optimized_buffer._underruns
                )
                
                # Check if we have enough data to process
                available_data = optimized_buffer.get_available_data()
                if available_data >= 32768:  # ~3 seconds worth of WebM data
                    logger.info(f"🕒 Optimized buffer segment ready for {connection_id} ({available_data} bytes)")
                    return await self._process_optimized_buffer(connection_id)
                else:
                    return {"success": True, "status": "buffering_optimized", "available_bytes": available_data}
        
        # Legacy buffer handling (fallback)
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
    
    async def _process_optimized_buffer(self, connection_id: str) -> Dict[str, Any]:
        """
        Process audio data from optimized buffer wrapper.
        
        This method handles audio processing using the optimized buffer system
        with adaptive sizing and memory pooling.
        """
        connection_data = await self._get_connection_data_by_original_id(connection_id)
        if not connection_data or not connection_data.voice_session_data:
            return {"success": False, "error": "Connection not found"}
        
        voice_session = connection_data.voice_session_data
        optimized_buffer = self._buffer_wrappers.get(connection_id)
        
        if not optimized_buffer:
            logger.warning(f"⚠️ No optimized buffer found for {connection_id}")
            return {"success": False, "error": "No optimized buffer"}
        
        # Mark as processing to prevent new chunks
        voice_session["processing_turn"] = True
        last_turn_id = f"turn_{voice_session['message_count'] + 1}_{int(time.time() * 1000)}"
        voice_session["last_turn_id"] = last_turn_id
        
        available_data = optimized_buffer.get_available_data()
        logger.info(f"🚀 Processing optimized buffer for {connection_id}: {available_data} bytes (turn: {last_turn_id})")
        
        try:
            # Read all available data from optimized buffer
            audio_data = await optimized_buffer.read(available_data)
            if not audio_data:
                logger.warning(f"⚠️ No data read from optimized buffer for {connection_id}")
                voice_session["processing_turn"] = False
                return {"success": False, "error": "No audio data"}
            
            logger.info(f"📦 Retrieved {len(audio_data)} bytes from optimized buffer")
            
            # Update buffer metrics with processing completion
            start_time = time.time()
            
            # Process the audio using existing voice service
            processing_result = await self.voice_service.process_audio_conversation(
                audio_data,
                voice_session["language"],
                voice_session["voice_type"],
                context={
                    "connection_id": connection_id,
                    "turn_id": last_turn_id,
                    "buffer_type": "optimized",
                    "data_size": len(audio_data)
                }
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update buffer metrics with processing performance
            await BufferIntegrationHelper.update_buffer_metrics_from_websocket(
                f"websocket_audio_{connection_id}",
                bytes_processed=len(audio_data),
                processing_time_ms=processing_time_ms,
                queue_size=optimized_buffer.get_available_data(),
                overflows=optimized_buffer._overflows,
                underruns=optimized_buffer._underruns
            )
            
            if processing_result.get("success", False):
                logger.info(f"✅ Optimized buffer processing successful for {connection_id}")
                
                # Send the response
                response_msg = {
                    "type": "audio_response",
                    "success": True,
                    "connection_id": connection_id,
                    "turn_id": last_turn_id,
                    "timestamp": time.time(),
                    "processing_time_ms": processing_time_ms,
                    "buffer_type": "optimized",
                    "data": processing_result
                }
                
                await self.send_message(connection_id, response_msg)
                
                # Update session counters
                voice_session["message_count"] += 1
                voice_session["_segments_processed"] += 1
                
            else:
                logger.error(f"❌ Optimized buffer processing failed for {connection_id}: {processing_result.get('error', 'Unknown error')}")
                
                # Send error response
                error_msg = {
                    "type": "processing_error",
                    "success": False,
                    "connection_id": connection_id,
                    "turn_id": last_turn_id,
                    "timestamp": time.time(),
                    "error": processing_result.get("error", "Processing failed"),
                    "buffer_type": "optimized"
                }
                
                await self.send_message(connection_id, error_msg)
            
            # Reset processing flag
            voice_session["processing_turn"] = False
            
            return {
                "success": processing_result.get("success", False),
                "turn_id": last_turn_id,
                "processing_time_ms": processing_time_ms,
                "buffer_type": "optimized",
                "bytes_processed": len(audio_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing optimized buffer for {connection_id}: {e}")
            
            # Reset processing flag
            voice_session["processing_turn"] = False
            
            # Send error response
            error_msg = {
                "type": "processing_error",
                "success": False,
                "connection_id": connection_id,
                "timestamp": time.time(),
                "error": f"Optimized buffer processing error: {str(e)}",
                "buffer_type": "optimized"
            }
            
            await self.send_message(connection_id, error_msg)
            
            return {"success": False, "error": f"Optimized buffer processing failed: {str(e)}"}

    async def _process_buffered_chunks(self, connection_id: str) -> Dict[str, Any]:
        """
        Process accumulated audio chunks as a complete WebM stream.
        
        This method handles the conversion of accumulated WebM chunks into
        a complete, decodable audio file for speech processing.
        """
        connection_data = await self._get_connection_data_by_original_id(connection_id)
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
            async def _compute_rms_and_dbfs(webm_bytes: bytes) -> tuple[float, float]:  # type: ignore
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
                        data = await decoder.decode_file_to_numpy(Path(src_path), normalize=True)
                        
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
                rms, dbfs = await _compute_rms_and_dbfs(concatenated_audio)
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
            connection_data = await self._get_connection_data_by_original_id(connection_id)
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
                current_connection_data = await self._get_connection_data_by_original_id(connection_id)
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
                connection_data = await self._get_connection_data_by_original_id(connection_id)
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
# Will be recreated with debug mode parameter in the endpoint
simple_ws_manager = None


def get_or_create_ws_manager(debug_mode: bool = False) -> SimpleVoiceWebSocketManager:
    """Get or create WebSocket manager with specified debug mode."""
    global simple_ws_manager
    
    # Create new manager if needed or debug mode changed
    if simple_ws_manager is None or simple_ws_manager.debug_mode != debug_mode:
        simple_ws_manager = SimpleVoiceWebSocketManager(debug_mode=debug_mode)
        logger.info(f"🔧 Created WebSocket manager with debug_mode={debug_mode}")
    
    return simple_ws_manager


@websocket_simple_voice_router.websocket("/simple-voice-chat")
async def websocket_simple_voice_chat(
    websocket: WebSocket,
    # Simplified parameters only
    language: str = Query("ar", description="Language: ar (Arabic) or en (English)"),
    voice_type: str = Query("female", description="Voice type: male or female"),
    session_id: Optional[str] = Query(None, description="Optional session ID"),
    debug_mode: bool = Query(False, description="Enable debug mode for detailed logging"),
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
    
    # Get or create WebSocket manager with debug mode
    ws_manager = get_or_create_ws_manager(debug_mode)
    
    logger.info(f"🎤 New simple voice WebSocket connection: {connection_id} (lang: {language}, voice: {voice_type}, debug: {debug_mode})")
    
    # Establish connection
    connected = await ws_manager.connect(
        websocket, 
        connection_id, 
        language, 
        voice_type, 
        session_id,
        debug_mode
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
                            pool_connection_id = getattr(ws_manager, f"_pool_mapping_{connection_id}", None)
                            if pool_connection_id:
                                pool = await ws_manager._get_connection_pool()
                                connection_data = pool.get_connection(pool_connection_id)
                                
                                if connection_data and connection_data.voice_session_data:
                                    voice_session = connection_data.voice_session_data
                                    
                                    # Lazy-init capture path
                                    if not voice_session.get("capture_file_path"):
                                        debug_dir = ws_manager._get_debug_audio_dir()
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
                        pool_connection_id = getattr(ws_manager, f"_pool_mapping_{connection_id}", None)
                        if pool_connection_id:
                            pool = await ws_manager._get_connection_pool()
                            connection_data = pool.get_connection(pool_connection_id)
                            
                            if connection_data and connection_data.voice_session_data:
                                voice_session = connection_data.voice_session_data
                                
                                if voice_session.get("vad_enabled", False) and not voice_session.get("processing_turn", False):
                                    # Use real-time VAD processing for smoother interaction
                                    await ws_manager.process_realtime_audio_chunk(
                                        connection_id,
                                        audio_data
                                    )
                                else:
                                    # Use traditional processing (complete audio at once)
                                    await ws_manager.process_audio_message(
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
                            await ws_manager.send_message(connection_id, {
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
        await ws_manager.disconnect(connection_id)


@websocket_simple_voice_router.get("/simple-voice-chat/status")
async def get_simple_voice_status(debug_info: bool = Query(False, description="Include debug information")):
    """Get simple voice service status."""
    try:
        # Get connection pool
        pool = get_websocket_pool()
        if not pool._health_check_task:
            await pool.start()
        
        pool_metrics = pool.get_metrics()
        
        # Get persistent model status
        persistent_model_status = {}
        ws_manager = get_or_create_ws_manager(False)  # Get manager to check status
        if ws_manager.persistent_model_manager:
            try:
                persistent_model_status = await ws_manager.persistent_model_manager.get_health_status()
            except Exception as e:
                logger.warning(f"Failed to get persistent model status: {e}")
                persistent_model_status = {"error": str(e)}
        
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
        
        # Get session manager statistics
        session_stats = ws_manager.session_manager.get_session_stats()
        
        result = {
            "service": "simple_voice_chat",
            "status": "available",
            "active_connections": len(active_connections_info),
            "connections": active_connections_info,
            "persistent_models": persistent_model_status,
            "session_management": session_stats,
            "debug_mode": ws_manager.debug_mode,
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
                "engine": "Edge TTS via SimpleVoiceService",
                "persistent_models_enabled": ws_manager._persistent_models_initialized,
                "adaptive_vad_enabled": ws_manager._vad_initialized,
                "debug_mode_available": True
            },
            "features": [
                "Ultra-fast voice responses",
                "Simplified parameter set",
                "Connection pooling for scalability",
                "Persistent model preloading",
                "Adaptive VAD processing",
                "Language-specific tuning",
                "Enhanced session management",
                "Conversation context preservation",
                "No complex configuration",
                "Direct Edge TTS integration",
                "Arabic and English only",
                "Real-time VAD processing",
                "Server-side turn-taking",
                "Gemini Live / GPT Voice style interaction",
                "Debug mode with detailed metrics",
                "Real-time debug events",
                "Pipeline stage tracking",
                "Audio quality analysis"
            ],
            "audio_formats": {
                "input": ["webm", "wav", "mp3"],
                "output": ["webm"]
            },
            "comparison_with_advanced": {
                "simple_endpoint": {
                    "route": "/ws/simple-voice-chat",
                    "parameters": 4,  # Updated to include debug_mode
                    "response_time": "< 2 seconds",
                    "memory_usage": "< 50MB",
                    "languages": "ar, en only",
                    "features": "Voice chat with persistent models + debug mode",
                    "connection_management": "Pooled",
                    "models": "Preloaded and cached",
                    "debug_capabilities": "Full pipeline debugging"
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
        
        # Add debug information if requested
        if debug_info and ws_manager.debug_mode:
            voice_service_stats = ws_manager.voice_service.get_processing_stats()
            result["debug_info"] = {
                "voice_service_debug_stats": voice_service_stats.get("debug_stats", {}),
                "recent_debug_events": len(ws_manager.voice_service.get_recent_debug_events()),
                "current_debug_summary_available": ws_manager.voice_service.current_debug_summary is not None
            }
        
        # Add performance monitoring data if available
        perf_service = get_performance_monitoring_service()
        if perf_service.is_enabled():
            dashboard_data = await perf_service.get_dashboard_data()
            if dashboard_data:
                result["performance_monitoring"] = {
                    "enabled": True,
                    "system_cpu": dashboard_data.get("system_summary", {}).get("system_cpu_usage_percent"),
                    "system_memory": dashboard_data.get("system_summary", {}).get("system_memory_usage_percent"),
                    "active_alerts": dashboard_data.get("alert_summary", {}).get("total_active", 0),
                    "anomaly_statistics": dashboard_data.get("anomaly_statistics", {})
                }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error getting simple voice status: {e}")
        return {
            "service": "simple_voice_chat",
            "status": "error",
            "error": str(e),
            "active_connections": 0
        }

