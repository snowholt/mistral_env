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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from ...services.voice.conversation.simple_voice_service import SimpleVoiceService
from ...services.voice.vad_service import RealTimeVADService, VADConfig, get_vad_service, initialize_vad_service

logger = logging.getLogger(__name__)

websocket_simple_voice_router = APIRouter(prefix="/ws", tags=["simple-voice"])

# Active WebSocket connections
simple_voice_connections: Dict[str, Dict[str, Any]] = {}


class SimpleVoiceWebSocketManager:
    """
    Enhanced WebSocket manager for simple voice conversations with VAD.
    
    Features:
    - Real-time VAD-driven turn-taking
    - Audio chunk buffering and processing
    - Server-side silence detection
    - Automatic turn completion
    - Gemini Live / GPT Voice style interaction
    """
    
    def __init__(self):
        self.voice_service = SimpleVoiceService()
        self._service_initialized = False
        
        # VAD service for real-time processing
        self.vad_service = None
        self._vad_initialized = False
        
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
        """Accept connection with minimal setup."""
        try:
            await websocket.accept()
            
            # Ensure voice service is ready
            await self._ensure_service_initialized()
            
            # Store connection info with minimal state
            connection_state = {
                "websocket": websocket,
                "language": language,  # Fixed: no auto-detection
                "voice_type": voice_type,  # Fixed: male/female
                "session_id": session_id or f"simple_{connection_id}",
                "connected_at": time.time(),
                "message_count": 0,
                "vad_enabled": self._vad_initialized,  # Track VAD availability
                "audio_buffer": [],  # Buffer for real-time chunks
                "processing_turn": False  # Track if we're processing a complete turn
            }
            
            simple_voice_connections[connection_id] = connection_state
            
            logger.info(f"✅ Simple voice WebSocket connected: {connection_id} (lang: {language}, voice: {voice_type})")
            
            # Send connection confirmation directly to websocket
            welcome_msg = {
                "type": "connection_established",
                "success": True,
                "connection_id": connection_id,
                "session_id": connection_state["session_id"],
                "timestamp": time.time(),
                "message": "Simple voice chat WebSocket connected successfully",
                "vad_enabled": connection_state["vad_enabled"],  # Send VAD availability to frontend
                "config": {
                    "language": language,
                    "voice_type": voice_type,
                    "target_response_time": "< 2 seconds",
                    "vad_config": {
                        "chunk_size_ms": 30,
                        "silence_threshold_ms": 500
                    } if connection_state["vad_enabled"] else None
                }
            }
            await websocket.send_text(json.dumps(welcome_msg))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to establish simple voice WebSocket connection {connection_id}: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Quick cleanup on disconnect."""
        if connection_id in simple_voice_connections:
            connection_info = simple_voice_connections[connection_id]
            session_duration = time.time() - connection_info["connected_at"]
            message_count = connection_info["message_count"]
            
            logger.info(f"🔌 Simple voice WebSocket disconnected: {connection_id} (duration: {session_duration:.1f}s, messages: {message_count})")
            
            # Clean up connection
            del simple_voice_connections[connection_id]
            
        # Clean up voice service if no active connections
        if not simple_voice_connections:
            logger.info("No active simple voice connections, cleaning up...")
            await self._cleanup_service()
    
    async def _cleanup_service(self):
        """Clean up voice service when no active connections."""
        try:
            # Wait a bit before cleanup in case of quick reconnections
            await asyncio.sleep(1)  # Reduced wait time for testing
            if not simple_voice_connections and self._service_initialized:
                await self.voice_service.cleanup()
                self._service_initialized = False
                logger.info("✅ SimpleVoiceService cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during simple voice service cleanup: {e}")
    
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
        if connection_id not in simple_voice_connections:
            logger.warning(f"⚠️ Attempted to send message to non-existent connection: {connection_id}")
            return False
        
        connection = simple_voice_connections[connection_id]
        websocket = connection["websocket"]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
                return True
            else:
                logger.warning(f"⚠️ Simple voice WebSocket connection {connection_id} is not connected")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to send message to simple voice connection {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def process_audio_message(
        self,
        connection_id: str,
        audio_data: bytes
    ) -> Dict[str, Any]:
        """Process audio with SimpleVoiceService for maximum speed."""
        if connection_id not in simple_voice_connections:
            return {"success": False, "error": "Connection not found"}
        
        connection = simple_voice_connections[connection_id]
        language = connection["language"]
        voice_type = connection["voice_type"]
        session_id = connection["session_id"]
        
        start_time = time.time()
        
        try:
            # Update message count
            connection["message_count"] += 1
            
            logger.info(f"🎤 Processing audio message {connection['message_count']} for {connection_id} (lang: {language}, voice: {voice_type})")
            
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
                response_data = {
                    "type": "voice_response",
                    "success": True,
                    "audio_base64": audio_base64,
                    "transcription": result.get("transcribed_text", ""),
                    "response_text": result.get("response_text", ""),
                    "language": result.get("language_detected", language),  # Use actual detected language
                    "voice_type": voice_type,
                    "response_time_ms": int(processing_time * 1000),
                    "session_id": session_id,
                    "message_count": connection["message_count"],
                    "timestamp": time.time()
                }
                
                # Send the response
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
        Process real-time audio chunk with VAD for smooth conversation.
        
        This method implements Gemini Live / GPT Voice style interaction:
        - Processes audio in small chunks (20-30ms)
        - Uses VAD to detect speech start/stop
        - Buffers audio during speech
        - Triggers turn completion on silence
        - Provides real-time feedback
        """
        if connection_id not in simple_voice_connections:
            return {"success": False, "error": "Connection not found"}
        
        connection = simple_voice_connections[connection_id]
        
        # Check if VAD is available
        if not connection.get("vad_enabled", False) or not self.vad_service:
            # Fallback to traditional processing
            logger.info("VAD not available, falling back to traditional processing")
            return await self.process_audio_message(connection_id, audio_data)
        
        try:
            # Detect audio format
            audio_format = self._detect_audio_format(audio_data)
            
            # Setup VAD callbacks for this connection if not already done
            if not hasattr(self, f'_vad_setup_{connection_id}'):
                await self._setup_vad_callbacks(connection_id)
                setattr(self, f'_vad_setup_{connection_id}', True)
            
            # Process chunk with VAD
            vad_result = await self.vad_service.process_audio_chunk(audio_data, audio_format)
            
            if not vad_result.get("success", False):
                logger.error(f"VAD processing failed: {vad_result.get('error', 'Unknown error')}")
                # Fallback to traditional processing
                return await self.process_audio_message(connection_id, audio_data)
            
            # Send real-time feedback to client
            current_state = vad_result.get("current_state", {})
            
            await self.send_message(connection_id, {
                "type": "vad_update",
                "success": True,
                "timestamp": time.time(),
                "state": {
                    "is_speaking": current_state.get("is_speaking", False),
                    "silence_duration_ms": current_state.get("silence_duration_ms", 0),
                    "buffered_chunks": current_state.get("buffered_chunks", 0)
                },
                "processing_time_ms": vad_result.get("processing_time_ms", 0)
            })
            
            return {
                "success": True,
                "processing_mode": "realtime_vad",
                "vad_result": vad_result
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing realtime audio chunk for {connection_id}: {e}")
            
            # Fallback to traditional processing on error
            logger.info("Falling back to traditional processing due to error")
            return await self.process_audio_message(connection_id, audio_data)
    
    async def _setup_vad_callbacks(self, connection_id: str):
        """Setup VAD callbacks for a specific connection."""
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
            if connection_id not in simple_voice_connections:
                logger.warning(f"⚠️ Turn complete for disconnected connection {connection_id} - skipping")
                try:
                    Path(audio_file_path).unlink()
                except Exception:
                    pass
                return
            
            connection = simple_voice_connections[connection_id]
            
            # 🛡️ CRITICAL FIX: Prevent duplicate turn processing
            if connection.get("processing_turn", False):
                logger.warning(f"🚫 Turn already being processed for {connection_id} - ignoring duplicate turn complete")
                try:
                    Path(audio_file_path).unlink()
                except Exception:
                    pass
                return
            
            # Mark connection as processing a turn immediately
            connection["processing_turn"] = True
            last_turn_id = f"turn_{connection['message_count'] + 1}_{int(time.time() * 1000)}"
            connection["last_turn_id"] = last_turn_id
            
            logger.info(f"🎯 Processing turn {last_turn_id} for {connection_id}")
            
            try:
                # Read the audio file created by VAD
                with open(audio_file_path, "rb") as f:
                    complete_audio_data = f.read()
                
                # Process the complete audio with voice service
                language = connection["language"]
                voice_type = connection["voice_type"]
                session_id = connection["session_id"]
                
                # Update message count
                connection["message_count"] += 1
                
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
                if connection_id not in simple_voice_connections:
                    logger.warning(f"⚠️ Connection {connection_id} disconnected during processing - aborting")
                    return
                
                current_connection = simple_voice_connections[connection_id]
                if current_connection.get("last_turn_id") != last_turn_id:
                    logger.warning(f"🚫 Turn ID mismatch for {connection_id} - aborting response (expected: {last_turn_id}, current: {current_connection.get('last_turn_id')})")
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
                    "message_count": connection["message_count"],
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
                if connection_id in simple_voice_connections:
                    simple_voice_connections[connection_id]["processing_turn"] = False
                    # Reset VAD service turn processing state to allow new turns
                    if self.vad_service:
                        self.vad_service.reset_turn_processing()
                    logger.info(f"🏁 Turn processing completed for {connection_id} (turn: {last_turn_id})")
        
        # Set the callbacks
        self.vad_service.set_callbacks(
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_turn_complete=on_turn_complete
        )


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
                        
                        # Check if we should use real-time VAD processing
                        connection_state = simple_voice_connections.get(connection_id, {})
                        
                        if connection_state.get("vad_enabled", False) and not connection_state.get("processing_turn", False):
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
    active_connections_info = []
    
    for conn_id, info in simple_voice_connections.items():
        active_connections_info.append({
            "connection_id": conn_id,
            "session_id": info.get("session_id", f"session_{conn_id}"),
            "language": info.get("language", "ar"),
            "voice_type": info.get("voice_type", "female"),
            "connected_at": info.get("connected_at", time.time()),
            "duration_seconds": time.time() - info.get("connected_at", time.time()),
            "message_count": info.get("message_count", 0)
        })
    
    return {
        "service": "simple_voice_chat",
        "status": "available",
        "active_connections": len(simple_voice_connections),
        "connections": active_connections_info,
        "performance": {
            "target_response_time": "< 2 seconds",
            "supported_languages": ["ar", "en"],
            "voice_types": ["male", "female"],
            "engine": "Edge TTS via SimpleVoiceService"
        },
        "features": [
            "Ultra-fast voice responses",
            "Simplified parameter set",
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
                "features": "Voice chat only"
            },
            "advanced_endpoint": {
                "route": "/ws/voice-conversation", 
                "parameters": "20+",
                "response_time": "5-8 seconds",
                "memory_usage": "3GB+",
                "languages": "Auto-detection + many",
                "features": "Voice cloning, filtering, etc."
            }
        }
    }
