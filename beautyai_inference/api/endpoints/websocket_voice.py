"""
Real-time WebSocket Voice-to-Voice Conversation Endpoint.

Provides real-time bidirectional voice conversation with streaming audio support:
- Send audio chunks via WebSocket
- Receive processed audio responses in real-time
- Maintain conversation context across messages
- Support for Arabic and multilingual conversations
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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from starlette.websockets import WebSocketState

from ..auth import AuthContext, get_auth_context_ws
from ..models import VoiceConversationSession
from ...services.voice_to_voice_service import VoiceToVoiceService
# Audio utilities will be handled inline
from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)

websocket_voice_router = APIRouter(prefix="/ws", tags=["websocket-voice"])

# Active WebSocket connections
active_connections: Dict[str, Dict[str, Any]] = {}


class WebSocketVoiceManager:
    """Manages WebSocket voice conversation connections."""
    
    def __init__(self):
        self.voice_service = VoiceToVoiceService()
        self.model_manager = ModelManager()
        
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        session_config: Dict[str, Any]
    ) -> bool:
        """Accept and register a new WebSocket connection."""
        try:
            await websocket.accept()
            
            # Store connection info
            active_connections[connection_id] = {
                "websocket": websocket,
                "session_id": session_config.get("session_id", f"ws_{connection_id}"),
                "config": session_config,
                "conversation_history": [],
                "connected_at": time.time(),
                "last_activity": time.time()
            }
            
            logger.info(f"WebSocket voice connection established: {connection_id}")
            
            # Send connection confirmation
            await self.send_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "session_id": active_connections[connection_id]["session_id"],
                "timestamp": time.time(),
                "message": "Voice conversation WebSocket connected successfully"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection {connection_id}: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Clean up connection and unload models if needed."""
        if connection_id in active_connections:
            connection_info = active_connections[connection_id]
            session_duration = time.time() - connection_info["connected_at"]
            
            logger.info(f"WebSocket voice connection closed: {connection_id} (duration: {session_duration:.1f}s)")
            
            # Clean up
            del active_connections[connection_id]
            
            # Unload models if no active connections
            if not active_connections:
                logger.info("No active WebSocket connections, cleaning up models")
                await self.cleanup_models()
    
    async def cleanup_models(self):
        """Unload models when no active connections."""
        try:
            # Only unload if no active connections for 30+ seconds
            await asyncio.sleep(30)
            if not active_connections:
                self.voice_service.unload_all_models()
                logger.info("Voice models unloaded due to inactivity")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send a message to a specific WebSocket connection."""
        if connection_id not in active_connections:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return False
        
        connection = active_connections[connection_id]
        websocket = connection["websocket"]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
                connection["last_activity"] = time.time()
                return True
            else:
                logger.warning(f"WebSocket connection {connection_id} is not connected")
                return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def process_voice_message(
        self, 
        connection_id: str, 
        audio_data: bytes, 
        audio_format: str = "webm"
    ) -> Dict[str, Any]:
        """Process voice input and generate audio response."""
        if connection_id not in active_connections:
            return {"error": "Connection not found"}
        
        connection = active_connections[connection_id]
        config = connection["config"]
        session_id = connection["session_id"]
        
        try:
            # Update activity timestamp
            connection["last_activity"] = time.time()
            
            # Send processing started message
            await self.send_message(connection_id, {
                "type": "processing_started",
                "timestamp": time.time(),
                "message": "Processing your voice message..."
            })
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # Prepare voice-to-voice request
                v2v_request = {
                    "audio_path": temp_audio_path,
                    "session_id": session_id,
                    "chat_history": connection["conversation_history"],
                    **config  # Include all configuration parameters
                }
                
                # Process voice-to-voice
                start_time = time.time()
                result = await self.voice_service.voice_to_voice_async(**v2v_request)
                processing_time = time.time() - start_time
                
                if result.get("success", False):
                    # Update conversation history
                    connection["conversation_history"].extend([
                        {"role": "user", "content": result.get("transcription", "")},
                        {"role": "assistant", "content": result.get("response_text", "")}
                    ])
                    
                    # Read generated audio file
                    audio_output_path = result.get("audio_output_path")
                    if audio_output_path and Path(audio_output_path).exists():
                        with open(audio_output_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        
                        # Encode audio as base64 for WebSocket transmission
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                        
                        # Send processing completed message with audio
                        await self.send_message(connection_id, {
                            "type": "voice_response",
                            "success": True,
                            "timestamp": time.time(),
                            "session_id": session_id,
                            "transcription": result.get("transcription", ""),
                            "response_text": result.get("response_text", ""),
                            "audio_base64": audio_base64,
                            "audio_format": result.get("audio_output_format", "wav"),
                            "audio_size_bytes": len(audio_bytes),
                            "processing_time_ms": processing_time * 1000,
                            "models_used": result.get("models_used", {}),
                            "metadata": {
                                "thinking_mode": result.get("thinking_mode", False),
                                "content_filter_applied": result.get("content_filter_applied", False),
                                "input_language": result.get("input_language", "auto"),
                                "output_language": result.get("output_language", "auto")
                            }
                        })
                        
                        # Clean up temporary audio file
                        try:
                            Path(audio_output_path).unlink()
                        except:
                            pass
                        
                        return {"success": True, "processing_time": processing_time}
                    
                    else:
                        # No audio generated
                        await self.send_message(connection_id, {
                            "type": "voice_response",
                            "success": False,
                            "timestamp": time.time(),
                            "error": "Audio generation failed",
                            "transcription": result.get("transcription", ""),
                            "response_text": result.get("response_text", "")
                        })
                        return {"success": False, "error": "Audio generation failed"}
                
                else:
                    # Processing failed
                    error_msg = result.get("error", "Voice processing failed")
                    await self.send_message(connection_id, {
                        "type": "voice_response", 
                        "success": False,
                        "timestamp": time.time(),
                        "error": error_msg
                    })
                    return {"success": False, "error": error_msg}
            
            finally:
                # Clean up temporary input file
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Error processing voice message for {connection_id}: {e}")
            await self.send_message(connection_id, {
                "type": "voice_response",
                "success": False,
                "timestamp": time.time(),
                "error": f"Processing error: {str(e)}"
            })
            return {"success": False, "error": str(e)}


# Global WebSocket manager instance
ws_voice_manager = WebSocketVoiceManager()


@websocket_voice_router.websocket("/voice-conversation")
async def websocket_voice_conversation(
    websocket: WebSocket,
    # Connection parameters
    session_id: Optional[str] = Query(None, description="Optional session ID for conversation continuity"),
    input_language: str = Query("auto", description="Input language (auto, ar, en, es, fr, de)"),
    output_language: str = Query("auto", description="Output language (auto, ar, en, es, fr, de)"),
    
    # Model configuration
    stt_model_name: str = Query("whisper-large-v3-turbo-arabic", description="Speech-to-text model"),
    tts_model_name: str = Query("coqui-tts-arabic", description="Text-to-speech model"),
    chat_model_name: str = Query("qwen3-unsloth-q4ks", description="Chat model"),
    
    # Audio settings
    speaker_voice: str = Query("female", description="TTS speaker voice"),
    emotion: str = Query("neutral", description="TTS emotion"),
    speech_speed: float = Query(1.0, description="TTS speech speed"),
    audio_output_format: str = Query("wav", description="Output audio format"),
    
    # Generation parameters
    temperature: Optional[float] = Query(None, description="Generation temperature"),
    top_p: Optional[float] = Query(None, description="Top-p sampling"),
    top_k: Optional[int] = Query(None, description="Top-k sampling"),
    repetition_penalty: Optional[float] = Query(None, description="Repetition penalty"),
    max_new_tokens: Optional[int] = Query(None, description="Maximum new tokens"),
    
    # Content filtering and thinking mode
    disable_content_filter: bool = Query(False, description="Disable content filtering"),
    content_filter_strictness: str = Query("balanced", description="Content filter strictness"),
    thinking_mode: bool = Query(False, description="Enable thinking mode"),
    preset: Optional[str] = Query(None, description="Generation preset"),
    
    # Authentication
    # auth: AuthContext = Depends(get_auth_context_ws)  # Commented out for now
):
    """
    🎤 Real-time Voice-to-Voice Conversation via WebSocket
    
    Provides bidirectional voice conversation with real-time audio streaming:
    
    📡 **Connection Flow:**
    1. Client connects to WebSocket with configuration parameters
    2. Server confirms connection and sends connection_id
    3. Client sends audio chunks as binary messages
    4. Server processes audio and sends back JSON responses with audio
    
    🎯 **Message Types:**
    
    **Incoming (Client → Server):**
    - Binary messages: Audio data (WebM, WAV, MP3, etc.)
    - Text messages: Control commands (JSON format)
    
    **Outgoing (Server → Client):**
    - `connection_established`: WebSocket connection confirmed
    - `processing_started`: Audio processing began
    - `voice_response`: Complete response with transcription and audio
    - `error`: Error occurred during processing
    
    🎵 **Audio Handling:**
    - **Input**: Any supported format (WebM recommended for browsers)
    - **Output**: Base64-encoded audio in JSON response
    - **Streaming**: Real-time processing and response
    
    💬 **Session Management:**
    - Automatic conversation history tracking
    - Persistent session across multiple audio exchanges
    - Session cleanup on disconnect
    
    🚀 **Usage Example:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/voice-conversation?preset=qwen_optimized');
    
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
    navigator.mediaDevices.getUserMedia({audio: true}).then(stream => {
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(e.data); // Send audio chunk
            }
        };
    });
    ```
    """
    
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    
    # Prepare session configuration
    session_config = {
        "session_id": session_id or f"ws_{connection_id}",
        "input_language": input_language,
        "output_language": output_language,
        "stt_model_name": stt_model_name,
        "tts_model_name": tts_model_name,
        "chat_model_name": chat_model_name,
        "speaker_voice": speaker_voice,
        "emotion": emotion,
        "speech_speed": speech_speed,
        "audio_output_format": audio_output_format,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "disable_content_filter": disable_content_filter,
        "content_filter_strictness": content_filter_strictness,
        "thinking_mode": thinking_mode,
        "preset": preset
    }
    
    # Establish connection
    connected = await ws_voice_manager.connect(websocket, connection_id, session_config)
    
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
                        logger.info(f"Received audio data: {len(audio_data)} bytes from {connection_id}")
                        
                        # Process voice message asynchronously
                        await ws_voice_manager.process_voice_message(
                            connection_id,
                            audio_data,
                            audio_format="webm"  # Default format for browsers
                        )
                    
                elif "text" in message:
                    # Control message received
                    try:
                        control_msg = json.loads(message["text"])
                        
                        if control_msg.get("type") == "ping":
                            # Respond to ping
                            await ws_voice_manager.send_message(connection_id, {
                                "type": "pong",
                                "timestamp": time.time()
                            })
                        
                        elif control_msg.get("type") == "update_config":
                            # Update session configuration
                            if connection_id in active_connections:
                                active_connections[connection_id]["config"].update(
                                    control_msg.get("config", {})
                                )
                                await ws_voice_manager.send_message(connection_id, {
                                    "type": "config_updated",
                                    "timestamp": time.time(),
                                    "message": "Configuration updated successfully"
                                })
                        
                        else:
                            logger.warning(f"Unknown control message type: {control_msg.get('type')}")
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in text message from {connection_id}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    
    finally:
        # Clean up connection
        await ws_voice_manager.disconnect(connection_id)


@websocket_voice_router.get("/voice-conversation/status")
async def websocket_voice_status():
    """Get status of WebSocket voice conversation service."""
    return {
        "service": "websocket_voice_conversation",
        "status": "available",
        "active_connections": len(active_connections),
        "connections": [
            {
                "connection_id": conn_id,
                "session_id": info["session_id"],
                "connected_at": info["connected_at"],
                "last_activity": info["last_activity"],
                "duration_seconds": time.time() - info["connected_at"]
            }
            for conn_id, info in active_connections.items()
        ],
        "supported_features": [
            "Real-time voice conversation",
            "Automatic language detection",
            "Session persistence",
            "Multiple audio formats",
            "Content filtering",
            "Thinking mode",
            "Generation presets"
        ],
        "audio_formats": {
            "input": ["webm", "wav", "mp3", "ogg", "flac", "m4a"],
            "output": ["wav", "mp3", "ogg"]
        }
    }
