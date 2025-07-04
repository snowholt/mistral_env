"""
Voice-to-Voice Service for BeautyAI Framework.

Integrates Speech-to-Text, Large Language Model, and Text-to-Speech 
to provide seamless voice conversations. This service connects the models
directly to minimize latency and improve performance.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO

from .base.base_service import BaseService
from .audio_transcription_service import AudioTranscriptionService
from .text_to_speech_service import TextToSpeechService
from .inference.chat_service import ChatService
from .inference.content_filter_service import ContentFilterService
from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class VoiceToVoiceService(BaseService):
    """
    Complete voice-to-voice conversation service.
    
    Pipeline: Audio Input → STT → LLM → TTS → Audio Output
    
    Features:
    - Direct model-to-model communication for minimal latency
    - Support for Arabic and English with Coqui TTS
    - Content filtering
    - Session management
    - Performance metrics tracking
    """
    
    def __init__(self, content_filter_strictness: str = "balanced"):
        """Initialize the voice-to-voice service."""
        super().__init__()
        
        # Core services
        self.stt_service = AudioTranscriptionService()
        self.tts_service = TextToSpeechService()
        self.chat_service = ChatService()
        self.content_filter = ContentFilterService(strictness=content_filter_strictness)
        self.model_manager = ModelManager()
        
        # Service status
        self.services_loaded = {
            "stt": False,
            "tts": False,
            "chat": False,
            "content_filter": True  # Always available
        }
        
        # Default configurations - use Coqui TTS instead of OuteTTS
        self.default_config = {
            "stt_model": "whisper-large-v3-turbo-arabic",
            "tts_model": "coqui-tts-arabic",  # Changed from OuteTTS
            "chat_model": "qwen3-unsloth-q4ks",
            "language": "ar",
            "speaker_voice": "female",
            "response_max_length": 256,
            "enable_content_filter": True
        }
        
        # Session management
        self.current_session = None
        self.conversation_history = []
        self.active_sessions = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "average_latency": 0.0,
            "success_rate": 0.0
        }
        
        # Output directory for audio files
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/voice_to_voice_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_models(
        self,
        stt_model: str = None,
        chat_model: str = None,
        tts_model: str = None,
        language: str = "ar"
    ) -> Dict[str, bool]:
        """
        Initialize all required models for voice-to-voice conversation.
        
        Args:
            stt_model: Speech-to-text model identifier
            chat_model: Chat model identifier
            tts_model: Text-to-speech model identifier
            language: Target language code
            
        Returns:
            Dict indicating success/failure for each model type
        """
        # Use defaults if not specified
        stt_model = stt_model or self.default_config["stt_model"]
        chat_model = chat_model or self.default_config["chat_model"]
        tts_model = tts_model or self.default_config["tts_model"]
        
        results = {}
        
        try:
            # Initialize STT Service
            logger.info(f"Initializing STT service with model: {stt_model}")
            results["stt"] = self.stt_service.initialize_model(stt_model)
            self.services_loaded["stt"] = results["stt"]
            
            # Initialize Chat Service
            logger.info(f"Initializing Chat service with model: {chat_model}")
            chat_success = self.chat_service.load_model(chat_model)
            results["chat"] = chat_success
            self.services_loaded["chat"] = chat_success
            
            # Initialize TTS Service
            logger.info(f"Initializing TTS service with model: {tts_model}")
            tts_success = self.tts_service.initialize_engine(tts_model)
            results["tts"] = tts_success
            self.services_loaded["tts"] = tts_success
            
            logger.info(f"Model initialization results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            return {"stt": False, "chat": False, "tts": False}

    def voice_to_voice_conversation(
        self,
        audio_file: str,
        session_id: str = None,
        language: str = "ar",
        speaker_voice: str = "female",
        response_max_length: int = 256,
        enable_content_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complete voice-to-voice conversation.
        
        Args:
            audio_file: Path to input audio file
            session_id: Conversation session identifier
            language: Target language for processing
            speaker_voice: Voice type for TTS output
            response_max_length: Maximum response length
            enable_content_filter: Whether to apply content filtering
            
        Returns:
            Dict containing conversation results and output paths
        """
        start_time = time.time()
        
        # Validate models are loaded
        if not self._validate_models_loaded():
            return {
                "success": False,
                "error": "Required models not loaded. Call initialize_models() first.",
                "transcription": None,
                "response": None,
                "audio_output": None,
                "processing_time": 0.0
            }
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Speech-to-Text
            logger.info(f"Starting STT for session {session_id}")
            transcription_result = self.stt_service.transcribe(
                audio_file=audio_file,
                language=language
            )
            
            if not transcription_result.get("success", False):
                return {
                    "success": False,
                    "error": f"STT failed: {transcription_result.get('error', 'Unknown error')}",
                    "transcription": None,
                    "response": None,
                    "audio_output": None,
                    "processing_time": time.time() - start_time
                }
            
            transcribed_text = transcription_result["transcription"]
            logger.info(f"Transcription successful: {transcribed_text[:50]}...")
            
            # Step 2: Content filtering (if enabled)
            if enable_content_filter:
                filter_result = self.content_filter.filter_content(transcribed_text)
                if not filter_result["is_safe"]:
                    return {
                        "success": False,
                        "error": f"Content filtered: {filter_result['reason']}",
                        "transcription": transcribed_text,
                        "response": None,
                        "audio_output": None,
                        "processing_time": time.time() - start_time
                    }
            
            # Step 3: Get conversation history for context
            conversation_history = self.get_session_history(session_id) or []
            
            # Step 4: Chat inference
            logger.info(f"Starting chat inference for session {session_id}")
            chat_result = self.chat_service.chat(
                message=transcribed_text,
                conversation_history=conversation_history,
                max_length=response_max_length,
                language=language
            )
            
            if not chat_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Chat failed: {chat_result.get('error', 'Unknown error')}",
                    "transcription": transcribed_text,
                    "response": None,
                    "audio_output": None,
                    "processing_time": time.time() - start_time
                }
            
            response_text = chat_result["response"]
            logger.info(f"Chat response: {response_text[:50]}...")
            
            # Step 5: Text-to-Speech
            logger.info(f"Starting TTS for session {session_id}")
            output_audio_path = self.output_dir / f"response_{session_id}_{int(time.time())}.wav"
            
            tts_result = self.tts_service.synthesize(
                text=response_text,
                output_path=str(output_audio_path),
                language=language,
                voice=speaker_voice
            )
            
            if not tts_result.get("success", False):
                return {
                    "success": False,
                    "error": f"TTS failed: {tts_result.get('error', 'Unknown error')}",
                    "transcription": transcribed_text,
                    "response": response_text,
                    "audio_output": None,
                    "processing_time": time.time() - start_time
                }
            
            # Step 6: Update session history
            new_messages = [
                {"role": "user", "content": transcribed_text},
                {"role": "assistant", "content": response_text}
            ]
            self._update_session(session_id, new_messages)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            self.performance_stats["total_requests"] += 1
            self.performance_stats["average_latency"] = (
                (self.performance_stats["average_latency"] * (self.performance_stats["total_requests"] - 1) + processing_time) /
                self.performance_stats["total_requests"]
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "transcription": transcribed_text,
                "response": response_text,
                "audio_output": str(output_audio_path),
                "processing_time": processing_time,
                "metadata": {
                    "language": language,
                    "speaker_voice": speaker_voice,
                    "content_filtered": enable_content_filter,
                    "audio_duration": transcription_result.get("duration", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in voice-to-voice conversation: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": None,
                "response": None,
                "audio_output": None,
                "processing_time": time.time() - start_time
            }

    def voice_to_voice_file(
        self,
        input_audio_path: str,
        output_audio_path: str = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process voice-to-voice conversion from file to file.
        
        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path for output audio file (optional)
            config: Configuration overrides
            
        Returns:
            Processing results with paths and metadata
        """
        if config is None:
            config = self.default_config.copy()
        
        # Generate output path if not provided
        if output_audio_path is None:
            timestamp = int(time.time())
            output_audio_path = str(self.output_dir / f"voice_output_{timestamp}.wav")
        
        # Use the main conversation method
        result = self.voice_to_voice_conversation(
            audio_file=input_audio_path,
            language=config.get("language", "ar"),
            speaker_voice=config.get("speaker_voice", "female"),
            response_max_length=config.get("response_max_length", 256),
            enable_content_filter=config.get("enable_content_filter", True)
        )
        
        # Copy audio to specified output path if different
        if result.get("success") and result.get("audio_output") != output_audio_path:
            try:
                import shutil
                shutil.copy2(result["audio_output"], output_audio_path)
                result["audio_output"] = output_audio_path
            except Exception as e:
                logger.warning(f"Could not copy to specified output path: {e}")
        
        return result

    def _validate_models_loaded(self) -> bool:
        """Check if all required models are loaded."""
        required_services = ["stt", "tts", "chat"]
        for service in required_services:
            if not self.services_loaded.get(service, False):
                logger.error(f"Service not loaded: {service}")
                return False
        return True

    def _update_session(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """Update conversation history for a session."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].extend(messages)

    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get conversation history for a session."""
        return self.active_sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False

    def get_models_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all loaded models."""
        return {
            "services_loaded": self.services_loaded.copy(),
            "stt_service": {
                "status": "loaded" if self.services_loaded.get("stt") else "not_loaded",
                "details": self.stt_service.get_status() if hasattr(self.stt_service, 'get_status') else {}
            },
            "tts_service": {
                "status": "loaded" if self.services_loaded.get("tts") else "not_loaded",
                "details": self.tts_service.get_status() if hasattr(self.tts_service, 'get_status') else {}
            },
            "chat_service": {
                "status": "loaded" if self.services_loaded.get("chat") else "not_loaded",
                "details": self.chat_service.get_status() if hasattr(self.chat_service, 'get_status') else {}
            },
            "active_sessions": len(self.active_sessions),
            "performance_stats": self.performance_stats.copy()
        }

    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        try:
            # Unload STT service
            if hasattr(self.stt_service, 'unload_model'):
                self.stt_service.unload_model()
            
            # Unload TTS service
            if hasattr(self.tts_service, 'unload_all_engines'):
                self.tts_service.unload_all_engines()
            
            # Unload chat models through model manager
            if hasattr(self.chat_service, 'unload_model'):
                self.chat_service.unload_model()
            
            # Reset service status
            self.services_loaded = {
                "stt": False,
                "tts": False,
                "chat": False,
                "content_filter": True
            }
            
            logger.info("All voice-to-voice models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "stt_service": self.stt_service.get_memory_stats() if hasattr(self.stt_service, 'get_memory_stats') else {},
            "tts_service": self.tts_service.get_memory_stats() if hasattr(self.tts_service, 'get_memory_stats') else {},
            "model_manager": self.model_manager.get_memory_usage() if hasattr(self.model_manager, 'get_memory_usage') else {}
        }
