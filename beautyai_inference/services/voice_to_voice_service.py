"""
Voice-to-Voice Service for BeautyAI Framework.

Integrates Speech-to-Text, Large Language Model, and Text-to-Speech 
to provide seamless voice conversations. This service connects the models
directly to minimize latency and improve performance.
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, BinaryIO
import io

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
    - Support for Arabic and English
    - Content filtering
    - Session management
    - Performance metrics tracking
    """
    
    def __init__(self, content_filter_strictness: str = "balanced"):
        super().__init__()
        self.model_manager = ModelManager()
        
        # Initialize sub-services
        self.stt_service = AudioTranscriptionService()
        self.tts_service = TextToSpeechService()
        self.chat_service = ChatService(content_filter_strictness)
        self.content_filter = ContentFilterService(strictness_level=content_filter_strictness)
        
        # Model loading status
        self.stt_model_loaded = False
        self.tts_model_loaded = False
        self.chat_model_loaded = False
        
        # Default models
        self.default_stt_model = "whisper-large-v3-turbo-arabic"
        self.default_tts_model = "xtts-v2"
        self.default_chat_model = "qwen3-unsloth-q4ks"
        
        # Session management
        self.active_sessions = {}
    
    def initialize_models(
        self,
        stt_model: str = None,
        tts_model: str = None,
        chat_model: str = None
    ) -> bool:
        """
        Initialize all required models for voice-to-voice conversation.
        
        Args:
            stt_model: Speech-to-Text model name
            tts_model: Text-to-Speech model name  
            chat_model: Chat model name
            
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            stt_model = stt_model or self.default_stt_model
            tts_model = tts_model or self.default_tts_model
            chat_model = chat_model or self.default_chat_model
            
            logger.info("Initializing voice-to-voice models...")
            
            # Load STT model
            logger.info(f"Loading STT model: {stt_model}")
            if not self.stt_service.load_whisper_model(stt_model):
                logger.error(f"Failed to load STT model: {stt_model}")
                return False
            self.stt_model_loaded = True
            logger.info("✓ STT model loaded")
            
            # Load TTS model
            logger.info(f"Loading TTS model: {tts_model}")
            if not self.tts_service.load_tts_model(tts_model):
                logger.error(f"Failed to load TTS model: {tts_model}")
                return False
            self.tts_model_loaded = True
            logger.info("✓ TTS model loaded")
            
            # Load chat model (through model manager)
            logger.info(f"Loading chat model: {chat_model}")
            app_config = AppConfig()
            app_config.models_file = "beautyai_inference/config/model_registry.json"
            app_config.load_model_registry()
            
            chat_model_config = app_config.model_registry.get_model(chat_model)
            if not chat_model_config:
                logger.error(f"Chat model configuration not found: {chat_model}")
                return False
            
            if not self.model_manager.is_model_loaded(chat_model):
                success = self.model_manager.load_model(chat_model_config)
                if not success:
                    logger.error(f"Failed to load chat model: {chat_model}")
                    return False
            
            self.chat_model_loaded = True
            logger.info("✓ Chat model loaded")
            
            logger.info("🎉 All voice-to-voice models initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def voice_to_voice_conversation(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        input_language: str = "ar",
        output_language: str = "ar",
        session_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        speaker_voice: Optional[str] = None,
        emotion: str = "neutral",
        speech_speed: float = 1.0,
        generation_config: Optional[Dict[str, Any]] = None,
        disable_content_filter: bool = False
    ) -> Dict[str, Any]:
        """
        Complete voice-to-voice conversation pipeline.
        
        Args:
            audio_bytes: Input audio data
            audio_format: Audio format (wav, mp3, etc.)
            input_language: Language of input audio
            output_language: Language for output audio
            session_id: Session identifier for conversation continuity
            chat_history: Previous conversation history
            speaker_voice: TTS speaker voice to use
            emotion: TTS emotion/style
            speech_speed: TTS speech speed
            generation_config: LLM generation parameters
            disable_content_filter: Whether to disable content filtering
            
        Returns:
            dict: Complete conversation result with metrics
        """
        start_time = time.time()
        result = {
            "success": False,
            "session_id": session_id or str(uuid.uuid4()),
            "transcription": "",
            "response_text": "",
            "response_audio_path": None,
            "response_audio_bytes": None,
            "input_language": input_language,
            "output_language": output_language,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Validate models are loaded
            if not self._validate_models_loaded():
                result["errors"].append("Not all required models are loaded")
                return result
            
            # Step 1: Speech-to-Text
            logger.info("🎙️ Starting speech-to-text conversion...")
            stt_start = time.time()
            
            transcription = self.stt_service.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                language=input_language
            )
            
            stt_end = time.time()
            stt_time = stt_end - stt_start
            
            if not transcription:
                result["errors"].append("Speech-to-text conversion failed")
                return result
            
            result["transcription"] = transcription
            logger.info(f"✓ Transcription: '{transcription[:100]}...' ({stt_time:.2f}s)")
            
            # Step 2: Content filtering (input)
            if not disable_content_filter:
                logger.info("🔒 Applying input content filter...")
                filter_result = self.content_filter.filter_input(transcription)
                if not filter_result.is_allowed:
                    result["errors"].append(f"Content filtered: {filter_result.reason}")
                    result["transcription"] = filter_result.filtered_content or ""
                    return result
            
            # Step 3: Chat completion
            logger.info("🤖 Generating chat response...")
            chat_start = time.time()
            
            # Build chat history
            messages = chat_history or []
            messages.append({"role": "user", "content": transcription})
            
            # Get chat model config
            app_config = AppConfig()
            app_config.models_file = "beautyai_inference/config/model_registry.json"
            app_config.load_model_registry()
            chat_model_config = app_config.model_registry.get_model(self.default_chat_model)
            
            # Generate response using direct model access for better performance
            model_instance = self.model_manager.get_loaded_model(self.default_chat_model)
            if not model_instance:
                result["errors"].append("Chat model not available")
                return result
            
            # Use generation config or defaults
            gen_config = generation_config or {
                "temperature": 0.7,
                "max_new_tokens": 512,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
            
            response_text = model_instance.chat(messages, **gen_config)
            
            chat_end = time.time()
            chat_time = chat_end - chat_start
            
            if not response_text:
                result["errors"].append("Chat response generation failed")
                return result
            
            result["response_text"] = response_text
            logger.info(f"✓ Chat response: '{response_text[:100]}...' ({chat_time:.2f}s)")
            
            # Step 4: Content filtering (output)
            if not disable_content_filter:
                logger.info("🔒 Applying output content filter...")
                filter_result = self.content_filter.filter_output(response_text)
                if not filter_result.is_allowed:
                    result["errors"].append(f"Response filtered: {filter_result.reason}")
                    result["response_text"] = filter_result.filtered_content or "I apologize, but I cannot provide that response."
            
            # Step 5: Text-to-Speech
            logger.info("🔊 Converting response to speech...")
            tts_start = time.time()
            
            response_audio_bytes = self.tts_service.text_to_speech_bytes(
                text=result["response_text"],
                language=output_language,
                speaker_voice=speaker_voice,
                emotion=emotion,
                speed=speech_speed
            )
            
            tts_end = time.time()
            tts_time = tts_end - tts_start
            
            if not response_audio_bytes:
                result["errors"].append("Text-to-speech conversion failed")
                return result
            
            result["response_audio_bytes"] = response_audio_bytes
            logger.info(f"✓ TTS completed ({tts_time:.2f}s, {len(response_audio_bytes)} bytes)")
            
            # Step 6: Update session history
            if session_id:
                self._update_session(session_id, messages + [{"role": "assistant", "content": response_text}])
            
            # Calculate total metrics
            total_time = time.time() - start_time
            result["metrics"] = {
                "total_time_seconds": total_time,
                "stt_time_seconds": stt_time,
                "chat_time_seconds": chat_time,
                "tts_time_seconds": tts_time,
                "transcription_length": len(transcription),
                "response_length": len(result["response_text"]),
                "audio_size_bytes": len(response_audio_bytes),
                "processing_efficiency": {
                    "stt_chars_per_second": len(transcription) / stt_time if stt_time > 0 else 0,
                    "chat_tokens_per_second": len(result["response_text"].split()) / chat_time if chat_time > 0 else 0,
                    "tts_chars_per_second": len(result["response_text"]) / tts_time if tts_time > 0 else 0
                }
            }
            
            result["success"] = True
            logger.info(f"🎉 Voice-to-voice conversation completed successfully! ({total_time:.2f}s total)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice-to-voice conversation: {e}")
            result["errors"].append(str(e))
            result["metrics"]["total_time_seconds"] = time.time() - start_time
            return result
    
    def voice_to_voice_file(
        self,
        input_audio_path: str,
        output_audio_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Voice-to-voice conversation from file input.
        
        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path to save output audio (optional)
            **kwargs: Additional parameters for voice_to_voice_conversation
            
        Returns:
            dict: Conversation result including output audio path
        """
        try:
            # Read input audio file
            with open(input_audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Detect audio format from file extension
            audio_format = input_audio_path.split('.')[-1].lower() if '.' in input_audio_path else 'wav'
            
            # Process voice-to-voice
            result = self.voice_to_voice_conversation(
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                **kwargs
            )
            
            # Save output audio if path provided
            if result["success"] and result["response_audio_bytes"] and output_audio_path:
                with open(output_audio_path, 'wb') as f:
                    f.write(result["response_audio_bytes"])
                result["response_audio_path"] = output_audio_path
                logger.info(f"Output audio saved to: {output_audio_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in file-based voice-to-voice: {e}")
            return {
                "success": False,
                "errors": [str(e)],
                "metrics": {}
            }
    
    def _validate_models_loaded(self) -> bool:
        """Validate that all required models are loaded."""
        if not self.stt_model_loaded:
            logger.error("STT model not loaded")
            return False
        if not self.tts_model_loaded:
            logger.error("TTS model not loaded") 
            return False
        if not self.chat_model_loaded:
            logger.error("Chat model not loaded")
            return False
        return True
    
    def _update_session(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """Update session history."""
        self.active_sessions[session_id] = {
            "messages": messages,
            "last_updated": time.time()
        }
    
    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get conversation history for a session."""
        session = self.active_sessions.get(session_id)
        return session["messages"] if session else None
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def get_models_status(self) -> Dict[str, Any]:
        """Get status of all loaded models."""
        return {
            "stt_model": {
                "loaded": self.stt_model_loaded,
                "model_name": self.stt_service.get_loaded_model_name()
            },
            "tts_model": {
                "loaded": self.tts_model_loaded,
                "model_name": self.tts_service.get_loaded_model_name()
            },
            "chat_model": {
                "loaded": self.chat_model_loaded,
                "model_name": self.default_chat_model
            }
        }
    
    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        try:
            if self.stt_model_loaded:
                self.stt_service.unload_model()
                self.stt_model_loaded = False
                
            if self.tts_model_loaded:
                self.tts_service.unload_model()
                self.tts_model_loaded = False
                
            if self.chat_model_loaded:
                # Unload chat model through model manager
                if self.model_manager.is_model_loaded(self.default_chat_model):
                    self.model_manager.unload_model(self.default_chat_model)
                self.chat_model_loaded = False
            
            logger.info("All voice-to-voice models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "stt_service": self.stt_service.get_memory_stats(),
            "tts_service": self.tts_service.get_memory_stats(),
            "model_manager": self.model_manager.get_memory_usage()
        }
