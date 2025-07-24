"""
Voice-to-Voice Service for BeautyAI Framework.

Integrates Speech-to-Text, Large Language Model, and Text-to-Sp            # Initialize STT service
            logger.info(f"Initializing STT service with model: {stt_model}")
            results["stt"] = self.stt_service.load_whisper_model(stt_model)h 
to provide seamless voice conversations. This service connects the models
directly to minimize latency and improve performance.
"""

import logging
import time
import uuid
import json
import io
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO, Union

from ...base.base_service import BaseService
from ..transcription.audio_transcription_service import WhisperTranscriptionService
from ..synthesis.unified_tts_service import UnifiedTTSService
from ...inference.chat_service import ChatService
from ...inference.content_filter_service import ContentFilterService
from ....config.config_manager import AppConfig, ModelConfig
from ....config.configuration_manager import ConfigurationManager
from ....core.model_manager import ModelManager
from ....utils.language_detection import language_detector, suggest_response_language

logger = logging.getLogger(__name__)


class AdvancedVoiceConversationService(BaseService):
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
    
    def __init__(self, content_filter_strictness: str = "disabled"):
        """Initialize the voice-to-voice service."""
        super().__init__()
        
        # Core services
        self.stt_service = WhisperTranscriptionService()
        self.tts_service = UnifiedTTSService()
        self.chat_service = ChatService()
        self.content_filter = ContentFilterService(strictness_level=content_filter_strictness)
        self.model_manager = ModelManager()
        
        # Configuration manager for registry-based configuration
        self.config_manager = ConfigurationManager()
        self.service_config = self.config_manager.get_service_config("advanced_voice_service")
        
        # Service status
        self.services_loaded = {
            "stt": False,
            "tts": False,
            "chat": False,
            "content_filter": True  # Always available
        }
        
        # Default configurations - enhanced for multilingual support
        self.default_config = {
            "stt_model": "whisper-large-v3-turbo-arabic",
            "tts_model": "coqui-tts-multilingual",  # Changed to multilingual for auto-detection
            "chat_model": "qwen3-unsloth-q4ks",
            "language": "auto",  # Changed from "ar" to "auto" for automatic detection
            "speaker_voice": "female",
            "response_max_length": 128,  # Reduced from 256 to keep audio under 1MB for WebSocket
            "enable_content_filter": False  # DISABLED content filtering globally by default
        }
        
        # Output directory for temporary files
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/voice_to_voice_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.active_sessions = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "average_latency": 0.0,
            "success_rate": 0.0
        }
    
    @staticmethod
    def _remove_thinking_content(text: str) -> str:
        """
        Remove thinking blocks from the model response before TTS processing.
        
        This ensures that only the final user-facing response is converted to speech,
        not the internal reasoning process.
        
        Args:
            text: Raw model response that may contain <think>...</think> blocks
            
        Returns:
            str: Clean response text without thinking content
        """
        if not text:
            return text
        
        logger.debug(f"Original text: {text[:100]}...")
        
        # Remove thinking blocks using regex (case insensitive, multiline)
        # This handles both <think>...</think> and any malformed variations
        thinking_pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(thinking_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any standalone <think> or </think> tags that might remain
        cleaned_text = re.sub(r'</?think>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Also handle common variations that might appear
        cleaned_text = re.sub(r'</?thinking>', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'</?thought>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Multiple newlines to single
        cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)  # Leading whitespace on lines
        cleaned_text = cleaned_text.strip()
        
        logger.debug(f"Cleaned text: {cleaned_text[:100]}...")
        
        # If the result is empty after cleaning, return a default response
        if not cleaned_text or cleaned_text.isspace():
            logger.warning("Response was empty after removing thinking content")
            default_response = "أعتذر، لم أتمكن من تقديم إجابة واضحة. هل يمكنك إعادة صياغة سؤالك؟"
            logger.info(f"Using default response: {default_response}")
            return default_response
        
        # Log the final cleaned text for verification
        logger.debug(f"✅ Thinking content removed. Clean response length: {len(cleaned_text)} chars")
        
        return cleaned_text
        
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
        language: str = "auto"
    ) -> Dict[str, bool]:
        """
        Initialize all required models for voice-to-voice conversation.
        
        Args:
            stt_model: Speech-to-text model identifier
            chat_model: Chat model identifier
            tts_model: Text-to-speech model identifier
            language: Target language code (auto for automatic detection)
            
        Returns:
            Dict indicating success/failure for each model type
        """
        # Use defaults if not specified
        stt_model = stt_model or self.default_config["stt_model"]
        chat_model = chat_model or self.default_config["chat_model"]
        
        # Select appropriate TTS model based on language
        if not tts_model:
            if language == "auto":
                # Use multilingual model for auto-detection
                tts_model = self.config_manager.get_coqui_model_config().get("model_name", "coqui-tts-multilingual")
                # Convert to service-friendly name
                if "multilingual" in tts_model or "xtts_v2" in tts_model:
                    tts_model = "coqui-tts-multilingual"
            else:
                # Use language-specific model from service configuration
                service_config = self.config_manager.get_service_config("advanced_voice_service")
                supported_languages = service_config.get("supported_languages", ["ar", "en"])
                
                if language in supported_languages:
                    # Use the configured Coqui model (typically multilingual)
                    coqui_config = self.config_manager.get_coqui_model_config()
                    tts_model = "coqui-tts-multilingual"  # Standardized name for the service
                else:
                    # Fallback to multilingual for unsupported languages
                    tts_model = "coqui-tts-multilingual"
        
        logger.info(f"Initializing models for language: {language}")
        logger.info(f"Selected TTS model: {tts_model}")
        
        results = {}
        
        try:
            # Initialize STT Service
            logger.info(f"Initializing STT service with model: {stt_model}")
            results["stt"] = self.stt_service.load_whisper_model(stt_model)
            self.services_loaded["stt"] = results["stt"]
            
            # Initialize Chat Service
            logger.info(f"Initializing Chat service with model: {chat_model}")
            chat_success = self.chat_service.load_model(chat_model)
            results["chat"] = chat_success
            self.services_loaded["chat"] = chat_success
            
            # Initialize TTS Service with language-specific model
            logger.info(f"Initializing TTS service with model: {tts_model}")
            tts_success = self.tts_service.load_tts_model(tts_model, "coqui")  # Force Coqui engine
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
        enable_content_filter: bool = False,
        **generation_kwargs  # Additional generation parameters
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
                # Convert FilterResult object to dict for compatibility
                filter_dict = {
                    "is_safe": filter_result.is_allowed,
                    "reason": filter_result.filter_reason or "Content not allowed"
                }
                if not filter_dict["is_safe"]:
                    return {
                        "success": False,
                        "error": f"Content filtered: {filter_dict['reason']}",
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
            
            # Step 5: Clean response text for TTS (remove thinking content)
            clean_response_text = self._remove_thinking_content(response_text)
            logger.info(f"Clean response for TTS: {clean_response_text[:50]}...")
            
            # Step 6: Text-to-Speech
            logger.info(f"Starting TTS for session {session_id}")
            output_audio_path = self.output_dir / f"response_{session_id}_{int(time.time())}.wav"
            
            # Extra safety: Always clean text before TTS
            final_clean_text = self._remove_thinking_content(clean_response_text)
            logger.info(f"Final TTS text (first 100 chars): {final_clean_text[:100]}...")
            
            tts_result = self.tts_service.generate_speech(
                text=final_clean_text,
                output_path=str(output_audio_path),
                voice=speaker_voice,
                language=language
            )
            
            if not tts_result.get("success"):
                return {
                    "success": False,
                    "error": f"TTS failed: {tts_result.get('error', 'Could not generate audio')}",
                    "transcription": transcribed_text,
                    "response": response_text,
                    "audio_output": None,
                    "processing_time": time.time() - start_time
                }
            
            tts_audio_path = tts_result.get("output_path")
            
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
                "audio_output": tts_audio_path,  # Use the actual path returned by TTS
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

    def voice_to_voice_bytes(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        session_id: str = None,
        input_language: str = "auto",
        output_language: str = "auto",
        speaker_voice: str = "female",
        enable_content_filter: bool = False,
        content_filter_strictness: str = "disabled",
        thinking_mode: bool = False,
        generation_config: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process voice-to-voice conversation from audio bytes with automatic language detection.
        
        Args:
            audio_bytes: Input audio as bytes
            audio_format: Audio format (wav, mp3, etc.)
            session_id: Conversation session identifier
            input_language: Language of input audio ("auto" for detection)
            output_language: Language for output audio ("auto" for matching input)
            speaker_voice: Voice type for TTS output
            enable_content_filter: Whether to apply content filtering
            content_filter_strictness: Content filter level
            thinking_mode: Enable thinking mode for LLM
            generation_config: Additional generation parameters
            
        Returns:
            Dict containing conversation results and output paths
        """
        start_time = time.time()
        
        if generation_config is None:
            generation_config = {}
        
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
            # Save audio bytes to temporary file for STT processing
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                # Step 1: Speech-to-Text with automatic language detection
                logger.info(f"Starting STT for session {session_id}")
                
                # Determine input language for STT
                stt_language = input_language if input_language != "auto" else "ar"  # Default fallback
                
                transcription_result = self.stt_service.transcribe(
                    audio_file=temp_audio_path,
                    language=stt_language
                )
                
                if not transcription_result.get("success", False):
                    return {
                        "success": False,
                        "error": f"STT failed: {transcription_result.get('error', 'Unknown error')}",
                        "transcription": "",
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "STT failed"}
                    }
                
                transcribed_text = transcription_result["transcription"]
                logger.info(f"Transcription successful: {transcribed_text[:50]}...")
                
                # 🔍 Language Detection Phase
                detected_input_language = None
                response_language = None
                
                if input_language == "auto" or output_language == "auto":
                    # Detect language from transcribed text
                    detected_language, confidence = language_detector.detect_language(transcribed_text)
                    detected_input_language = detected_language
                    logger.info(f"🌍 Detected input language: {detected_language} (confidence: {confidence:.3f})")
                    
                    # Get conversation history for better language detection
                    conversation_history = self.get_session_history(session_id) or []
                    suggested_language = suggest_response_language(transcribed_text, conversation_history)
                    
                    # Determine final output language
                    if output_language == "auto":
                        response_language = suggested_language
                        logger.info(f"🌍 Auto-determined response language: {response_language}")
                    else:
                        response_language = output_language
                else:
                    # Use specified languages
                    detected_input_language = input_language
                    response_language = output_language
                    logger.info(f"🌍 Using specified languages: input={input_language}, output={output_language}")
                
                # 🚫 VOICE-TO-VOICE: Add \no_think prefix by default for faster responses
                # This ensures voice conversations are quick and don't include thinking content
                if not transcribed_text.startswith("/no_think") and "/think" not in transcribed_text.lower():
                    # Add \no_think prefix to disable thinking mode by default in voice conversations
                    transcribed_text = f"/no_think {transcribed_text}"
                    thinking_mode = False
                    logger.info("🚫 Added /no_think prefix for voice-to-voice speed optimization")
                elif "/no_think" in transcribed_text:
                    thinking_mode = False
                    # Remove /no_think from text but keep thinking_mode=False
                    transcribed_text = transcribed_text.replace("/no_think", "").strip()
                elif "/think" in transcribed_text.lower():
                    # User explicitly requested thinking mode - honor it
                    thinking_mode = True
                    transcribed_text = transcribed_text.replace("/think", "").strip()
                    logger.info("🧠 User explicitly requested thinking mode via /think command")
                
                # Step 2: Content filtering (if enabled)
                if enable_content_filter:
                    # Update content filter strictness
                    self.content_filter.strictness = content_filter_strictness
                    filter_result = self.content_filter.filter_content(transcribed_text)
                    # Convert FilterResult object to dict for compatibility
                    filter_dict = {
                        "is_safe": filter_result.is_allowed,
                        "reason": filter_result.filter_reason or "Content not allowed"
                    }
                    if not filter_dict["is_safe"]:
                        return {
                            "success": False,
                            "error": f"Content filtered: {filter_dict['reason']}",
                            "transcription": transcribed_text,
                            "response": None,
                            "audio_output": None,
                            "processing_time": time.time() - start_time
                        }
                
                # Step 3: Get conversation history for context
                conversation_history = self.get_session_history(session_id) or []
                
                # Step 4: Chat inference with enhanced parameters
                logger.info(f"Starting chat inference for session {session_id}")
                
                # 🚫 VOICE-TO-VOICE: Process transcribed text for thinking mode
                # By default, add \no_think prefix for faster voice responses
                processed_text = transcribed_text
                thinking_override = None
                
                if "/no_think" in transcribed_text.lower():
                    processed_text = transcribed_text.replace("/no_think", "").strip()
                    thinking_override = False
                    logger.info("User requested no thinking mode via /no_think command")
                elif "/think" in transcribed_text.lower():
                    processed_text = transcribed_text.replace("/think", "").strip()
                    thinking_override = True
                    logger.info("User requested thinking mode via /think command")
                else:
                    # 🚫 DEFAULT BEHAVIOR: Add \no_think for voice-to-voice speed
                    if not processed_text.startswith("/no_think"):
                        processed_text = f"/no_think {processed_text}"
                        thinking_override = False
                        logger.info("🚫 Added /no_think prefix by default for voice-to-voice speed optimization")
                
                # Determine final thinking mode
                final_thinking_mode = thinking_override if thinking_override is not None else thinking_mode
                
                # Prepare chat parameters with full generation config and language matching
                # Sanitize generation_config to remove None values that cause engine errors
                sanitized_generation_config = {}
                if generation_config:
                    sanitized_generation_config = {k: v for k, v in generation_config.items() if v is not None}
                
                # Voice-optimized defaults for better performance
                voice_chat_defaults = {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 128  # Shorter responses for voice conversations
                }
                
                # Apply defaults for missing parameters
                for key, default_value in voice_chat_defaults.items():
                    if key not in sanitized_generation_config:
                        sanitized_generation_config[key] = default_value
                
                chat_params = {
                    "message": processed_text,
                    "conversation_history": conversation_history,
                    "max_length": sanitized_generation_config.get("max_new_tokens", 128),
                    "language": response_language,  # Use detected/determined response language
                    "thinking_mode": final_thinking_mode,
                    **sanitized_generation_config  # Include sanitized generation parameters
                }
                
                logger.info(f"Chat parameters: thinking_mode={final_thinking_mode}, content_filter={enable_content_filter}, response_language={response_language}")
                logger.info(f"🔍 Debug - Full chat_params: {chat_params}")
                
                try:
                    logger.info("🚀 Calling chat service...")
                    chat_result = self.chat_service.chat(**chat_params)
                    logger.info(f"✅ Chat service returned: success={chat_result.get('success', False)}")
                except Exception as chat_error:
                    import traceback
                    logger.error(f"❌ Chat service error: {chat_error}")
                    logger.error(f"❌ Chat service traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Chat service failed: {str(chat_error)}",
                        "transcription": transcribed_text,
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "Chat service failed"}
                    }
                
                if not chat_result.get("success", False):
                    return {
                        "success": False,
                        "error": f"Chat failed: {chat_result.get('error', 'Unknown error')}",
                        "transcription": transcribed_text,
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "Chat failed"}
                    }
                
                response_text = chat_result["response"]
                logger.info(f"Chat response: {response_text[:50]}...")
                
                # Step 5: Clean response text for TTS (remove thinking content)
                clean_response_text = self._remove_thinking_content(response_text)
                logger.info(f"Clean response for TTS: {clean_response_text[:50]}...")
                
                # Step 6: Text-to-Speech with matching language
                logger.info(f"Starting TTS for session {session_id} in language: {response_language}")
                output_audio_path = self.output_dir / f"response_{session_id}_{int(time.time())}.wav"
                
                try:
                    # Extra safety: Always clean text before TTS
                    final_clean_text = self._remove_thinking_content(clean_response_text)
                    logger.info(f"Final TTS text (first 100 chars): {final_clean_text[:100]}...")
                    
                    # Generate audio using the enhanced generate_speech method
                    tts_result = self.tts_service.generate_speech(
                        text=final_clean_text,
                        output_path=str(output_audio_path),
                        voice=speaker_voice,
                        language=response_language  # Use the determined response language
                    )
                    
                    if not tts_result.get("success"):
                        logger.error(f"TTS generation failed: {tts_result.get('error', 'Unknown error')}")
                        return {
                            "success": False,
                            "error": f"TTS failed: {tts_result.get('error', 'Could not generate audio')}",
                            "transcription": transcribed_text,
                            "response_text": response_text,
                            "audio_output_path": None,
                            "audio_output_bytes": None,
                            "audio_output_format": "wav",
                            "session_id": session_id,
                            "processing_time": time.time() - start_time,
                            "models_used": {
                                "stt": "whisper-large-v3-turbo-arabic", 
                                "chat": "qwen3-unsloth-q4ks", 
                                "tts": self.tts_service.current_model or "unknown"
                            },
                            "metadata": {"error": "TTS generation failed"}
                        }
                    
                    tts_audio_path = tts_result.get("output_path")
                    
                    # Also generate audio bytes for direct WebSocket usage
                    logger.info(f"Generating TTS bytes for WebSocket transmission...")
                    tts_audio_bytes = self.tts_service.text_to_speech_bytes(
                        text=final_clean_text,
                        language=response_language,
                        speaker_voice=speaker_voice
                    )
                except Exception as tts_error:
                    logger.error(f"TTS service error: {tts_error}")
                    return {
                        "success": False,
                        "error": f"TTS service failed: {str(tts_error)}",
                        "transcription": transcribed_text,
                        "response_text": response_text,
                        "audio_output_path": None,
                        "audio_output_bytes": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "TTS service failed"}
                    }
                
                if not tts_audio_path and not tts_audio_bytes:
                    return {
                        "success": False,
                        "error": f"TTS failed: Could not generate audio",
                        "transcription": transcribed_text,
                        "response_text": response_text,
                        "audio_output_path": None,
                        "audio_output_bytes": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "TTS failed"}
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
                    "audio_output": tts_audio_path,  # Use the actual path returned by TTS
                    "processing_time": processing_time,
                    "metadata": {
                        "language": input_language,  # Fixed: use input_language instead of undefined language
                        "speaker_voice": speaker_voice,
                        "content_filtered": enable_content_filter,
                        "audio_duration": transcription_result.get("duration", 0.0)
                    }
                }
            
            finally:
                # Clean up temporary file
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
            
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

    def voice_to_voice_bytes(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        session_id: str = None,
        input_language: str = "auto",
        output_language: str = "auto",
        speaker_voice: str = "female",
        enable_content_filter: bool = False,
        content_filter_strictness: str = "disabled",
        thinking_mode: bool = False,
        generation_config: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process voice-to-voice conversation from audio bytes with automatic language detection.
        
        Args:
            audio_bytes: Input audio as bytes
            audio_format: Audio format (wav, mp3, etc.)
            session_id: Conversation session identifier
            input_language: Language of input audio ("auto" for detection)
            output_language: Language for output audio ("auto" for matching input)
            speaker_voice: Voice type for TTS output
            enable_content_filter: Whether to apply content filtering
            content_filter_strictness: Content filter level
            thinking_mode: Enable thinking mode for LLM
            generation_config: Additional generation parameters
            
        Returns:
            Dict containing conversation results and output paths
        """
        start_time = time.time()
        
        if generation_config is None:
            generation_config = {}
        
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
            # Save audio bytes to temporary file for STT processing
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                # Step 1: Speech-to-Text with automatic language detection
                logger.info(f"Starting STT for session {session_id}")
                
                # Determine input language for STT
                stt_language = input_language if input_language != "auto" else "ar"  # Default fallback
                
                transcription_result = self.stt_service.transcribe(
                    audio_file=temp_audio_path,
                    language=stt_language
                )
                
                if not transcription_result.get("success", False):
                    return {
                        "success": False,
                        "error": f"STT failed: {transcription_result.get('error', 'Unknown error')}",
                        "transcription": "",
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "STT failed"}
                    }
                
                transcribed_text = transcription_result["transcription"]
                logger.info(f"Transcription successful: {transcribed_text[:50]}...")
                
                # 🔍 Language Detection Phase
                detected_input_language = None
                response_language = None
                
                if input_language == "auto" or output_language == "auto":
                    # Detect language from transcribed text
                    detected_language, confidence = language_detector.detect_language(transcribed_text)
                    detected_input_language = detected_language
                    logger.info(f"🌍 Detected input language: {detected_language} (confidence: {confidence:.3f})")
                    
                    # Get conversation history for better language detection
                    conversation_history = self.get_session_history(session_id) or []
                    suggested_language = suggest_response_language(transcribed_text, conversation_history)
                    
                    # Determine final output language
                    if output_language == "auto":
                        response_language = suggested_language
                        logger.info(f"🌍 Auto-determined response language: {response_language}")
                    else:
                        response_language = output_language
                else:
                    # Use specified languages
                    detected_input_language = input_language
                    response_language = output_language
                    logger.info(f"🌍 Using specified languages: input={input_language}, output={output_language}")
                
                # 🚫 VOICE-TO-VOICE: Add \no_think prefix by default for faster responses
                # This ensures voice conversations are quick and don't include thinking content
                if not transcribed_text.startswith("/no_think") and "/think" not in transcribed_text.lower():
                    # Add \no_think prefix to disable thinking mode by default in voice conversations
                    transcribed_text = f"/no_think {transcribed_text}"
                    thinking_mode = False
                    logger.info("🚫 Added /no_think prefix for voice-to-voice speed optimization")
                elif "/no_think" in transcribed_text:
                    thinking_mode = False
                    # Remove /no_think from text but keep thinking_mode=False
                    transcribed_text = transcribed_text.replace("/no_think", "").strip()
                elif "/think" in transcribed_text.lower():
                    # User explicitly requested thinking mode - honor it
                    thinking_mode = True
                    transcribed_text = transcribed_text.replace("/think", "").strip()
                    logger.info("🧠 User explicitly requested thinking mode via /think command")
                
                # Step 2: Content filtering (if enabled)
                if enable_content_filter:
                    # Update content filter strictness
                    self.content_filter.strictness = content_filter_strictness
                    filter_result = self.content_filter.filter_content(transcribed_text)
                    # Convert FilterResult object to dict for compatibility
                    filter_dict = {
                        "is_safe": filter_result.is_allowed,
                        "reason": filter_result.filter_reason or "Content not allowed"
                    }
                    if not filter_dict["is_safe"]:
                        return {
                            "success": False,
                            "error": f"Content filtered: {filter_dict['reason']}",
                            "transcription": transcribed_text,
                            "response": None,
                            "audio_output": None,
                            "processing_time": time.time() - start_time
                        }
                
                # Step 3: Get conversation history for context
                conversation_history = self.get_session_history(session_id) or []
                
                # Step 4: Chat inference with enhanced parameters
                logger.info(f"Starting chat inference for session {session_id}")
                
                # 🚫 VOICE-TO-VOICE: Process transcribed text for thinking mode
                # By default, add \no_think prefix for faster voice responses
                processed_text = transcribed_text
                thinking_override = None
                
                if "/no_think" in transcribed_text.lower():
                    processed_text = transcribed_text.replace("/no_think", "").strip()
                    thinking_override = False
                    logger.info("User requested no thinking mode via /no_think command")
                elif "/think" in transcribed_text.lower():
                    processed_text = transcribed_text.replace("/think", "").strip()
                    thinking_override = True
                    logger.info("User requested thinking mode via /think command")
                else:
                    # 🚫 DEFAULT BEHAVIOR: Add \no_think for voice-to-voice speed
                    if not processed_text.startswith("/no_think"):
                        processed_text = f"/no_think {processed_text}"
                        thinking_override = False
                        logger.info("🚫 Added /no_think prefix by default for voice-to-voice speed optimization")
                
                # Determine final thinking mode
                final_thinking_mode = thinking_override if thinking_override is not None else thinking_mode
                
                # Prepare chat parameters with full generation config and language matching
                # Sanitize generation_config to remove None values that cause engine errors
                sanitized_generation_config = {}
                if generation_config:
                    sanitized_generation_config = {k: v for k, v in generation_config.items() if v is not None}
                
                # Voice-optimized defaults for better performance
                voice_chat_defaults = {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 128  # Shorter responses for voice conversations
                }
                
                # Apply defaults for missing parameters
                for key, default_value in voice_chat_defaults.items():
                    if key not in sanitized_generation_config:
                        sanitized_generation_config[key] = default_value
                
                chat_params = {
                    "message": processed_text,
                    "conversation_history": conversation_history,
                    "max_length": sanitized_generation_config.get("max_new_tokens", 128),
                    "language": response_language,  # Use detected/determined response language
                    "thinking_mode": final_thinking_mode,
                    **sanitized_generation_config  # Include sanitized generation parameters
                }
                
                logger.info(f"Chat parameters: thinking_mode={final_thinking_mode}, content_filter={enable_content_filter}, response_language={response_language}")
                logger.info(f"🔍 Debug - Full chat_params: {chat_params}")
                
                try:
                    logger.info("🚀 Calling chat service...")
                    chat_result = self.chat_service.chat(**chat_params)
                    logger.info(f"✅ Chat service returned: success={chat_result.get('success', False)}")
                except Exception as chat_error:
                    import traceback
                    logger.error(f"❌ Chat service error: {chat_error}")
                    logger.error(f"❌ Chat service traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Chat service failed: {str(chat_error)}",
                        "transcription": transcribed_text,
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "Chat service failed"}
                    }
                
                if not chat_result.get("success", False):
                    return {
                        "success": False,
                        "error": f"Chat failed: {chat_result.get('error', 'Unknown error')}",
                        "transcription": transcribed_text,
                        "response_text": "",
                        "audio_output_path": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "Chat failed"}
                    }
                
                response_text = chat_result["response"]
                logger.info(f"Chat response: {response_text[:50]}...")
                
                # Step 5: Clean response text for TTS (remove thinking content)
                clean_response_text = self._remove_thinking_content(response_text)
                logger.info(f"Clean response for TTS: {clean_response_text[:50]}...")
                
                # Step 6: Text-to-Speech with matching language
                logger.info(f"Starting TTS for session {session_id} in language: {response_language}")
                output_audio_path = self.output_dir / f"response_{session_id}_{int(time.time())}.wav"
                
                try:
                    # Extra safety: Always clean text before TTS
                    final_clean_text = self._remove_thinking_content(clean_response_text)
                    logger.info(f"Final TTS text (first 100 chars): {final_clean_text[:100]}...")
                    
                    # Generate audio using the enhanced generate_speech method
                    tts_result = self.tts_service.generate_speech(
                        text=final_clean_text,
                        output_path=str(output_audio_path),
                        voice=speaker_voice,
                        language=response_language  # Use the determined response language
                    )
                    
                    if not tts_result.get("success"):
                        logger.error(f"TTS generation failed: {tts_result.get('error', 'Unknown error')}")
                        return {
                            "success": False,
                            "error": f"TTS failed: {tts_result.get('error', 'Could not generate audio')}",
                            "transcription": transcribed_text,
                            "response_text": response_text,
                            "audio_output_path": None,
                            "audio_output_bytes": None,
                            "audio_output_format": "wav",
                            "session_id": session_id,
                            "processing_time": time.time() - start_time,
                            "models_used": {
                                "stt": "whisper-large-v3-turbo-arabic", 
                                "chat": "qwen3-unsloth-q4ks", 
                                "tts": self.tts_service.current_model or "unknown"
                            },
                            "metadata": {"error": "TTS generation failed"}
                        }
                    
                    tts_audio_path = tts_result.get("output_path")
                    
                    # Also generate audio bytes for direct WebSocket usage
                    logger.info(f"Generating TTS bytes for WebSocket transmission...")
                    tts_audio_bytes = self.tts_service.text_to_speech_bytes(
                        text=final_clean_text,
                        language=response_language,
                        speaker_voice=speaker_voice
                    )
                except Exception as tts_error:
                    logger.error(f"TTS service error: {tts_error}")
                    return {
                        "success": False,
                        "error": f"TTS service failed: {str(tts_error)}",
                        "transcription": transcribed_text,
                        "response_text": response_text,
                        "audio_output_path": None,
                        "audio_output_bytes": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "TTS service failed"}
                    }
                
                if not tts_audio_path and not tts_audio_bytes:
                    return {
                        "success": False,
                        "error": f"TTS failed: Could not generate audio",
                        "transcription": transcribed_text,
                        "response_text": response_text,
                        "audio_output_path": None,
                        "audio_output_bytes": None,
                        "audio_output_format": "wav",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "models_used": {
                            "stt": "whisper-large-v3-turbo-arabic", 
                            "chat": "qwen3-unsloth-q4ks", 
                            "tts": "coqui-tts-arabic"
                        },
                        "metadata": {"error": "TTS failed"}
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
                    "response": response_text,  # Consistent with WebSocket expectation
                    "audio_output": tts_audio_path,
                    "audio_output_path": tts_audio_path,  # Add both keys for compatibility
                    "audio_output_bytes": tts_audio_bytes,  # Add bytes for WebSocket direct transmission
                    "audio_output_format": "wav",  # Add format information
                    "processing_time": processing_time,
                    "detected_input_language": detected_input_language,
                    "response_language": response_language,
                    "language_auto_detected": input_language == "auto" or output_language == "auto",
                    "models_used": {  # Add models used information
                        "stt": getattr(self.stt_service, 'current_model_name', 'unknown'),
                        "chat": getattr(self.chat_service, 'current_model_name', 'unknown'),
                        "tts": getattr(self.tts_service, 'current_model_name', 'unknown')
                    },
                    "metadata": {
                        "input_language": detected_input_language or input_language,
                        "output_language": response_language,
                        "speaker_voice": speaker_voice,
                        "thinking_mode": final_thinking_mode,
                        "content_filter_applied": enable_content_filter,
                        "content_filter_strictness": content_filter_strictness,
                        "generation_config": sanitized_generation_config,
                        "language_detection_confidence": chat_result.get("language_confidence", 1.0)
                    }
                }
                
            finally:
                # Clean up temporary file
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Voice-to-voice conversation failed: {e}")
            import traceback
            logger.error(f"Voice-to-voice traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "response_text": "",
                "audio_output": None,
                "audio_output_path": None,
                "audio_output_format": None,
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
            enable_content_filter=config.get("enable_content_filter", True)
        )
        
        # Copy audio to specified output path if different
        if result.get("success") and result.get("audio_output") != output_audio_path:
            try:
                import shutil
                shutil.copy2(result["audio_output"], output_audio_path)
            except Exception as e:
                logger.warning(f"Could not copy to specified output path: {e}")
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
    
    async def voice_to_voice_async(
        self,
        audio_path: str,
        stt_model_name: str = None,
        tts_model_name: str = None,
        chat_model_name: str = None,
        session_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        input_language: str = "auto",
        output_language: str = "auto",
        speaker_voice: str = "female",
        emotion: str = "neutral",
        speech_speed: float = 1.0,
        audio_output_format: str = "wav",
        disable_content_filter: bool = True,
        content_filter_strictness: str = "disabled",
        thinking_mode: bool = False,
        preset: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async version of voice_to_voice for WebSocket support.
        
        This method provides the same functionality as voice_to_voice_bytes
        but is designed for async/await usage in WebSocket connections.
        
        Args:
            audio_path: Path to input audio file
            stt_model_name: Speech-to-text model name
            tts_model_name: Text-to-speech model name  
            chat_model_name: Chat model name
            session_id: Session ID for conversation continuity
            chat_history: Previous conversation messages
            input_language: Input audio language (auto-detect if "auto")
            output_language: Output audio language (auto-match if "auto")
            speaker_voice: TTS voice type (female, male, neutral)
            emotion: TTS emotion (neutral, happy, sad, professional)
            speech_speed: TTS speech speed multiplier
            audio_output_format: Output audio format (wav, mp3, ogg)
            disable_content_filter: Whether to disable content filtering
            content_filter_strictness: Content filter level
            thinking_mode: Whether to enable thinking mode
            preset: Generation preset name
            **generation_params: Additional LLM parameters
            
        Returns:
            Dict containing:
            - success: bool
            - transcription: str
            - response_text: str  
            - audio_output_path: str
            - audio_output_base64: bytes
            - audio_output_format: str
            - session_id: str
            - models_used: Dict[str, str]
            - processing_time: float
            - thinking_mode: bool
            - content_filter_applied: bool
            - input_language: str
            - output_language: str
            - error: Optional[str]
        """
        import asyncio
        
        try:
            # Run the synchronous voice_to_voice_bytes in a thread pool
            # to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # Read audio file to bytes for voice_to_voice_bytes method
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Determine audio format from file extension
            audio_format = Path(audio_path).suffix.lstrip('.').lower() or 'wav'
            
            # Prepare arguments for voice_to_voice_bytes
            kwargs_for_sync = {
                "audio_bytes": audio_bytes,
                "audio_format": audio_format,
                "session_id": session_id,
                "chat_history": chat_history,
                "input_language": input_language,
                "output_language": output_language,
                "speaker_voice": speaker_voice,
                "emotion": emotion,
                "speech_speed": speech_speed,
                "audio_output_format": audio_output_format,
                "enable_content_filter": not disable_content_filter,
                "content_filter_strictness": content_filter_strictness,
                "thinking_mode": thinking_mode,
                "preset": preset,
                "generation_config": generation_config,
                **kwargs  # Include any additional parameters
            }
            
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: self.voice_to_voice_bytes(**kwargs_for_sync)
            )
            
            # Ensure audio_output_base64 is set for WebSocket compatibility
            if result.get("success") and result.get("audio_output_bytes"):
                result["audio_output_base64"] = result["audio_output_bytes"]
            
            return result
            
        except Exception as e:
            logger.error(f"Async voice-to-voice processing failed: {e}")
            import traceback
            logger.error(f"Async voice-to-voice traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Async processing failed: {str(e)}",
                "transcription": "",
                "response_text": "",
                "audio_output_path": "",
                "audio_output_base64": None,
                "session_id": session_id or "",
                "processing_time": 0.0,
                "input_language": input_language,
                "output_language": output_language
            }

    def _validate_models_loaded(self) -> bool:
        """
        Validate that all required models are loaded.
        
        Returns:
            bool: True if all models are loaded, False otherwise
        """
        try:
            # Check if services are loaded
            stt_loaded = self.services_loaded.get("stt", False)
            chat_loaded = self.services_loaded.get("chat", False)
            tts_loaded = self.services_loaded.get("tts", False)
            
            # Additional validation - check if services actually have loaded models
            if stt_loaded and hasattr(self.stt_service, 'whisper_model'):
                stt_loaded = self.stt_service.whisper_model is not None
            
            if chat_loaded and hasattr(self.chat_service, 'model'):
                chat_loaded = self.chat_service.model is not None
                
            if tts_loaded and hasattr(self.tts_service, 'tts_model'):
                tts_loaded = self.tts_service.tts_model is not None
            
            all_loaded = stt_loaded and chat_loaded and tts_loaded
            
            if not all_loaded:
                logger.warning(f"Model validation failed - STT: {stt_loaded}, Chat: {chat_loaded}, TTS: {tts_loaded}")
            
            return all_loaded
            
        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return False

    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages or None if session not found
        """
        if not session_id:
            return None
            
        return self.active_sessions.get(session_id, {}).get("history", [])

    def update_session_history(self, session_id: str, user_message: str, assistant_response: str):
        """
        Update conversation history for a session.
        
        Args:
            session_id: Session identifier
            user_message: User's message to add
            assistant_response: Assistant's response to add
        """
        if not session_id:
            return
            
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {"history": []}
        
        # Add user message
        self.active_sessions[session_id]["history"].append({
            "role": "user",
            "content": user_message
        })
        
        # Add assistant response
        self.active_sessions[session_id]["history"].append({
            "role": "assistant", 
            "content": assistant_response
        })
        
        # Keep only last 20 messages to prevent memory bloat
        if len(self.active_sessions[session_id]["history"]) > 20:
            self.active_sessions[session_id]["history"] = self.active_sessions[session_id]["history"][-20:]

    def clear_session_history(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["history"] = []

    def _update_session(self, session_id: str, new_messages: List[Dict[str, str]]):
        """
        Update session with new messages.
        
        Args:
            session_id: Session identifier
            new_messages: List of messages to add
        """
        if not session_id:
            return
            
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {"history": []}
        
        # Add all new messages
        self.active_sessions[session_id]["history"].extend(new_messages)
        
        # Keep only last 20 messages to prevent memory bloat
        if len(self.active_sessions[session_id]["history"]) > 20:
            self.active_sessions[session_id]["history"] = self.active_sessions[session_id]["history"][-20:]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "stt_service": self.stt_service.get_memory_stats() if hasattr(self.stt_service, 'get_memory_stats') else {},
            "tts_service": self.tts_service.get_memory_stats() if hasattr(self.tts_service, 'get_memory_stats') else {},
            "model_manager": self.model_manager.get_memory_usage() if hasattr(self.model_manager, 'get_memory_usage') else {}
        }
