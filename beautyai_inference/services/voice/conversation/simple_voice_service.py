"""
Simple Voice Conversation Service using Edge TTS.

This service provides a lightweight, fast voice conversation implementation
using Microsoft Edge TTS for speech synthesis and Whisper for transcription.
Optimized for <2 second response times.

Author: BeautyAI Framework
Date: 2025-01-23
"""

import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import edge_tts

from ....config.configuration_manager import ConfigurationManager

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class VoiceMapping:
    """Voice mapping configuration for Edge TTS."""
    language: str
    gender: str
    voice_id: str
    display_name: str


class SimpleVoiceService:
    """
    Simple voice conversation service using Edge TTS.
    
    This service provides fast voice-to-voice conversation capabilities
    using Microsoft Edge TTS for speech synthesis and Whisper for transcription.
    Designed for minimal latency (<2 seconds) and Arabic/English support.
    
    Features:
    - Direct Edge TTS integration (no abstraction layers)
    - Arabic and English voice support
    - Optimized for speed and simplicity
    - Built-in audio processing
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Simple Voice Conversation Service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigurationManager()
        
        # Get service configuration from registry
        self.service_config = self.config_manager.get_service_config("simple_voice_service")
        
        # Service configuration
        self.temp_dir = Path(tempfile.gettempdir()) / "beautyai_simple_voice"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Core services - will be initialized later
        self.transcription_service = None
        self.chat_service = None
        
        # Voice mappings from configuration
        self.voice_mappings = self._setup_voice_mappings_from_config()
        
        # Default settings from configuration
        self.default_arabic_voice = self.config_manager.get_edge_tts_voice("arabic", "female")
        self.default_english_voice = self.config_manager.get_edge_tts_voice("english", "female")
        self.speech_rate = "+0%"
        self.speech_pitch = "+0Hz"
        
        self.logger.info("SimpleVoiceService initialized with registry configuration")
    
    def _setup_voice_mappings_from_config(self) -> Dict[str, VoiceMapping]:
        """Set up voice mappings from configuration manager."""
        mappings = {}
        
        try:
            # Get Edge TTS voices from configuration
            arabic_voices = self.config_manager.get_edge_tts_voices_for_language("arabic")
            english_voices = self.config_manager.get_edge_tts_voices_for_language("english")
            
            # Setup Arabic voices
            if "male" in arabic_voices:
                mappings["ar_male"] = VoiceMapping("ar-SA", "male", arabic_voices["male"], "Arabic Male")
            if "female" in arabic_voices:
                mappings["ar_female"] = VoiceMapping("ar-SA", "female", arabic_voices["female"], "Arabic Female")
            
            # Setup English voices
            if "male" in english_voices:
                mappings["en_male"] = VoiceMapping("en-US", "male", english_voices["male"], "English Male")
            if "female" in english_voices:
                mappings["en_female"] = VoiceMapping("en-US", "female", english_voices["female"], "English Female")
            
            # Add fallbacks from legacy setup if config is incomplete
            if not mappings:
                self.logger.warning("No voices found in configuration, using fallback mappings")
                return self._setup_fallback_voice_mappings()
            
            self.logger.info(f"Loaded {len(mappings)} voice mappings from configuration")
            return mappings
            
        except Exception as e:
            self.logger.error(f"Error loading voice mappings from config: {e}")
            return self._setup_fallback_voice_mappings()
    
    def _setup_fallback_voice_mappings(self) -> Dict[str, VoiceMapping]:
        """Set up fallback voice mappings if configuration fails."""
        return {
            # Arabic voices
            "ar_female": VoiceMapping("ar-SA", "female", "ar-SA-ZariyahNeural", "Zariyah (Arabic Female)"),
            "ar_male": VoiceMapping("ar-SA", "male", "ar-SA-HamedNeural", "Hamed (Arabic Male)"),
            
            # English voices  
            "en_female": VoiceMapping("en-US", "female", "en-US-AriaNeural", "Aria (English Female)"),
            "en_male": VoiceMapping("en-US", "male", "en-US-GuyNeural", "Guy (English Male)"),
        }
    
    async def initialize(self) -> None:
        """Initialize the service and load required models."""
        try:
            self.logger.info("Initializing SimpleVoiceService...")
            
            # Initialize transcription service - placeholder for now
            # self.transcription_service = WhisperTranscriptionService()
            self.logger.info("Transcription service placeholder - will be implemented")
            
            # Test Edge TTS availability with configured voice
            await self._test_edge_tts()
            
            self.logger.info("SimpleVoiceService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleVoiceService: {e}")
            raise Exception(f"Service initialization failed: {e}")
    
    async def _test_edge_tts(self) -> None:
        """Test Edge TTS functionality with configured voice."""
        try:
            # Test with a simple phrase using configured default voice
            communicate = edge_tts.Communicate("Test", self.default_arabic_voice)
            test_file = self.temp_dir / "test_edge_tts.wav"
            
            await communicate.save(str(test_file))
            
            if test_file.exists():
                test_file.unlink()  # Clean up
                self.logger.info(f"Edge TTS test successful with voice: {self.default_arabic_voice}")
            else:
                raise Exception("Edge TTS test failed - no output file generated")
                
        except Exception as e:
            self.logger.error(f"Edge TTS test failed: {e}")
            raise Exception(f"Edge TTS is not available: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Clean up transcription service
            if self.transcription_service:
                # Add cleanup if available
                pass
            
            # Clean up temporary files
            if self.temp_dir.exists():
                for file in self.temp_dir.iterdir():
                    try:
                        file.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete temp file {file}: {e}")
            
            self.logger.info("SimpleVoiceService cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_available_voices(self) -> Dict[str, Dict[str, str]]:
        """
        Get available voice mappings.
        
        Returns:
            Dictionary of available voices with their metadata
        """
        return {
            voice_key: {
                "language": mapping.language,
                "gender": mapping.gender,
                "voice_id": mapping.voice_id,
                "display_name": mapping.display_name
            }
            for voice_key, mapping in self.voice_mappings.items()
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for voice selection.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code ('ar' or 'en')
        """
        # Simple heuristic: check for Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return 'en'  # Default to English for non-alphabetic text
        
        arabic_ratio = arabic_chars / total_chars
        return 'ar' if arabic_ratio > 0.3 else 'en'
    
    def _select_voice(self, language: str = None, gender: str = "female") -> str:
        """
        Select appropriate voice based on language and gender.
        
        Args:
            language: Language code ('ar' or 'en'), auto-detected if None
            gender: Voice gender ('female' or 'male')
            
        Returns:
            Edge TTS voice ID
        """
        if language is None:
            language = 'ar'  # Default to Arabic
        
        voice_key = f"{language}_{gender}"
        
        if voice_key in self.voice_mappings:
            return self.voice_mappings[voice_key].voice_id
        
        # Fallback to default voices from configuration
        if language == 'ar':
            return self.default_arabic_voice
        else:
            return self.default_english_voice
    
    async def process_voice_message(
        self,
        audio_data: bytes,
        chat_model: str = "qwen-3",
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        gender: str = "female"
    ) -> Dict[str, Any]:
        """
        Process voice message: transcribe audio -> generate response -> synthesize speech.
        
        Args:
            audio_data: Raw audio data in bytes
            chat_model: Name of the chat model to use for response generation
            voice_id: Specific voice ID to use (overrides auto-selection)
            language: Target language ('ar' or 'en'), auto-detected if None
            gender: Voice gender preference ('female' or 'male')
            
        Returns:
            Dictionary containing:
                - transcribed_text: The transcribed user input
                - response_text: The AI generated response
                - audio_file_path: Path to the generated audio response
                - processing_time: Total processing time in seconds
                - voice_used: Voice ID that was used
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Processing voice message...")
            
            # Step 1: Save audio data to temporary file for processing
            audio_input_path = await self._save_audio_data(audio_data)
            
            # Step 2: Transcribe audio to text
            transcribed_text = await self._transcribe_audio(audio_data)
            self.logger.info(f"Transcribed: {transcribed_text}")
            
            # Step 3: Detect language if not provided
            if language is None:
                language = self._detect_language(transcribed_text)
            
            # Step 4: Generate AI response using chat model
            response_text = await self._generate_chat_response(transcribed_text)
            self.logger.info(f"AI Response: {response_text}")
            
            # Step 5: Select appropriate voice
            selected_voice = voice_id or self._select_voice(language, gender)
            
            # Step 6: Synthesize speech using Edge TTS
            audio_output_path = await self._synthesize_speech(response_text, selected_voice)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up input file
            audio_input_path.unlink(missing_ok=True)
            
            result = {
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "audio_file_path": str(audio_output_path),
                "processing_time": processing_time,
                "voice_used": selected_voice,
                "language_detected": language
            }
            
            self.logger.info(f"Voice processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing voice message: {e}")
            raise Exception(f"Voice processing failed: {e}")
    
    async def _save_audio_data(self, audio_data: bytes) -> Path:
        """Save audio data to a temporary file."""
        audio_file = self.temp_dir / f"input_{uuid.uuid4().hex}.wav"
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        return audio_file
    
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribes audio data using Whisper transcription service.
        
        Args:
            audio_data: Raw audio data in bytes format
            
        Returns:
            Transcribed text from the audio
        """
        try:
            # Initialize transcription service if needed
            if self.transcription_service is None:
                from beautyai_inference.services.voice.transcription.audio_transcription_service import WhisperTranscriptionService
                self.transcription_service = WhisperTranscriptionService()
                
                # Load the Whisper model
                model_loaded = self.transcription_service.load_whisper_model("whisper-large-v3-turbo-arabic")
                if not model_loaded:
                    logger.warning("Failed to load Arabic Whisper model, trying base model...")
                    if not self.transcription_service.load_whisper_model("whisper-base"):
                        logger.error("Failed to load any Whisper model")
                        return "Sorry, I couldn't understand the audio."
            
            # Use the real transcription service
            result = self.transcription_service.transcribe_audio_bytes(audio_data, audio_format="wav", language="ar")
            logger.info(f"Transcribed audio: {result}")
            return result if result else "Sorry, I couldn't understand the audio."
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "Sorry, I couldn't understand the audio."
    
    async def _generate_chat_response(self, text: str) -> str:
        """
        Generates chat response using the actual chat service.
        
        Args:
            text: User input text to respond to
            
        Returns:
            Generated response text
        """
        try:
            # Initialize chat service if needed
            if self.chat_service is None:
                from beautyai_inference.services.inference.chat_service import ChatService
                self.chat_service = ChatService()
                
                # Load default Arabic model
                success = self.chat_service.load_model("qwen3-unsloth-q4ks")  # Default model from registry
                if not success:
                    logger.warning("Failed to load default model, trying alternatives...")
                    # Try alternative models
                    alternative_models = ["qwen3-model", "deepseek-r1-qwen-14b-multilingual", "qwen3-official-q4km"]
                    for model in alternative_models:
                        if self.chat_service.load_model(model):
                            logger.info(f"Successfully loaded alternative model: {model}")
                            break
                    else:
                        logger.error("Failed to load any model")
                        return "I'm sorry, I'm currently unable to process your request due to a technical issue."
            
            # Use the real chat service
            result = self.chat_service.chat(
                message=text,
                max_length=256,
                language="auto"  # Auto-detect language
            )
            
            if result.get("success"):
                response = result.get("response", "")
                logger.info(f"Generated chat response: {response[:100]}...")
                return response
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Chat service error: {error_msg}")
                return "I apologize, but I'm having trouble processing your request right now. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    async def _synthesize_speech(self, text: str, voice_id: str) -> Path:
        """
        Synthesize speech using Edge TTS.
        
        Args:
            text: Text to synthesize
            voice_id: Edge TTS voice ID
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Create output file path
            output_file = self.temp_dir / f"output_{uuid.uuid4().hex}.wav"
            
            # Create Edge TTS communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=self.speech_rate,
                pitch=self.speech_pitch
            )
            
            # Generate speech and save to file
            await communicate.save(str(output_file))
            
            if not output_file.exists():
                raise Exception(f"Failed to generate speech file: {output_file}")
            
            self.logger.info(f"Speech synthesized: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise Exception(f"Failed to synthesize speech: {e}")
    
    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        gender: str = "female"
    ) -> Path:
        """
        Convert text to speech using Edge TTS.
        
        Args:
            text: Text to convert to speech
            voice_id: Specific voice ID (overrides auto-selection)
            language: Target language ('ar' or 'en'), auto-detected if None
            gender: Voice gender preference ('female' or 'male')
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Detect language if not provided
            if language is None:
                language = self._detect_language(text)
            
            # Select voice
            selected_voice = voice_id or self._select_voice(language, gender)
            
            # Synthesize speech
            return await self._synthesize_speech(text, selected_voice)
            
        except Exception as e:
            self.logger.error(f"Text-to-speech conversion failed: {e}")
            raise Exception(f"TTS conversion failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and health info."""
        temp_files = list(self.temp_dir.glob("*")) if self.temp_dir.exists() else []
        
        return {
            "service_name": "SimpleVoiceService",
            "edge_tts_available": True,  # We tested this during initialization
            "temp_directory": str(self.temp_dir),
            "temp_files_count": len(temp_files),
            "available_voices": len(self.voice_mappings),
            "default_arabic_voice": self.default_arabic_voice,
            "default_english_voice": self.default_english_voice,
            "service_config": self.service_config.get("performance_config", {})
        }
