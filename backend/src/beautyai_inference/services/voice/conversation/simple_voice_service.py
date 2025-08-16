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
from ....services.voice.utils.text_cleaning import sanitize_tts_text

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
            config: Optional configuration dictionary (deprecated, uses voice registry)
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Use centralized voice configuration
        from ....config.voice_config_loader import get_voice_config
        self.voice_config = get_voice_config()
        
        # Service configuration
        self.temp_dir = Path(tempfile.gettempdir()) / "beautyai_simple_voice"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Core services - will be initialized later
        self.transcription_service = None
        self.chat_service = None
        
        # Voice mappings from voice registry
        self.voice_mappings = self._setup_voice_mappings_from_registry()
        
        # Default settings from voice registry
        self.default_arabic_voice = self.voice_config.get_voice_id("arabic", "female")
        self.default_english_voice = self.voice_config.get_voice_id("english", "female")
        self.speech_rate = "+0%"
        self.speech_pitch = "+0Hz"
        
        # Audio configuration from registry
        self.audio_config = self.voice_config.get_audio_config()
        
        self.logger.info("SimpleVoiceService initialized with voice registry configuration")
    
    def _setup_voice_mappings_from_registry(self) -> Dict[str, VoiceMapping]:
        """Set up voice mappings from voice registry."""
        mappings = {}
        
        try:
            supported_languages = self.voice_config.get_supported_languages()
            
            for language in supported_languages:
                voice_types = self.voice_config.get_voice_types(language)
                
                for gender in voice_types:
                    voice_id = self.voice_config.get_voice_id(language, gender)
                    # Create mapping key
                    lang_code = "ar" if language == "arabic" else "en"
                    mapping_key = f"{lang_code}_{gender}"
                    
                    mappings[mapping_key] = VoiceMapping(
                        language=f"{lang_code}-SA" if lang_code == "ar" else "en-US",
                        gender=gender,
                        voice_id=voice_id,
                        display_name=f"{language.title()} {gender.title()}"
                    )
            
            self.logger.info(f"Loaded {len(mappings)} voice mappings from voice registry")
            return mappings
            
        except Exception as e:
            self.logger.error(f"Error loading voice mappings from registry: {e}")
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
            
            # Pre-load required models for faster response times
            await self._preload_required_models()
            
            # Test Edge TTS availability with configured voice
            await self._test_edge_tts()
            
            self.logger.info("SimpleVoiceService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleVoiceService: {e}")
            raise Exception(f"Service initialization failed: {e}")
    
    async def _preload_required_models(self) -> None:
        """Pre-load required models for voice processing to avoid delays."""
        try:
            self.logger.info("Pre-loading voice processing models...")
            
            # Pre-load transcription service with voice registry model
            if self.transcription_service is None:
                from beautyai_inference.services.voice.transcription.transformers_whisper_service import TransformersWhisperService
                self.transcription_service = TransformersWhisperService()
            
            # Use voice registry default STT model
            model_loaded = self.transcription_service.load_whisper_model()  # Uses voice registry default
            if not model_loaded:
                self.logger.warning("Failed to load voice registry STT model, will load on demand")
            else:
                stt_config = self.voice_config.get_stt_model_config()
                self.logger.info(f"✅ Voice registry STT model pre-loaded: {stt_config.model_id}")
            
            # Pre-load chat service with fastest model for 24/7 service
            if self.chat_service is None:
                from beautyai_inference.services.inference.chat_service import ChatService
                self.chat_service = ChatService()
                
                # Load the fastest model for persistent 24/7 service
                success = self.chat_service.load_default_model_from_config()  # This will load qwen3-unsloth-q4ks
                if success:
                    self.logger.info("✅ Fastest chat model (qwen3-unsloth-q4ks) pre-loaded for 24/7 service")
                else:
                    self.logger.warning("Failed to pre-load fastest chat model, will load on demand")
            
            self.logger.info("🚀 Voice processing models pre-loading completed")
            
        except Exception as e:
            self.logger.error(f"Error during model pre-loading: {e}")
            # Don't raise exception, allow graceful fallback to on-demand loading
    
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
    
    def _detect_language(self, text: str, fallback_language: str = "en") -> str:
        """
        Simple language detection for voice selection.
        
        Args:
            text: Input text to analyze
            fallback_language: Language to use if detection fails ("ar" or "en")
            
        Returns:
            Language code ('ar' or 'en')
        """
        # Simple heuristic: check for Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            # Use the fallback language instead of hardcoded English
            self.logger.info(f"No alphabetic characters found, using fallback language: {fallback_language}")
            return fallback_language
        
        arabic_ratio = arabic_chars / total_chars
        detected = 'ar' if arabic_ratio > 0.3 else 'en'
        self.logger.info(f"Language detection: {detected} (Arabic ratio: {arabic_ratio:.2f}, fallback: {fallback_language})")
        return detected
    
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
        audio_format: str = "wav",  # Add audio_format parameter
        chat_model: str = "qwen-3",
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        gender: str = "female"
    ) -> Dict[str, Any]:
        """
        Process voice message: transcribe audio -> generate response -> synthesize speech.
        
        Args:
            audio_data: Raw audio data in bytes
            audio_format: Format of the audio data (webm, wav, mp3, etc.)
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
            self.logger.info(f"Processing voice message: format={audio_format}, language={language}, gender={gender}")
            
            # Step 1: Save audio data to temporary file for processing (keeping original format info)
            audio_input_path = await self._save_audio_data(audio_data, audio_format)
            
            # Step 2: Transcribe audio to text with correct format and language specification
            transcribed_text = await self._transcribe_audio(audio_data, audio_format, language)
            if transcribed_text.startswith("Sorry"):
                # Handle transcription failure gracefully - use language-specific fallback
                logger.warning("Transcription failed, using fallback response")
                transcribed_text = "صوت غير واضح" if language == "ar" else "unclear audio"
                
            self.logger.info(f"Transcribed: {transcribed_text}")

            # NEW: If transcription is unclear, do NOT generate AI response to avoid infinite loop of filler phrase.
            unclear_markers = ["unclear audio", "صوت غير واضح", "unclear audio /no_think", "unclear audio /no_think".strip()]
            base_transcribed_lower = transcribed_text.lower()
            if any(m in base_transcribed_lower for m in ["unclear audio", "صوت غير واضح"]) or transcribed_text.startswith("unclear audio /no_think"):
                guidance = "أعد المحاولة وتحدث بوضوح خلال ثانيتين" if (language or 'ar') == 'ar' else "Please try again and speak clearly for 2 seconds"
                self.logger.info("Skipping AI generation due to unclear transcription; sending guidance only")
                # Return minimal result without TTS to prevent loop
                return {
                    "transcribed_text": transcribed_text,
                    "response_text": guidance,
                    "response_text_clean": guidance,
                    "audio_file_path": None,
                    "processing_time": time.time() - start_time,
                    "voice_used": None,
                    "language_detected": language or 'ar'
                }
            
            # Step 3: Detect language if not provided, but respect user's choice
            if language is None:
                detected_language = self._detect_language(transcribed_text, fallback_language="en")
                self.logger.info(f"Auto-detected language: {detected_language}")
            else:
                detected_language = language
                self.logger.info(f"Using specified language: {detected_language}")
            
            # Step 4: Generate AI response using chat model with language specification
            response_text = await self._generate_chat_response(transcribed_text, target_language=detected_language)
            if response_text.startswith("عذراً") or response_text.startswith("Sorry"):
                # Handle chat failure gracefully - provide language-appropriate default response
                logger.warning("Chat generation failed, using fallback response")
                if detected_language == "ar":
                    response_text = "أهلاً بك! كيف يمكنني مساعدتك اليوم؟"
                else:
                    response_text = "Hello! How can I help you today?"
                
            self.logger.info(f"AI Response: {response_text}")
            
            # Step 5: Select appropriate voice
            selected_voice = voice_id or self._select_voice(detected_language, gender)
            
            # Step 6: Synthesize speech using Edge TTS (WebM format for browser compatibility)
            audio_output_path = await self._synthesize_speech(response_text, selected_voice, output_format="webm")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up input file
            audio_input_path.unlink(missing_ok=True)
            
            result = {
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "response_text_clean": response_text,  # Already cleaned in _generate_chat_response
                "audio_file_path": str(audio_output_path),
                "processing_time": processing_time,
                "voice_used": selected_voice,
                "language_detected": detected_language  # Use detected_language instead of language
            }
            
            self.logger.info(f"Voice processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing voice message: {e}")
            raise Exception(f"Voice processing failed: {e}")
    
    async def _save_audio_data(self, audio_data: bytes, audio_format: str = "wav") -> Path:
        """Save audio data to a temporary file with appropriate extension."""
        # Use the correct file extension based on the actual audio format
        file_extension = audio_format.lower()
        if file_extension == "webm":
            file_extension = "webm"
        elif file_extension in ["wav", "wave"]:
            file_extension = "wav"
        elif file_extension in ["mp3", "mpeg"]:
            file_extension = "mp3"
        else:
            file_extension = "wav"  # Default fallback
            
        audio_file = self.temp_dir / f"input_{uuid.uuid4().hex}.{file_extension}"
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        return audio_file
    
    async def _transcribe_audio(self, audio_data: bytes, audio_format: str = "wav", language: Optional[str] = None) -> str:
        """
        Transcribes audio data using Whisper transcription service.
        
        Args:
            audio_data: Raw audio data in bytes format
            audio_format: Format of the audio data (webm, wav, mp3, etc.)
            language: Language for transcription ('ar', 'en', or None for auto-detect)
            
        Returns:
            Transcribed text from the audio with /no_think suffix added
        """
        try:
            # Initialize transcription service if needed (fallback for non-pre-loaded case)
            if self.transcription_service is None:
                from beautyai_inference.services.voice.transcription.transformers_whisper_service import TransformersWhisperService
                self.transcription_service = TransformersWhisperService()
                
                # Use voice registry model
                model_loaded = self.transcription_service.load_whisper_model()
                if not model_loaded:
                    logger.warning("Failed to load voice registry STT model")
                    return "Sorry, I couldn't understand the audio."
            
            # Use the transcription service with the correct audio format and language
            result = self.transcription_service.transcribe_audio_bytes(
                audio_data, 
                audio_format=audio_format,  # Use the actual format instead of config
                language=language  # Use the specified language instead of hardcoded "ar"
            )
            
            # Add /no_think to disable thinking mode for voice conversations
            if result and result.strip():
                transcribed_text = result.strip() + " /no_think"
                logger.info(f"Transcribed audio with /no_think: {transcribed_text}")
                return transcribed_text
            else:
                return "unclear audio /no_think"
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "unclear audio /no_think"
    
    def _clean_thinking_blocks(self, text: str) -> str:
        """
        Remove <think>...</think> blocks from the response text.
        
        Args:
            text: Raw response text that may contain thinking blocks
            
        Returns:
            Clean text with thinking blocks removed
        """
        import re
        
        # Remove <think>...</think> blocks (including multiline content)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Remove multiple newlines
        cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
        
        self.logger.debug(f"Cleaned thinking blocks from response: {len(text)} -> {len(cleaned_text)} chars")
        return cleaned_text
    
    async def _generate_chat_response(self, text: str, target_language: str = "auto") -> str:
        """
        Generates chat response using the actual chat service.
        
        Args:
            text: User input text to respond to
            target_language: Target language for response ("ar", "en", or "auto")
            
        Returns:
            Generated response text
        """
        try:
            # Initialize chat service if needed (fallback for non-pre-loaded case)
            if self.chat_service is None:
                from beautyai_inference.services.inference.chat_service import ChatService
                self.chat_service = ChatService()
                
                # Try to load persistent default model first
                success = self.chat_service.load_default_model_from_config()
                if not success:
                    logger.warning("Failed to load default model, trying registry alternatives...")
                    # Try alternative models from registry
                    alternative_models = ["qwen3-unsloth-q4ks", "qwen3-model", "deepseek-r1-qwen-14b-multilingual", "qwen3-official-q4km"]
                    for model in alternative_models:
                        if self.chat_service.load_model(model):
                            logger.info(f"Successfully loaded alternative model: {model}")
                            break
                    else:
                        logger.error("Failed to load any model")
                        if target_language == "ar":
                            return "عذراً، أواجه مشكلة تقنية حالياً. من فضلك حاول مرة أخرى."
                        else:
                            return "Sorry, I'm experiencing technical difficulties. Please try again."
            
            # Create optimized message for fast responses in simple voice mode
            if target_language == "ar":
                optimized_message = f"أجب بإيجاز بالعربية: {text}"
            else:
                optimized_message = f"Answer briefly in English: {text}"
            logger.info(f"Optimized message: {optimized_message[:100]}... (target_language: {target_language})")
            
            # Use the real chat service with specified target language
            # Use low-latency wrapper that enforces no thinking
            result = self.chat_service.chat_fast(
                message=optimized_message,
                max_length=128,
                language=target_language,
            )
            
            if result.get("success"):
                raw_response = result.get("response", "")
                # Clean thinking blocks from the response before TTS
                # Already disable thinking in chat_fast, but perform defensive clean & emoji strip
                clean_response = sanitize_tts_text(raw_response)
                logger.info(f"Generated chat response for {target_language}: {clean_response[:100]}...")
                return clean_response
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Chat service error: {error_msg}")
                if target_language == "ar":
                    return "عذراً، أواجه صعوبة في معالجة طلبك الآن. من فضلك حاول مرة أخرى."
                else:
                    return "Sorry, I'm having trouble processing your request. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            if target_language == "ar":
                return "عذراً، أواجه صعوبة في معالجة طلبك الآن. من فضلك حاول مرة أخرى."
            else:
                return "Sorry, I'm having trouble processing your request. Please try again."
    
    async def _synthesize_speech(self, text: str, voice_id: str, output_format: str = "wav") -> Path:
        """
        Synthesize speech using Edge TTS with flexible output format.
        
        Args:
            text: Text to synthesize
            voice_id: Edge TTS voice ID
            output_format: Output format ('wav' or 'webm')
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Create temporary WAV file first (Edge TTS always outputs WAV)
            temp_wav_file = self.temp_dir / f"temp_{uuid.uuid4().hex}.wav"
            
            # Create Edge TTS communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=self.speech_rate,
                pitch=self.speech_pitch
            )
            
            # Generate speech and save to temporary WAV file
            await communicate.save(str(temp_wav_file))
            
            if not temp_wav_file.exists():
                raise Exception(f"Failed to generate speech file: {temp_wav_file}")

            # Edge TTS current python client (edge_tts) streams MPEG (MP3) data even when we give a .wav
            # filename because the output format is hard-coded (e.g. audio-24khz-48kbitrate-mono-mp3).
            # Our downstream pipeline (and tests) expect a *real* RIFF/WAV file when output_format == 'wav'.
            # Detect MP3 by header (0xFF 0xF3 / 0xFF 0xFB) and convert to PCM WAV via ffmpeg.
            try:
                with open(temp_wav_file, "rb") as _f:
                    header = _f.read(4)
                # MPEG audio frame sync: 11 bits set => 0xFFE, often starting bytes 0xFF 0xF3 / 0xFF 0xFB / 0xFF 0xF2
                is_mpeg_frame = len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0 and not header.startswith(b"RIFF")
                if output_format.lower() == "wav" and is_mpeg_frame:
                    converted_path = self.temp_dir / f"converted_{uuid.uuid4().hex}.wav"
                    import subprocess
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_wav_file),
                        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(converted_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0 or not converted_path.exists():
                        self.logger.warning(
                            "TTS WAV conversion failed (rc=%s). stderr=%s", result.returncode, result.stderr.strip()[:300]
                        )
                    else:
                        try:
                            temp_wav_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                        temp_wav_file = converted_path
                        self.logger.info("Converted MP3-like TTS stream to PCM WAV: %s", converted_path)
            except Exception as conv_e:
                self.logger.warning("TTS format detection/conversion error (non-fatal): %s", conv_e)
            
            # If WebM output is requested, convert from WAV to WebM
            if output_format.lower() == "webm":
                output_file = self.temp_dir / f"output_{uuid.uuid4().hex}.webm"
                await self._convert_audio_to_webm(temp_wav_file, output_file)
                
                # Clean up temporary WAV file
                temp_wav_file.unlink(missing_ok=True)
                
                self.logger.info(f"Speech synthesized as WebM: {output_file}")
                return output_file
            else:
                # Return WAV file (rename for consistency)
                output_file = self.temp_dir / f"output_{uuid.uuid4().hex}.wav"
                temp_wav_file.rename(output_file)
                
                self.logger.info(f"Speech synthesized as WAV: {output_file}")
                return output_file
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise Exception(f"Failed to synthesize speech: {e}")
    
    async def _convert_audio_to_webm(self, input_wav_path: Path, output_webm_path: Path):
        """
        Convert WAV audio to WebM format with browser-compatible settings using ffmpeg.
        
        Args:
            input_wav_path: Path to input WAV file
            output_webm_path: Path to output WebM file
        """
        try:
            import subprocess
            
            # Use ffmpeg to convert WAV to WebM with Opus codec
            # Settings optimized for browser compatibility:
            # - 24kHz sample rate (good quality, not too large)
            # - 64kbps bitrate (good quality, small size)
            # - Opus codec (best browser support)
            # - Mono channel (smaller file size)
            cmd = [
                'ffmpeg',
                '-i', str(input_wav_path),          # Input WAV file
                '-c:a', 'libopus',                  # Use Opus codec
                '-b:a', '64k',                      # 64kbps bitrate
                '-ar', '24000',                     # 24kHz sample rate
                '-ac', '1',                         # Mono audio
                '-y',                               # Overwrite output file
                str(output_webm_path)               # Output WebM file
            ]
            
            # Run ffmpeg conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"ffmpeg conversion failed: {result.stderr}")
            
            if not output_webm_path.exists():
                raise Exception(f"Output file not created: {output_webm_path}")
            
            self.logger.info(f"Successfully converted audio: {input_wav_path} -> {output_webm_path}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg conversion timed out")
            raise Exception("Audio conversion timed out")
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            raise Exception(f"Failed to convert audio to WebM: {e}")
    
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
            
            # Synthesize speech (temporarily using WAV for stability)
            return await self._synthesize_speech(text, selected_voice, output_format="wav")
            
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
