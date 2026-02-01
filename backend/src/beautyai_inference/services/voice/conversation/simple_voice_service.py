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
import os
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - optional dependency
    resample_poly = None

import edge_tts
from ..utils.audio import to_float_mono_16k, ensure_sample_rate
from ....services.voice.utils.text_cleaning import sanitize_tts_text

from ....config.configuration_manager import ConfigurationManager
from ....api.schemas.debug_schemas import (
    PipelineDebugSummary, TranscriptionDebugData, LLMDebugData, TTSDebugData,
    DebugEvent, PipelineStage, DebugLevel, AudioDebugInfo
)

# Configure logger
LOGGER_NAME = "beautyai.voice.simple_voice_service"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.propagate = True


def _log_pcm_stats(
    logger: logging.Logger,
    label: str,
    sample_rate: Optional[int],
    channels: int,
    audio_bytes: bytes
) -> None:
    """Log PCM statistics for debugging STT sample-rate mismatches."""
    sr = ensure_sample_rate(sample_rate)
    samples = len(audio_bytes) // 2
    duration = samples / float(sr) if sr else 0.0

    if samples:
        int16_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = int16_audio.astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(float_audio))) if float_audio.size else 0.0
        rms = float(np.sqrt(np.mean(float_audio ** 2))) if float_audio.size else 0.0
    else:
        peak = 0.0
        rms = 0.0

    message = (
        f"[PCM] label={label} sr={sr} ch={channels} dtype=int16 "
        f"samples={samples} duration_s={duration:.3f} peak={peak:.4f} rms={rms:.4f}"
    )
    logger.info(message)
    diagnostics_logger = logging.getLogger("beautyai.voice.diagnostics")
    if diagnostics_logger is not logger:
        diagnostics_logger.info(message)


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, debug_mode: bool = False):
        """
        Initialize the Simple Voice Conversation Service.
        
        Args:
            config: Optional configuration dictionary (deprecated, uses voice registry)
            debug_mode: Enable debug mode for detailed metrics and logging
        """
        self.config = config or {}
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        
        # Debug configuration
        self.debug_mode = debug_mode
        self.debug_events: List[DebugEvent] = []
        self.debug_callback: Optional[Callable] = None
        self.current_debug_summary: Optional[PipelineDebugSummary] = None
        self.stage_timings: Dict[PipelineStage, float] = {}
        
        # Use centralized voice configuration
        from ....config.voice_config_loader import get_voice_config
        self.voice_config = get_voice_config()
        
        # Service configuration
        self.temp_dir = Path(tempfile.gettempdir()) / "beautyai_simple_voice"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Core services - will be initialized later
        self.transcription_service = None
        self.chat_service = None
        self.chat_default_model_name: Optional[str] = None
        self.chat_default_model_config: Optional[Any] = None
        self.chat_default_generation_config: Dict[str, Any] = {}
        self.chat_session_id: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []
        self.chat_last_language: Optional[str] = None
        
        # UPDATED: Reference to persistent Whisper engine from ModelManager
        self.persistent_whisper_engine = None
        
        # Persistent model manager reference (for enhanced performance)
        self.persistent_model_manager = None
        
        # Voice mappings from voice registry
        self.voice_mappings = self._setup_voice_mappings_from_registry()
        
        # Default settings from voice registry
        self.default_arabic_voice = self.voice_config.get_voice_id("ar", "female")
        self.default_english_voice = self.voice_config.get_voice_id("en", "female")
        self.speech_rate = "+0%"
        self.speech_pitch = "+0Hz"
        
        # Audio configuration from registry
        self.audio_config = self.voice_config.get_audio_config()

        # Streaming STT configuration
        self._stt_sample_rate_hint: int = 16000

        # Chat persona configuration (default to beauty/medical domain)
        self.chat_persona: str = os.getenv("VOICE_CHAT_PERSONA", "beauty").lower()
        self.disable_content_filter: bool = False
        
        self.logger.info("SimpleVoiceService initialized with voice registry configuration")
        
        if self.debug_mode:
            self.logger.info("🔍 Debug mode enabled for SimpleVoiceService")

    def _cache_chat_defaults(self) -> None:
        """Capture default chat model metadata for fast reuse in the voice pipeline."""
        if not self.chat_service:
            return

        try:
            default_name: Optional[str] = None
            default_config: Optional[Any] = None
            generation_config: Dict[str, Any] = {}

            if hasattr(self.chat_service, "get_default_model_info"):
                default_name, default_config = self.chat_service.get_default_model_info()

            if hasattr(self.chat_service, "get_default_generation_config"):
                generation_config = self.chat_service.get_default_generation_config()

            if not generation_config:
                generation_config = {
                    "max_new_tokens": 256,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "do_sample": True,
                    "thinking_mode": False
                }

            if default_name:
                self.logger.debug(
                    "Cached chat defaults: model=%s, max_new_tokens=%s",
                    default_name,
                    generation_config.get("max_new_tokens")
                )

            self.chat_default_model_name = default_name
            self.chat_default_model_config = default_config
            self.chat_default_generation_config = generation_config

        except Exception as exc:
            self.logger.warning(f"Failed to cache chat defaults: {exc}")

    def configure_chat_persona(self, persona: str, disable_content_filter: bool = False) -> None:
        """Configure chat persona (e.g., general vs beauty) for voice sessions."""
        if persona:
            self.chat_persona = persona.lower()
        self.disable_content_filter = disable_content_filter
        self._apply_chat_persona_overrides()

    def _apply_chat_persona_overrides(self) -> None:
        """Apply persona-specific prompt or safety overrides when chat service is ready."""
        if not self.chat_service:
            return

        applied_marker = getattr(self.chat_service, "_voice_persona_applied", None)
        if applied_marker == self.chat_persona:
            return

        prompt_builder = getattr(self.chat_service, "prompt_builder", None)

        if self.chat_persona == "general" and prompt_builder and hasattr(prompt_builder, "override_system_prompt"):
            prompt_builder.override_system_prompt(
                "en",
                "You are a friendly, conversational AI assistant. Respond in clear English "
                "with helpful, respectful answers across any topic."
            )
            prompt_builder.override_system_prompt(
                "ar",
                "أنت مساعد ذكاء اصطناعي ودود يقدم إجابات محترمة ومفيدة في أي موضوع باحترافية ووضوح."
            )

        setattr(self.chat_service, "_voice_persona_applied", self.chat_persona)

    def _prepare_generation_config(self, target_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Construct per-request generation settings for the voice chat flow.
        
        Language-aware token limiting:
        - Arabic: max 60 tokens (~150 chars) to fit Saudi XTTS 166 char limit
        - English: max 150 tokens for Edge TTS (no char limit)
        
        Args:
            target_language: Target response language (ar, en, auto)
        """
        base = dict(self.chat_default_generation_config or {})

        # Language-aware max_tokens for TTS compatibility
        # Arabic XTTS has 166 char limit per inference, ~2.5 chars/token average
        # English Edge TTS has no practical limit
        if target_language == "ar":
            # Arabic: 60 tokens * ~2.5 chars = ~150 chars (safe under 166 limit)
            voice_max_tokens = 60
            logger.debug(f"Arabic TTS limit applied: max_new_tokens={voice_max_tokens}")
        else:
            # English or auto: 150 tokens for natural responses
            voice_max_tokens = 150

        base.setdefault("max_new_tokens", voice_max_tokens)
        try:
            base["max_new_tokens"] = min(int(base["max_new_tokens"]), voice_max_tokens)
        except (TypeError, ValueError):
            base["max_new_tokens"] = voice_max_tokens
            
        base.setdefault("temperature", 0.3)
        base.setdefault("top_p", 0.95)
        base.setdefault("do_sample", True)

        # Thinking mode must stay disabled for voice
        base["thinking_mode"] = False
        
        # Check for dev mode to disable safeguards (for testing)
        if os.getenv('DISABLE_SYSTEM_PROMPT_SAFEGUARDS', '').lower() in ('1', 'true', 'yes'):
            base["disable_system_prompt_safeguards"] = True

        # Voice responses should not be too random
        try:
            base["temperature"] = float(base.get("temperature", 0.3))
        except (TypeError, ValueError):
            base["temperature"] = 0.3
        try:
            base["top_p"] = float(base.get("top_p", 0.95))
        except (TypeError, ValueError):
            base["top_p"] = 0.95
        base["top_p"] = min(max(base["top_p"], 0.0), 1.0)

        return base
    
    def set_debug_callback(self, callback: Callable[[DebugEvent], None]) -> None:
        """Set callback function for real-time debug events."""
        self.debug_callback = callback
        if self.debug_mode:
            self.logger.info("🔍 Debug callback registered")
    
    def _emit_debug_event(self, stage: PipelineStage, level: DebugLevel, message: str,
                          data: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        """Emit a debug event."""
        if not self.debug_mode:
            return

        event = DebugEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            stage=stage,
            level=level,
            message=message,
            data=data or {}
        )
        
        self.debug_events.append(event)
        
        # Keep only recent events to prevent memory issues
        if len(self.debug_events) > 1000:
            self.debug_events = self.debug_events[-800:]
        
        # Call debug callback if set
        if self.debug_callback:
            try:
                import asyncio
                import inspect
                
                # Check if callback is a coroutine function
                if inspect.iscoroutinefunction(self.debug_callback):
                    # Create a task to run async callback without awaiting
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self.debug_callback(event))
                    except RuntimeError:
                        # No event loop running, skip async callback
                        self.logger.debug("No event loop running, skipping async debug callback")
                else:
                    # Regular function call
                    self.debug_callback(event)
            except Exception as e:
                self.logger.warning(f"Debug callback error: {e}")
        
        # Log debug event
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"🔍 [{stage.upper()}] {message}")
    
    def _start_stage_timing(self, stage: PipelineStage) -> float:
        """Start timing for a pipeline stage."""
        start_time = time.time()
        if self.debug_mode:
            self._emit_debug_event(stage, DebugLevel.DEBUG, f"Starting {stage} stage")
        return start_time
    
    def _end_stage_timing(self, stage: PipelineStage, start_time: float) -> float:
        """End timing for a pipeline stage and return duration in ms."""
        duration_ms = (time.time() - start_time) * 1000
        self.stage_timings[stage] = duration_ms
        
        if self.debug_mode:
            self._emit_debug_event(
                stage, DebugLevel.INFO, 
                f"Completed {stage} stage", 
                {"duration_ms": duration_ms},
                duration_ms
            )
        
        return duration_ms
    
    def get_debug_summary(self, session_id: str = "", connection_id: str = "", turn_id: str = "") -> Optional[PipelineDebugSummary]:
        """Get current debug summary."""
        if not self.debug_mode or not self.current_debug_summary:
            return None
            
        # Update summary with current session info
        if session_id:
            self.current_debug_summary.session_id = session_id
        if connection_id:
            self.current_debug_summary.connection_id = connection_id
        if turn_id:
            self.current_debug_summary.turn_id = turn_id
            
        return self.current_debug_summary
    
    def get_recent_debug_events(self, limit: int = 100) -> List[DebugEvent]:
        """Get recent debug events."""
        if not self.debug_mode:
            return []
        return self.debug_events[-limit:]
    
    def clear_debug_data(self) -> None:
        """Clear accumulated debug data."""
        self.debug_events.clear()
        self.stage_timings.clear()
        self.current_debug_summary = None
        if self.debug_mode:
            self.logger.info("🔍 Debug data cleared")
    
    def set_persistent_model_manager(self, persistent_model_manager):
        """
        Set the persistent model manager for enhanced performance.
        
        Args:
            persistent_model_manager: PersistentModelManager instance
        """
        self.persistent_model_manager = persistent_model_manager
        self.logger.info("Persistent model manager connected to SimpleVoiceService")
    
    def _setup_voice_mappings_from_registry(self) -> Dict[str, VoiceMapping]:
        """Set up voice mappings from voice registry."""
        mappings = {}
        
        try:
            supported_languages = self.voice_config.get_supported_languages()
            
            for language in supported_languages:
                voice_types = self.voice_config.get_voice_types(language)
                
                for gender in voice_types:
                    voice_id = self.voice_config.get_voice_id(language, gender)
                    # Create mapping key - language is already "ar" or "en"
                    lang_code = language  # language is already in correct format ("ar" or "en")
                    mapping_key = f"{lang_code}_{gender}"
                    
                    # Get proper display name for language
                    display_lang = "Arabic" if lang_code == "ar" else "English"
                    
                    mappings[mapping_key] = VoiceMapping(
                        language=f"{lang_code}-SA" if lang_code == "ar" else "en-US",
                        gender=gender,
                        voice_id=voice_id,
                        display_name=f"{display_lang} {gender.title()}"
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
            
            # Test Edge TTS availability with configured voice (non-blocking)
            try:
                await self._test_edge_tts()
            except Exception as e:
                # Don't fail initialization if Edge TTS test fails
                # TTS will be tested again during actual usage
                self.logger.warning(f"Edge TTS test failed during initialization: {e}")
                self.logger.info("Continuing initialization - Edge TTS will be tested during first use")
            
            self.logger.info("SimpleVoiceService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleVoiceService: {e}")
            raise Exception(f"Service initialization failed: {e}")
    
    async def _preload_required_models(self) -> None:
        """Pre-load required models for voice processing to avoid delays."""
        try:
            self.logger.info("Pre-loading voice processing models...")
            
            # UPDATED: Try to use persistent model manager first for better performance
            if self.persistent_model_manager:
                self.logger.info("Using persistent model manager for optimized model access")
                
                # Get preloaded Whisper model from persistent manager
                try:
                    whisper_engine = self.persistent_model_manager.get_whisper_model()
                    if whisper_engine:
                        self.persistent_whisper_engine = whisper_engine
                        self.logger.info("✅ Using preloaded Whisper model from persistent manager")
                    else:
                        self.logger.warning("No preloaded Whisper model available from persistent manager")
                except Exception as e:
                    self.logger.warning(f"Failed to get Whisper model from persistent manager: {e}")
                
                # Get preloaded LLM from persistent manager
                try:
                    llm_engine = self.persistent_model_manager.get_llm_model()
                    if llm_engine and self.chat_service is None:
                        # Use the preloaded LLM for chat service
                        from beautyai_inference.services.inference.chat_service import ChatService
                        self.chat_service = ChatService()
                        self._cache_chat_defaults()
                        
                        # Set the preloaded model in chat service if supported
                        if hasattr(self.chat_service, 'set_model_engine'):
                            self.chat_service.set_model_engine(llm_engine)
                            self.logger.info("✅ Using preloaded LLM from persistent manager")
                        else:
                            # Fallback to regular loading
                            success = self.chat_service.load_default_model_from_config()
                            if success:
                                self.logger.info("✅ Loaded default chat model (fallback)")
                                self._cache_chat_defaults()
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM model from persistent manager: {e}")
            
            # Fallback to ModelManager if persistent manager not available or failed
            if self.persistent_whisper_engine is None:
                self.logger.info("Falling back to ModelManager for Whisper model loading")
                from ....core.model_manager import ModelManager
                model_manager = ModelManager()
                
                # Get persistent Whisper model from ModelManager
                whisper_engine = model_manager.get_streaming_whisper()  # Uses voice registry default
                if whisper_engine:
                    stt_config = self.voice_config.get_stt_model_config()
                    self.logger.info(f"✅ Persistent Whisper model loaded: {stt_config.model_id}")
                    
                    # Store reference to the persistent engine
                    self.persistent_whisper_engine = whisper_engine
                else:
                    self.logger.warning("Failed to load persistent Whisper model, will use factory fallback")
                    self.persistent_whisper_engine = None
            
            # Pre-load chat service with fastest model for 24/7 service (if not already done)
            if self.chat_service is None:
                from beautyai_inference.services.inference.chat_service import ChatService
                self.chat_service = ChatService()
                self._cache_chat_defaults()
                
                # Load the fastest model for persistent 24/7 service
                success = self.chat_service.load_default_model_from_config()  # This will load qwen3-unsloth-q4ks
                if success:
                    self.logger.info("✅ Fastest chat model (qwen3-unsloth-q4ks) pre-loaded for 24/7 service")
                    self._cache_chat_defaults()
                else:
                    self.logger.warning("Failed to pre-load fastest chat model, will load on demand")
            
            # Ensure defaults are cached even if chat_service was pre-existing
            if self.chat_service:
                self._cache_chat_defaults()

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
            
            # Add timeout and better error handling
            import asyncio
            
            # Run with timeout to prevent hanging
            await asyncio.wait_for(communicate.save(str(test_file)), timeout=10.0)
            
            # Give a brief moment for file system to sync
            await asyncio.sleep(0.1)
            
            if test_file.exists() and test_file.stat().st_size > 0:
                test_file.unlink()  # Clean up
                self.logger.info(f"Edge TTS test successful with voice: {self.default_arabic_voice}")
            else:
                # More specific error information
                if test_file.exists():
                    size = test_file.stat().st_size
                    self.logger.error(f"Edge TTS test failed - file exists but is empty (size: {size})")
                    test_file.unlink()  # Clean up empty file
                else:
                    self.logger.error(f"Edge TTS test failed - no output file generated at: {test_file}")
                raise Exception("Edge TTS test failed - no valid output file generated")
                
        except asyncio.TimeoutError:
            self.logger.error("Edge TTS test failed: timeout after 10 seconds")
            raise Exception("Edge TTS is not available: timeout during test")
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
    
    def _analyze_audio_debug_info(self, audio_data: bytes, audio_format: str) -> AudioDebugInfo:
        """Analyze audio data for debug information."""
        try:
            import struct
            import math
            
            size_bytes = len(audio_data)
            
            # Basic format detection
            channels = 1  # Default assumption
            sample_rate = 16000  # Default assumption
            bit_depth = 16  # Default assumption
            
            # Try to extract more detailed info for WAV files
            if audio_format.lower() == "wav" and len(audio_data) > 44:
                try:
                    # Parse WAV header
                    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                        # Extract sample rate (bytes 24-27)
                        sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                        # Extract channels (bytes 22-23)
                        channels = struct.unpack('<H', audio_data[22:24])[0]
                        # Extract bit depth (bytes 34-35)
                        bit_depth = struct.unpack('<H', audio_data[34:36])[0]
                except Exception as e:
                    self.logger.debug(f"WAV header parsing failed: {e}")
            
            # Estimate duration
            duration_ms = 0.0
            if audio_format.lower() == "wav" and sample_rate > 0:
                # For WAV: duration = data_size / (sample_rate * channels * bytes_per_sample)
                data_size = max(0, size_bytes - 44)  # Subtract header size
                bytes_per_sample = bit_depth // 8
                if bytes_per_sample > 0:
                    duration_ms = (data_size / (sample_rate * channels * bytes_per_sample)) * 1000
            else:
                # Rough estimation for other formats (WebM, MP3)
                # Assume ~64kbps encoding
                duration_ms = (size_bytes * 8) / (64 * 1000) * 1000
            
            # Basic RMS calculation for audio level (simplified)
            rms_level = None
            peak_level = None
            
            if audio_format.lower() == "wav" and len(audio_data) > 44:
                try:
                    # Extract audio samples from WAV data
                    audio_samples = audio_data[44:]  # Skip header
                    if len(audio_samples) >= 2:
                        # Convert to 16-bit samples
                        sample_count = len(audio_samples) // 2
                        if sample_count > 0:
                            samples = struct.unpack(f'<{sample_count}h', audio_samples[:sample_count*2])
                            
                            # Calculate RMS and peak
                            sum_squares = sum(s*s for s in samples)
                            rms_level = math.sqrt(sum_squares / len(samples)) / 32768.0  # Normalize to 0-1
                            peak_level = max(abs(s) for s in samples) / 32768.0  # Normalize to 0-1
                except Exception as e:
                    self.logger.debug(f"Audio level calculation failed: {e}")
            
            return AudioDebugInfo(
                format=audio_format,
                duration_ms=duration_ms,
                size_bytes=size_bytes,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                rms_level=rms_level,
                peak_level=peak_level,
                conversion_applied=False,
                normalization_applied=False
            )
            
        except Exception as e:
            self.logger.warning(f"Audio analysis failed: {e}")
            return AudioDebugInfo(
                format=audio_format,
                duration_ms=0.0,
                size_bytes=len(audio_data),
                sample_rate=16000,
                channels=1
            )
    
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
    
    async def process_transcription(
        self,
        audio_data: bytes,
        audio_format: str = "pcm",
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Legacy API shim that routes to process_voice_message."""
        metadata = metadata or {}
        return await self.process_voice_message(
            audio_data=audio_data,
            audio_format=metadata.get("audio_format", audio_format),
            chat_model=metadata.get("chat_model", "qwen-3"),
            voice_id=metadata.get("voice_id"),
            language=language or metadata.get("language"),
            gender=metadata.get("gender", "female"),
            conversation_context=metadata.get("conversation_context"),
            debug_context=metadata.get("debug_context"),
        )

    async def process_voice_message(
        self,
        audio_data: bytes,
        audio_format: str = "wav",  # Add audio_format parameter
        chat_model: str = "qwen-3",
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        gender: str = "female",
        conversation_context: Optional[str] = None,  # Add conversation context
        debug_context: Optional[Dict[str, Any]] = None  # Add debug context
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
            conversation_context: Previous conversation context for better responses
            
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
        
        # Initialize debug tracking
        if self.debug_mode:
            self.current_debug_summary = PipelineDebugSummary(
                total_processing_time_ms=0.0,
                success=False,
                debug_events=[],
                stage_timings={},
                completed_stages=[]
            )
            
            self._emit_debug_event(
                PipelineStage.UPLOAD, 
                DebugLevel.INFO, 
                f"Starting voice message processing",
                {
                    "audio_format": audio_format,
                    "audio_size_bytes": len(audio_data),
                    "language": language,
                    "gender": gender,
                    "chat_model": chat_model
                }
            )
        
        try:
            self.logger.info(f"Processing voice message: format={audio_format}, language={language}, gender={gender}")
            
            # Step 1: Save audio data to temporary file for processing (keeping original format info)
            audio_input_path = await self._save_audio_data(audio_data, audio_format)
            
            # Step 2: Transcribe audio to text with correct format and language specification
            stt_start_time = self._start_stage_timing(PipelineStage.STT)
            
            # Analyze audio for debug info
            audio_debug_info = None
            if self.debug_mode:
                audio_debug_info = self._analyze_audio_debug_info(audio_data, audio_format)
            
            transcribed_text = await self._transcribe_audio(audio_data, audio_format, language)
            stt_duration_ms = self._end_stage_timing(PipelineStage.STT, stt_start_time)
            
            # Create STT debug data
            if self.debug_mode:
                transcription_quality = "unclear" if any(marker in transcribed_text.lower() for marker in ["unclear audio", "صوت غير واضح"]) else "ok"
                
                stt_debug = TranscriptionDebugData(
                    transcribed_text=transcribed_text,
                    language_detected=None,  # Will be set later
                    confidence_score=None,  # Could be added if available
                    processing_time_ms=stt_duration_ms,
                    model_used="whisper-large-v3-turbo",  # Default model
                    audio_duration_ms=audio_debug_info.duration_ms if audio_debug_info else 0.0,
                    audio_format=audio_format,
                    audio_info=audio_debug_info,
                    errors=[],
                    warnings=[]
                )
                
                self.current_debug_summary.transcription_data = stt_debug
                self.current_debug_summary.completed_stages.append(PipelineStage.STT)
            
            if transcribed_text.startswith("Sorry"):
                # Handle transcription failure gracefully - use language-specific fallback
                logger.warning("Transcription failed, using fallback response")
                transcribed_text = "صوت غير واضح" if language == "ar" else "unclear audio"
                
                if self.debug_mode:
                    self._emit_debug_event(
                        PipelineStage.STT, 
                        DebugLevel.WARNING, 
                        "Transcription failed, using fallback"
                    )
                
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
            
            # Step 4: Generate AI response using chat model with language specification and context
            llm_start_time = self._start_stage_timing(PipelineStage.LLM)
            
            response_text = await self._generate_chat_response(
                transcribed_text, 
                target_language=detected_language,
                conversation_context=conversation_context
            )
            
            llm_duration_ms = self._end_stage_timing(PipelineStage.LLM, llm_start_time)
            
            # Create LLM debug data
            if self.debug_mode:
                # Calculate token estimates (rough)
                input_tokens = len(transcribed_text.split()) + (len(conversation_context.split()) if conversation_context else 0)
                output_tokens = len(response_text.split())
                tokens_per_second = output_tokens / (llm_duration_ms / 1000) if llm_duration_ms > 0 else 0
                
                llm_debug = LLMDebugData(
                    response_text=response_text,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    processing_time_ms=llm_duration_ms,
                    model_used=chat_model,
                    temperature=0.3,  # Default from voice service
                    thinking_mode=False,  # Disabled for voice
                    errors=[],
                    warnings=[]
                )
                
                self.current_debug_summary.llm_data = llm_debug
                self.current_debug_summary.completed_stages.append(PipelineStage.LLM)
            if response_text.startswith("عذراً") or response_text.startswith("Sorry"):
                # Handle chat failure gracefully - provide language-appropriate default response
                logger.warning("Chat generation failed, using fallback response")
                if detected_language == "ar":
                    response_text = sanitize_tts_text("أهلاً بك! كيف يمكنني مساعدتك اليوم؟")
                else:
                    response_text = sanitize_tts_text("Hello! How can I help you today?")
            else:
                response_text = sanitize_tts_text(response_text)
                
            self.logger.info(f"AI Response: {response_text}")
            
            # Step 5: Select appropriate voice
            selected_voice = voice_id or self._select_voice(detected_language, gender)
            
            # Step 6: Synthesize speech using Edge TTS (WebM format for browser compatibility)
            tts_start_time = self._start_stage_timing(PipelineStage.TTS)
            
            audio_output_path = await self._synthesize_speech(response_text, selected_voice, output_format="webm")
            
            tts_duration_ms = self._end_stage_timing(PipelineStage.TTS, tts_start_time)
            
            # Create TTS debug data
            if self.debug_mode:
                audio_size_bytes = None
                audio_duration_ms = None
                
                if audio_output_path and audio_output_path.exists():
                    audio_size_bytes = audio_output_path.stat().st_size
                    # Rough duration estimate for WebM (assume ~64kbps)
                    audio_duration_ms = (audio_size_bytes * 8) / (64 * 1000) * 1000
                
                tts_debug = TTSDebugData(
                    audio_length_ms=audio_duration_ms,
                    voice_used=selected_voice,
                    processing_time_ms=tts_duration_ms,
                    output_format="webm",
                    text_length=len(response_text),
                    speech_rate="medium",  # Default
                    errors=[],
                    warnings=[]
                )
                
                self.current_debug_summary.tts_data = tts_debug
                self.current_debug_summary.completed_stages.append(PipelineStage.TTS)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up input file
            audio_input_path.unlink(missing_ok=True)
            
            # Finalize debug summary
            if self.debug_mode:
                total_processing_time_ms = (time.time() - start_time) * 1000
                self.current_debug_summary.total_processing_time_ms = total_processing_time_ms
                self.current_debug_summary.stage_timings = self.stage_timings.copy()
                self.current_debug_summary.success = True
                self.current_debug_summary.completed_stages.append(PipelineStage.COMPLETE)
                
                # Update detected language in STT debug if available
                if self.current_debug_summary.transcription_data:
                    self.current_debug_summary.transcription_data.language_detected = detected_language
                
                self._emit_debug_event(
                    PipelineStage.COMPLETE, 
                    DebugLevel.INFO, 
                    "Voice processing completed successfully",
                    {
                        "total_time_ms": total_processing_time_ms,
                        "stt_time_ms": self.stage_timings.get(PipelineStage.STT, 0),
                        "llm_time_ms": self.stage_timings.get(PipelineStage.LLM, 0), 
                        "tts_time_ms": self.stage_timings.get(PipelineStage.TTS, 0),
                        "transcribed_text": transcribed_text,
                        "response_text": response_text,
                        "voice_used": selected_voice,
                        "language_detected": detected_language
                    },
                    total_processing_time_ms
                )
            
            result = {
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "response_text_clean": response_text,
                "audio_file_path": str(audio_output_path),
                "processing_time": processing_time,
                "voice_used": selected_voice,
                "language_detected": detected_language,  # Use detected_language instead of language
                "chat_session_id": self.chat_session_id,
                "debug_summary": self.current_debug_summary if self.debug_mode else None
            }
            
            self.logger.info(f"Voice processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle debug error tracking
            if self.debug_mode and self.current_debug_summary:
                total_processing_time_ms = (time.time() - start_time) * 1000
                self.current_debug_summary.total_processing_time_ms = total_processing_time_ms
                self.current_debug_summary.stage_timings = self.stage_timings.copy()
                self.current_debug_summary.success = False
                self.current_debug_summary.error_message = str(e)
                
                # Log the error for debugging
                self.logger.error(f"Pipeline processing failed: {str(e)}")
                
                self._emit_debug_event(
                    self.current_debug_summary.failed_stage or PipelineStage.UPLOAD,
                    DebugLevel.ERROR,
                    f"Voice processing failed: {str(e)}",
                    {"error": str(e), "total_time_ms": total_processing_time_ms}
                )
            
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
        elif file_extension == "pcm":
            # For raw PCM data, we need to create a proper WAV file with header
            return await self._save_pcm_as_wav(audio_data)
        else:
            file_extension = "wav"  # Default fallback
            
        audio_file = self.temp_dir / f"input_{uuid.uuid4().hex}.{file_extension}"
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        return audio_file
    
    async def _save_pcm_as_wav(self, pcm_data: bytes) -> Path:
        """
        Convert raw PCM data to WAV file with proper header.
        
        This handles PCM data from client-side audio decoding (browser Web Audio API)
        which typically produces 16-bit mono samples at 16kHz.
        
        Args:
            pcm_data: Raw PCM samples as bytes (16-bit signed integers)
            
        Returns:
            Path to created WAV file
        """
        import struct
        
        # Assume 16-bit mono samples at 16kHz (common for speech)
        sample_rate = 16000
        num_channels = 1
        bytes_per_sample = 2  # 16-bit
        
        # Calculate WAV file parameters
        num_samples = len(pcm_data) // bytes_per_sample
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        
        # Create WAV file with proper header
        wav_file = self.temp_dir / f"pcm_input_{uuid.uuid4().hex}.wav"
        
        with open(wav_file, 'wb') as f:
            # WAV header
            f.write(b'RIFF')  # ChunkID
            f.write(struct.pack('<I', 36 + len(pcm_data)))  # ChunkSize
            f.write(b'WAVE')  # Format

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        audio_format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Transcribe raw audio bytes for WebRTC adapter compatibility."""
        try:
            effective_format = audio_format or "pcm"
            sample_rate_hint = None
            if metadata:
                sample_rate_hint = metadata.get("sample_rate") or metadata.get("sample_rate_hz")
            transcript = await self._transcribe_audio(
                audio_data,
                audio_format=effective_format,
                language=language,
                sample_rate_hint=sample_rate_hint,
                metadata=metadata
            )
            return {
                "success": True,
                "transcription": transcript,
                "language": language
            }
        except Exception as e:
            self.logger.error(f"Transcription error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_chat_response(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        language: Optional[str] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate chat response compatible with WebRTC adapter expectations."""
        try:
            target_language = language or self._detect_language(
                user_message,
                fallback_language="en"
            )
            response_text = await self._generate_chat_response(
                user_message,
                target_language=target_language,
                conversation_context=conversation_context
            )
            clean_response = sanitize_tts_text(response_text)
            response_language = self.chat_last_language or target_language
            return {
                "success": True,
                "response": clean_response,
                "language": response_language,
                "session_id": self.chat_session_id
            }
        except Exception as e:
            self.logger.error(f"Chat generation error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def synthesize_speech(
        self,
        text: str,
        language: Optional[str] = None,
        gender: str = "female"
    ) -> Dict[str, Any]:
        """Synthesize speech and return raw audio bytes for WebRTC adapter."""
        try:
            voice_language = language or self._detect_language(
                text,
                fallback_language="en"
            )
            voice_id = self._select_voice(voice_language, gender)
            audio_path = await self._synthesize_speech(
                text,
                voice_id,
                output_format="wav"
            )

            audio_bytes = audio_path.read_bytes()

            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                self.logger.debug(
                    "Failed to remove temporary TTS file: %s",
                    audio_path
                )

            return {
                "success": True,
                "audio_data": audio_bytes,
                "audio_format": "wav",
                "voice_id": voice_id
            }
        except Exception as e:
            self.logger.error(f"Speech synthesis error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

            
            # fmt sub-chunk
            f.write(b'fmt ')  # Subchunk1ID
            f.write(struct.pack('<I', 16))  # Subchunk1Size (PCM)
            f.write(struct.pack('<H', 1))   # AudioFormat (PCM)
            f.write(struct.pack('<H', num_channels))  # NumChannels
            f.write(struct.pack('<I', sample_rate))   # SampleRate
            f.write(struct.pack('<I', byte_rate))     # ByteRate
            f.write(struct.pack('<H', block_align))   # BlockAlign
            f.write(struct.pack('<H', 16))  # BitsPerSample
            
            # data sub-chunk
            f.write(b'data')  # Subchunk2ID
            f.write(struct.pack('<I', len(pcm_data)))  # Subchunk2Size
            f.write(pcm_data)  # PCM samples
        
        logger.info(f"🎵 Converted raw PCM data ({len(pcm_data)} bytes) to WAV file: {wav_file}")
        return wav_file
    
    def _prepare_audio_for_stt(
        self,
        audio_bytes: bytes,
        sample_rate_hint: Optional[int] = None,
        target_sample_rate: int = 16000,
        silence_threshold_db: float = -40.0,
        silence_padding_ms: int = 100
    ) -> Tuple[np.ndarray, int, float, float]:
        """Convert PCM bytes to normalized float audio ready for STT."""
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32), target_sample_rate, 0.0, 0.0

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        except ValueError:
            self.logger.warning("[STT] Unable to interpret audio bytes as int16 PCM; skipping preprocessing")
            return np.zeros(0, dtype=np.float32), target_sample_rate, 0.0, 0.0

        if audio_int16.size == 0:
            return np.zeros(0, dtype=np.float32), target_sample_rate, 0.0, 0.0

        sample_rate = sample_rate_hint or getattr(self, "_stt_sample_rate_hint", target_sample_rate)
        normalized_audio, normalized_rate = to_float_mono_16k(audio_int16, sample_rate)
        sample_rate = normalized_rate or target_sample_rate

        trimmed_audio = self._trim_silence(
            normalized_audio,
            sample_rate=sample_rate,
            threshold_db=silence_threshold_db,
            padding_ms=silence_padding_ms
        )

        peak_level = float(np.max(np.abs(trimmed_audio))) if trimmed_audio.size else 0.0
        rms_level = float(np.sqrt(np.mean(trimmed_audio ** 2))) if trimmed_audio.size else 0.0

        return trimmed_audio.astype(np.float32), sample_rate, peak_level, rms_level

    def _resample_audio(self, audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using polyphase filtering when available."""
        if audio.size == 0 or source_sr == target_sr:
            return audio

        if resample_poly is not None:
            try:
                from math import gcd

                ratio_gcd = gcd(source_sr, target_sr)
                up = target_sr // ratio_gcd
                down = source_sr // ratio_gcd
                return resample_poly(audio, up, down)
            except Exception as exc:  # pragma: no cover - fallback path
                self.logger.debug(f"[STT] resample_poly failed ({exc}), falling back to linear interpolation")

        # Fallback linear interpolation
        if source_sr <= 0 or target_sr <= 0:
            return audio

        duration = audio.size / float(source_sr)
        target_length = max(int(duration * target_sr), 1)
        source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio)

    def _trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0,
        padding_ms: int = 100
    ) -> np.ndarray:
        """Trim silence from both ends of the signal while preserving padding."""
        if audio.size == 0 or sample_rate <= 0:
            return audio

        threshold = 10 ** (threshold_db / 20.0)
        magnitude = np.abs(audio)
        active_indices = np.where(magnitude >= threshold)[0]

        if active_indices.size == 0:
            return audio

        padding = int(sample_rate * padding_ms / 1000)
        start = max(int(active_indices[0]) - padding, 0)
        end = min(int(active_indices[-1]) + padding + 1, audio.size)
        trimmed = audio[start:end]

        # Keep original audio if trimming results in extremely short clip (<100ms)
        min_samples = int(sample_rate * 0.1)
        if trimmed.size < min_samples:
            return audio

        return trimmed

    def _float_audio_to_pcm_bytes(self, audio: np.ndarray) -> bytes:
        """Convert normalized float audio back to int16 PCM bytes."""
        if audio.size == 0:
            return b""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    async def _transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: Optional[str] = None,
        sample_rate_hint: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Transcribes audio data using persistent Whisper model from ModelManager.
        
        UPDATED: Now uses persistent Whisper engine from ModelManager for better performance.
        
        Args:
            audio_data: Raw audio data in bytes format
            audio_format: Format of the audio data (webm, wav, mp3, etc.)
            language: Language for transcription ('ar', 'en', or None for auto-detect)
            
        Returns:
            Transcribed text from the audio with /no_think suffix added
        """
        try:
            language_hint = language or "en"
            effective_format = (audio_format or "pcm").lower()

            # Default sample-rate hint can be overridden by upstream metadata later
            sample_rate_hint = sample_rate_hint or getattr(self, "_stt_sample_rate_hint", 16000)

            processed_audio = None
            peak_level = 0.0
            rms_level = 0.0
            duration_sec = 0.0

            # Only custom-preprocess PCM (and WAV when we can parse header)
            audio_bytes_for_engine = audio_data

            if effective_format == "wav":
                try:
                    import io
                    import wave

                    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                        sample_rate_hint = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                    audio_bytes_for_engine = frames
                    effective_format = "pcm"
                except Exception as exc:  # pragma: no cover - WAV parsing fallback
                    self.logger.debug(f"[STT] WAV header parsing failed ({exc}); using raw bytes")

            if effective_format == "pcm":
                processed_audio, sample_rate_used, peak_level, rms_level = self._prepare_audio_for_stt(
                    audio_bytes_for_engine,
                    sample_rate_hint=sample_rate_hint
                )

                self._stt_sample_rate_hint = sample_rate_used or 16000
                duration_sec = (
                    processed_audio.size / float(sample_rate_used)
                    if processed_audio.size and sample_rate_used else 0.0
                )

                self.logger.info(
                    "[STT-IN] sr=%d, dtype=float32, len=%d, peak=%.4f, rms=%.4f",
                    sample_rate_used,
                    processed_audio.size,
                    peak_level,
                    rms_level
                )

                prepared_bytes = self._float_audio_to_pcm_bytes(processed_audio)
                if not prepared_bytes:
                    self.logger.warning("[STT] Prepared audio is empty after preprocessing")
                    return "unclear audio /no_think"

                try:
                    last_dump_path = self.temp_dir / "last_stt_input.wav"
                    with wave.open(str(last_dump_path), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate_used)
                        wav_file.writeframes(prepared_bytes)
                except Exception as exc:
                    self.logger.debug("[STT] Failed to write latest STT dump: %s", exc)
            else:
                prepared_bytes = audio_bytes_for_engine
                if sample_rate_hint:
                    duration_sec = len(prepared_bytes) / float(sample_rate_hint * 2)
                self.logger.info(
                    "[STT-IN] format=%s, len_bytes=%d (no custom preprocessing)",
                    effective_format,
                    len(prepared_bytes)
                )
                self._stt_sample_rate_hint = sample_rate_hint or self._stt_sample_rate_hint

            dump_path = None
            if metadata:
                dump_path = metadata.get("dump_path") or metadata.get("stt_dump_path")
            if not dump_path:
                dump_path = os.getenv("BEAUTYAI_DUMP_STT_INPUT")
            if dump_path:
                try:
                    dump_file = Path(dump_path)
                    dump_file.parent.mkdir(parents=True, exist_ok=True)
                    sample_rate_to_write = ensure_sample_rate(self._stt_sample_rate_hint)
                    import wave

                    with wave.open(str(dump_file), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate_to_write)
                        wav_file.writeframes(prepared_bytes)
                    _log_pcm_stats(
                        self.logger,
                        label="stt-in",
                        sample_rate=sample_rate_to_write,
                        channels=1,
                        audio_bytes=prepared_bytes,
                    )

                    self.logger.info("[STT] Dumped prepared audio to %s", dump_file)
                except Exception as exc:
                    self.logger.warning("[STT] Failed to dump prepared audio: %s", exc)

            # UPDATED: Use persistent Whisper engine if available
            whisper_engine = self.persistent_whisper_engine

            if whisper_engine is None:
                logger.warning("Persistent Whisper engine not available, using factory fallback")
                if self.transcription_service is None:
                    from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service
                    self.transcription_service = create_transcription_service()

                    model_loaded = self.transcription_service.load_whisper_model()
                    if not model_loaded:
                        logger.warning("Failed to load voice registry STT model")
                        return "Sorry, I couldn't understand the audio."

                result = self.transcription_service.transcribe_audio_bytes(
                    prepared_bytes,
                    audio_format="pcm",
                    language=language_hint
                )
            else:
                logger.debug("Using persistent Whisper engine for transcription")
                result = whisper_engine.transcribe_audio_bytes(
                    prepared_bytes,
                    audio_format="pcm",
                    language=language_hint
                )

            raw_text = result.strip() if result else ""
            self.logger.info("[STT-OUT] text=\"%s\"", raw_text)
            self.logger.info("[STT] duration_sec=%.3f, lang_hint=%s", duration_sec, language_hint)

            if raw_text:
                transcribed_text = raw_text + " /no_think"
                logger.info(f"Transcribed audio with /no_think: {transcribed_text}")
                return transcribed_text

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
    
    async def _generate_chat_response(self, text: str, target_language: str = "auto", conversation_context: Optional[str] = None) -> str:
        """
        Generates chat response using the actual chat service with conversation context.
        
        Args:
            text: User input text to respond to
            target_language: Target language for response ("ar", "en", or "auto")
            conversation_context: Previous conversation context for better responses
            
        Returns:
            Generated response text
        """
        try:
            # Initialize chat service if needed (fallback for non-pre-loaded case)
            if self.chat_service is None:
                from beautyai_inference.services.inference.chat_service import ChatService
                self.chat_service = ChatService()
                self._cache_chat_defaults()

            # Ensure a chat model is ready
            if self.chat_default_model_config is None:
                load_success = self.chat_service.load_default_model_from_config()
                if load_success:
                    self._cache_chat_defaults()

            if self.chat_default_model_config is None:
                logger.error("Chat model configuration unavailable; cannot generate response")
                if target_language == "ar":
                    return sanitize_tts_text("عذراً، أواجه مشكلة تقنية حالياً. من فضلك حاولي مرة أخرى لاحقاً.")
                return sanitize_tts_text("Sorry, I'm having technical difficulties right now. Please try again later.")

            # Refresh cached values in case registry changed after initialization
            self._cache_chat_defaults()
            self._apply_chat_persona_overrides()
            
            # Create optimized message for fast responses in simple voice mode
            if self.chat_persona == "general":
                if conversation_context:
                    if target_language == "ar":
                        optimized_message = f"سياق إضافي: {conversation_context}\n\nرجاءً قدم رداً واضحاً ومفيداً: {text}"
                    else:
                        optimized_message = (
                            f"Additional context: {conversation_context}\n\n"
                            f"Please respond helpfully in {target_language or 'English'}: {text}"
                        )
                else:
                    if target_language == "ar":
                        optimized_message = f"يرجى الرد بوضوح وبأسلوب ودود: {text}"
                    else:
                        optimized_message = f"Please respond helpfully in {target_language or 'English'}: {text}"
            else:
                if conversation_context:
                    if target_language == "ar":
                        optimized_message = f"السياق: {conversation_context}\n\nأجب بإيجاز بالعربية: {text}"
                    else:
                        optimized_message = f"Context: {conversation_context}\n\nAnswer briefly in English: {text}"
                else:
                    if target_language == "ar":
                        optimized_message = f"أجب بإيجاز بالعربية: {text}"
                    else:
                        optimized_message = f"Answer briefly in English: {text}"
                    
            logger.info(f"Optimized message with context: {optimized_message[:100]}... (target_language: {target_language})")
            
            generation_config = self._prepare_generation_config(target_language=target_language)

            if target_language not in (None, "auto"):
                response_language_hint = target_language
            elif self.chat_last_language:
                response_language_hint = self.chat_last_language
            else:
                response_language_hint = None

            history_snapshot = self.chat_history[:] if self.chat_history else None
            session_id_snapshot = self.chat_session_id

            loop = asyncio.get_running_loop()

            def invoke_chat() -> Tuple[str, str, List[Dict[str, str]], str]:
                return self.chat_service.chat(
                    message=optimized_message,
                    model_name=self.chat_default_model_name or getattr(self.chat_default_model_config, "name", "default"),
                    model_config=self.chat_default_model_config,
                    generation_config=generation_config,
                    conversation_history=history_snapshot,
                    response_language=response_language_hint,
                    session_id=session_id_snapshot,
                    disable_content_filter=self.disable_content_filter
                )

            try:
                response_text, detected_language, updated_history, new_session_id = await loop.run_in_executor(
                    None,
                    invoke_chat
                )
            except Exception as chat_error:
                logger.error(f"ChatService invocation failed: {chat_error}")
                if target_language == "ar":
                    return sanitize_tts_text("عذراً، حدث خطأ أثناء توليد الرد. من فضلك حاولي مرة أخرى لاحقاً.")
                return sanitize_tts_text("Sorry, I ran into an error while generating a response. Please try again later.")

            # Persist session state for subsequent turns
            if new_session_id:
                self.chat_session_id = new_session_id
            if updated_history is not None:
                self.chat_history = updated_history
                max_turn_pairs = 20
                if len(self.chat_history) > max_turn_pairs * 2:
                    self.chat_history = self.chat_history[-max_turn_pairs * 2:]
            if detected_language:
                self.chat_last_language = detected_language

            clean_response = sanitize_tts_text(response_text)
            if not clean_response:
                logger.warning("ChatService returned empty response after sanitization; using fallback")
                if (detected_language or target_language) == "ar":
                    return sanitize_tts_text("عذراً، لم أفهم سؤالك بشكل كافٍ. هل يمكنك إعادة صياغته؟")
                return sanitize_tts_text("I’m sorry, I didn’t quite catch that. Could you please rephrase?")

            logger.info(f"Generated chat response for {target_language}: {clean_response[:100]}...")
            return clean_response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            if target_language == "ar":
                return sanitize_tts_text("عذراً، أواجه صعوبة في معالجة طلبك الآن. من فضلك حاول مرة أخرى.")
            else:
                return sanitize_tts_text("Sorry, I'm having trouble processing your request. Please try again.")
    
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
        gender: str = "female",
        debug_context: Optional[Dict[str, Any]] = None
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
        
        stats = {
            "service_name": "SimpleVoiceService",
            "edge_tts_available": True,  # We tested this during initialization
            "temp_directory": str(self.temp_dir),
            "temp_files_count": len(temp_files),
            "available_voices": len(self.voice_mappings),
            "default_arabic_voice": self.default_arabic_voice,
            "default_english_voice": self.default_english_voice,
            "debug_mode": self.debug_mode
        }
        
        # Add debug statistics if available
        if self.debug_mode:
            stats["debug_stats"] = {
                "total_debug_events": len(self.debug_events),
                "stage_timings": self.stage_timings.copy(),
                "recent_events_count": len([e for e in self.debug_events if (datetime.utcnow() - e.timestamp).seconds < 300]),
                "has_current_summary": self.current_debug_summary is not None,
                "debug_callback_set": self.debug_callback is not None
            }
        
        return stats
