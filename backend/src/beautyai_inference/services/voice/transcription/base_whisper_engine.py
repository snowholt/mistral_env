"""
Base Whisper Transcription Engine for BeautyAI Framework.

This base class provides common functionality for all specialized Whisper engines,
including GPU detection, memory management, audio preprocessing, and error handling.
All specific engine implementations should inherit from this base class.

Author: BeautyAI Framework
Date: 2025-01-30
"""

import logging
import tempfile
import time
import io
import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

import torch
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class BaseWhisperEngine(ABC):
    """
    Abstract base class for all Whisper transcription engines.
    
    Provides common functionality including:
    - GPU/CPU device detection and optimization
    - Audio preprocessing and format handling
    - Memory management and cleanup
    - Error handling and logging
    - Common interface compliance
    """
    
    def __init__(self):
        """Initialize base engine with hardware detection and optimization."""
        # Hardware detection and optimization
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Model components (to be set by subclasses)
        self.model = None
        self.processor = None
        self.loaded_model_name = None
        self.model_id = None
        
        # Performance tracking
        self.load_time = None
        self.last_inference_time = None
        # Runtime stats (extended by subclasses)
        self._runtime_stats: Dict[str, Any] = {
            "feature_extractions": 0,
            "generate_calls": 0,
            "total_feature_ms": 0.0,
            "total_generate_ms": 0.0,
            "consecutive_failures": 0,
            "circuit_open_events": 0,
        }
        
        # GPU optimization settings
        if self.device.startswith("cuda"):
            self._setup_gpu_optimizations()
        
        logger.info(f"BaseWhisperEngine initialized - Device: {self.device}, Dtype: {self.torch_dtype}")
    
    def _setup_gpu_optimizations(self):
        """Configure GPU-specific optimizations."""
        try:
            # Enable optimizations for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Log GPU information
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory_gb:.1f}GB")
                
        except Exception as e:
            logger.warning(f"GPU optimization setup failed: {e}")
    
    @abstractmethod
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Engine-specific model loading implementation.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _get_engine_name(self) -> str:
        """Return the name of this engine for logging/identification."""
        pass
    
    def load_whisper_model(self, model_name: str = None) -> bool:
        """
        Load a Whisper model for transcription.
        
        UPDATED: Now singleton-aware - checks ModelManager for existing instances
        before loading new models to improve performance and reduce memory usage.
        
        Args:
            model_name: Registry name of the model to load (optional, uses voice registry default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # CRITICAL DEBUG: Check if this engine is being managed by ModelManager FIRST
            managed_flag = getattr(self, '_managed_by_model_manager', False)
            logger.info(f"🔍 DEBUG: {self._get_engine_name()} managed flag = {managed_flag}")
            
            if managed_flag:
                logger.info(f"✅ {self._get_engine_name()} is managed by ModelManager - proceeding with direct loading")
                # Skip ModelManager checks and proceed directly to model loading
                # This prevents circular dependency: ModelManager -> load_whisper_model -> ModelManager
                pass  # Continue to direct loading section
            else:
                logger.info(f"🔍 {self._get_engine_name()} is NOT managed - checking ModelManager for existing models")
                # Only check ModelManager if this engine is NOT already managed
                try:
                    from ....core.model_manager import ModelManager
                    model_manager = ModelManager()
                    
                    # Check if there's already a persistent Whisper model loaded
                    persistent_engine = model_manager.get_streaming_whisper(model_name)
                    
                    if persistent_engine is not None and persistent_engine != self:
                        # Found existing persistent model - share its loaded model
                        if hasattr(persistent_engine, 'model') and persistent_engine.model is not None:
                            logger.info(f"🔗 Sharing model from persistent Whisper engine")
                            self.model = persistent_engine.model
                            self.processor = persistent_engine.processor
                            self.loaded_model_name = persistent_engine.loaded_model_name
                            self.model_id = persistent_engine.model_id
                            self.load_time = time.time() - start_time
                            logger.info(f"✅ {self._get_engine_name()} sharing persistent model in {self.load_time:.2f}s")
                            return True
                    
                except Exception as e:
                    logger.debug(f"ModelManager check failed: {e}, proceeding with direct loading")
            
            # Check if we already have a loaded model
            if self.model is not None:
                logger.info(f"✅ {self._get_engine_name()} model already loaded")
                return True
            
            # Proceed with direct model loading (original behavior)
            logger.info(f"🔄 Loading {self._get_engine_name()} directly")
            
            # Get model configuration from voice registry
            from ....config.voice_config_loader import get_voice_config
            voice_config = get_voice_config()
            
            if model_name is None:
                stt_config = voice_config.get_stt_model_config()
                actual_model_id = stt_config.model_id
                model_name = voice_config._config["default_models"]["stt"]
                logger.info(f"Using default STT model from voice registry: {model_name} -> {actual_model_id}")
            else:
                # Get specific model configuration
                model_config = voice_config._config["models"].get(model_name)
                if not model_config:
                    logger.error(f"Model '{model_name}' not found in voice registry")
                    return False
                actual_model_id = model_config["model_id"]
            
            # Validate engine compatibility
            if not self._validate_model_compatibility(model_name, voice_config):
                return False
            
            logger.info(f"Loading {self._get_engine_name()} with model: {actual_model_id}")
            
            # Call engine-specific loading implementation
            success = self._load_model_implementation(actual_model_id)
            
            if success:
                self.loaded_model_name = model_name
                self.model_id = actual_model_id
                self.load_time = time.time() - start_time
                logger.info(f"✅ {self._get_engine_name()} loaded successfully in {self.load_time:.2f}s")
                return True
            else:
                logger.error(f"❌ {self._get_engine_name()} loading failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading {self._get_engine_name()}: {e}")
            return False
    
    def _validate_model_compatibility(self, model_name: str, voice_config) -> bool:
        """
        Validate that the requested model is compatible with this engine.
        
        Args:
            model_name: Name of the model to validate
            voice_config: Voice configuration object
            
        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            model_config = voice_config._config["models"].get(model_name)
            if not model_config:
                logger.error(f"Model '{model_name}' not found in registry")
                return False
            
            expected_engine = model_config.get("engine_type")
            engine_name = self._get_engine_name().lower().replace(" ", "_")
            
            if expected_engine != engine_name:
                logger.warning(f"Engine mismatch: model '{model_name}' expects '{expected_engine}', got '{engine_name}'")
                # Allow loading but warn - useful for testing
            
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility validation failed: {e}")
            return False
    
    def transcribe_audio_bytes(
        self, 
        audio_bytes: bytes, 
        audio_format: str = "wav", 
        language: str = "ar"
    ) -> str:
        """
        Transcribe audio from bytes (main interface method).
        
        Args:
            audio_bytes: Audio data as bytes
            audio_format: Format of the audio data
            language: Language code for transcription
            
        Returns:
            Transcribed text or empty string on failure
        """
        try:
            if not self.model:
                logger.error(f"{self._get_engine_name()} model not loaded. Call load_whisper_model() first.")
                return ""
            
            # DEBUG: Log which engine is handling the request
            engine_name = self._get_engine_name()
            if hasattr(self, '_runtime_stats') and (self._runtime_stats.get('generate_calls', 0) % 10) == 0:
                logger.info(f"[engine-trace] {engine_name} handling transcription request")
            
            start_time = time.time()
            
            # Preprocess audio
            audio_array = self._prepare_audio_for_inference(audio_bytes, audio_format)
            
            # Call engine-specific transcription
            result = self._transcribe_implementation(audio_array, language)
            
            # Track performance
            self.last_inference_time = time.time() - start_time
            
            if result:
                logger.debug(f"{self._get_engine_name()} transcription completed in {self.last_inference_time:.2f}s: '{result[:100]}...'")
                return result
            else:
                logger.warning(f"{self._get_engine_name()} returned empty transcription")
                return ""
                
        except Exception as e:
            logger.error(f"{self._get_engine_name()} transcription failed: {e}")
            return ""

    # ---------------- Runtime Stats -----------------
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Return a shallow copy of current runtime statistics for observability.

        Returns:
            dict: metrics including counts and cumulative timings.
        """
        try:
            return dict(self._runtime_stats)
        except Exception:  # pragma: no cover - defensive
            return {}
    
    @abstractmethod
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Engine-specific transcription implementation.
        
        Args:
            audio_array: Preprocessed audio as numpy array
            language: Language code
            
        Returns:
            Transcribed text
        """
        pass
    
    def _prepare_audio_for_inference(self, audio_bytes: bytes, audio_format: str, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Prepare audio data for Whisper inference with optimal resampling.
        
        Args:
            audio_bytes: Raw audio bytes
            audio_format: Format of the audio (wav, mp3, webm, etc.)
            target_sample_rate: Target sample rate for Whisper (16kHz)
            
        Returns:
            numpy array of audio samples at 16kHz, mono, float32 in [-1, 1]
        """
        try:
            # Handle raw PCM first (common in streaming)
            if self._is_raw_pcm(audio_bytes, audio_format):
                return self._process_raw_pcm(audio_bytes, target_sample_rate)
            
            # Handle other formats using librosa/pydub
            return self._process_formatted_audio(audio_bytes, audio_format, target_sample_rate)
            
        except Exception as e:
            logger.error(f"Audio preparation failed: {e}")
            raise Exception(f"Audio preparation failed: {e}")
    
    def _is_raw_pcm(self, audio_bytes: bytes, audio_format: str) -> bool:
        """
        Detect if audio is raw PCM without WAV headers.
        
        Args:
            audio_bytes: Audio data
            audio_format: Specified format
            
        Returns:
            bool: True if raw PCM, False otherwise
        """
        # Check for WAV header
        has_wav_header = (len(audio_bytes) >= 12 and 
                         audio_bytes[:4] == b"RIFF" and 
                         audio_bytes[8:12] == b"WAVE")
        
        # If format suggests WAV but no header, it's likely raw PCM
        return (audio_format.lower() in ("wav", "pcm", "pcm_raw") and not has_wav_header)
    
    def _process_raw_pcm(self, audio_bytes: bytes, target_sample_rate: int) -> np.ndarray:
        """
        Process raw PCM audio bytes.
        
        Args:
            audio_bytes: Raw PCM data (assumed 16kHz mono int16 little-endian)
            target_sample_rate: Target sample rate
            
        Returns:
            Processed audio array
        """
        try:
            # Convert int16 PCM to float32
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"Processed raw PCM: {audio_array.shape[0]} samples")
            return audio_array
            
        except Exception as e:
            logger.error(f"Raw PCM processing failed: {e}")
            raise
    
    def _process_formatted_audio(self, audio_bytes: bytes, audio_format: str, target_sample_rate: int) -> np.ndarray:
        """
        Process formatted audio (WAV, MP3, WebM, etc.) using librosa/pydub.
        
        Args:
            audio_bytes: Audio data with proper headers
            audio_format: Audio format
            target_sample_rate: Target sample rate
            
        Returns:
            Processed audio array
        """
        try:
            import librosa
            from pydub import AudioSegment
            
            # Use pydub for WebM/Ogg formats
            if audio_format.lower() in ['webm', 'ogg']:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format.lower())
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                if audio_segment.channels == 2:
                    audio_array = audio_array.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
                audio_array = audio_array / np.iinfo(np.int16).max  # Normalize to [-1, 1]
                sample_rate = audio_segment.frame_rate
            else:
                # Use librosa for other formats
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_bytes), 
                    sr=None,
                    mono=True,
                    dtype=np.float32
                )
            
            logger.debug(f"Original audio: {sample_rate}Hz, {len(audio_array)} samples")
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=target_sample_rate,
                    res_type='kaiser_best'
                )
                logger.debug(f"Resampled to: {target_sample_rate}Hz")
            
            # Normalize and ensure minimum length
            return self._normalize_and_pad_audio(audio_array, target_sample_rate)
            
        except Exception as e:
            logger.error(f"Formatted audio processing failed: {e}")
            raise
    
    def _normalize_and_pad_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio levels and pad if too short.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            
        Returns:
            Normalized and padded audio array
        """
        # Light normalization to prevent clipping
        if np.max(np.abs(audio_array)) > 0:
            max_val = np.max(np.abs(audio_array))
            if max_val > 0.95:
                audio_array = audio_array * (0.95 / max_val)
        
        # Ensure minimum length (Whisper works better with some minimum duration)
        min_samples = sample_rate * 0.1  # 0.1 seconds minimum
        if len(audio_array) < min_samples:
            padding = np.zeros(int(min_samples - len(audio_array)), dtype=np.float32)
            audio_array = np.concatenate([audio_array, padding])
            logger.debug(f"Padded short audio to {len(audio_array)} samples")
        
        return audio_array
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self.loaded_model_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.model:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "engine_name": self._get_engine_name(),
            "model_name": self.loaded_model_name,
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "load_time": self.load_time,
            "last_inference_time": self.last_inference_time,
            "managed_by_model_manager": getattr(self, '_managed_by_model_manager', False)
        }
        
        # Add GPU memory info if available
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            info.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            })
        
        return info
    
    def set_managed_by_model_manager(self, managed: bool = True) -> None:
        """
        Mark this engine as managed by ModelManager to prevent redundant loading.
        
        Args:
            managed: True if managed by ModelManager, False otherwise
        """
        self._managed_by_model_manager = managed
        if managed:
            logger.debug(f"{self._get_engine_name()} marked as managed by ModelManager")
    
    def cleanup(self):
        """Clean up model and free memory resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.loaded_model_name = None
            self.model_id = None
            logger.info(f"✅ {self._get_engine_name()} cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during {self._get_engine_name()} cleanup: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "ogg", "flac", "m4a", "webm", "aac", "opus", "pcm"]
    
    def validate_audio_format(self, format_or_filename: str) -> bool:
        """
        Validate if an audio format is supported.
        
        Args:
            format_or_filename: Either a format string or filename
            
        Returns:
            bool: True if format is supported, False otherwise
        """
        if "." in format_or_filename:
            format_str = format_or_filename.split('.')[-1].lower()
        else:
            format_str = format_or_filename.lower()
            
        return format_str in self.get_supported_formats()