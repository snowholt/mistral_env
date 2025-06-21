"""
Audio Transcription Service for BeautyAI Framework.

Handles audio processing and transcription using Whisper models.
Supports multiple audio formats and provides integration with the chat system.
"""
import logging
import tempfile
import os
from typing import Dict, Any, Optional, BinaryIO
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .base.base_service import BaseService
from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class AudioTranscriptionService(BaseService):
    """Service for audio transcription using Whisper models."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.whisper_processor = None
        self.whisper_model = None
        self.loaded_model_name = None
        
    def load_whisper_model(self, model_name: str = "whisper-large-v3-turbo-arabic") -> bool:
        """
        Load a Whisper model for transcription.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get model configuration
            app_config = AppConfig()
            app_config.models_file = "beautyai_inference/config/model_registry.json"
            app_config.load_model_registry()
            
            model_config = app_config.model_registry.get_model(model_name)
            if not model_config:
                logger.error(f"Model configuration for '{model_name}' not found.")
                return False
            
            # Load processor and model
            logger.info(f"Loading Whisper model: {model_config.model_id}")
            self.whisper_processor = WhisperProcessor.from_pretrained(model_config.model_id)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_config.model_id)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.whisper_model = self.whisper_model.cuda()
                logger.info("Whisper model moved to GPU")
            
            self.loaded_model_name = model_name
            logger.info(f"Whisper model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}")
            return False
    
    def transcribe_audio_file(self, audio_file_path: str, language: str = "ar") -> Optional[str]:
        """
        Transcribe an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code for transcription (default: "ar" for Arabic)
            
        Returns:
            str: Transcribed text, or None if failed
        """
        try:
            if not self.whisper_model or not self.whisper_processor:
                logger.error("Whisper model not loaded. Call load_whisper_model() first.")
                return None
            
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return None
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_file_path)
            logger.info(f"Audio file loaded: {audio_file_path}, Sample rate: {sample_rate}")
            
            # Resample to 16kHz if needed (Whisper requirement)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                logger.info(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
                sample_rate = target_sample_rate
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.info("Converted stereo to mono")
            
            # Prepare input features
            input_features = self.whisper_processor(
                waveform.squeeze().numpy(), 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features
            
            # Move to GPU if model is on GPU
            if next(self.whisper_model.parameters()).is_cuda:
                input_features = input_features.cuda()
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
            
            logger.info(f"Transcription completed: {transcription[:100]}...")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str = "wav", language: str = "ar") -> Optional[str]:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            audio_format: Audio format (wav, mp3, ogg, flac, m4a, wma, webm, etc.)
            language: Language code for transcription (default: "ar" for Arabic)
            
        Returns:
            str: Transcribed text, or None if failed
        """
        try:
            if not self.whisper_model or not self.whisper_processor:
                logger.error("Whisper model not loaded. Call load_whisper_model() first.")
                return None
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Transcribe using the temporary file
                transcription = self.transcribe_audio_file(temp_file_path, language)
                return transcription
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during audio bytes transcription: {e}")
            return None
    
    def transcribe_audio_stream(self, audio_stream: BinaryIO, audio_format: str = "wav", language: str = "ar") -> Optional[str]:
        """
        Transcribe audio from a stream.
        
        Args:
            audio_stream: Audio stream/file-like object
            audio_format: Audio format (wav, mp3, ogg, flac, m4a, wma, webm, etc.)
            language: Language code for transcription (default: "ar" for Arabic)
            
        Returns:
            str: Transcribed text, or None if failed
        """
        try:
            # Read stream to bytes
            audio_bytes = audio_stream.read()
            return self.transcribe_audio_bytes(audio_bytes, audio_format, language)
            
        except Exception as e:
            logger.error(f"Error during audio stream transcription: {e}")
            return None
    
    def is_model_loaded(self) -> bool:
        """Check if a Whisper model is currently loaded."""
        return self.whisper_model is not None and self.whisper_processor is not None
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded Whisper model."""
        return self.loaded_model_name
    
    def unload_model(self) -> None:
        """Unload the current Whisper model to free memory."""
        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
                
            if self.whisper_processor is not None:
                del self.whisper_processor
                self.whisper_processor = None
                
            self.loaded_model_name = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Whisper model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading Whisper model: {e}")
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "ogg", "flac", "m4a", "wma", "webm"]
    
    def validate_audio_format(self, format_or_filename: str) -> bool:
        """
        Validate if an audio format is supported.
        
        Args:
            format_or_filename: Either a format string or filename
            
        Returns:
            bool: True if format is supported, False otherwise
        """
        if "." in format_or_filename:
            # Extract extension from filename
            format_str = Path(format_or_filename).suffix.lstrip(".").lower()
        else:
            format_str = format_or_filename.lower()
            
        return format_str in self.get_supported_formats()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the transcription service."""
        stats = {
            "model_loaded": self.is_model_loaded(),
            "loaded_model_name": self.loaded_model_name,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_memory_cached": torch.cuda.memory_cached()
            })
            
        return stats
