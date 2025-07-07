"""
Audio Transcription Service for BeautyAI Framework.

Handles audio processing and transcription using Whisper models.
Supports multiple audio formats and provides integration with the chat system.
"""
import logging
import tempfile
import os
import time
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
    
    def transcribe(self, audio_file: str = None, audio_bytes: bytes = None, language: str = "ar") -> Dict[str, Any]:
        """
        Transcribe audio with a unified interface for voice-to-voice service.
        
        Args:
            audio_file: Path to audio file (if using file)
            audio_bytes: Audio bytes (if using bytes)
            language: Language code for transcription
            
        Returns:
            Dict with success status and transcription result
        """
        try:
            if audio_file:
                transcription = self.transcribe_audio_file(audio_file, language)
            elif audio_bytes:
                transcription = self.transcribe_audio_bytes(audio_bytes, "wav", language)
            else:
                return {"success": False, "error": "No audio input provided", "transcription": None}
            
            if transcription:
                return {"success": True, "transcription": transcription}
            else:
                return {"success": False, "error": "Transcription failed", "transcription": None}
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"success": False, "error": str(e), "transcription": None}
    
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
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy and flatten
            audio_array = waveform.squeeze().numpy()
            
            # Process with Whisper
            inputs = self.whisper_processor(
                audio_array, 
                sampling_rate=target_sample_rate, 
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            if torch.cuda.is_available() and next(self.whisper_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate transcription with enhanced parameters for better Arabic accuracy
            generation_kwargs = {
                "input_features": inputs["input_features"],
                "task": "transcribe",
                "max_new_tokens": 350,  # Optimized for Arabic text length
                "return_timestamps": False,
                "do_sample": False,     # Use deterministic decoding for consistency
                "num_beams": 3,         # Improved beam search for better accuracy
                "temperature": 0.0,     # Most conservative for accurate transcription
                "length_penalty": 1.0,  # Encourage complete transcription
                "early_stopping": False, # Prevent premature stopping
                "no_repeat_ngram_size": 3,  # Reduce repetition artifacts
            }
            
            # Add language parameter if specified
            if language and language != "auto":
                generation_kwargs["language"] = language
                logger.info(f"Transcribing with language: {language}")
            else:
                logger.info("Transcribing with automatic language detection")
            
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(**generation_kwargs)
            
            # Decode transcription
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Transcription completed: '{transcription[:100]}...'")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
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
            import tempfile
            import subprocess
            
            if not self.whisper_model or not self.whisper_processor:
                logger.error("Whisper model not loaded. Call load_whisper_model() first.")
                return None
            
            # Create temporary file for audio conversion in tests directory
            tests_dir = Path(__file__).parent.parent.parent / "tests"
            tests_dir.mkdir(exist_ok=True)
            
            temp_input_path = tests_dir / f"temp_audio_{int(time.time() * 1000)}.{audio_format}"
            with open(temp_input_path, 'wb') as temp_file:
                temp_file.write(audio_bytes)
            
            try:
                # Convert to WAV format using ffmpeg if not already WAV
                if audio_format.lower() != 'wav':
                    temp_wav_path = temp_input_path.with_suffix('.wav')
                    
                    # Use ffmpeg to convert audio format
                    conversion_cmd = [
                        'ffmpeg', '-i', str(temp_input_path), 
                        '-acodec', 'pcm_s16le',  # Use PCM 16-bit encoding
                        '-ar', '16000',  # Set sample rate to 16kHz
                        '-ac', '1',      # Convert to mono
                        '-y',            # Overwrite output file
                        str(temp_wav_path)
                    ]
                    
                    result = subprocess.run(
                        conversion_cmd, 
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.warning(f"FFmpeg conversion failed, trying direct processing: {result.stderr}")
                        # Fall back to direct processing
                        transcription = self.transcribe_audio_file(str(temp_input_path), language)
                    else:
                        transcription = self.transcribe_audio_file(str(temp_wav_path), language)
                        # Clean up converted file
                        try:
                            temp_wav_path.unlink()
                        except:
                            pass
                else:
                    transcription = self.transcribe_audio_file(str(temp_input_path), language)
                
                return transcription
                    
            finally:
                # Clean up temporary input file
                try:
                    if temp_input_path.exists():
                        temp_input_path.unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Audio bytes transcription failed: {e}")
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
            
            # Reset stream position if possible
            if hasattr(audio_stream, 'seek'):
                audio_stream.seek(0)
            
            return self.transcribe_audio_bytes(audio_bytes, audio_format, language)
            
        except Exception as e:
            logger.error(f"Audio stream transcription failed: {e}")
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
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.loaded_model_name = None
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
            # Extract format from filename
            format_str = format_or_filename.split('.')[-1].lower()
        else:
            # Direct format string
            format_str = format_or_filename.lower()
            
        return format_str in self.get_supported_formats()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the transcription service."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = {
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "model_loaded": self.is_model_loaded(),
                "loaded_model": self.loaded_model_name
            }
            
            # Add GPU memory if available
            if torch.cuda.is_available() and self.whisper_model is not None:
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                stats.update({
                    "gpu_memory_allocated_mb": gpu_memory,
                    "gpu_memory_cached_mb": gpu_cached,
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
