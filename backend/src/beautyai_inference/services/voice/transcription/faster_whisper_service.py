"""
Optimized Audio Transcription Service using Faster-Whisper for BeautyAI Framework.

This service provides up to 4x faster transcription with less memory usage compared to 
the traditional transformers-based Whisper implementation.

Key Features:
- GPU-optimized with INT8 quantization for 40% less VRAM usage
- Batch processing support for multiple audio files
- Built-in VAD (Voice Activity Detection) for better audio processing
- WebM format support with PyAV integration (no external FFmpeg required)
- Arabic language optimized models
"""
import logging
import tempfile
import os
import time
from typing import Dict, Any, Optional, BinaryIO, List, Tuple
from pathlib import Path

import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

from ...base.base_service import BaseService
from ....config.config_manager import AppConfig, ModelConfig
from ....core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class FasterWhisperTranscriptionService(BaseService):
    """Optimized service for audio transcription using Faster-Whisper."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.whisper_model = None
        self.batched_model = None
        self.loaded_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        
        logger.info(f"FasterWhisper initialized - Device: {self.device}, Compute: {self.compute_type}")
        
    def load_whisper_model(self, model_name: str = "whisper-turbo-arabic") -> bool:
        """
        Load a Whisper model for transcription using Faster-Whisper.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Map model names to faster-whisper compatible models
            model_mapping = {
                "whisper-turbo-arabic": "turbo",  # Fastest, good Arabic support
                "whisper-large-v3-turbo-arabic": "large-v3-turbo",  # High accuracy
                "whisper-base": "base",  # Lightweight
                "whisper-large-v3": "large-v3",  # High accuracy
                "whisper-distil-large-v3": "distil-large-v3"  # Balanced speed/accuracy
            }
            
            # Get the actual model identifier
            actual_model = model_mapping.get(model_name, "turbo")
            
            logger.info(f"Loading Faster-Whisper model: {actual_model}")
            
            # Load model with optimized settings
            self.whisper_model = WhisperModel(
                model_size_or_path=actual_model,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache
                local_files_only=False
            )
            
            # Create batched inference pipeline for better performance
            self.batched_model = BatchedInferencePipeline(model=self.whisper_model)
            
            self.loaded_model_name = model_name
            logger.info(f"Faster-Whisper model '{model_name}' ({actual_model}) loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model '{model_name}': {e}")
            return False
    
    def transcribe(self, audio_file: str = None, audio_bytes: bytes = None, language: str = "ar") -> Dict[str, Any]:
        """
        Transcribe audio with optimized faster-whisper backend.
        
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
                transcription = self.transcribe_audio_bytes(audio_bytes, "webm", language)
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
        Transcribe an audio file using Faster-Whisper.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code for transcription (default: "ar" for Arabic)
            
        Returns:
            str: Transcribed text, or None if failed
        """
        try:
            if not self.whisper_model:
                logger.error("Faster-Whisper model not loaded. Call load_whisper_model() first.")
                return None
            
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return None
            
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Transcribe with optimized parameters for Arabic
            segments, info = self.batched_model.transcribe(
                audio=audio_file_path,
                batch_size=8,  # Optimized batch size for speed
                beam_size=3,   # Reduced beam size for faster processing
                language=language if language != "auto" else None,
                task="transcribe",
                temperature=0.0,  # Deterministic for consistency
                condition_on_previous_text=False,  # Faster processing
                compression_ratio_threshold=2.4,  # Default threshold
                log_prob_threshold=-1.0,  # Default threshold
                no_speech_threshold=0.6,  # Adjusted for better silence detection
                initial_prompt=None,
                word_timestamps=False,  # Disabled for faster processing
                vad_filter=True,  # Enable built-in VAD
                vad_parameters={
                    "min_silence_duration_ms": 500,  # More responsive than default 2000ms
                    "speech_pad_ms": 30  # Reduced padding for faster response
                }
            )
            
            # Collect all segments
            transcription_segments = []
            for segment in segments:
                transcription_segments.append(segment.text.strip())
            
            # Join all segments
            transcription = " ".join(transcription_segments).strip()
            
            if transcription:
                logger.info(f"Transcription completed successfully: '{transcription[:100]}...'")
                logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
                return transcription
            else:
                logger.warning("No transcription generated")
                return None
            
        except Exception as e:
            logger.error(f"Audio file transcription failed: {e}")
            return None
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str = "webm", language: str = "ar") -> Optional[str]:
        """
        Transcribe audio from bytes using Faster-Whisper with built-in PyAV support.
        
        Args:
            audio_bytes: Audio data as bytes
            audio_format: Audio format (webm, wav, mp3, ogg, flac, m4a, etc.)
            language: Language code for transcription (default: "ar" for Arabic)
            
        Returns:
            str: Transcribed text, or None if failed
        """
        try:
            if not self.whisper_model:
                logger.error("Faster-Whisper model not loaded. Call load_whisper_model() first.")
                return None
            
            # Create temporary file for audio processing
            tests_dir = Path(__file__).parent.parent.parent / "tests" / "faster_whisper_temp"
            tests_dir.mkdir(parents=True, exist_ok=True)
            
            temp_input_path = tests_dir / f"temp_audio_{int(time.time() * 1000)}.{audio_format}"
            
            try:
                # Write bytes to temporary file
                with open(temp_input_path, 'wb') as temp_file:
                    temp_file.write(audio_bytes)
                
                logger.info(f"Processing {len(audio_bytes)} bytes of {audio_format} audio")
                
                # Faster-Whisper handles format conversion internally via PyAV
                # No need for external FFmpeg conversion
                transcription = self.transcribe_audio_file(str(temp_input_path), language)
                
                return transcription
                    
            finally:
                # Clean up temporary file
                try:
                    if temp_input_path.exists():
                        temp_input_path.unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Audio bytes transcription failed: {e}")
            return None
    
    def transcribe_batch(self, audio_files: List[str], language: str = "ar") -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files in batch for better efficiency.
        
        Args:
            audio_files: List of paths to audio files
            language: Language code for transcription
            
        Returns:
            List of transcription results
        """
        try:
            if not self.batched_model:
                logger.error("Batched model not available. Load a model first.")
                return []
            
            results = []
            for audio_file in audio_files:
                result = {
                    "file": audio_file,
                    "transcription": self.transcribe_audio_file(audio_file, language),
                    "success": True
                }
                if not result["transcription"]:
                    result["success"] = False
                    result["error"] = "Transcription failed"
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            return []
    
    def is_model_loaded(self) -> bool:
        """Check if a Faster-Whisper model is currently loaded."""
        return self.whisper_model is not None
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded Faster-Whisper model."""
        return self.loaded_model_name
    
    def unload_model(self) -> None:
        """Unload the current Faster-Whisper model to free memory."""
        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            if self.batched_model is not None:
                del self.batched_model
                self.batched_model = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.loaded_model_name = None
            logger.info("Faster-Whisper model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading Faster-Whisper model: {e}")
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats (broader than transformers version)."""
        return ["wav", "mp3", "ogg", "flac", "m4a", "wma", "webm", "aac", "opus"]
    
    def validate_audio_format(self, format_or_filename: str) -> bool:
        """
        Validate if an audio format is supported by Faster-Whisper.
        
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
        """Get memory statistics for the Faster-Whisper transcription service."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = {
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "model_loaded": self.is_model_loaded(),
                "loaded_model": self.loaded_model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "backend": "faster-whisper"
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.whisper_model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.loaded_model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "backend": "faster-whisper",
            "supports_batch": True,
            "supports_vad": True,
            "supports_formats": self.get_supported_formats()
        }
