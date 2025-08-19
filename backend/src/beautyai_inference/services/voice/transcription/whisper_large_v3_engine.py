"""
Whisper Large v3 High-Accuracy Transcription Engine for BeautyAI Framework.

This engine implements the full-featured openai/whisper-large-v3 model optimized for
maximum accuracy with proper GPU acceleration, Flash Attention 2 support, and
chunked long-form processing capabilities.

Key Features:
- Full 1.55B parameter model with 32 decoder layers
- Flash Attention 2 / SDPA optimization  
- Chunked long-form processing for >30s audio
- Maximum accuracy configuration
- Robust error handling and fallback mechanisms

Author: BeautyAI Framework
Date: 2025-01-30
"""

import logging
import time
from typing import Dict, Any, Optional
import numpy as np

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

from .base_whisper_engine import BaseWhisperEngine

logger = logging.getLogger(__name__)


class WhisperLargeV3Engine(BaseWhisperEngine):
    """
    High-accuracy Whisper Large v3 transcription engine.
    
    Optimized for maximum transcription accuracy using the full 1.55B parameter model
    with advanced GPU optimizations including Flash Attention 2 and SDPA.
    """
    
    def __init__(self):
        """Initialize Whisper Large v3 engine with accuracy-focused configuration."""
        super().__init__()
        
        # Engine-specific configuration
        self.pipe = None
        self.supports_flash_attention = self._check_flash_attention_support()
        self.supports_chunked_processing = True
        self.max_accuracy_mode = True
        
        # Model configuration
        self.low_cpu_mem_usage = True
        self.use_safetensors = True
        self.attn_implementation = self._select_attention_implementation()
        
        logger.info(f"WhisperLargeV3Engine initialized - Flash Attention: {self.supports_flash_attention}, "
                   f"Attention: {self.attn_implementation}")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_large_v3"
    
    def _check_flash_attention_support(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _select_attention_implementation(self) -> str:
        """Select the best available attention implementation."""
        if self.supports_flash_attention and self.device.startswith("cuda"):
            return "flash_attention_2"
        else:
            return "sdpa"  # Scaled Dot-Product Attention (PyTorch default)
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load Whisper Large v3 model with optimized settings.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading Whisper Large v3 model: {model_id}")
            logger.info(f"Using attention implementation: {self.attn_implementation}")
            
            # Load model with optimal configuration
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                use_safetensors=self.use_safetensors,
                attn_implementation=self.attn_implementation
            )
            self.model.to(self.device)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Create optimized pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,  # Enable chunked processing for long audio
                batch_size=16 if self.device.startswith("cuda") else 4,
                model_kwargs={
                    "use_safetensors": self.use_safetensors,
                    "low_cpu_mem_usage": self.low_cpu_mem_usage,
                    "attn_implementation": self.attn_implementation
                }
            )
            
            logger.info("✅ Whisper Large v3 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper Large v3 model: {e}")
            return False
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform high-accuracy transcription using Whisper Large v3.
        
        Args:
            audio_array: Preprocessed audio array (16kHz mono float32)
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        try:
            # Prepare audio input for pipeline
            audio_input = {
                "array": audio_array,
                "sampling_rate": 16000
            }
            
            # Configure generation parameters for maximum accuracy
            generate_kwargs = self._get_generation_parameters(language)
            
            # Perform transcription
            if generate_kwargs:
                result = self.pipe(audio_input, generate_kwargs=generate_kwargs)
            else:
                result = self.pipe(audio_input)
            
            # Extract and clean transcription
            transcribed_text = result.get("text", "").strip() if result else ""
            
            # Handle common Whisper artifacts
            transcribed_text = self._clean_transcription_output(transcribed_text)
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper Large v3 transcription failed: {e}")
            # Attempt fallback without generation parameters
            try:
                logger.info("Attempting fallback transcription...")
                audio_input = {"array": audio_array, "sampling_rate": 16000}
                result = self.pipe(audio_input)
                transcribed_text = result.get("text", "").strip() if result else ""
                return self._clean_transcription_output(transcribed_text)
            except Exception as fallback_e:
                logger.error(f"Fallback transcription also failed: {fallback_e}")
                return ""
    
    def _get_generation_parameters(self, language: str) -> Dict[str, Any]:
        """
        Get optimized generation parameters for maximum accuracy.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary of generation parameters
        """
        # Base parameters optimized for accuracy
        params = {
            "max_new_tokens": 448,
            "num_beams": 5,  # Higher beam search for accuracy
            "condition_on_prev_tokens": True,  # Use context for better accuracy
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": False,  # Disable for faster processing
            "task": "transcribe"  # Always transcribe (don't translate)
        }
        
        # Add language specification if provided
        if language and language != "auto":
            if language == "ar":
                params["language"] = "arabic"
            elif language == "en":
                params["language"] = "english"
            else:
                params["language"] = language
        
        return params
    
    def _clean_transcription_output(self, text: str) -> str:
        """
        Clean common Whisper artifacts from transcription output.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
        
        # Remove common Whisper artifacts
        artifacts = [
            "unclear audio",
            "inaudible",
            "music playing",
            "[music]",
            "[applause]",
            "[laughter]"
        ]
        
        text_lower = text.lower()
        for artifact in artifacts:
            if text_lower.startswith(artifact):
                logger.warning(f"Removing Whisper artifact: '{artifact}'")
                return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def transcribe_audio_file(self, audio_file_path: str, language: str = "ar") -> str:
        """
        Transcribe audio from a file path with chunked processing.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        try:
            if not self.pipe:
                logger.error("Whisper Large v3 model not loaded")
                return ""
            
            logger.info(f"Transcribing file: {audio_file_path}")
            
            # Use pipeline with file path (automatic chunking for long files)
            generate_kwargs = self._get_generation_parameters(language)
            
            if generate_kwargs:
                result = self.pipe(audio_file_path, generate_kwargs=generate_kwargs)
            else:
                result = self.pipe(audio_file_path)
            
            transcribed_text = result.get("text", "").strip() if result else ""
            transcribed_text = self._clean_transcription_output(transcribed_text)
            
            logger.info(f"File transcription completed: '{transcribed_text[:100]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Whisper Large v3 model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_size": "1.55B parameters",
                "architecture": "32 decoder layers, 128 Mel bins", 
                "attention_implementation": self.attn_implementation,
                "supports_flash_attention": self.supports_flash_attention,
                "supports_chunked_processing": self.supports_chunked_processing,
                "max_accuracy_mode": self.max_accuracy_mode,
                "pipeline_batch_size": getattr(self.pipe, 'batch_size', 'N/A') if self.pipe else 'N/A'
            })
        
        return base_info
    
    def cleanup(self):
        """Enhanced cleanup for Whisper Large v3 resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            
            # Call base cleanup
            super().cleanup()
            
            logger.info("✅ Whisper Large v3 cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Whisper Large v3 cleanup: {e}")