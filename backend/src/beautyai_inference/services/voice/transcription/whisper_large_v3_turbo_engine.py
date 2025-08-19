"""
Whisper Large v3 Turbo Speed-Optimized Transcription Engine for BeautyAI Framework.

This engine implements the openai/whisper-large-v3-turbo model optimized for maximum speed
with torch.compile support, static cache, and minimal beam search for 4x faster inference
while maintaining good accuracy.

Key Features:
- Pruned 809M parameter model with 4 decoder layers
- torch.compile optimization with static cache
- Speed-optimized generation parameters
- SDPA attention implementation
- Warmup routine for consistent performance

Author: BeautyAI Framework  
Date: 2025-01-30
"""

import logging
import time
from typing import Dict, Any, Optional
import numpy as np

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

from .base_whisper_engine import BaseWhisperEngine

logger = logging.getLogger(__name__)


class WhisperLargeV3TurboEngine(BaseWhisperEngine):
    """
    Speed-optimized Whisper Large v3 Turbo transcription engine.
    
    Optimized for maximum speed using the pruned 809M parameter model with
    torch.compile optimization and aggressive speed-focused parameters.
    """
    
    def __init__(self):
        """Initialize Whisper Large v3 Turbo engine with speed-focused configuration."""
        super().__init__()
        
        # Engine-specific configuration
        self.pipe = None
        self.compiled_model = None
        self.is_warmed_up = False
        self.supports_torch_compile = self._check_torch_compile_support()
        self.static_cache_enabled = True
        
        # Model configuration
        self.low_cpu_mem_usage = True
        self.use_safetensors = True
        self.attn_implementation = "sdpa"  # SDPA is optimal for turbo + compile
        
        logger.info(f"WhisperLargeV3TurboEngine initialized - Torch Compile: {self.supports_torch_compile}, "
                   f"Static Cache: {self.static_cache_enabled}")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_large_v3_turbo"
    
    def _check_torch_compile_support(self) -> bool:
        """Check if torch.compile is available and recommended."""
        try:
            # torch.compile is available in PyTorch 2.0+
            if hasattr(torch, 'compile') and self.device.startswith("cuda"):
                return True
            return False
        except Exception:
            return False
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load Whisper Large v3 Turbo model with speed optimizations.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading Whisper Large v3 Turbo model: {model_id}")
            logger.info(f"Torch compile enabled: {self.supports_torch_compile}")
            
            # Set precision for optimal compile performance
            if self.supports_torch_compile:
                torch.set_float32_matmul_precision("high")
            
            # Load model with optimal configuration
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                use_safetensors=self.use_safetensors,
                attn_implementation=self.attn_implementation
            )
            self.model.to(self.device)
            
            # Configure static cache and compile model if supported
            if self.supports_torch_compile:
                self._setup_torch_compile()
            
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
                batch_size=16 if self.device.startswith("cuda") else 4,
                model_kwargs={
                    "use_safetensors": self.use_safetensors,
                    "low_cpu_mem_usage": self.low_cpu_mem_usage
                }
            )
            
            # Perform warmup if torch.compile is enabled
            if self.supports_torch_compile:
                self._warmup_model()
            
            logger.info("✅ Whisper Large v3 Turbo model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper Large v3 Turbo model: {e}")
            return False
    
    def _setup_torch_compile(self):
        """Configure torch.compile with optimal settings for Whisper Turbo."""
        try:
            logger.info("Setting up torch.compile optimization...")
            
            # Enable static cache for consistent performance
            self.model.generation_config.cache_implementation = "static"
            self.model.generation_config.max_new_tokens = 256
            
            # Compile the forward pass with optimal settings
            self.model.forward = torch.compile(
                self.model.forward, 
                mode="reduce-overhead",  # Optimal for inference
                fullgraph=True
            )
            
            logger.info("✅ torch.compile setup completed")
            
        except Exception as e:
            logger.warning(f"torch.compile setup failed, continuing without: {e}")
            self.supports_torch_compile = False
    
    def _warmup_model(self, warmup_steps: int = 2):
        """
        Perform warmup inference to optimize torch.compile performance.
        
        Args:
            warmup_steps: Number of warmup iterations
        """
        try:
            logger.info(f"Performing {warmup_steps} warmup steps...")
            
            # Create dummy audio for warmup (1 second of silence)
            dummy_audio = np.zeros(16000, dtype=np.float32)
            audio_input = {"array": dummy_audio, "sampling_rate": 16000}
            
            # Warmup iterations
            for i in range(warmup_steps):
                with sdpa_kernel(SDPBackend.MATH):
                    _ = self.pipe(
                        audio_input.copy(), 
                        generate_kwargs={
                            "min_new_tokens": 32,
                            "max_new_tokens": 32,
                            "num_beams": 1,
                            "temperature": 0.0
                        }
                    )
                logger.debug(f"Warmup step {i+1}/{warmup_steps} completed")
            
            self.is_warmed_up = True
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            self.is_warmed_up = False
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform speed-optimized transcription using Whisper Large v3 Turbo.
        
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
            
            # Configure generation parameters for maximum speed
            generate_kwargs = self._get_speed_optimized_parameters(language)
            
            # Perform transcription with SDPA kernel for torch.compile compatibility
            if self.supports_torch_compile:
                with sdpa_kernel(SDPBackend.MATH):
                    result = self.pipe(audio_input, generate_kwargs=generate_kwargs)
            else:
                result = self.pipe(audio_input, generate_kwargs=generate_kwargs)
            
            # Extract and clean transcription
            transcribed_text = result.get("text", "").strip() if result else ""
            transcribed_text = self._clean_transcription_output(transcribed_text)
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper Large v3 Turbo transcription failed: {e}")
            # Attempt minimal fallback
            try:
                logger.info("Attempting minimal fallback transcription...")
                audio_input = {"array": audio_array, "sampling_rate": 16000}
                result = self.pipe(audio_input, generate_kwargs={"num_beams": 1, "temperature": 0.0})
                transcribed_text = result.get("text", "").strip() if result else ""
                return self._clean_transcription_output(transcribed_text)
            except Exception as fallback_e:
                logger.error(f"Fallback transcription also failed: {fallback_e}")
                return ""
    
    def _get_speed_optimized_parameters(self, language: str) -> Dict[str, Any]:
        """
        Get speed-optimized generation parameters.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary of generation parameters optimized for speed
        """
        # Base parameters optimized for speed
        params = {
            "max_new_tokens": 256,  # Reduced for speed
            "num_beams": 1,  # Greedy search for maximum speed
            "condition_on_prev_tokens": False,  # Disable for speed
            "compression_ratio_threshold": 2.4,
            "temperature": 0.0,  # Deterministic for speed
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": False,
            "task": "transcribe"
        }
        
        # Add static cache parameters if enabled
        if self.static_cache_enabled and self.supports_torch_compile:
            params.update({
                "min_new_tokens": 32,  # Help with static cache
                "max_new_tokens": 256
            })
        
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
        
        # Remove common Whisper artifacts (same as base implementation)
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Whisper Large v3 Turbo model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_size": "809M parameters (pruned from 1.55B)",
                "architecture": "4 decoder layers (down from 32)",
                "torch_compile_enabled": self.supports_torch_compile,
                "static_cache_enabled": self.static_cache_enabled,
                "is_warmed_up": self.is_warmed_up,
                "attention_implementation": self.attn_implementation,
                "speed_optimized": True,
                "pipeline_batch_size": getattr(self.pipe, 'batch_size', 'N/A') if self.pipe else 'N/A'
            })
        
        return base_info
    
    def cleanup(self):
        """Enhanced cleanup for Whisper Large v3 Turbo resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            
            if self.compiled_model is not None:
                del self.compiled_model
                self.compiled_model = None
                
            self.is_warmed_up = False
            
            # Call base cleanup
            super().cleanup()
            
            logger.info("✅ Whisper Large v3 Turbo cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Whisper Large v3 Turbo cleanup: {e}")