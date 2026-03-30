"""
Whisper Byne Arabic v3 Transcription Engine for BeautyAI Framework.

This engine implements the Byne/whisper-large-v3-arabic model - a fine-tuned
Whisper Large v3 optimized for Arabic language transcription.

Key Features:
- Full 1.55B parameter model fine-tuned for Arabic
- Uses Transformers pipeline for compatibility (not faster-whisper)
- GPU-accelerated inference with float16
- Higher accuracy than turbo models for Arabic

Author: BeautyAI Framework  
Date: 2025-01-17
"""

import logging
import time
import os
from typing import Dict, Any, Optional
import numpy as np

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from .base_whisper_engine import BaseWhisperEngine

logger = logging.getLogger(__name__)


class WhisperByneArabicEngine(BaseWhisperEngine):
    """
    Byne Arabic-optimized Whisper Large v3 transcription engine.
    
    Uses the Byne/whisper-large-v3-arabic model which is fine-tuned on Arabic data
    for improved accuracy over the base Whisper model.
    
    This engine uses the Transformers backend (not faster-whisper) since the
    Byne model is a HuggingFace checkpoint, not a CTranslate2 model.
    """
    
    # Default model ID for this engine
    DEFAULT_MODEL_ID = "Byne/whisper-large-v3-arabic"
    
    def __init__(self):
        """Initialize Whisper Byne Arabic engine."""
        super().__init__()
        
        # Configure Triton cache directory
        try:
            triton_cache_dir = "/home/lumi/beautyai/logs/triton_cache"
            os.makedirs(triton_cache_dir, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
        except Exception as e:
            logger.warning(f"Failed to set TRITON_CACHE_DIR: {e}")

        # Engine-specific configuration
        self.pipe = None
        self.use_safetensors = True
        self.attn_implementation = "sdpa"  # Use SDPA attention for efficiency
        
        logger.info(f"WhisperByneArabicEngine initialized - Device: {self.device}, Dtype: {self.torch_dtype}")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_byne_arabic"
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load Whisper Byne Arabic model using Transformers.
        
        Args:
            model_id: Hugging Face model identifier (defaults to Byne/whisper-large-v3-arabic)
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Use default model ID if not specified or if it's a registry name
            if not model_id or "/" not in model_id:
                model_id = self.DEFAULT_MODEL_ID
            
            logger.info(f"Loading Whisper Byne Arabic model: {model_id}")
            logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
            
            # Model loading configuration
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "use_safetensors": self.use_safetensors,
                "attn_implementation": self.attn_implementation,
            }
            
            # Load the model - try different dtype approaches for compatibility
            try:
                # Try with dtype parameter first (newer transformers)
                model_kwargs["dtype"] = self.torch_dtype
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    **model_kwargs
                )
            except TypeError:
                # Fallback to torch_dtype for older transformers versions
                model_kwargs.pop("dtype", None)
                model_kwargs["torch_dtype"] = self.torch_dtype
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    **model_kwargs
                )
            
            # Move model to GPU
            self.model.to(self.device)
            logger.info(f"Model loaded to {self.device}")
            
            # Load processor (tokenizer + feature extractor)
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Create the transcription pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_cached = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
            
            logger.info("✅ Whisper Byne Arabic model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper Byne Arabic model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform Arabic-optimized transcription using Whisper Byne.
        
        Args:
            audio_array: Preprocessed audio array (16kHz mono float32)
            language: Language code for transcription (should be 'ar' for Arabic)
            
        Returns:
            Transcribed text
        """
        try:
            # Track stats
            self._runtime_stats["generate_calls"] = self._runtime_stats.get("generate_calls", 0) + 1
            
            start_time = time.time()
            
            # Normalize language hint
            normalized_language = self._normalize_language_hint(language)
            # Default to Arabic since this is an Arabic-specialized model
            language_for_generation = normalized_language or "arabic"
            
            logger.info(f"[WHISPER-BYNE] Transcribing with language: {language_for_generation}")
            
            if self.pipe is None:
                logger.error("Whisper pipeline not initialized")
                return ""
            
            # Prepare audio input
            audio_input = {
                "array": audio_array,
                "sampling_rate": 16000,
            }
            
            # Build generate_kwargs for optimal Arabic transcription
            # Note: max_new_tokens + decoder_input_ids (4 tokens) must not exceed max_target_positions (448)
            generate_kwargs = {
                "language": language_for_generation,
                "task": "transcribe",
                "max_new_tokens": 440,  # Leave room for decoder special tokens (was 448, caused overflow)
                "num_beams": 1,  # Greedy for speed, still good accuracy
                "do_sample": False,
                "temperature": 0.0,
            }
            
            # Try to get forced decoder IDs for more robust language enforcement
            forced_decoder_ids = None
            if self.processor is not None:
                try:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language_for_generation,
                        task="transcribe",
                    )
                    if forced_decoder_ids:
                        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
                        logger.debug(f"[WHISPER-BYNE] Using forced decoder IDs for {language_for_generation}")
                except Exception as e:
                    logger.debug(f"Could not get forced decoder IDs: {e}")
            
            # Run transcription
            result = self.pipe(
                audio_input,
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
            
            # Extract and clean text
            transcribed_text = result.get("text", "").strip() if result else ""
            transcribed_text = self._clean_arabic_transcription(transcribed_text)
            
            # Track timing
            inference_time = time.time() - start_time
            self._runtime_stats["total_generate_ms"] = self._runtime_stats.get("total_generate_ms", 0) + (inference_time * 1000)
            
            logger.info(f"[WHISPER-BYNE] Transcription ({inference_time:.2f}s): '{transcribed_text[:80]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper Byne Arabic transcription failed: {e}")
            self._runtime_stats["consecutive_failures"] = self._runtime_stats.get("consecutive_failures", 0) + 1
            return ""
    
    def _clean_arabic_transcription(self, text: str) -> str:
        """
        Clean Arabic transcription output.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common Whisper artifacts that might appear
        artifacts = [
            "[Music]",
            "[music]",
            "(music)",
            "[MUSIC]",
            "[Silence]",
            "[silence]",
            "(silence)",
        ]
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Whisper Byne Arabic model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_type": "whisper_byne_arabic",
                "model_size": "1.55B parameters",
                "architecture": "Whisper Large v3 (fine-tuned for Arabic)",
                "fine_tuned_on": "Arabic speech data",
                "backend": "transformers",
                "attention_implementation": self.attn_implementation,
                "language_specialized": "Arabic",
                "pipeline_active": self.pipe is not None,
            })
        
        return base_info
    
    def cleanup(self):
        """Clean up Whisper Byne Arabic resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            
            # Call base cleanup for model and processor
            super().cleanup()
            
            logger.info("✅ WhisperByneArabicEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during WhisperByneArabicEngine cleanup: {e}")
