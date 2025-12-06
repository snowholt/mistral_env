"""
Whisper Genius Arabic Transcription Engine for BeautyAI Framework.

This engine implements the Genius AI fine-tuned Arabic Whisper model
optimized for Arabic speech recognition with high accuracy.

Key Features:
- Fine-tuned on Arabic speech dataset
- Local model loading (no internet required)
- Based on Whisper Large v3 architecture
- Optimized for Arabic dialects

Model Path: /home/lumi/.cache/geniusai-arabic-models/2025-12-06/whisper/whisper/

Author: BeautyAI Framework
Date: 2025-12-06
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from .base_whisper_engine import BaseWhisperEngine

logger = logging.getLogger(__name__)

# Default path for Genius AI Arabic Whisper model
GENIUS_WHISPER_DEFAULT_PATH = Path("/home/lumi/.cache/geniusai-arabic-models/2025-12-06/whisper/whisper")


class WhisperGeniusArabicEngine(BaseWhisperEngine):
    """
    Genius AI fine-tuned Arabic Whisper transcription engine.
    
    Optimized for Arabic speech recognition using a locally fine-tuned
    Whisper model with enhanced accuracy for Arabic dialects.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize Whisper Genius Arabic engine.
        
        Args:
            model_path: Optional custom path to the model directory.
                       Defaults to GENIUS_WHISPER_DEFAULT_PATH.
        """
        super().__init__()
        
        # Model configuration
        self.model_path = model_path or GENIUS_WHISPER_DEFAULT_PATH
        self.pipe = None
        self.is_local_model = True
        
        # Verify model exists
        if not self.model_path.exists():
            logger.warning(f"Genius Whisper model not found at: {self.model_path}")
            logger.warning("Model will attempt to load from this path when load_whisper_model() is called")
        else:
            model_size = self._get_model_size()
            logger.info(f"WhisperGeniusArabicEngine initialized - Model path: {self.model_path}")
            logger.info(f"Model size: {model_size:.2f}GB")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_genius_arabic"
    
    def _get_model_size(self) -> float:
        """Calculate the total size of model files in GB."""
        try:
            total_size = 0
            for file in self.model_path.iterdir():
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024 ** 3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load Genius Arabic Whisper model from local path.
        
        Args:
            model_id: Model identifier (ignored for local models, uses self.model_path)
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Use local path instead of model_id
            local_path = str(self.model_path)
            
            logger.info(f"Loading Genius Arabic Whisper model from: {local_path}")
            
            # Validate model files exist
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]
            for file in required_files:
                file_path = self.model_path / file
                if not file_path.exists():
                    logger.error(f"Required model file not found: {file_path}")
                    return False
            
            model_size = self._get_model_size()
            logger.info(f"Model file size validated: {model_size:.2f}GB")
            
            # Load model with optimal configuration for local model
            logger.info("Loading Whisper model from local files...")
            
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    local_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    local_files_only=True,  # Force local loading
                )
            except Exception as e:
                logger.warning(f"Failed to load with safetensors, trying without: {e}")
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    local_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            
            self.model.to(self.device)
            logger.info(f"Model loaded to device: {self.device}")
            
            # Load processor
            logger.info("Loading processor/tokenizer...")
            self.processor = AutoProcessor.from_pretrained(
                local_path,
                local_files_only=True,
            )
            
            # Create pipeline
            logger.info("Creating inference pipeline...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device,
            )
            
            # Apply Arabic-specific optimizations
            self._apply_arabic_optimizations()
            
            logger.info("✅ Genius Arabic Whisper model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Genius Arabic Whisper model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_arabic_optimizations(self):
        """Apply Arabic-specific optimizations to the model."""
        try:
            logger.info("Applying Arabic fine-tuned optimizations...")
            
            # Set generation config for Arabic
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.language = "arabic"
                self.model.generation_config.task = "transcribe"
                logger.info("✅ Arabic language preset applied to generation config")
            
            logger.info("✅ Arabic optimizations applied")
            
        except Exception as e:
            logger.warning(f"Arabic optimization setup failed: {e}")
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform transcription using Genius Arabic Whisper model.
        
        Args:
            audio_array: Preprocessed audio array (16kHz mono float32)
            language: Language code for transcription (defaults to Arabic)
            
        Returns:
            Transcribed text
        """
        try:
            # Update runtime stats
            self._runtime_stats["generate_calls"] = self._runtime_stats.get("generate_calls", 0) + 1
            
            # Force Arabic for this engine (it's fine-tuned specifically for Arabic)
            target_language = "arabic"
            
            if language and language.lower() not in ("ar", "arabic", "auto"):
                logger.warning(f"Genius Arabic Whisper is optimized for Arabic. Requested: {language}, using: arabic")
            
            if self.pipe is None:
                logger.error("Whisper pipeline not initialized")
                return ""
            
            logger.debug(f"[WHISPER-GENIUS] Transcribing {len(audio_array)} samples in Arabic mode")
            
            # Prepare audio input
            audio_input = {
                "array": audio_array,
                "sampling_rate": 16000,
            }
            
            # Build generate_kwargs optimized for Arabic
            generate_kwargs = {
                "language": target_language,
                "task": "transcribe",
                "max_new_tokens": 256,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.0,
                "condition_on_prev_tokens": False,
            }
            
            # Get forced decoder IDs for Arabic
            try:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=target_language,
                    task="transcribe",
                )
                if forced_decoder_ids:
                    generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
            except Exception as e:
                logger.debug(f"Could not get decoder prompt IDs: {e}")
            
            # Run inference
            start_time = time.time()
            result = self.pipe(
                audio_input,
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
            inference_time = time.time() - start_time
            
            # Update stats
            self._runtime_stats["total_generate_ms"] = (
                self._runtime_stats.get("total_generate_ms", 0) + (inference_time * 1000)
            )
            
            transcribed_text = result.get("text", "").strip() if result else ""
            
            logger.debug(f"[WHISPER-GENIUS] Transcription ({inference_time*1000:.0f}ms): '{transcribed_text[:100]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Genius Arabic Whisper transcription failed: {e}")
            self._runtime_stats["consecutive_failures"] = (
                self._runtime_stats.get("consecutive_failures", 0) + 1
            )
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Genius Arabic Whisper model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_source": "local",
                "model_path": str(self.model_path),
                "model_size_gb": self._get_model_size(),
                "fine_tuned_for": "Arabic",
                "base_model": "Whisper Large v3",
                "pipeline_available": self.pipe is not None,
            })
        
        return base_info
    
    def cleanup(self):
        """Clean up Genius Arabic Whisper resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            
            # Call base cleanup
            super().cleanup()
            
            logger.info("✅ Genius Arabic Whisper cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Genius Arabic Whisper cleanup: {e}")
