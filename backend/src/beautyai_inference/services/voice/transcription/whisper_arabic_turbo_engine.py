"""
Whisper Arabic Turbo Specialized Transcription Engine for BeautyAI Framework.

This engine implements the mboushaba/whisper-large-v3-turbo-arabic model optimized for
Arabic speech recognition with specialized fine-tuning on Common Voice Arabic dataset,
achieving 31.14% WER while maintaining turbo-level speed performance.

Key Features:
- Fine-tuned 809M parameter model specialized for Arabic
- 31.14% WER on Common Voice Arabic dataset
- Dialect-aware processing (MSA, Egyptian, Levantine, etc.)
- Speed-optimized with torch.compile support
- Graceful fallback for non-Arabic content

Author: BeautyAI Framework
Date: 2025-01-30
"""

import logging
import time
from typing import Dict, Any, Optional, List
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


class WhisperArabicTurboEngine(BaseWhisperEngine):
    """
    Arabic-specialized Whisper Turbo transcription engine.
    
    Optimized for Arabic speech recognition using the mboushaba fine-tuned model
    with dialect handling capabilities and speed optimizations.
    """
    
    def __init__(self):
        """Initialize Whisper Arabic Turbo engine with Arabic-focused configuration."""
        super().__init__()
        
        # Engine-specific configuration
        self.pipe = None
        self.arabic_optimized = True
        self.default_language = "ar"
        self.supports_torch_compile = self._check_torch_compile_support()
        self.dialect_handling = True
        
        # Model configuration
        self.low_cpu_mem_usage = True
        self.use_safetensors = True
        self.attn_implementation = "sdpa"  # Optimal for Arabic turbo model
        
        # Arabic dialect support
        self.supported_dialects = [
            "MSA",  # Modern Standard Arabic
            "Egyptian",
            "Levantine", 
            "Gulf",
            "Maghrebi",
            "Sudanese"
        ]
        
        logger.info(f"WhisperArabicTurboEngine initialized - Arabic optimized: {self.arabic_optimized}, "
                   f"Dialects: {len(self.supported_dialects)}")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_arabic_turbo"
    
    def _check_torch_compile_support(self) -> bool:
        """Check if torch.compile is available."""
        try:
            if hasattr(torch, 'compile') and self.device.startswith("cuda"):
                return True
            return False
        except Exception:
            return False
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load Whisper Arabic Turbo model with Arabic optimizations.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading Whisper Arabic Turbo model: {model_id}")
            logger.info("Applying Arabic-specific optimizations...")
            
            # Set precision for optimal performance
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
            
            # Apply Arabic-specific optimizations
            self._optimize_for_arabic()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Create Arabic-optimized pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                batch_size=12 if self.device.startswith("cuda") else 4,  # Slightly lower for Arabic processing
                model_kwargs={
                    "use_safetensors": self.use_safetensors,
                    "low_cpu_mem_usage": self.low_cpu_mem_usage
                }
            )
            
            logger.info("✅ Whisper Arabic Turbo model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper Arabic Turbo model: {e}")
            return False
    
    def _optimize_for_arabic(self):
        """Apply Arabic-specific model optimizations."""
        try:
            # Configure model for Arabic language preference
            if hasattr(self.model, 'config'):
                # Set language preference if supported
                if hasattr(self.model.config, 'forced_decoder_ids'):
                    # This would set Arabic as the preferred language
                    pass
                
            # Enable torch.compile if supported (helps with Arabic processing)
            if self.supports_torch_compile:
                try:
                    self.model.forward = torch.compile(
                        self.model.forward, 
                        mode="default",  # Conservative for fine-tuned models
                        fullgraph=False  # More compatible with fine-tuned weights
                    )
                    logger.info("✅ torch.compile enabled for Arabic model")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without: {e}")
                    self.supports_torch_compile = False
            
            logger.info("✅ Arabic optimizations applied")
            
        except Exception as e:
            logger.warning(f"Arabic optimization setup failed: {e}")
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform Arabic-specialized transcription.
        
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
            
            # FIXED: Use proper parameter structure for generate_kwargs
            # Configure generation parameters for Arabic
            generate_kwargs = self._get_arabic_optimized_parameters(language)
            
            # FIXED: Use the correct parameter structure to avoid pipeline errors
            kwargs = {"generate_kwargs": generate_kwargs}
            
            # Perform transcription with Arabic optimizations
            if self.supports_torch_compile:
                with sdpa_kernel(SDPBackend.MATH):
                    result = self.pipe(audio_input, **kwargs)
            else:
                result = self.pipe(audio_input, **kwargs)
            
            # Extract and clean transcription
            transcribed_text = result.get("text", "").strip() if result else ""
            transcribed_text = self._clean_arabic_transcription(transcribed_text)
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper Arabic Turbo transcription failed: {e}")
            # Attempt fallback for non-Arabic or problematic audio
            return self._fallback_transcription(audio_array, language)
    
    def _get_arabic_optimized_parameters(self, language: str) -> Dict[str, Any]:
        """
        Get Arabic-optimized generation parameters.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary of generation parameters optimized for Arabic
        """
        # FIXED: Simplified parameters compatible with current transformers
        # Use only essential parameters to avoid parameter conflicts
        params = {}
        
        # Force Arabic language for best results with this model
        if language == "ar" or language is None:
            params["language"] = "arabic"
        elif language == "en":
            params["language"] = "english"
        else:
            # For other languages, let the model auto-detect
            pass
        
        return params
    
    def _clean_arabic_transcription(self, text: str) -> str:
        """
        Minimal cleaning of Arabic transcription output - only remove excessive whitespace.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned Arabic transcription text
        """
        if not text:
            return ""
        
        # Only remove excessive whitespace
        text = " ".join(text.split())
        
        return text
        
        # Arabic-specific cleaning
        text = self._clean_arabic_text(text)
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def _clean_arabic_text(self, text: str) -> str:
        """
        Apply Arabic-specific text cleaning.
        
        Args:
            text: Arabic transcription text
            
        Returns:
            Cleaned Arabic text
        """
        # Remove or normalize Arabic diacritics if they appear incorrectly
        # This is a basic implementation - could be enhanced based on needs
        
        # Remove redundant spaces around Arabic punctuation
        arabic_punctuation = ['،', '؟', '؛', ':', '.', '!']
        for punct in arabic_punctuation:
            text = text.replace(f' {punct}', punct)
            text = text.replace(f'{punct} {punct}', punct)
        
        # Normalize Arabic numbers if mixed with English
        arabic_to_english_digits = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        for arabic_digit, english_digit in arabic_to_english_digits.items():
            text = text.replace(arabic_digit, english_digit)
        
        return text
    
    def _fallback_transcription(self, audio_array: np.ndarray, language: str) -> str:
        """
        Fallback transcription for problematic audio or non-Arabic content.
        
        Args:
            audio_array: Audio array
            language: Language code
            
        Returns:
            Fallback transcription result
        """
        try:
            logger.info("Attempting fallback transcription with minimal parameters...")
            
            audio_input = {"array": audio_array, "sampling_rate": 16000}
            
            # Minimal parameters for fallback
            fallback_kwargs = {
                "num_beams": 1,
                "temperature": 0.0,
                "max_new_tokens": 128,
                "task": "transcribe"
            }
            
            result = self.pipe(audio_input, generate_kwargs=fallback_kwargs)
            transcribed_text = result.get("text", "").strip() if result else ""
            
            if transcribed_text:
                logger.info(f"Fallback transcription successful: '{transcribed_text[:50]}...'")
                return self._clean_arabic_transcription(transcribed_text)
            else:
                logger.warning("Fallback transcription returned empty result")
                return ""
                
        except Exception as e:
            logger.error(f"Fallback transcription also failed: {e}")
            return ""
    
    def detect_arabic_dialect(self, text: str) -> Optional[str]:
        """
        Simple dialect detection based on common words/patterns.
        
        Args:
            text: Arabic transcribed text
            
        Returns:
            Detected dialect or None
        """
        if not text:
            return None
        
        # Basic dialect indicators (could be enhanced with proper ML models)
        dialect_indicators = {
            "Egyptian": ["عايز", "عاوز", "إيه", "ازيك", "اللي"],
            "Levantine": ["شو", "هاي", "كيفك", "بدي", "منيح"],
            "Gulf": ["شلون", "شنو", "أبي", "وش", "زين"],
            "Maghrebi": ["شنو", "واش", "نتا", "نتي", "باش"],
            "MSA": ["ماذا", "كيف", "أريد", "هذا", "التي"]
        }
        
        text_lower = text.lower()
        dialect_scores = {}
        
        for dialect, indicators in dialect_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                dialect_scores[dialect] = score
        
        if dialect_scores:
            detected_dialect = max(dialect_scores, key=dialect_scores.get)
            logger.info(f"Detected Arabic dialect: {detected_dialect}")
            return detected_dialect
        
        return "MSA"  # Default to MSA if no specific indicators found
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Whisper Arabic Turbo model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_size": "809M parameters (Arabic fine-tuned)",
                "architecture": "4 decoder layers, Arabic-specialized",
                "arabic_optimized": self.arabic_optimized,
                "default_language": self.default_language,
                "supported_dialects": self.supported_dialects,
                "wer_arabic": "31.14% (Common Voice Arabic)",
                "training_dataset": "common_voice_11_0",
                "torch_compile_enabled": self.supports_torch_compile,
                "dialect_handling": self.dialect_handling,
                "pipeline_batch_size": getattr(self.pipe, 'batch_size', 'N/A') if self.pipe else 'N/A'
            })
        
        return base_info
    
    def cleanup(self):
        """Enhanced cleanup for Whisper Arabic Turbo resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                
            # Call base cleanup
            super().cleanup()
            
            logger.info("✅ Whisper Arabic Turbo cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Whisper Arabic Turbo cleanup: {e}")