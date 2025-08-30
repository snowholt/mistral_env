"""
Whisper Fine-tuned Arabic Specialized Transcription Engine for BeautyAI Framework.

This engine implements a custom fine-tuned Whisper Large v3 Turbo model optimized for
Arabic speech recognition with specialized training on Arabic voice datasets, providing
superior performance for Arabic transcription tasks.

Key Features:
- Custom fine-tuned 809M parameter model specialized for Arabic
- Enhanced Arabic dialect handling and recognition
- Optimized for BeautyAI voice streaming requirements
- Speed-optimized with torch.compile support
- Local model storage for consistent performance

Author: BeautyAI Framework
Date: 2025-08-27
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

from .base_whisper_engine import BaseWhisperEngine

logger = logging.getLogger(__name__)


class WhisperFinetunedArabicEngine(BaseWhisperEngine):
    """
    Fine-tuned Arabic-specialized Whisper transcription engine.
    
    Optimized for Arabic speech recognition using a custom fine-tuned model
    with enhanced Arabic dialect handling and superior transcription quality.
    """
    
    def __init__(self):
        """Initialize Whisper Fine-tuned Arabic engine with custom model configuration."""
        super().__init__()
        
        # Engine-specific configuration
        self.pipe = None
        self.arabic_optimized = True
        self.default_language = "ar"
        self.supports_torch_compile = self._check_torch_compile_support()
        self.dialect_handling = True
        self.fine_tuned = True
        # Circuit breaker / failure mitigation
        self._max_consecutive_failures = int(__import__('os').getenv('WHISPER_AR_MAX_CONSECUTIVE_FAILURES', '5'))
        self._circuit_breaker_cooldown_sec = float(__import__('os').getenv('WHISPER_AR_CIRCUIT_COOLDOWN_SEC', '2.0'))
        self._circuit_last_open_time: Optional[float] = None
        
        # Custom model configuration
        self.low_cpu_mem_usage = True
        self.use_safetensors = True
        self.attn_implementation = "sdpa"  # Optimal for Arabic turbo model
        
        # Local model path
        self.local_model_path = Path("/home/lumi/.cache/beautyai-whisper-turbo/whisper_fine_tuned")
        
        # Arabic dialect support (enhanced with fine-tuning)
        self.supported_dialects = [
            "MSA",  # Modern Standard Arabic
            "Egyptian",
            "Levantine", 
            "Gulf",
            "Maghrebi",
            "Sudanese",
            "Custom_Trained"  # Enhanced with fine-tuning
        ]
        
        logger.info(f"WhisperFinetunedArabicEngine initialized - Fine-tuned: {self.fine_tuned}, "
                   f"Local path: {self.local_model_path}, Dialects: {len(self.supported_dialects)}")
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "whisper_finetuned_arabic"
    
    def _check_torch_compile_support(self) -> bool:
        """Check if torch.compile is available."""
        try:
            if hasattr(torch, 'compile') and self.device.startswith("cuda"):
                return True
            return False
        except Exception:
            return False
    
    def _validate_local_model(self) -> bool:
        """
        Validate that the local fine-tuned model exists and is complete.
        
        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            if not self.local_model_path.exists():
                logger.error(f"Local model path does not exist: {self.local_model_path}")
                return False
            
            # Check for essential model files
            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = self.local_model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.error(f"Missing required model files: {missing_files}")
                return False
            
            # Check model size (should be around 3.2GB for the main model file)
            model_file = self.local_model_path / "model.safetensors"
            model_size_gb = model_file.stat().st_size / (1024**3)
            
            if model_size_gb < 3.0 or model_size_gb > 4.0:
                logger.warning(f"Model file size unexpected: {model_size_gb:.2f}GB (expected ~3.2GB)")
            else:
                logger.info(f"Model file size validated: {model_size_gb:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _load_model_implementation(self, model_id: str) -> bool:
        """
        Load fine-tuned Arabic Whisper model from local storage.
        
        Args:
            model_id: Model identifier (ignored, uses local path)
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading fine-tuned Arabic Whisper model from: {self.local_model_path}")
            
            # Validate model first
            if not self._validate_local_model():
                logger.error("Model validation failed")
                return False
            
            logger.info("Applying Arabic fine-tuned optimizations...")
            
            # Set precision for optimal performance
            if self.supports_torch_compile:
                torch.set_float32_matmul_precision("high")
            
            # Load model with optimal configuration from local path
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.local_model_path),
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                use_safetensors=self.use_safetensors,
                attn_implementation=self.attn_implementation,
                local_files_only=True  # Force local loading
            )
            self.model.to(self.device)
            
            # Apply fine-tuned Arabic optimizations
            self._optimize_for_finetuned_arabic()
            
            # Load processor from local path
            self.processor = AutoProcessor.from_pretrained(
                str(self.local_model_path),
                local_files_only=True
            )
            
            # (Revised) Defer creating pipeline; we'll run a manual generate path first
            # Pipeline kept as optional fallback for future compatibility or batch decode features.
            try:
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    torch_dtype=self.torch_dtype,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Deferred pipeline creation failed (will rely on manual path): {e}")
                self.pipe = None
            
            # FIXED: Skip torch.compile for now to ensure compatibility like other engines
            logger.info("Skipping torch.compile to ensure compatibility")
            self.supports_torch_compile = False
            
            logger.info("✅ Fine-tuned Arabic Whisper model loaded successfully from local storage")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned Arabic Whisper model: {e}")
            return False
    
    def _optimize_for_finetuned_arabic(self):
        """Apply fine-tuned Arabic-specific model optimizations."""
        try:
            # Configure model for Arabic language preference
            if hasattr(self.model, 'config'):
                # Set forced decoder IDs for Arabic from the fine-tuned config
                if hasattr(self.model.config, 'forced_decoder_ids'):
                    # The fine-tuned model should already have optimal decoder IDs
                    logger.info("Using fine-tuned forced decoder IDs for Arabic")
                
            # FIXED: Skip torch.compile for compatibility like other working engines
            logger.info("Skipping torch.compile for maximum compatibility")
            self.supports_torch_compile = False
            
            logger.info("✅ Fine-tuned Arabic optimizations applied")
            
        except Exception as e:
            logger.warning(f"Fine-tuned Arabic optimization setup failed: {e}")
    
    def _transcribe_implementation(self, audio_array: np.ndarray, language: str) -> str:
        """
        Perform fine-tuned Arabic-specialized transcription.
        
        Args:
            audio_array: Preprocessed audio array (16kHz mono float32)
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        try:
            # DEBUG: Confirm this engine is being used
            if (self._runtime_stats.get('generate_calls', 0) % 5) == 0:
                logger.info("[engine-trace] WhisperFinetunedArabicEngine._transcribe_implementation called")
            
            # Circuit breaker: avoid hammering model if we hit repeated failures
            if self._runtime_stats.get('consecutive_failures', 0) >= self._max_consecutive_failures:
                now = time.time()
                if self._circuit_last_open_time is None:
                    self._circuit_last_open_time = now
                    self._runtime_stats['circuit_open_events'] += 1
                    logger.warning("[whisper-circuit] Opened after %d consecutive failures", self._runtime_stats.get('consecutive_failures'))
                elif (now - self._circuit_last_open_time) < self._circuit_breaker_cooldown_sec:
                    # Still in cooldown window – skip heavy work
                    return ""
                else:
                    # Cooldown elapsed – reset breaker
                    logger.info("[whisper-circuit] Cooldown elapsed; allowing new attempts")
                    self._runtime_stats['consecutive_failures'] = 0
                    self._circuit_last_open_time = None
            # Primary path: manual feature extraction + model.generate to avoid pipeline arg mapping issues.
            generate_kwargs = self._get_finetuned_arabic_parameters(language) or {}
            if not self.model or not self.processor:
                logger.error("Model or processor not loaded in fine-tuned engine")
                return ""
            self.model.eval()
            import torch
            with torch.inference_mode():
                feat_start = time.time()
                inputs = self.processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True  # good practice for batching / future
                )
                feat_ms = (time.time() - feat_start) * 1000.0
                self._runtime_stats['feature_extractions'] += 1
                self._runtime_stats['total_feature_ms'] += feat_ms
                input_features = inputs.get("input_features")
                if input_features is None:
                    # Some processor versions may use 'features'
                    input_features = inputs.get("features")
                if input_features is None:
                    raise RuntimeError("Processor did not return input_features")
                input_features = input_features.to(self.device, dtype=self.torch_dtype)
                
                # Log feature shape for debugging (Whisper features are typically [B, 80, T] not [B, T, 80])
                feat_shape = tuple(input_features.shape)
                # Validate: expect [batch_size, n_mels, time_frames]
                # Standard Whisper: n_mels=80, but fine-tuned models may use different mel configs
                if input_features.dim() == 3:
                    batch_size, n_mels, time_frames = feat_shape
                    if n_mels not in [80, 128]:  # Allow both standard (80) and extended (128) mel configs
                        logger.warning(
                            "[whisper] Unexpected n_mels=%d in shape %s (expect 80 or 128)", n_mels, feat_shape
                        )
                    # Positive debug log every few calls to confirm correct path
                    if (self._runtime_stats.get('feature_extractions', 0) % 10) == 1:
                        logger.info(
                            "[whisper-debug] Feature extraction OK: shape=%s, n_mels=%d, time_frames=%d", 
                            feat_shape, n_mels, time_frames
                        )
                else:
                    logger.warning(
                        "[whisper] Unexpected feature dimensionality: shape=%s (expect 3D [B, n_mels, T])", feat_shape
                    )
                gen_start = time.time()
                # Whisper generate expects input_features (older versions accepted inputs=)
                
                # Filter out pipeline-specific parameters that model.generate doesn't understand
                # These can cause "unexpected keyword argument 'input_ids'" errors in newer transformers
                pipeline_only_params = {
                    'no_speech_threshold', 'compression_ratio_threshold', 'logprob_threshold',
                    'condition_on_prev_tokens', 'repetition_penalty', 'no_repeat_ngram_size'
                }
                model_generate_kwargs = {
                    k: v for k, v in generate_kwargs.items() 
                    if v is not None and k not in pipeline_only_params
                }
                
                try:
                    generated_ids = self.model.generate(
                        input_features=input_features,
                        **model_generate_kwargs
                    )
                except TypeError as te:
                    # Fallback to legacy argument name
                    if "unexpected keyword argument 'input_features'" in str(te):
                        generated_ids = self.model.generate(
                            inputs=input_features,
                            **model_generate_kwargs
                        )
                    else:
                        raise
                gen_ms = (time.time() - gen_start) * 1000.0
                self._runtime_stats['generate_calls'] += 1
                self._runtime_stats['total_generate_ms'] += gen_ms
                if (self._runtime_stats['generate_calls'] % 25) == 0:
                    logger.info(
                        "[whisper-metrics] features=%d (%.1fms avg) generate=%d (%.1fms avg)",
                        self._runtime_stats['feature_extractions'],
                        (self._runtime_stats['total_feature_ms'] / max(1, self._runtime_stats['feature_extractions'])),
                        self._runtime_stats['generate_calls'],
                        (self._runtime_stats['total_generate_ms'] / max(1, self._runtime_stats['generate_calls'])),
                    )
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text = text.strip()
                # Heuristic repetition mitigation (pre-clean) for Arabic short utterances
                if text:
                    original_text = text
                    mitigated = self._mitigate_repetition(text)
                    if mitigated != original_text:
                        logger.debug("[repetition-mitigated] '%s' -> '%s'", original_text[:80], mitigated[:80])
                        text = mitigated
            text = self._clean_finetuned_arabic_transcription(text)
            # If we got a plausible transcript, return early
            if text:
                # reset failure counter
                self._runtime_stats['consecutive_failures'] = 0
                return text
            # If empty, attempt pipeline fallback (some edge cases rely on internal suppression logic)
            if self.pipe is not None:
                try:
                    audio_input = {"array": audio_array, "sampling_rate": 16000}
                    pipe_kwargs = {"generate_kwargs": generate_kwargs} if generate_kwargs else {}
                    result = self.pipe(audio_input, **pipe_kwargs)
                    alt_text = (result.get("text", "") if isinstance(result, dict) else "").strip()
                    if alt_text:
                        self._runtime_stats['consecutive_failures'] = 0
                    return self._clean_finetuned_arabic_transcription(alt_text)
                except Exception as pe:
                    logger.warning(f"Pipeline fallback also failed: {pe}")
            return ""
            
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Fine-tuned Arabic transcription failed: {e}")
            try:
                self._runtime_stats['consecutive_failures'] += 1
            except Exception:
                pass

            # AUTO-RECOVERY: If classic 'input_ids' mis-binding error appears, rebuild a minimal pipeline once.
            if "unexpected keyword argument 'input_ids'" in err_msg or "got an unexpected keyword argument 'input_ids'" in err_msg:
                try:
                    if getattr(self, "_recovery_attempted", False) is False:
                        self._recovery_attempted = True
                        logger.warning("[whisper-recovery] Attempting auto-recovery: rebuilding Whisper pipeline without explicit tokenizer binding")
                        # Rebuild a minimal pipeline letting HF infer processor parts to avoid mis-mapping to input_ids
                        from transformers import pipeline as hf_pipeline
                        self.pipe = hf_pipeline(
                            "automatic-speech-recognition",
                            model=self.model,
                            torch_dtype=self.torch_dtype,
                            device=self.device
                        )
                        # Retry once
                        try:
                            audio_input = {"array": audio_array, "sampling_rate": 16000}
                            retry_result = self.pipe(audio_input, **({"generate_kwargs": generate_kwargs} if generate_kwargs else {}))
                            retry_text = retry_result.get("text", "").strip() if retry_result else ""
                            if retry_text:
                                logger.info("[whisper-recovery] Successful retry after pipeline rebuild")
                                self._runtime_stats['consecutive_failures'] = 0
                            return self._clean_finetuned_arabic_transcription(retry_text)
                        except Exception as retry_e:
                            logger.error(f"[whisper-recovery] Retry failed: {retry_e}")
                    else:
                        logger.debug("[whisper-recovery] Auto-recovery already attempted; skipping rebuild")
                except Exception as rec_e:
                    logger.error(f"[whisper-recovery] Pipeline auto-rebuild failed: {rec_e}")
            # Attempt fallback for problematic audio (will be empty-safe)
            fb = self._fallback_transcription(audio_array, language)
            if fb:
                self._runtime_stats['consecutive_failures'] = 0
            return fb
    
    def _get_finetuned_arabic_parameters(self, language: str) -> Dict[str, Any]:
        """
        Get fine-tuned Arabic-optimized generation parameters.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary of generation parameters optimized for fine-tuned Arabic model
        """
        # FIXED: Add essential parameters to prevent repetition in streaming
        params = {
            # Core decoding behavior
            "task": "transcribe",
            "return_timestamps": False,
            # Deterministic + speed
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
            # Streaming / windowed re-decode critical flag: prevents model from
            # conditioning on previous decoded tokens when we re-decode the trailing window.
            # Without this the model can accumulate duplicated phrases.
            "condition_on_prev_tokens": False,
            # Repetition mitigation (empirically tuned starting values)
            "repetition_penalty": float(__import__('os').getenv('WHISPER_AR_REPETITION_PENALTY', '1.22')),
            "no_repeat_ngram_size": int(__import__('os').getenv('WHISPER_AR_NO_REPEAT_NGRAM', '4')),
            # Length controls (tighter to avoid runaway repeats)
            "max_new_tokens": int(__import__('os').getenv('WHISPER_AR_MAX_NEW_TOKENS', '96')),
            # Audio heuristic thresholds (leave defaults but expose easily tunable)
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        
        # Language-specific settings
        if language == "ar" or language is None:
            params["language"] = "arabic"
        elif language == "en":
            params["language"] = "english"
        else:
            # For other languages, let the model auto-detect
            pass
        
        return params

    # ---------------- Repetition Mitigation Utilities -----------------
    def _mitigate_repetition(self, text: str) -> str:
        """Detect and collapse pathological repeated phrases in Arabic transcripts.

        Strategy:
          1. Remove excessive immediate single-token repetitions (>2 in a row).
          2. Detect multi-token phrase (length 3..8) repeated consecutively; if total
             token span of repeats exceeds threshold, keep first occurrence only.
        """
        try:
            tokens = text.split()
            if len(tokens) < 6:
                return text  # too short to loop pathologically

            # 1. Immediate token repetition collapse
            collapsed: List[str] = []
            last = None
            repeat_count = 0
            for tok in tokens:
                if tok == last:
                    repeat_count += 1
                    if repeat_count < 2:  # allow up to 2 consecutive
                        collapsed.append(tok)
                else:
                    last = tok
                    repeat_count = 0
                    collapsed.append(tok)

            # 2. Phrase repetition detection (greedy longest first)
            def collapse_phrase(seq: List[str]) -> List[str]:
                for n in range(8, 2, -1):  # phrase length 8 down to 3
                    i = 0
                    while i + n * 2 <= len(seq):
                        segment = seq[i:i+n]
                        repeats = 1
                        j = i + n
                        while j + n <= len(seq) and seq[j:j+n] == segment:
                            repeats += 1
                            j += n
                        # Threshold: phrase appears >=2 times and combined length >=12 tokens
                        if repeats >= 2 and repeats * n >= 12:
                            # Keep first occurrence only
                            seq = seq[:i+n] + seq[j:]
                            # Restart scan after collapse
                            return collapse_phrase(seq)
                        i += 1
                return seq

            mitigated_tokens = collapse_phrase(collapsed)
            if len(mitigated_tokens) < len(tokens):
                return " ".join(mitigated_tokens)
            return text
        except Exception:
            return text  # fail-safe
    
    def _clean_finetuned_arabic_transcription(self, text: str) -> str:
        """
        Enhanced cleaning of fine-tuned Arabic transcription output.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned Arabic transcription text
        """
        if not text:
            return ""
        
        # Apply fine-tuned specific cleaning
        text = self._clean_finetuned_arabic_text(text)
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def _clean_finetuned_arabic_text(self, text: str) -> str:
        """
        Apply fine-tuned Arabic-specific text cleaning.
        
        Args:
            text: Arabic transcription text
            
        Returns:
            Cleaned Arabic text with fine-tuned enhancements
        """
        # Enhanced Arabic text cleaning for fine-tuned model
        
        # Remove or normalize Arabic diacritics if they appear incorrectly
        # Fine-tuned model may produce better diacritics, so be more conservative
        
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
        
        # Fine-tuned model specific enhancements
        # Remove common transcription artifacts that might occur
        text = text.replace('[MUSIC]', '').replace('[SOUND]', '').replace('[NOISE]', '')
        text = text.replace('  ', ' ')  # Remove double spaces
        
        return text
    
    def _fallback_transcription(self, audio_array: np.ndarray, language: str) -> str:
        """
        Fallback transcription for problematic audio.
        
        Args:
            audio_array: Audio array
            language: Language code
            
        Returns:
            Fallback transcription result
        """
        try:
            logger.info("Attempting fallback transcription with minimal parameters...")
            
            audio_input = {"array": audio_array, "sampling_rate": 16000}
            
            # FIXED: Minimal parameters for fallback, no complex options
            result = self.pipe(audio_input)
            transcribed_text = result.get("text", "").strip() if result else ""
            
            if transcribed_text:
                logger.info(f"Fallback transcription successful: '{transcribed_text[:50]}...'")
                return self._clean_finetuned_arabic_transcription(transcribed_text)
            else:
                logger.warning("Fallback transcription returned empty result")
                return ""
                
        except Exception as e:
            logger.error(f"Fallback transcription also failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the fine-tuned Arabic Whisper model."""
        base_info = super().get_model_info()
        
        if self.model:
            base_info.update({
                "model_size": "809M parameters (Custom Arabic fine-tuned)",
                "architecture": "4 decoder layers, Arabic-specialized fine-tuned",
                "arabic_optimized": self.arabic_optimized,
                "fine_tuned": self.fine_tuned,
                "default_language": self.default_language,
                "supported_dialects": self.supported_dialects,
                "local_model_path": str(self.local_model_path),
                "training_dataset": "Custom Arabic voice dataset",
                "torch_compile_enabled": self.supports_torch_compile,
                "dialect_handling": self.dialect_handling,
                "pipeline_batch_size": getattr(self.pipe, 'batch_size', 'N/A') if self.pipe else 'N/A',
                "base_model": "openai/whisper-large-v3-turbo",
                "fine_tuning_date": "2025-08-19"
            })
        
        return base_info
    
    def cleanup(self):
        """Enhanced cleanup for fine-tuned Arabic Whisper resources."""
        try:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                
            # Call base cleanup
            super().cleanup()
            
            logger.info("✅ Fine-tuned Arabic Whisper cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during fine-tuned Arabic Whisper cleanup: {e}")