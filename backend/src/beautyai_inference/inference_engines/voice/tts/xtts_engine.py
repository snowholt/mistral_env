"""
XTTS v2 Text-to-Speech Engine for BeautyAI Framework.

This engine uses the official Coqui XTTS v2 multilingual model which properly
supports Arabic and 16 other languages with voice cloning support.

Key Features:
- Official XTTS v2 multilingual model (proper Arabic support)
- Voice cloning from reference audio
- Streaming audio generation  
- GPU-accelerated inference

Model: tts_models/multilingual/multi-dataset/xtts_v2

Author: BeautyAI Framework
Date: 2025-12-09 (Updated to use official XTTS v2)
"""

import logging
import os
import io
import wave
import tempfile
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import numpy as np

import torch

# Monkey-patch torch.load for PyTorch 2.6+ compatibility with TTS library
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ....core.model_interface import ModelInterface
from ....config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

# Official XTTS v2 model name
OFFICIAL_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Try to import TTS library
try:
    from TTS.api import TTS as CoquiTTS
    TTS_AVAILABLE = True
except ImportError:
    logger.warning("Coqui TTS library not available. Install with: pip install TTS")
    TTS_AVAILABLE = False


class XTTSEngine(ModelInterface):
    """
    Coqui XTTS v2 Text-to-Speech engine for multilingual voice synthesis.
    
    Uses the official XTTS v2 multilingual model which properly supports:
    - Arabic, English, Spanish, French, German, Italian, Portuguese
    - Polish, Turkish, Russian, Dutch, Czech, Chinese, Japanese, Korean, etc.
    
    Features:
    - Voice cloning from reference audio
    - High-quality neural synthesis
    - GPU-accelerated inference
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, model_path: Optional[Path] = None):
        """
        Initialize XTTS engine.
        
        Args:
            model_config: Optional model configuration
            model_path: Optional custom path (ignored - uses official model)
        """
        if not TTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS library is required. Install with: pip install TTS"
            )
        
        self.config = model_config
        self.model = None
        self.tts = None  # TTS API object
        
        # Device configuration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Audio configuration
        self.output_sample_rate = 24000  # XTTS v2 outputs at 24kHz
        
        # Reference speaker for voice cloning (not used with TTS.api)
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.default_speaker_wav = None
        
        # Streaming configuration
        self._chunk_size_ms = 40  # 40ms chunks for low latency
        
        logger.info(f"XTTSEngine initialized - Using official model: {OFFICIAL_XTTS_MODEL}")
        logger.info(f"Device: {self.device}")
    
    def load_model(self) -> bool:
        """Load the official XTTS v2 model via TTS API."""
        try:
            logger.info(f"Loading official XTTS v2 model: {OFFICIAL_XTTS_MODEL}")
            
            # Initialize TTS with official multilingual XTTS v2
            # gpu=True will use CUDA if available
            use_gpu = torch.cuda.is_available()
            
            logger.info(f"Initializing CoquiTTS (GPU: {use_gpu}, Device: {self.device})...")
            self.tts = CoquiTTS(OFFICIAL_XTTS_MODEL, gpu=use_gpu)
            
            # The TTS API handles model loading internally
            # Set model reference for compatibility
            self.model = self.tts
            
            logger.info(f"✅ XTTS v2 model loaded successfully (output: {self.output_sample_rate}Hz)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load XTTS model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self) -> bool:
        """Unload the model and free resources."""
        try:
            if self.tts is not None:
                del self.tts
                self.tts = None
            
            if self.model is not None:
                del self.model
                self.model = None
            
            self.default_speaker_wav = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ XTTS model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unloading XTTS model: {e}")
            return False
    
    def set_speaker_reference(self, speaker_wav_path: str) -> bool:
        """
        Set a reference speaker for voice cloning.
        
        Args:
            speaker_wav_path: Path to a WAV file of the reference speaker
            
        Returns:
            bool: True if successful
        """
        try:
            if self.tts is None:
                logger.error("Model not loaded. Call load_model() first.")
                return False
            
            if not os.path.exists(speaker_wav_path):
                logger.error(f"Speaker reference file not found: {speaker_wav_path}")
                return False
            
            self.default_speaker_wav = speaker_wav_path
            logger.info(f"✅ Speaker reference set: {speaker_wav_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set speaker reference: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate speech from text and return path to audio file."""
        return self.text_to_speech(prompt, **kwargs)
    
    def text_to_speech(
        self,
        text: str,
        language: str = "ar",
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convert text to speech using XTTS v2.
        
        Args:
            text: Text to convert to speech
            language: Language code (ar, en, etc.)
            speaker_wav: Optional path to speaker reference audio
            output_path: Optional path to save the audio file
            **kwargs: Additional parameters
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            if self.tts is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            logger.debug(f"Generating TTS for: '{text[:50]}...' (lang={language})")
            
            # Determine speaker reference
            ref_wav = speaker_wav or self.default_speaker_wav
            
            if ref_wav is None:
                raise RuntimeError(
                    "XTTS requires a speaker reference. Call set_speaker_reference() first, "
                    "or provide speaker_wav parameter."
                )
            
            if not os.path.exists(ref_wav):
                raise RuntimeError(f"Speaker reference file not found: {ref_wav}")
            
            # Create output path
            if output_path is None:
                output_dir = Path(tempfile.gettempdir()) / "beautyai_tts"
                output_dir.mkdir(exist_ok=True)
                text_hash = abs(hash(text)) % 100000
                output_path = str(output_dir / f"xtts_{language}_{text_hash}.wav")
            
            # Generate speech using TTS API
            # This method handles speaker cloning internally
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_wav,
                language=language,
            )
            
            logger.debug(f"TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            raise
    
    async def stream_tts_chunks(
        self,
        text: str,
        language: str = "ar",
        speaker_wav: Optional[str] = None,
        chunk_size_ms: int = 40,
        target_sample_rate: int = 16000,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as small chunks for duplex voice communication.
        
        The TTS API generates audio synchronously, so we generate the full
        audio and then stream it in chunks.
        
        Args:
            text: Text to convert to speech
            language: Language code
            speaker_wav: Optional path to speaker reference audio
            chunk_size_ms: Size of each audio chunk in milliseconds
            target_sample_rate: Target sample rate for output
            **kwargs: Additional parameters
            
        Yields:
            bytes: PCM16 audio chunks
        """
        try:
            if self.tts is None:
                raise RuntimeError("Model not loaded")
            
            text = text.strip()
            if not text:
                return
            
            logger.debug(f"Streaming TTS: '{text[:50]}...' (lang={language})")
            
            # Generate full audio first
            audio_path = self.text_to_speech(text, language, speaker_wav)
            
            # Read and stream chunks
            with wave.open(audio_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                samples_per_chunk = int((sample_rate * chunk_size_ms) / 1000)
                
                while True:
                    chunk = wav_file.readframes(samples_per_chunk)
                    if not chunk:
                        break
                    
                    # Resample if needed
                    if sample_rate != target_sample_rate:
                        chunk = await self._resample_chunk(
                            chunk, sample_rate, target_sample_rate
                        )
                    
                    yield chunk
            
            # Cleanup temp file
            try:
                os.remove(audio_path)
            except:
                pass
                    
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            raise
    
    async def _resample_chunk(
        self,
        audio_data: bytes,
        source_rate: int,
        target_rate: int
    ) -> bytes:
        """Simple audio resampling for chunks."""
        if source_rate == target_rate:
            return audio_data
        
        import array
        source_samples = array.array('h')
        source_samples.frombytes(audio_data)
        
        ratio = source_rate / target_rate
        target_length = int(len(source_samples) / ratio)
        
        target_samples = array.array('h')
        for i in range(target_length):
            source_idx = i * ratio
            base_idx = int(source_idx)
            
            if base_idx + 1 < len(source_samples):
                frac = source_idx - base_idx
                sample = source_samples[base_idx] * (1 - frac) + source_samples[base_idx + 1] * frac
                target_samples.append(int(sample))
            elif base_idx < len(source_samples):
                target_samples.append(source_samples[base_idx])
        
        return target_samples.tobytes()
    
    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers (for XTTS, this depends on reference audio)."""
        if self.default_speaker_wav:
            return [self.default_speaker_wav]
        return []
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", 
                "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Not applicable for TTS engine."""
        raise NotImplementedError("Chat not supported for TTS engine")
    
    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Benchmark TTS generation."""
        import time
        start_time = time.time()
        
        result_path = self.text_to_speech(prompt, **kwargs)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        chars_per_second = len(prompt) / generation_time if generation_time > 0 else 0
        
        return {
            "generation_time": generation_time,
            "characters_per_second": chars_per_second,
            "audio_file": result_path,
            "engine": "xtts",
            "sample_rate": self.output_sample_rate,
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        stats = {
            "memory_used_mb": 0.0,
            "gpu_memory_used_mb": 0.0,
        }
        
        if torch.cuda.is_available() and self.tts is not None:
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return stats
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.tts is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Coqui XTTS v2 Multilingual",
            "type": "neural_tts",
            "model_name": OFFICIAL_XTTS_MODEL,
            "output_sample_rate": self.output_sample_rate,
            "languages": self.get_supported_languages(),
            "gpu_required": True,
            "device": self.device,
            "loaded": self.tts is not None,
            "has_speaker_reference": self.default_speaker_wav is not None,
        }
