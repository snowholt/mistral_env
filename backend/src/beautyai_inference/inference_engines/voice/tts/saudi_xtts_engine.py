"""
Saudi XTTS Text-to-Speech Engine for BeautyAI Framework.

This engine uses the fine-tuned Saudi Arabic XTTS v2 model from HuggingFace:
https://huggingface.co/AhmedEladl/saudi-tts

Key Features:
- Native Saudi Arabic dialect support
- Voice cloning from reference audio
- DeepSpeed acceleration for faster inference
- Pre-computed speaker embeddings (avoiding cold start)
- Automatic fallback to Edge TTS on failure

Model: AhmedEladl/saudi-tts (XTTS v2 fine-tuned)
Output: 24kHz WAV audio

Author: BeautyAI Framework
Date: 2026-01-16
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

# HuggingFace model ID
SAUDI_TTS_MODEL_ID = "AhmedEladl/saudi-tts"

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "beautyai-models" / "saudi-tts"

# Default speaker reference directory (relative to backend folder)
# Path: backend/speakers/saudi-female/reference.wav
# Engine path: backend/src/beautyai_inference/inference_engines/voice/tts/saudi_xtts_engine.py
# Need 6 parent levels: tts -> voice -> inference_engines -> beautyai_inference -> src -> backend
_ENGINE_FILE = Path(__file__).resolve()
_BACKEND_DIR = _ENGINE_FILE.parent.parent.parent.parent.parent.parent  # 6 levels up to backend/
DEFAULT_SPEAKER_DIR = _BACKEND_DIR / "speakers" / "saudi-female"

# Try to import TTS library (low-level XTTS classes)
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    TTS_AVAILABLE = True
except ImportError:
    logger.warning("Coqui TTS library not available. Install with: pip install TTS")
    TTS_AVAILABLE = False


class SaudiXTTSEngine(ModelInterface):
    """
    Saudi Arabic XTTS v2 Text-to-Speech engine.
    
    Uses the fine-tuned model from AhmedEladl/saudi-tts which is optimized
    for Saudi Arabic dialect. Features DeepSpeed acceleration and 
    pre-computed speaker embeddings to minimize latency.
    
    Supports:
    - Arabic (Saudi dialect) - Primary use case
    - Voice cloning from reference audio
    - GPU-accelerated inference with DeepSpeed
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        model_path: Optional[Path] = None,
        speaker_wav_path: Optional[Path] = None,
        use_deepspeed: bool = True,
    ):
        """
        Initialize Saudi XTTS engine.
        
        Args:
            model_config: Optional model configuration
            model_path: Path to downloaded model directory (default: ~/.cache/beautyai-models/saudi-tts)
            speaker_wav_path: Path to reference speaker WAV file
            use_deepspeed: Whether to enable DeepSpeed acceleration (default: True)
        """
        if not TTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS library is required. Install with: pip install TTS"
            )
        
        self.config = model_config
        self.model: Optional[Xtts] = None
        self.xtts_config: Optional[XttsConfig] = None
        
        # Model paths
        self.model_path = model_path or DEFAULT_CACHE_DIR
        self.speaker_wav_path = speaker_wav_path or (DEFAULT_SPEAKER_DIR / "reference.wav")
        
        # DeepSpeed configuration
        self.use_deepspeed = use_deepspeed and torch.cuda.is_available()
        
        # Device configuration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Audio configuration (XTTS v2 outputs at 24kHz)
        self.output_sample_rate = 24000
        
        # Pre-computed speaker conditioning (set during load_model)
        self.gpt_cond_latent: Optional[torch.Tensor] = None
        self.speaker_embedding: Optional[torch.Tensor] = None
        
        # Streaming configuration
        self._chunk_size_ms = 40  # 40ms chunks for low latency
        
        logger.info(f"SaudiXTTSEngine initialized")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Speaker ref: {self.speaker_wav_path}")
        logger.info(f"  DeepSpeed: {self.use_deepspeed}")
        logger.info(f"  Device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Load the Saudi XTTS model and pre-compute speaker embeddings.
        
        This method:
        1. Loads the XTTS config from config.json
        2. Initializes the model with DeepSpeed acceleration
        3. Pre-computes speaker conditioning latents for zero cold-start
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            model_dir = Path(self.model_path)
            
            # Verify model files exist
            config_path = model_dir / "config.json"
            vocab_path = model_dir / "vocab.json"
            
            if not config_path.exists():
                logger.error(f"Model config not found: {config_path}")
                logger.error("Please run the download script: python scripts/download_saudi_tts.py")
                return False
            
            if not vocab_path.exists():
                logger.error(f"Vocabulary file not found: {vocab_path}")
                return False
            
            logger.info(f"Loading Saudi XTTS model from: {model_dir}")
            
            # Load XTTS configuration
            self.xtts_config = XttsConfig()
            self.xtts_config.load_json(str(config_path))
            
            # Initialize model from config
            logger.info("Initializing XTTS model...")
            self.model = Xtts.init_from_config(self.xtts_config)
            
            # Load checkpoint with DeepSpeed if available
            logger.info(f"Loading checkpoint (DeepSpeed: {self.use_deepspeed})...")
            self.model.load_checkpoint(
                self.xtts_config,
                checkpoint_dir=str(model_dir),
                use_deepspeed=self.use_deepspeed,
                vocab_path=str(vocab_path),
            )
            
            # Move to GPU
            if torch.cuda.is_available():
                self.model.cuda()
                logger.info(f"Model moved to CUDA: {self.device}")
            
            # Pre-compute speaker conditioning (critical for avoiding cold start!)
            if not self._precompute_speaker_conditioning():
                logger.warning("Speaker conditioning failed - TTS will require speaker_wav per call")
            
            logger.info(f"✅ Saudi XTTS model loaded successfully (output: {self.output_sample_rate}Hz)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Saudi XTTS model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _precompute_speaker_conditioning(self) -> bool:
        """
        Pre-compute speaker conditioning latents from reference audio.
        
        This is called once during model loading to avoid the latency
        of computing embeddings on every TTS request.
        
        Returns:
            bool: True if successful
        """
        try:
            speaker_path = Path(self.speaker_wav_path)
            
            if not speaker_path.exists():
                logger.warning(f"Speaker reference not found: {speaker_path}")
                logger.warning("TTS calls will require explicit speaker_wav parameter")
                return False
            
            logger.info(f"Computing speaker latents from: {speaker_path}")
            
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[str(speaker_path)]
            )
            
            logger.info("✅ Speaker conditioning pre-computed (zero cold-start ready)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to compute speaker conditioning: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload the model and free GPU memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            self.xtts_config = None
            self.gpt_cond_latent = None
            self.speaker_embedding = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Saudi XTTS model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unloading Saudi XTTS model: {e}")
            return False
    
    def set_speaker_reference(self, speaker_wav_path: str) -> bool:
        """
        Update the speaker reference and re-compute conditioning latents.
        
        Args:
            speaker_wav_path: Path to a WAV file of the reference speaker
            
        Returns:
            bool: True if successful
        """
        try:
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return False
            
            if not os.path.exists(speaker_wav_path):
                logger.error(f"Speaker reference file not found: {speaker_wav_path}")
                return False
            
            self.speaker_wav_path = Path(speaker_wav_path)
            return self._precompute_speaker_conditioning()
            
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
        temperature: float = 0.75,
        **kwargs
    ) -> str:
        """
        Convert text to speech using Saudi XTTS.
        
        Args:
            text: Text to convert to speech (Arabic)
            language: Language code (default: "ar" for Arabic)
            speaker_wav: Optional path to override speaker reference
            output_path: Optional path to save the audio file
            temperature: Generation temperature (default: 0.75)
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            str: Path to the generated audio file
        """
        import torchaudio
        
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            logger.debug(f"Saudi TTS generating: '{text[:50]}...' (lang={language})")
            
            # Determine speaker conditioning
            gpt_cond = self.gpt_cond_latent
            speaker_emb = self.speaker_embedding
            
            # If custom speaker_wav provided, compute new conditioning
            if speaker_wav and os.path.exists(speaker_wav):
                logger.debug(f"Using custom speaker reference: {speaker_wav}")
                gpt_cond, speaker_emb = self.model.get_conditioning_latents(
                    audio_path=[speaker_wav]
                )
            elif gpt_cond is None or speaker_emb is None:
                raise RuntimeError(
                    "No speaker conditioning available. Either call set_speaker_reference() "
                    "or provide speaker_wav parameter."
                )
            
            # Generate speech
            out = self.model.inference(
                text,
                language,
                gpt_cond,
                speaker_emb,
                temperature=temperature,
            )
            
            # Create output path
            if output_path is None:
                output_dir = Path(tempfile.gettempdir()) / "beautyai_tts"
                output_dir.mkdir(exist_ok=True)
                text_hash = abs(hash(text)) % 100000
                output_path = str(output_dir / f"saudi_tts_{language}_{text_hash}.wav")
            
            # Save audio (24kHz WAV)
            audio_tensor = torch.tensor(out["wav"]).unsqueeze(0)
            torchaudio.save(output_path, audio_tensor, self.output_sample_rate)
            
            logger.debug(f"Saudi TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Saudi TTS generation failed: {e}")
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
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            text = text.strip()
            if not text:
                return
            
            logger.debug(f"Streaming Saudi TTS: '{text[:50]}...' (lang={language})")
            
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
            logger.error(f"Streaming Saudi TTS error: {e}")
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
        """Get available speakers."""
        if self.speaker_wav_path and Path(self.speaker_wav_path).exists():
            return [str(self.speaker_wav_path)]
        return []
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages (primarily Arabic)."""
        return ["ar"]
    
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
            "engine": "saudi_xtts",
            "model": SAUDI_TTS_MODEL_ID,
            "sample_rate": self.output_sample_rate,
            "deepspeed_enabled": self.use_deepspeed,
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        stats = {
            "memory_used_mb": 0.0,
            "gpu_memory_used_mb": 0.0,
        }
        
        if torch.cuda.is_available() and self.model is not None:
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return stats
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def has_speaker_conditioning(self) -> bool:
        """Check if speaker conditioning is pre-computed."""
        return self.gpt_cond_latent is not None and self.speaker_embedding is not None
