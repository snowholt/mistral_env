"""
Chatterbox Multilingual Text-to-Speech Engine for BeautyAI Framework.

This engine uses ResembleAI's Chatterbox Multilingual TTS model which supports
23 languages including Arabic and English with zero-shot voice cloning.

Key Features:
- 23 language support (Arabic, English, French, German, Spanish, etc.)
- Zero-shot voice cloning from reference audio (10s clip)
- GPU-accelerated inference (CUDA)
- Exaggeration and CFG weight control for expressive speech
- ~500M parameter Llama backbone

Model: ResembleAI/chatterbox (Multilingual variant)
HuggingFace: https://huggingface.co/ResembleAI/chatterbox
GitHub: https://github.com/resemble-ai/chatterbox

Author: BeautyAI Framework
Date: 2026-01-30
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
import torchaudio

from ....core.model_interface import ModelInterface
from ....config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

# Supported languages for Chatterbox Multilingual
CHATTERBOX_LANGUAGES = [
    "ar",  # Arabic
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "fi",  # Finnish
    "fr",  # French
    "he",  # Hebrew
    "hi",  # Hindi
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "ms",  # Malay
    "nl",  # Dutch
    "no",  # Norwegian
    "pl",  # Polish
    "pt",  # Portuguese
    "ru",  # Russian
    "sv",  # Swedish
    "sw",  # Swahili
    "tr",  # Turkish
    "zh",  # Chinese
]

# === FIX: Redirect spacy_pkuseg cache to writable directory ===
# Must be set BEFORE any chatterbox imports, as pkuseg reads config at import time
# NOTE: Systemd service has ProtectSystem=strict, so we must use a path in ReadWritePaths (backend/)
# Go up 6 levels to reach backend/ from backend/src/beautyai_inference/inference_engines/voice/tts/chatterbox_engine.py
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
_CACHE_BASE = os.path.join(_BACKEND_ROOT, "cache", "chatterbox")
_PKUSEG_CACHE_DIR = os.path.join(_CACHE_BASE, "pkuseg")
os.makedirs(_PKUSEG_CACHE_DIR, exist_ok=True)
os.environ["PKUSEG_HOME"] = _PKUSEG_CACHE_DIR
# Also set SPACY_DATA just in case
_SPACY_DATA_DIR = os.path.join(_CACHE_BASE, "spacy_data")
os.makedirs(_SPACY_DATA_DIR, exist_ok=True)
os.environ["SPACY_DATA"] = _SPACY_DATA_DIR

# Try to import Chatterbox library
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    logger.warning("Chatterbox TTS library not available. Install with: pip install chatterbox-tts")
    CHATTERBOX_AVAILABLE = False


class ChatterboxMultilingualEngine(ModelInterface):
    """
    ResembleAI Chatterbox Multilingual Text-to-Speech engine.
    
    Uses the Chatterbox Multilingual model (500M params) which supports:
    - 23 languages including Arabic, English, French, German, etc.
    - Zero-shot voice cloning from 10-second reference clip
    - Exaggeration control for expressive/dramatic speech
    - CFG weight for pacing control
    
    Output: 24kHz WAV audio
    VRAM: ~4-6GB on GPU
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Chatterbox Multilingual engine.
        
        Args:
            model_config: Optional model configuration
            model_path: Optional custom path for model cache
            device: Device to run on ("cuda" or "cpu")
            cache_dir: Optional custom cache directory for model files
        """
        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "Chatterbox TTS library is required. Install with: pip install chatterbox-tts"
            )
        
        self.config = model_config
        self.model_path = model_path
        self.model = None
        
        # Device configuration
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
        
        # Custom cache directory for model storage
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "beautyai-models" / "chatterbox")
        
        # Audio configuration (Chatterbox outputs 24kHz)
        self.output_sample_rate = 24000
        
        # Default speaker reference for voice cloning
        self.default_speaker_wav = None
        
        # Default generation parameters
        self.default_exaggeration = 0.5  # 0.0-1.0, higher = more expressive
        self.default_cfg_weight = 0.5    # 0.0-1.0, lower = slower pacing
        
        # Streaming configuration
        self._chunk_size_ms = 40  # 40ms chunks for low latency
        
        logger.info(f"ChatterboxMultilingualEngine initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Cache dir: {self.cache_dir}")
        logger.info(f"  Languages: {len(CHATTERBOX_LANGUAGES)} supported")
    
    def load_model(self) -> bool:
        """Load the Chatterbox Multilingual model."""
        try:
            logger.info("Loading Chatterbox Multilingual TTS model...")
            logger.info(f"  Device: {self.device}")
            
            # Set custom cache directory if specified
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                # Set HuggingFace cache environment variables
                os.environ["HF_HOME"] = self.cache_dir
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.cache_dir, "transformers")
                os.environ["HF_DATASETS_CACHE"] = os.path.join(self.cache_dir, "datasets")
                logger.info(f"  HF_HOME set to: {self.cache_dir}")
            
            # Load the multilingual model
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            
            # Store the sample rate from model
            if hasattr(self.model, 'sr'):
                self.output_sample_rate = self.model.sr
            
            logger.info(f"✅ Chatterbox Multilingual model loaded successfully")
            logger.info(f"  Output sample rate: {self.output_sample_rate}Hz")
            logger.info(f"  Model device: {self.device}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Chatterbox model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self) -> bool:
        """Unload the model and free resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            self.default_speaker_wav = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            logger.info("✅ Chatterbox model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unloading Chatterbox model: {e}")
            return False
    
    def set_speaker_reference(self, speaker_wav_path: str) -> bool:
        """
        Set a reference speaker for voice cloning.
        
        Args:
            speaker_wav_path: Path to a WAV file of the reference speaker (~10 seconds)
            
        Returns:
            bool: True if successful
        """
        try:
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
        language: str = "en",
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Convert text to speech using Chatterbox Multilingual.
        
        Args:
            text: Text to convert to speech
            language: Language code (ar, en, fr, de, etc.) - see CHATTERBOX_LANGUAGES
            speaker_wav: Optional path to speaker reference audio (~10s clip)
            output_path: Optional path to save the audio file
            exaggeration: Expressiveness control (0.0-1.0, default 0.5)
                         Higher = more expressive/dramatic
            cfg_weight: CFG weight for pacing (0.0-1.0, default 0.5)
                       Lower = slower, more deliberate pacing
            **kwargs: Additional parameters
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Validate language
            lang_code = language.lower()[:2]  # Normalize to 2-char code
            if lang_code not in CHATTERBOX_LANGUAGES:
                logger.warning(f"Language '{language}' not in supported list, using 'en'")
                lang_code = "en"
            
            logger.debug(f"Generating TTS: '{text[:50]}...' (lang={lang_code})")
            
            # Determine speaker reference
            ref_wav = speaker_wav or self.default_speaker_wav
            
            # Set generation parameters
            exag = exaggeration if exaggeration is not None else self.default_exaggeration
            cfg = cfg_weight if cfg_weight is not None else self.default_cfg_weight
            
            # Generate audio
            # Chatterbox API: model.generate(text, language_id=lang, audio_prompt_path=ref_wav)
            generate_kwargs = {
                "text": text,
                "language_id": lang_code,
            }
            
            # Add speaker reference if available
            if ref_wav and os.path.exists(ref_wav):
                generate_kwargs["audio_prompt_path"] = ref_wav
                logger.debug(f"Using speaker reference: {ref_wav}")
            
            # Add exaggeration/cfg if supported (check model attributes)
            if hasattr(self.model, 'generate'):
                # Check signature for additional params
                import inspect
                sig = inspect.signature(self.model.generate)
                if 'exaggeration' in sig.parameters:
                    generate_kwargs["exaggeration"] = exag
                if 'cfg_weight' in sig.parameters:
                    generate_kwargs["cfg_weight"] = cfg
            
            # Generate the audio tensor
            wav_tensor = self.model.generate(**generate_kwargs)
            
            # Create output path
            if output_path is None:
                output_dir = Path(tempfile.gettempdir()) / "beautyai_tts"
                output_dir.mkdir(exist_ok=True)
                text_hash = abs(hash(text)) % 100000
                output_path = str(output_dir / f"chatterbox_{lang_code}_{text_hash}.wav")
            
            # Save the audio file
            torchaudio.save(output_path, wav_tensor, self.model.sr)
            
            logger.debug(f"TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def stream_tts_chunks(
        self,
        text: str,
        language: str = "en",
        speaker_wav: Optional[str] = None,
        chunk_size_ms: int = 40,
        target_sample_rate: int = 16000,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as small chunks for duplex voice communication.
        
        Chatterbox generates audio synchronously, so we generate the full
        audio and then stream it in chunks with optional resampling.
        
        Args:
            text: Text to convert to speech
            language: Language code (ar, en, fr, etc.)
            speaker_wav: Optional path to speaker reference audio
            chunk_size_ms: Size of each audio chunk in milliseconds (default 40ms)
            target_sample_rate: Target sample rate for output (default 16kHz for WebRTC)
            exaggeration: Expressiveness control (0.0-1.0)
            cfg_weight: CFG weight for pacing (0.0-1.0)
            **kwargs: Additional parameters
            
        Yields:
            bytes: PCM16 audio chunks at target_sample_rate
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            text = text.strip()
            if not text:
                return
            
            logger.debug(f"Streaming Chatterbox TTS: '{text[:50]}...' (lang={language})")
            
            # Normalize language code
            lang_code = language.lower()[:2]
            if lang_code not in CHATTERBOX_LANGUAGES:
                lang_code = "en"
            
            # Determine speaker reference
            ref_wav = speaker_wav or self.default_speaker_wav
            
            # Set generation parameters
            exag = exaggeration if exaggeration is not None else self.default_exaggeration
            cfg = cfg_weight if cfg_weight is not None else self.default_cfg_weight
            
            # Generate full audio (wrapped in executor to not block event loop)
            loop = asyncio.get_event_loop()
            
            def _generate():
                generate_kwargs = {
                    "text": text,
                    "language_id": lang_code,
                }
                if ref_wav and os.path.exists(ref_wav):
                    generate_kwargs["audio_prompt_path"] = ref_wav
                
                # Add optional params if supported
                import inspect
                sig = inspect.signature(self.model.generate)
                if 'exaggeration' in sig.parameters:
                    generate_kwargs["exaggeration"] = exag
                if 'cfg_weight' in sig.parameters:
                    generate_kwargs["cfg_weight"] = cfg
                
                return self.model.generate(**generate_kwargs)
            
            wav_tensor = await loop.run_in_executor(None, _generate)
            
            # Get audio as numpy array
            # wav_tensor shape: (1, num_samples) or (num_samples,)
            if wav_tensor.dim() == 2:
                audio_np = wav_tensor.squeeze(0).cpu().numpy()
            else:
                audio_np = wav_tensor.cpu().numpy()
            
            source_rate = self.model.sr
            
            # Resample if needed
            if source_rate != target_sample_rate:
                audio_np = await self._resample_audio(audio_np, source_rate, target_sample_rate)
            
            # Convert to PCM16 bytes
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Calculate chunk size in bytes
            bytes_per_sample = 2  # int16
            samples_per_chunk = int((target_sample_rate * chunk_size_ms) / 1000)
            chunk_size_bytes = samples_per_chunk * bytes_per_sample
            
            # Stream chunks
            offset = 0
            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + chunk_size_bytes]
                if chunk:
                    yield chunk
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(chunk_size_ms / 1000 * 0.8)  # 80% of chunk duration
                offset += chunk_size_bytes
            
            logger.debug(f"Streaming complete: {len(audio_bytes)} bytes total")
                    
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _resample_audio(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio from source_rate to target_rate.
        
        Args:
            audio: Audio samples as numpy array
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio as numpy array
        """
        try:
            from scipy import signal
            
            # Calculate resampling ratio
            ratio = target_rate / source_rate
            target_length = int(len(audio) * ratio)
            
            # Use scipy's resample for better quality
            resampled = signal.resample(audio, target_length)
            
            return resampled.astype(np.float32)
            
        except ImportError:
            # Fallback to simple linear interpolation
            ratio = source_rate / target_rate
            target_length = int(len(audio) / ratio)
            
            indices = np.arange(target_length) * ratio
            base_indices = indices.astype(int)
            fracs = indices - base_indices
            
            # Handle boundary
            base_indices = np.clip(base_indices, 0, len(audio) - 2)
            
            resampled = audio[base_indices] * (1 - fracs) + audio[base_indices + 1] * fracs
            return resampled.astype(np.float32)
    
    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers (depends on reference audio for Chatterbox)."""
        speakers = []
        if self.default_speaker_wav:
            speakers.append(self.default_speaker_wav)
        return speakers
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return CHATTERBOX_LANGUAGES.copy()
    
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
        
        # Get audio duration
        audio_duration = 0.0
        try:
            import wave
            with wave.open(result_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_duration = frames / float(rate)
        except:
            pass
        
        return {
            "generation_time": generation_time,
            "characters_per_second": chars_per_second,
            "audio_duration": audio_duration,
            "real_time_factor": generation_time / audio_duration if audio_duration > 0 else 0,
            "audio_file": result_path,
            "engine": "chatterbox_multilingual",
            "sample_rate": self.output_sample_rate,
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        stats = {
            "memory_used_mb": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_cached_mb": 0.0,
        }
        
        if torch.cuda.is_available() and self.model is not None:
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return stats
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Chatterbox Multilingual TTS",
            "type": "neural_tts",
            "model_id": "ResembleAI/chatterbox",
            "variant": "multilingual",
            "parameters": "500M",
            "output_sample_rate": self.output_sample_rate,
            "languages": self.get_supported_languages(),
            "language_count": len(CHATTERBOX_LANGUAGES),
            "gpu_required": False,  # Can run on CPU, but slow
            "device": self.device,
            "loaded": self.model is not None,
            "has_speaker_reference": self.default_speaker_wav is not None,
            "features": [
                "zero-shot voice cloning",
                "23 language support",
                "exaggeration control",
                "cfg weight control",
                "Arabic support",
            ],
            "cache_dir": self.cache_dir,
        }
