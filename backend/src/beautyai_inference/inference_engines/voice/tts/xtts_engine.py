"""
XTTS v2 Text-to-Speech Engine for BeautyAI Framework.

This engine implements the Coqui XTTS v2 model fine-tuned for Arabic TTS
with voice cloning and streaming support.

Key Features:
- Fine-tuned on Arabic speech dataset
- Local model loading (no internet required)
- Voice cloning support
- Streaming audio generation
- GPU-accelerated inference

Model Path: /home/lumi/.cache/geniusai-arabic-models/2025-12-06/xtts/

Author: BeautyAI Framework
Date: 2025-12-06
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

from ....core.model_interface import ModelInterface
from ....config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

# Default path for Genius AI Arabic XTTS model
GENIUS_XTTS_DEFAULT_PATH = Path(
    "/home/lumi/.cache/geniusai-arabic-models/2025-12-06/xtts/"
    "GPT_XTTS_v2.0_LJSpeech_FT-December-01-2025_08+00PM-2b091fe"
)

# Try to import TTS library
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    TTS_AVAILABLE = True
except ImportError:
    logger.warning("Coqui TTS library not available. Install with: pip install TTS")
    TTS_AVAILABLE = False


class XTTSEngine(ModelInterface):
    """
    Coqui XTTS v2 Text-to-Speech engine for Arabic voice synthesis.
    
    Features:
    - Fine-tuned Arabic voice model
    - Voice cloning from reference audio
    - Streaming audio generation
    - GPU-accelerated inference
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, model_path: Optional[Path] = None):
        """
        Initialize XTTS engine.
        
        Args:
            model_config: Optional model configuration
            model_path: Optional custom path to the model directory
        """
        if not TTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS library is required. Install with: pip install TTS"
            )
        
        self.config = model_config
        self.model_path = model_path or GENIUS_XTTS_DEFAULT_PATH
        self.model = None
        self.xtts_config = None
        
        # Device configuration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Audio configuration (from model config)
        self.output_sample_rate = 24000  # XTTS v2 outputs at 24kHz
        
        # Reference speaker for voice cloning
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.default_speaker_wav = None
        
        # Streaming configuration
        self._chunk_size_ms = 40  # 40ms chunks for low latency
        
        # Verify model exists
        if not self.model_path.exists():
            logger.warning(f"XTTS model not found at: {self.model_path}")
        else:
            model_size = self._get_model_size()
            logger.info(f"XTTSEngine initialized - Model path: {self.model_path}")
            logger.info(f"Model size: {model_size:.2f}GB, Device: {self.device}")
    
    def _get_model_size(self) -> float:
        """Calculate the total size of model files in GB."""
        try:
            total_size = 0
            for file in self.model_path.iterdir():
                if file.is_file() and file.suffix in ['.pth', '.pt', '.bin']:
                    total_size += file.stat().st_size
            return total_size / (1024 ** 3)
        except Exception:
            return 0.0
    
    def load_model(self) -> bool:
        """Load the XTTS model."""
        try:
            logger.info(f"Loading XTTS model from: {self.model_path}")
            
            # Validate model files
            model_file = self.model_path / "model.pth"
            config_file = self.model_path / "config.json"
            vocab_file = self.model_path / "vocab.json"
            
            if not model_file.exists():
                # Try best_model variant
                best_model = list(self.model_path.glob("best_model_*.pth"))
                if best_model:
                    model_file = best_model[0]
                    logger.info(f"Using best model: {model_file.name}")
                else:
                    logger.error(f"Model file not found in {self.model_path}")
                    return False
            
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return False
            
            # Load configuration
            logger.info("Loading XTTS configuration...")
            self.xtts_config = XttsConfig()
            self.xtts_config.load_json(str(config_file))
            
            # Initialize model
            logger.info("Initializing XTTS model...")
            self.model = Xtts.init_from_config(self.xtts_config)
            
            # Load checkpoint
            logger.info(f"Loading model checkpoint: {model_file.name}")
            self.model.load_checkpoint(
                self.xtts_config,
                checkpoint_path=str(model_file),
                vocab_path=str(vocab_file) if vocab_file.exists() else None,
                eval=True,
                use_deepspeed=False,
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Get output sample rate from config
            if hasattr(self.xtts_config, 'audio') and 'output_sample_rate' in self.xtts_config.audio:
                self.output_sample_rate = self.xtts_config.audio['output_sample_rate']
            
            logger.info(f"✅ XTTS model loaded successfully (output: {self.output_sample_rate}Hz)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load XTTS model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self) -> bool:
        """Unload the model and free resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.speaker_embedding is not None:
                del self.speaker_embedding
                self.speaker_embedding = None
            
            if self.gpt_cond_latent is not None:
                del self.gpt_cond_latent
                self.gpt_cond_latent = None
            
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
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return False
            
            logger.info(f"Computing speaker embedding from: {speaker_wav_path}")
            
            # Compute speaker conditioning
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                audio_path=speaker_wav_path,
                gpt_cond_len=self.xtts_config.gpt_cond_len if hasattr(self.xtts_config, 'gpt_cond_len') else 12,
                gpt_cond_chunk_len=self.xtts_config.gpt_cond_chunk_len if hasattr(self.xtts_config, 'gpt_cond_chunk_len') else 4,
            )
            
            self.default_speaker_wav = speaker_wav_path
            logger.info("✅ Speaker embedding computed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to compute speaker embedding: {e}")
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
        Convert text to speech using XTTS.
        
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
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            logger.debug(f"Generating TTS for: '{text[:50]}...' (lang={language})")
            
            # Get speaker conditioning
            gpt_cond_latent = self.gpt_cond_latent
            speaker_embedding = self.speaker_embedding
            
            if speaker_wav:
                # Compute conditioning for provided speaker
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=speaker_wav,
                    gpt_cond_len=12,
                    gpt_cond_chunk_len=4,
                )
            elif gpt_cond_latent is None:
                # No speaker reference, use default voice
                logger.warning("No speaker reference set. Using model's default voice characteristics.")
                # For XTTS, we need speaker conditioning. Generate a synthetic one if needed.
                # This is a fallback - ideally a reference speaker should be set.
                raise RuntimeError(
                    "XTTS requires a speaker reference. Call set_speaker_reference() first, "
                    "or provide speaker_wav parameter."
                )
            
            # Generate speech
            output = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
            )
            
            # Get audio array
            audio_array = output['wav']
            
            # Convert to proper format if needed
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            # Normalize and convert to int16
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Create output path
            if output_path is None:
                output_dir = Path(tempfile.gettempdir()) / "beautyai_tts"
                output_dir.mkdir(exist_ok=True)
                text_hash = abs(hash(text)) % 100000
                output_path = str(output_dir / f"xtts_{language}_{text_hash}.wav")
            
            # Save as WAV
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.output_sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
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
            
            logger.debug(f"Streaming TTS: '{text[:50]}...' (lang={language})")
            
            # Get speaker conditioning
            gpt_cond_latent = self.gpt_cond_latent
            speaker_embedding = self.speaker_embedding
            
            if speaker_wav:
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=speaker_wav,
                    gpt_cond_len=12,
                    gpt_cond_chunk_len=4,
                )
            
            if gpt_cond_latent is None:
                raise RuntimeError("No speaker reference available")
            
            # Use streaming inference if available
            if hasattr(self.model, 'inference_stream'):
                async for chunk in self._stream_inference(
                    text, language, gpt_cond_latent, speaker_embedding,
                    chunk_size_ms, target_sample_rate
                ):
                    yield chunk
            else:
                # Fallback: Generate full audio and chunk it
                logger.debug("Streaming not available, using chunked full generation")
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
    
    async def _stream_inference(
        self,
        text: str,
        language: str,
        gpt_cond_latent,
        speaker_embedding,
        chunk_size_ms: int,
        target_sample_rate: int
    ) -> AsyncGenerator[bytes, None]:
        """Stream inference using XTTS streaming capability."""
        try:
            # XTTS v2 streaming inference
            chunks_iterator = self.model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
                stream_chunk_size=chunk_size_ms,
            )
            
            for chunk_audio in chunks_iterator:
                if isinstance(chunk_audio, torch.Tensor):
                    chunk_audio = chunk_audio.cpu().numpy()
                
                # Normalize and convert to int16
                chunk_audio = np.clip(chunk_audio, -1.0, 1.0)
                chunk_int16 = (chunk_audio * 32767).astype(np.int16)
                
                # Resample if needed
                if self.output_sample_rate != target_sample_rate:
                    chunk_bytes = await self._resample_chunk(
                        chunk_int16.tobytes(),
                        self.output_sample_rate,
                        target_sample_rate
                    )
                else:
                    chunk_bytes = chunk_int16.tobytes()
                
                yield chunk_bytes
                
        except Exception as e:
            logger.error(f"Stream inference error: {e}")
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
            "name": "Coqui XTTS v2 (Genius Arabic Fine-tuned)",
            "type": "neural_tts",
            "model_path": str(self.model_path),
            "model_size_gb": self._get_model_size(),
            "output_sample_rate": self.output_sample_rate,
            "languages": self.get_supported_languages(),
            "gpu_required": True,
            "device": self.device,
            "loaded": self.model is not None,
            "has_speaker_reference": self.speaker_embedding is not None,
        }
