"""
XTTS v2 Text-to-Speech engine for BeautyAI Framework.
Integrates Coqui XTTS-v2 for high-quality multilingual TTS generation.
"""

import logging
import tempfile
import os
import io
import time
import wave
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch
import torchaudio
import numpy as np

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    XTTS_AVAILABLE = True
except ImportError:
    logger.warning("TTS library not available - using mock implementation for testing")
    # Mock classes for testing when TTS is not available
    class TTS:
        def __init__(self, model_name=None, progress_bar=False, gpu=False):
            self.model_name = model_name
            self.gpu = gpu
            self.languages = ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"]
            
        def tts_to_file(self, text, file_path, speaker=None, language="en", emotion="neutral", speed=1.0):
            # Create a mock audio file for testing (improved speech-like synthesis)
            import wave
            import math
            import struct
            import random
            
            with wave.open(file_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)  # Standard sample rate
                
                # Create audio based on text length
                words = text.split()
                duration = max(2.0, len(words) * 0.3 + 0.5)  # 0.3s per word + pauses
                num_samples = int(22050 * duration)
                
                # Generate more realistic speech-like audio
                audio_data = []
                word_index = 0
                silence_duration = 0.1  # Pause between words
                
                for i in range(num_samples):
                    t = i / 22050
                    
                    # Determine if we're in a word or silence
                    word_time = word_index * 0.3
                    in_word = (t - word_time) % 0.4 < 0.3  # 0.3s word, 0.1s silence
                    
                    if not in_word or word_index >= len(words):
                        # Silence between words
                        sample = 0
                    else:
                        # Generate speech-like formants for the current word
                        word = words[word_index]
                        
                        # Base fundamental frequency (varies by language/speaker)
                        f0 = 150 + (len(word) % 100)  # 150-250 Hz
                        
                        # Add prosodic variation
                        prosody = 1.0 + 0.3 * math.sin(2 * math.pi * t * 2)
                        f0 *= prosody
                        
                        # Generate harmonics (formant-like structure)
                        amplitude = 0.0
                        for harmonic in range(1, 6):  # First 5 harmonics
                            freq = f0 * harmonic
                            if freq < 11000:  # Within audible range
                                formant_strength = 1.0 / harmonic  # Weaker higher harmonics
                                amplitude += formant_strength * math.sin(2 * math.pi * freq * (t - word_time))
                        
                        # Add vowel-like formant emphasis
                        vowel_emphasis = 1.0 + 0.5 * math.sin(2 * math.pi * 800 * (t - word_time))
                        amplitude *= vowel_emphasis
                        
                        # Envelope for natural onset/offset
                        word_local_time = (t - word_time) % 0.3
                        envelope = 1.0
                        if word_local_time < 0.05:  # Attack
                            envelope = word_local_time / 0.05
                        elif word_local_time > 0.25:  # Release
                            envelope = (0.3 - word_local_time) / 0.05
                        
                        amplitude *= envelope * 0.2  # Scale down
                        
                        # Convert to sample
                        sample = int(amplitude * 32767)
                        sample = max(-32767, min(32767, sample))  # Clamp
                    
                    audio_data.append(struct.pack('<h', sample))
                    
                    # Update word index
                    if t > (word_index + 1) * 0.4 and word_index < len(words) - 1:
                        word_index += 1
                
                wav_file.writeframes(b''.join(audio_data))
            return file_path
            
        def tts(self, text, speaker=None, language="en", emotion="neutral", speed=1.0):
            # Return mock audio bytes
            return b"mock_audio_data"
    
    XttsConfig = None
    Xtts = None
    XTTS_AVAILABLE = False


class XTTSEngine(ModelInterface):
    """
    XTTS v2 Text-to-Speech engine for high-quality multilingual speech synthesis.
    
    Features:
    - Multilingual support (Arabic, English, and many more)
    - Voice cloning capabilities
    - High-quality neural synthesis
    - Real-time inference
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the XTTS engine with a model configuration."""
        self.config = model_config
        self.model = None
        self.tts_api = None
        self.mock_mode = not XTTS_AVAILABLE
        
        if not XTTS_AVAILABLE:
            logger.warning(
                "TTS library not available. Running in mock mode. "
                "To enable full TTS functionality, install with: pip install 'TTS>=0.22.0' "
                "Note: TTS requires Python < 3.12. Current Python version may not be supported."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 22050  # XTTS default sample rate
        
        # Language mapping for XTTS
        self.language_mapping = {
            "ar": "ar",  # Arabic
            "en": "en",  # English
            "es": "es",  # Spanish
            "fr": "fr",  # French
            "de": "de",  # German
            "it": "it",  # Italian
            "pt": "pt",  # Portuguese
            "pl": "pl",  # Polish
            "tr": "tr",  # Turkish
            "ru": "ru",  # Russian
            "nl": "nl",  # Dutch
            "cs": "cs",  # Czech
            "zh": "zh-cn",  # Chinese
            "ja": "ja",  # Japanese
            "hu": "hu",  # Hungarian
            "ko": "ko",  # Korean
        }
        
        # Default voices for each language
        self.default_voices = {
            "ar": "ar_speaker_0",  # Arabic default voice
            "en": "en_speaker_0",  # English default voice
        }

    def load_model(self) -> None:
        """Load the XTTS model into memory."""
        if self.mock_mode:
            logger.info("XTTS running in mock mode - TTS library not available")
            # Create mock TTS API for testing
            self.tts_api = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=False
            )
            logger.info("Mock XTTS model loaded successfully")
            return
            
        try:
            logger.info(f"Loading XTTS model: {self.config.model_id}")
            
            # Initialize TTS API with XTTS-v2
            self.tts_api = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            
            logger.info(f"XTTS model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise

    def unload_model(self) -> None:
        """Unload the model from memory and free resources."""
        try:
            if self.tts_api is not None:
                del self.tts_api
                self.tts_api = None
            
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("XTTS model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading XTTS model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate speech from text.
        This method returns the path to the generated audio file.
        """
        return self.text_to_speech(prompt, **kwargs)

    def text_to_speech(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        output_path: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        **kwargs
    ) -> str:
        """
        Convert text to speech using XTTS.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            output_path: Path to save the audio file (optional)
            emotion: Emotion/style for the voice (neutral, happy, sad, etc.)
            speed: Speech speed multiplier (0.5-2.0)
            **kwargs: Additional TTS parameters
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            if self.tts_api is None:
                raise RuntimeError("XTTS model not loaded. Call load_model() first.")
            
            # Map language code
            xtts_language = self.language_mapping.get(language, "en")
            
            # Use default voice if none specified
            if speaker_voice is None:
                speaker_voice = self.default_voices.get(language, "en_speaker_0")
            
            # Create output path if not provided
            if output_path is None:
                # Use tests directory instead of temp directory
                tests_dir = Path(__file__).parent.parent.parent / "tests"
                tests_dir.mkdir(exist_ok=True)
                output_path = tests_dir / f"tts_output_{hash(text)}.wav"
                output_path = str(output_path)
            
            logger.info(f"Generating TTS for language: {xtts_language}, speaker: {speaker_voice}")
            
            if self.mock_mode:
                # Mock mode: create a realistic speech-like audio file
                logger.info("Mock mode: Creating speech-like audio file")
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create a more realistic speech-like audio file
                import wave
                import math
                import struct
                import random
                
                with wave.open(output_path, 'w') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    
                    # Create realistic speech timing
                    words = text.split()
                    duration = max(2.0, len(words) * 0.4 + 1.0)  # More realistic timing
                    num_samples = int(self.sample_rate * duration)
                    
                    # Language-specific voice characteristics
                    if xtts_language == "ar":
                        base_f0 = 180  # Slightly higher for Arabic
                        formant_shift = 1.1
                    else:
                        base_f0 = 150  # English default
                        formant_shift = 1.0
                    
                    # Generate realistic speech-like audio
                    audio_data = []
                    current_word = 0
                    
                    for i in range(num_samples):
                        t = i / self.sample_rate
                        
                        # Calculate word timing
                        word_duration = 0.4  # seconds per word
                        silence_duration = 0.1  # pause between words
                        total_word_time = word_duration + silence_duration
                        
                        current_word_index = int(t / total_word_time)
                        local_time = (t % total_word_time)
                        
                        if current_word_index >= len(words) or local_time > word_duration:
                            # Silence period or end of text
                            sample = 0
                        else:
                            # Generate speech for current word
                            word = words[current_word_index]
                            word_progress = local_time / word_duration
                            
                            # Dynamic fundamental frequency with prosody
                            f0 = base_f0 * (1.0 + 0.3 * math.sin(2 * math.pi * word_progress))
                            
                            # Add sentence-level prosody
                            sentence_progress = current_word_index / max(1, len(words) - 1)
                            sentence_prosody = 1.0 + 0.2 * math.sin(math.pi * sentence_progress)
                            f0 *= sentence_prosody
                            
                            # Generate multiple formants for speech-like quality
                            amplitude = 0.0
                            
                            # Fundamental frequency
                            amplitude += 0.6 * math.sin(2 * math.pi * f0 * t)
                            
                            # First formant (vowel-like, around 800Hz)
                            f1 = 800 * formant_shift
                            amplitude += 0.4 * math.sin(2 * math.pi * f1 * t)
                            
                            # Second formant (consonant-like, around 1200Hz)
                            f2 = 1200 * formant_shift
                            amplitude += 0.3 * math.sin(2 * math.pi * f2 * t)
                            
                            # Third formant (clarity, around 2400Hz)
                            f3 = 2400 * formant_shift
                            amplitude += 0.2 * math.sin(2 * math.pi * f3 * t)
                            
                            # Word-specific variations based on word content
                            word_hash = hash(word) % 1000
                            freq_variation = 1.0 + (word_hash / 5000.0)  # Slight frequency variation
                            amplitude *= freq_variation
                            
                            # Natural envelope (attack, sustain, release)
                            if word_progress < 0.1:  # Attack (10%)
                                envelope = word_progress / 0.1
                            elif word_progress > 0.8:  # Release (20%)
                                envelope = (1.0 - word_progress) / 0.2
                            else:  # Sustain
                                envelope = 1.0
                            
                            # Add some natural variation
                            variation = 1.0 + 0.1 * math.sin(2 * math.pi * 5 * t)  # 5Hz variation
                            
                            amplitude *= envelope * variation * 0.25  # Scale to reasonable level
                            
                            # Convert to 16-bit sample
                            sample = int(amplitude * 32767)
                            sample = max(-32767, min(32767, sample))  # Clamp to valid range
                        
                        audio_data.append(struct.pack('<h', sample))
                    
                    wav_file.writeframes(b''.join(audio_data))
                
                logger.info(f"Mock TTS audio saved to: {output_path}")
                logger.info(f"Generated {duration:.1f}s of speech-like audio for {len(words)} words")
                return output_path
            
            # Real TTS generation
            wav = self.tts_api.tts(
                text=text,
                language=xtts_language,
                speaker=speaker_voice,
                emotion=emotion,
                speed=speed
            )
            
            # Convert to tensor and save
            if isinstance(wav, np.ndarray):
                wav_tensor = torch.from_numpy(wav).unsqueeze(0)
            else:
                wav_tensor = torch.tensor(wav).unsqueeze(0)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio file
            torchaudio.save(output_path, wav_tensor, self.sample_rate)
            
            logger.info(f"TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during TTS generation: {e}")
            raise

    def text_to_speech_bytes(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        audio_format: str = "wav",
        **kwargs
    ) -> bytes:
        """
        Convert text to speech and return audio as bytes.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            audio_format: Output audio format (wav, mp3)
            **kwargs: Additional TTS parameters
            
        Returns:
            bytes: Audio data as bytes
        """
        try:
            # Generate audio file
            temp_path = self.text_to_speech(text, language, speaker_voice, **kwargs)
            
            # Read file as bytes
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error generating TTS bytes: {e}")
            raise

    def text_to_speech_stream(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        **kwargs
    ) -> io.BytesIO:
        """
        Convert text to speech and return as stream.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            **kwargs: Additional TTS parameters
            
        Returns:
            io.BytesIO: Audio stream
        """
        try:
            # Get audio bytes
            audio_bytes = self.text_to_speech_bytes(text, language, speaker_voice, **kwargs)
            
            # Create stream
            audio_stream = io.BytesIO(audio_bytes)
            audio_stream.seek(0)
            
            return audio_stream
            
        except Exception as e:
            logger.error(f"Error creating TTS stream: {e}")
            raise

    def get_available_speakers(self, language: str = None) -> List[str]:
        """
        Get available speakers for the specified language.
        
        Args:
            language: Language code (optional)
            
        Returns:
            List[str]: List of available speaker names
        """
        try:
            if self.tts_api is None:
                return []
            
            # Return language-specific speakers or all speakers
            if language:
                language = self.language_mapping.get(language, language)
                # For XTTS, speakers are usually generic
                return [f"{language}_speaker_0", f"{language}_speaker_1"]
            else:
                # Return all available speakers
                speakers = []
                for lang in self.language_mapping.values():
                    speakers.extend([f"{lang}_speaker_0", f"{lang}_speaker_1"])
                return speakers
                
        except Exception as e:
            logger.error(f"Error getting available speakers: {e}")
            return []

    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for TTS.
        
        Returns:
            List[str]: List of supported language codes
        """
        return list(self.language_mapping.keys())

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response in a conversation (delegates to TTS for voice output).
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Returns:
            str: Path to generated audio file
        """
        # Extract the last user message for TTS
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise ValueError("No user message found in conversation")
        
        # Convert to speech
        return self.text_to_speech(user_message, **kwargs)

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark on the TTS model.
        
        Args:
            prompt: Text to use for benchmarking
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        try:
            start_time = time.time()
            
            # Generate speech
            output_path = self.text_to_speech(prompt, **kwargs)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Get audio file size
            file_size = 0
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
            
            # Calculate metrics
            text_length = len(prompt)
            audio_duration = file_size / (self.sample_rate * 2)  # Approximate duration
            
            return {
                "generation_time_ms": generation_time * 1000,
                "text_length": text_length,
                "audio_file_size_bytes": file_size,
                "audio_duration_seconds": audio_duration,
                "chars_per_second": text_length / generation_time if generation_time > 0 else 0,
                "device": self.device,
                "model_name": self.config.name,
                "sample_rate": self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error during TTS benchmark: {e}")
            return {"error": str(e)}

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """
        Stream a chat response (for TTS, this is similar to regular chat).
        
        Args:
            messages: List of conversation messages
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            str: Path to generated audio file
        """
        # For TTS, streaming doesn't make much sense, so we just do regular generation
        result = self.chat(messages, **kwargs)
        
        if callback:
            callback(result)
        
        return result

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = {
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": process.memory_percent(),
            }
            
            # Add GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                stats.update({
                    "gpu_memory_allocated_mb": gpu_memory,
                    "gpu_memory_cached_mb": gpu_cached,
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.tts_api is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.config.name,
            "model_id": self.config.model_id,
            "engine_type": "xtts",
            "device": self.device,
            "sample_rate": self.sample_rate,
            "supported_languages": self.get_supported_languages(),
            "is_loaded": self.is_model_loaded(),
            "model_config": {
                "name": self.config.name,
                "model_id": self.config.model_id,
                "engine_type": self.config.engine_type,
                "quantization": self.config.quantization,
                "dtype": self.config.dtype
            }
        }
