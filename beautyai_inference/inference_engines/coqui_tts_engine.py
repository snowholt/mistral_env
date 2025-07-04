"""
Coqui TTS Engine for BeautyAI Framework.
Implements high-quality text-to-speech using Coqui TTS with native Arabic support.
"""

import logging
import os
import time
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    COQUI_TTS_AVAILABLE = True
    logger.info("Coqui TTS library is available")
except ImportError:
    COQUI_TTS_AVAILABLE = False
    logger.warning("Coqui TTS library not available. Install with: pip install coqui-tts")

class CoquiTTSEngine(ModelInterface):
    """Coqui TTS Engine for high-quality neural text-to-speech synthesis."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the Coqui TTS engine."""
        self.config = model_config
        self.tts = None
        self.model_loaded = False
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Arabic TTS Models (prioritized for Arabic language)
        self.arabic_models = {
            "arabic_female_premium": "tts_models/ar/tn_arabicspeech/vits",      # Best Arabic model
            "arabic_multilingual": "tts_models/multilingual/multi-dataset/xtts_v2",  # Multilingual with Arabic
            "arabic_fairseq": "tts_models/ar/tn_arabicspeech/fairseq",          # Alternative Arabic model
        }
        
        # Available models by language
        self.language_models = {
            "ar": {
                "female": "tts_models/ar/tn_arabicspeech/vits",
                "male": "tts_models/ar/tn_arabicspeech/vits",  # Same model, different speaker
                "neutral": "tts_models/ar/tn_arabicspeech/vits",
                "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
            },
            "en": {
                "female": "tts_models/en/ljspeech/vits",
                "male": "tts_models/en/ek1/tacotron2",
                "neutral": "tts_models/en/ljspeech/vits",
                "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
            },
            "es": {
                "female": "tts_models/es/mai/tacotron2-DDC",
                "male": "tts_models/es/mai/tacotron2-DDC",
                "neutral": "tts_models/es/mai/tacotron2-DDC",
                "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
            },
            "fr": {
                "female": "tts_models/fr/mai/tacotron2-DDC",
                "male": "tts_models/fr/mai/tacotron2-DDC", 
                "neutral": "tts_models/fr/mai/tacotron2-DDC",
                "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
            },
            "de": {
                "female": "tts_models/de/thorsten/vits",
                "male": "tts_models/de/thorsten/vits",
                "neutral": "tts_models/de/thorsten/vits",
                "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
            }
        }
        
        # Speaker configurations for supported models
        self.speaker_configs = {
            "tts_models/ar/tn_arabicspeech/vits": {
                "speakers": ["female_voice", "male_voice"],
                "default_speaker": "female_voice"
            },
            "tts_models/multilingual/multi-dataset/xtts_v2": {
                "speakers": ["Claribel Dervla", "Daisy Studious", "Gracie Wise"],
                "default_speaker": "Claribel Dervla"
            }
        }
        
        # Supported languages
        self.supported_languages = ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja"]
        
        # Output directory for TTS files
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/coqui_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        """Load the Coqui TTS model."""
        if not COQUI_TTS_AVAILABLE:
            raise RuntimeError("Coqui TTS library not available. Install with: pip install coqui-tts")
        
        try:
            logger.info(f"Loading Coqui TTS model: {self.config.model_id}")
            
            # Determine the best model to load based on configuration
            if self.config.model_id in self.arabic_models.values():
                model_name = self.config.model_id
            elif "arabic" in self.config.model_id.lower():
                model_name = self.arabic_models["arabic_female_premium"]
            else:
                # Default to Arabic model for BeautyAI
                model_name = self.arabic_models["arabic_female_premium"]
            
            logger.info(f"Initializing Coqui TTS with model: {model_name}")
            logger.info(f"Device: {self.device}")
            
            # Initialize TTS with the selected model
            self.tts = TTS(model_name=model_name, gpu=(self.device == "cuda"))
            self.current_model = model_name
            
            self.model_loaded = True
            logger.info("✅ Coqui TTS model loaded successfully")
            logger.info(f"✅ Current model: {self.current_model}")
            
            # List available speakers for this model
            self._log_available_speakers()
            
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS model: {e}")
            raise RuntimeError(f"Failed to load Coqui TTS model: {e}")

    def unload_model(self) -> None:
        """Unload the Coqui TTS model."""
        try:
            if self.tts:
                # Clean up TTS resources
                del self.tts
                self.tts = None
                
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model_loaded = False
            self.current_model = None
            logger.info("Coqui TTS model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading Coqui TTS model: {e}")

    def _get_model_for_language(self, language: str, speaker_voice: str = "female") -> str:
        """Get the appropriate model for the given language and voice."""
        if language in self.language_models:
            lang_models = self.language_models[language]
            
            # For Arabic, prioritize native Arabic models
            if language == "ar":
                if speaker_voice in lang_models:
                    return lang_models[speaker_voice]
                else:
                    return lang_models["female"]  # Default to female
            else:
                # For other languages, use available models
                if speaker_voice in lang_models:
                    return lang_models[speaker_voice]
                else:
                    return lang_models.get("neutral", lang_models.get("female", lang_models["multilingual"]))
        else:
            # Fallback to multilingual model
            return "tts_models/multilingual/multi-dataset/xtts_v2"

    def _get_speaker_name(self, model_name: str, speaker_voice: str = "female") -> Optional[str]:
        """Get the appropriate speaker name for the model."""
        if model_name in self.speaker_configs:
            config = self.speaker_configs[model_name]
            speakers = config["speakers"]
            
            if speaker_voice.lower() == "female" and any("female" in s.lower() for s in speakers):
                return next((s for s in speakers if "female" in s.lower()), speakers[0])
            elif speaker_voice.lower() == "male" and any("male" in s.lower() for s in speakers):
                return next((s for s in speakers if "male" in s.lower()), speakers[0])
            else:
                return config["default_speaker"]
        return None

    def text_to_speech(
        self, 
        text: str, 
        language: str = "ar", 
        output_path: str = None,
        speaker_voice: str = "female",
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> str:
        """Convert text to speech and save to file."""
        if not self.model_loaded:
            raise RuntimeError("Coqui TTS model not loaded. Call load_model() first.")
        
        try:
            logger.info(f"Generating speech for text: '{text[:50]}...' (language: {language})")
            
            # Determine if we need to switch models
            target_model = self._get_model_for_language(language, speaker_voice)
            
            if target_model != self.current_model:
                logger.info(f"Switching from {self.current_model} to {target_model}")
                self.tts = TTS(model_name=target_model, gpu=(self.device == "cuda"))
                self.current_model = target_model
            
            # Get speaker name if model supports multiple speakers
            speaker_name = self._get_speaker_name(self.current_model, speaker_voice)
            
            # Generate output path
            if not output_path:
                timestamp = int(time.time())
                output_path = self.output_dir / f"coqui_tts_{language}_{timestamp}.wav"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech
            logger.info(f"Using model: {self.current_model}")
            if speaker_name:
                logger.info(f"Using speaker: {speaker_name}")
                
            # Call TTS synthesis
            if speaker_name and hasattr(self.tts, 'tts_to_file'):
                # Model with speaker support
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker=speaker_name,
                    language=language if language in ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja"] else None
                )
            else:
                # Model without speaker support or simple synthesis
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    language=language if language in ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja"] else None
                )
            
            if output_path.exists():
                logger.info(f"Speech saved to: {output_path}")
                return str(output_path)
            else:
                raise RuntimeError("TTS generation completed but output file not found")
                
        except Exception as e:
            logger.error(f"Coqui TTS generation failed: {e}")
            raise RuntimeError(f"Coqui TTS generation failed: {e}")

    def text_to_speech_bytes(
        self, 
        text: str, 
        language: str = "ar",
        speaker_voice: str = "female",
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> bytes:
        """Convert text to speech and return audio bytes."""
        if not self.model_loaded:
            raise RuntimeError("Coqui TTS model not loaded. Call load_model() first.")
        
        try:
            # Generate to temporary file and read bytes
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            result_path = self.text_to_speech(
                text=text,
                language=language,
                output_path=temp_path,
                speaker_voice=speaker_voice,
                emotion=emotion,
                speed=speed
            )
            
            # Read bytes
            with open(result_path, "rb") as f:
                audio_bytes = f.read()
            
            # Cleanup
            os.unlink(result_path)
            
            return audio_bytes
                
        except Exception as e:
            logger.error(f"Coqui TTS bytes generation failed: {e}")
            raise RuntimeError(f"Coqui TTS bytes generation failed: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text-to-speech (compatibility method)."""
        output_path = kwargs.get('output_path', None)
        language = kwargs.get('language', 'ar')
        speaker_voice = kwargs.get('speaker_voice', 'female')
        
        return self.text_to_speech(
            text=prompt,
            language=language,
            output_path=output_path,
            speaker_voice=speaker_voice
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate speech from chat messages."""
        # Extract the last user message
        if messages:
            last_message = messages[-1].get('content', '')
            return self.generate(last_message, **kwargs)
        return ""

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Stream speech generation (Coqui TTS doesn't support streaming, so we generate normally)."""
        result = self.chat(messages, **kwargs)
        if callback:
            callback(result)
        return result

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Benchmark Coqui TTS performance."""
        if not self.model_loaded:
            raise RuntimeError("Coqui TTS model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Generate speech for benchmarking
            output_path = kwargs.get('output_path', f"benchmark_coqui_{int(time.time())}.wav")
            result_path = self.text_to_speech(
                text=prompt,
                language=kwargs.get('language', 'ar'),
                output_path=output_path,
                speaker_voice=kwargs.get('speaker_voice', 'female')
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            characters_per_second = len(prompt) / generation_time if generation_time > 0 else 0
            
            # Get file size if output exists
            file_size = 0
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
            
            return {
                "generation_time": generation_time,
                "characters_per_second": characters_per_second,
                "text_length": len(prompt),
                "output_file": result_path,
                "file_size_bytes": file_size,
                "success": True,
                "engine": "Coqui TTS",
                "model": self.current_model,
                "language": kwargs.get('language', 'ar'),
                "speaker": kwargs.get('speaker_voice', 'female')
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "generation_time": end_time - start_time,
                "characters_per_second": 0,
                "text_length": len(prompt),
                "output_file": None,
                "file_size_bytes": 0,
                "success": False,
                "error": str(e),
                "engine": "Coqui TTS",
                "model": self.current_model or "unknown"
            }

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import psutil
        
        stats = {
            "system_memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "system_memory_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
                stats.update({
                    "gpu_memory_used_gb": gpu_memory,
                    "gpu_memory_cached_gb": gpu_memory_cached,
                    "gpu_available": True
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
                stats["gpu_available"] = False
        else:
            stats["gpu_available"] = False
            
        return stats

    def get_available_speakers(self, language: str = "ar") -> List[str]:
        """Get available speakers for a language."""
        if language in self.language_models:
            return list(self.language_models[language].keys())
        else:
            return ["female", "male", "neutral"]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.config.name,
            "model_id": self.config.model_id,
            "engine_type": "Coqui TTS",
            "current_model": self.current_model,
            "device": self.device,
            "supported_languages": self.supported_languages,
            "gpu_required": False,  # Works on CPU too
            "python_compatibility": "3.8+",
            "loaded": self.model_loaded,
            "neural_synthesis": True,
            "real_time": True,
            "arabic_optimized": True
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model_loaded

    def supports_language(self, language: str) -> bool:
        """Check if the engine supports a specific language."""
        return language in self.supported_languages

    def _log_available_speakers(self) -> None:
        """Log available speakers for the current model."""
        try:
            if self.current_model in self.speaker_configs:
                speakers = self.speaker_configs[self.current_model]["speakers"]
                logger.info(f"✅ Available speakers for {self.current_model}: {speakers}")
            else:
                logger.info(f"✅ Model {self.current_model} loaded (single speaker)")
        except Exception as e:
            logger.warning(f"Could not log speakers: {e}")

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by language."""
        return self.language_models

    def switch_model(self, language: str, speaker_voice: str = "female") -> bool:
        """Switch to a different model for optimal language/speaker combination."""
        try:
            target_model = self._get_model_for_language(language, speaker_voice)
            
            if target_model != self.current_model:
                logger.info(f"Switching TTS model from {self.current_model} to {target_model}")
                self.tts = TTS(model_name=target_model, gpu=(self.device == "cuda"))
                self.current_model = target_model
                logger.info(f"✅ Successfully switched to {target_model}")
                return True
            else:
                logger.info(f"Already using optimal model: {self.current_model}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def create_voice_clone(self, audio_file_path: str, clone_name: str = "custom_clone") -> bool:
        """
        Create a voice clone from an audio sample (XTTS v2 models only).
        
        Args:
            audio_file_path: Path to audio sample for cloning
            clone_name: Name for the voice clone
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if we're using XTTS v2 which supports voice cloning
            if "xtts_v2" not in self.current_model:
                logger.warning("Voice cloning requires XTTS v2 model. Switching...")
                if not self.switch_model("multilingual", "multilingual"):
                    return False
            
            logger.info(f"Creating voice clone '{clone_name}' from: {audio_file_path}")
            
            # Verify audio file exists
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return False
            
            # Store clone information (Coqui TTS handles cloning during synthesis)
            if not hasattr(self, 'voice_clones'):
                self.voice_clones = {}
            
            self.voice_clones[clone_name] = {
                "audio_path": audio_file_path,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Voice clone '{clone_name}' registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create voice clone: {e}")
            return False

    def use_voice_clone(self, text: str, clone_name: str, output_path: str = None) -> str:
        """
        Generate speech using a voice clone.
        
        Args:
            text: Text to synthesize
            clone_name: Name of the voice clone to use
            output_path: Optional output path
            
        Returns:
            str: Path to generated audio file
        """
        try:
            if not hasattr(self, 'voice_clones') or clone_name not in self.voice_clones:
                raise ValueError(f"Voice clone '{clone_name}' not found")
            
            clone_info = self.voice_clones[clone_name]
            speaker_wav = clone_info["audio_path"]
            
            # Generate output path
            if not output_path:
                timestamp = int(time.time())
                output_path = self.output_dir / f"clone_{clone_name}_{timestamp}.wav"
            
            logger.info(f"Generating speech with voice clone: {clone_name}")
            
            # Use XTTS v2 for voice cloning
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language="ar"  # Can be adjusted based on text
            )
            
            logger.info(f"✅ Voice clone synthesis completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Voice clone synthesis failed: {e}")
            raise RuntimeError(f"Voice clone synthesis failed: {e}")
