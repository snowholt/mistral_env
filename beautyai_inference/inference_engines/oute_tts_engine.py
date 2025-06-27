"""
OuteTTS Engine for BeautyAI Framework.
Implements text-to-speech using the OuteTTS library with neural speech synthesis.
"""

import logging
import os
import time
import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

try:
    import outetts
    OUTETTS_AVAILABLE = True
    logger.info("OuteTTS library is available")
except ImportError:
    OUTETTS_AVAILABLE = False
    logger.warning("OuteTTS library not available. Install with: pip install outetts")

class OuteTTSEngine(ModelInterface):
    """OuteTTS Engine for neural text-to-speech synthesis."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the OuteTTS engine."""
        self.config = model_config
        self.interface = None
        self.model_loaded = False
        self.current_speaker = None
        
        # Configuration for OuteTTS
        self.model_version = outetts.Models.VERSION_1_0_SIZE_1B
        self.backend = outetts.Backend.LLAMACPP
        self.quantization = outetts.LlamaCppQuantization.FP16
        
        # Speaker configurations
        self.available_speakers = {
            "en": {
                "female": "EN-FEMALE-1-NEUTRAL",
                "male": "EN-MALE-1-NEUTRAL",
                "neutral": "EN-FEMALE-1-NEUTRAL"
            },
            "ar": {
                "female": "AR-FEMALE-1-NEUTRAL",
                "male": "AR-MALE-1-NEUTRAL", 
                "neutral": "AR-FEMALE-1-NEUTRAL"
            },
            "es": {
                "female": "ES-FEMALE-1-NEUTRAL",
                "male": "ES-MALE-1-NEUTRAL",
                "neutral": "ES-FEMALE-1-NEUTRAL"
            },
            "fr": {
                "female": "FR-FEMALE-1-NEUTRAL",
                "male": "FR-MALE-1-NEUTRAL",
                "neutral": "FR-FEMALE-1-NEUTRAL"
            },
            "de": {
                "female": "DE-FEMALE-1-NEUTRAL",
                "male": "DE-MALE-1-NEUTRAL",
                "neutral": "DE-FEMALE-1-NEUTRAL"
            }
        }
        
        # Supported languages
        self.supported_languages = ["en", "ar", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja"]

    def load_model(self) -> None:
        """Load the OuteTTS model."""
        if not OUTETTS_AVAILABLE:
            raise RuntimeError("OuteTTS library not available. Install with: pip install outetts")
        
        try:
            logger.info(f"Loading OuteTTS model: {self.config.model_id}")
            
            # Configure the model
            model_config = outetts.ModelConfig.auto_config(
                model=self.model_version,
                backend=self.backend,
                quantization=self.quantization
            )
            
            # Initialize the interface
            logger.info("Initializing OuteTTS interface...")
            self.interface = outetts.Interface(config=model_config)
            
            # Load a default speaker
            logger.info("Loading default speaker...")
            self.current_speaker = self.interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
            
            self.model_loaded = True
            logger.info("✅ OuteTTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OuteTTS model: {e}")
            raise RuntimeError(f"Failed to load OuteTTS model: {e}")

    def unload_model(self) -> None:
        """Unload the OuteTTS model."""
        try:
            if self.interface:
                # Clean up resources
                del self.interface
                self.interface = None
                
            if self.current_speaker:
                del self.current_speaker
                self.current_speaker = None
                
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model_loaded = False
            logger.info("OuteTTS model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading OuteTTS model: {e}")

    def _get_speaker_id(self, language: str, speaker_voice: str = "female") -> str:
        """Get the appropriate speaker ID for the given language and voice."""
        if language in self.available_speakers:
            speaker_dict = self.available_speakers[language]
            return speaker_dict.get(speaker_voice, speaker_dict.get("female", "EN-FEMALE-1-NEUTRAL"))
        else:
            # Fallback to English
            return self.available_speakers["en"].get(speaker_voice, "EN-FEMALE-1-NEUTRAL")

    def text_to_speech(
        self, 
        text: str, 
        language: str = "en", 
        output_path: str = None,
        speaker_voice: str = "female",
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> str:
        """Convert text to speech and save to file."""
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
        
        try:
            # Get the appropriate speaker
            speaker_id = self._get_speaker_id(language, speaker_voice)
            speaker = self.interface.load_default_speaker(speaker_id)
            
            # Generate speech
            logger.info(f"Generating speech for text: '{text[:50]}...'")
            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=outetts.GenerationType.CHUNKED,
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=50
                    ),
                )
            )
            
            # Save to file
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output.save(str(output_file))
                logger.info(f"Speech saved to: {output_path}")
                return str(output_file)
            else:
                # Generate a default filename
                timestamp = int(time.time())
                default_path = f"outetts_output_{timestamp}.wav"
                output.save(default_path)
                return default_path
                
        except Exception as e:
            logger.error(f"OuteTTS generation failed: {e}")
            raise RuntimeError(f"OuteTTS generation failed: {e}")

    def text_to_speech_bytes(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: str = "female",
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> bytes:
        """Convert text to speech and return audio bytes."""
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
        
        try:
            # Get the appropriate speaker
            speaker_id = self._get_speaker_id(language, speaker_voice)
            speaker = self.interface.load_default_speaker(speaker_id)
            
            # Generate speech
            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=outetts.GenerationType.CHUNKED,
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=50
                    ),
                )
            )
            
            # Save to a temporary file and read bytes
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output.save(temp_file.name)
                with open(temp_file.name, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(temp_file.name)
                return audio_bytes
                
        except Exception as e:
            logger.error(f"OuteTTS bytes generation failed: {e}")
            raise RuntimeError(f"OuteTTS bytes generation failed: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text-to-speech (compatibility method)."""
        output_path = kwargs.get('output_path', None)
        language = kwargs.get('language', 'en')
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
        """Stream speech generation (OuteTTS doesn't support streaming, so we generate normally)."""
        result = self.chat(messages, **kwargs)
        if callback:
            callback(result)
        return result

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Benchmark OuteTTS performance."""
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Generate speech for benchmarking
            output_path = kwargs.get('output_path', f"benchmark_outetts_{int(time.time())}.wav")
            result_path = self.text_to_speech(
                text=prompt,
                language=kwargs.get('language', 'en'),
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
                "engine": "OuteTTS",
                "model": self.config.model_id,
                "language": kwargs.get('language', 'en'),
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
                "engine": "OuteTTS",
                "model": self.config.model_id
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

    def get_available_speakers(self, language: str = "en") -> List[str]:
        """Get available speakers for a language."""
        if language in self.available_speakers:
            return list(self.available_speakers[language].values())
        else:
            return list(self.available_speakers["en"].values())

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.config.name,
            "model_id": self.config.model_id,
            "engine_type": "OuteTTS",
            "backend": "LlamaCpp",
            "quantization": "FP16",
            "supported_languages": self.supported_languages,
            "gpu_required": True,
            "python_compatibility": "3.8+",
            "loaded": self.model_loaded,
            "neural_synthesis": True,
            "real_time": True
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model_loaded

    def supports_language(self, language: str) -> bool:
        """Check if the engine supports a specific language."""
        return language in self.supported_languages
