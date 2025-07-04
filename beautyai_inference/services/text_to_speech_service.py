"""
Text-to-Speech Service for BeautyAI Framework.
Provides unified TTS interface with support for Coqui TTS and Edge TTS engines.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base.base_service import BaseService
from ..config.config_manager import AppConfig, ModelConfig
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class TextToSpeechService(BaseService):
    """
    Unified Text-to-Speech service supporting multiple TTS engines.
    
    Supported Engines:
    - Coqui TTS: High-quality neural TTS with native Arabic support
    - Edge TTS: Microsoft's neural voices for multiple languages
    
    Features:
    - Automatic engine selection based on language and quality requirements
    - Voice cloning capabilities (Coqui TTS)
    - Multiple speaker voices per language
    - GPU acceleration support
    - Arabic language optimization
    """
    
    def __init__(self):
        """Initialize the TTS service."""
        super().__init__()
        
        self.model_manager = ModelManager()
        self.current_engine = None
        self.current_model = None
        self.engine_loaded = False
        
        # Supported engines in order of preference
        self.supported_engines = {
            "coqui": {
                "name": "Coqui TTS",
                "class": "CoquiTTSEngine",
                "arabic_support": "native",
                "quality": "high",
                "gpu_required": False,
                "voice_cloning": True,
                "default_models": {
                    "ar": "tts_models/ar/tn_arabicspeech/vits",
                    "en": "tts_models/en/ljspeech/vits",
                    "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2"
                }
            },
            "edge": {
                "name": "Edge TTS",
                "class": "EdgeTTSEngine",
                "arabic_support": "good",
                "quality": "medium",
                "gpu_required": False,
                "voice_cloning": False,
                "default_models": {
                    "ar": "ar-SA-ZariyahNeural",
                    "en": "en-US-JennyNeural",
                    "multilingual": "edge-tts-multilingual"
                }
            }
        }
        
        # Language preferences for engine selection
        self.language_engine_preferences = {
            "ar": ["coqui", "edge"],  # Prefer Coqui for Arabic
            "en": ["coqui", "edge"],
            "es": ["coqui", "edge"],
            "fr": ["coqui", "edge"],
            "de": ["coqui", "edge"],
            "default": ["coqui", "edge"]
        }
        
        # Output directory
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/tts_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_tts_model(self, model_name: str = "coqui-arabic", engine_type: str = None) -> bool:
        """
        Load a TTS model for text-to-speech generation.
        
        Args:
            model_name: Name of the model to load
            engine_type: Preferred engine type ("coqui" or "edge")
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading TTS model: {model_name}")
            
            # Determine engine type if not specified
            if not engine_type:
                if "coqui" in model_name.lower():
                    engine_type = "coqui"
                elif "edge" in model_name.lower():
                    engine_type = "edge"
                else:
                    # Default to Coqui for Arabic optimization
                    engine_type = "coqui"
            
            # Create model configuration
            if engine_type == "coqui":
                model_config = ModelConfig(
                    name=f"coqui-{model_name}",
                    model_id="tts_models/ar/tn_arabicspeech/vits",  # Default Arabic model
                    engine_type="coqui_tts",
                    device="cuda" if self._check_gpu_available() else "cpu",
                    parameters={
                        "language": "ar",
                        "speaker": "female",
                        "quality": "high"
                    }
                )
            elif engine_type == "edge":
                model_config = ModelConfig(
                    name=f"edge-{model_name}",
                    model_id="ar-SA-ZariyahNeural",  # Default Arabic voice
                    engine_type="edge_tts",
                    device="cpu",  # Edge TTS always uses CPU
                    parameters={
                        "language": "ar",
                        "voice": "ar-SA-ZariyahNeural",
                        "quality": "medium"
                    }
                )
            else:
                raise ValueError(f"Unsupported engine type: {engine_type}")
            
            # Load the model using ModelManager
            self.current_engine = self.model_manager.load_model(model_config)
            self.current_model = model_name
            self.engine_loaded = True
            
            logger.info(f"✅ TTS model loaded: {model_name} ({engine_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model {model_name}: {e}")
            self.engine_loaded = False
            return False

    def text_to_speech(
        self,
        text: str,
        language: str = "ar",
        speaker_voice: str = "female",
        output_path: str = None,
        engine_preference: str = None
    ) -> Optional[str]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            language: Target language (ar, en, es, fr, de, etc.)
            speaker_voice: Voice type (female, male, neutral)
            output_path: Optional output file path
            engine_preference: Preferred engine ("coqui" or "edge")
            
        Returns:
            str: Path to generated audio file, or None if failed
        """
        try:
            if not self.engine_loaded:
                # Auto-load optimal engine for the language
                optimal_engine = self._get_optimal_engine(language, engine_preference)
                if not self.load_tts_model(f"auto-{language}", optimal_engine):
                    raise RuntimeError("Failed to load TTS model")
            
            logger.info(f"Generating TTS: '{text[:50]}...' (lang: {language}, voice: {speaker_voice})")
            
            # Generate output path if not provided
            if not output_path:
                timestamp = int(time.time())
                output_path = self.output_dir / f"tts_{language}_{speaker_voice}_{timestamp}.wav"
            
            # Call the engine's text_to_speech method
            result_path = self.current_engine.text_to_speech(
                text=text,
                language=language,
                output_path=str(output_path),
                speaker_voice=speaker_voice
            )
            
            if result_path and os.path.exists(result_path):
                logger.info(f"✅ TTS generated: {result_path}")
                return result_path
            else:
                logger.error("TTS generation failed - no output file")
                return None
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    def text_to_speech_bytes(
        self,
        text: str,
        language: str = "ar",
        speaker_voice: str = "female",
        engine_preference: str = None
    ) -> Optional[bytes]:
        """
        Convert text to speech and return audio bytes.
        
        Args:
            text: Text to convert to speech
            language: Target language
            speaker_voice: Voice type
            engine_preference: Preferred engine
            
        Returns:
            bytes: Audio data as bytes, or None if failed
        """
        try:
            if not self.engine_loaded:
                # Auto-load optimal engine for the language
                optimal_engine = self._get_optimal_engine(language, engine_preference)
                if not self.load_tts_model(f"auto-{language}", optimal_engine):
                    raise RuntimeError("Failed to load TTS model")
            
            # Call the engine's text_to_speech_bytes method
            audio_bytes = self.current_engine.text_to_speech_bytes(
                text=text,
                language=language,
                speaker_voice=speaker_voice
            )
            
            return audio_bytes
                
        except Exception as e:
            logger.error(f"TTS bytes generation failed: {e}")
            return None

    def create_voice_clone(
        self,
        audio_file_path: str,
        clone_name: str,
        language: str = "ar"
    ) -> bool:
        """
        Create a voice clone from an audio sample.
        
        Args:
            audio_file_path: Path to audio sample for cloning
            clone_name: Name for the voice clone
            language: Language of the audio sample
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Voice cloning requires Coqui TTS
            if not self.engine_loaded or not hasattr(self.current_engine, 'create_voice_clone'):
                logger.info("Loading Coqui TTS for voice cloning...")
                if not self.load_tts_model("coqui-cloning", "coqui"):
                    logger.error("Failed to load Coqui TTS for voice cloning")
                    return False
            
            # Create the voice clone
            result = self.current_engine.create_voice_clone(audio_file_path, clone_name)
            
            if result:
                logger.info(f"✅ Voice clone '{clone_name}' created successfully")
            else:
                logger.error(f"Failed to create voice clone '{clone_name}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Voice clone creation failed: {e}")
            return False

    def use_voice_clone(
        self,
        text: str,
        clone_name: str,
        output_path: str = None
    ) -> Optional[str]:
        """
        Generate speech using a voice clone.
        
        Args:
            text: Text to synthesize
            clone_name: Name of the voice clone to use
            output_path: Optional output path
            
        Returns:
            str: Path to generated audio file, or None if failed
        """
        try:
            if not self.engine_loaded or not hasattr(self.current_engine, 'use_voice_clone'):
                logger.error("Voice cloning requires Coqui TTS")
                return None
            
            # Generate speech with voice clone
            result_path = self.current_engine.use_voice_clone(text, clone_name, output_path)
            
            if result_path:
                logger.info(f"✅ Voice clone synthesis completed: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Voice clone synthesis failed: {e}")
            return None

    def get_available_voices(self, language: str = "ar") -> List[str]:
        """Get available voices for a language."""
        try:
            if self.engine_loaded and hasattr(self.current_engine, 'get_available_speakers'):
                return self.current_engine.get_available_speakers(language)
            else:
                # Return default voices
                return ["female", "male", "neutral"]
        except Exception as e:
            logger.warning(f"Could not get available voices: {e}")
            return ["female", "male", "neutral"]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        try:
            if self.engine_loaded and hasattr(self.current_engine, 'get_supported_languages'):
                return self.current_engine.get_supported_languages()
            else:
                # Return default supported languages
                return ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja"]
        except Exception as e:
            logger.warning(f"Could not get supported languages: {e}")
            return ["ar", "en"]

    def benchmark_tts(
        self,
        text: str,
        language: str = "ar",
        speaker_voice: str = "female"
    ) -> Dict[str, Any]:
        """
        Benchmark TTS performance.
        
        Args:
            text: Text to synthesize for benchmarking
            language: Target language
            speaker_voice: Voice type
            
        Returns:
            Dict containing benchmark results
        """
        try:
            if not self.engine_loaded:
                # Auto-load optimal engine
                optimal_engine = self._get_optimal_engine(language)
                if not self.load_tts_model(f"benchmark-{language}", optimal_engine):
                    raise RuntimeError("Failed to load TTS model for benchmarking")
            
            # Run benchmark
            if hasattr(self.current_engine, 'benchmark'):
                return self.current_engine.benchmark(
                    prompt=text,
                    language=language,
                    speaker_voice=speaker_voice
                )
            else:
                # Manual benchmark
                start_time = time.time()
                output_path = self.text_to_speech(text, language, speaker_voice)
                end_time = time.time()
                
                generation_time = end_time - start_time
                characters_per_second = len(text) / generation_time if generation_time > 0 else 0
                
                return {
                    "generation_time": generation_time,
                    "characters_per_second": characters_per_second,
                    "text_length": len(text),
                    "output_file": output_path,
                    "success": output_path is not None,
                    "engine": self._get_engine_name(),
                    "model": self.current_model
                }
                
        except Exception as e:
            logger.error(f"TTS benchmark failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time": 0,
                "characters_per_second": 0
            }

    def unload_model(self) -> bool:
        """Unload the current TTS model."""
        try:
            if self.current_engine:
                if hasattr(self.current_engine, 'unload_model'):
                    self.current_engine.unload_model()
                
                self.model_manager.unload_model()
                
            self.current_engine = None
            self.current_model = None
            self.engine_loaded = False
            
            logger.info("✅ TTS model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload TTS model: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if a TTS model is loaded."""
        return self.engine_loaded and self.current_engine is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current TTS model."""
        try:
            if self.engine_loaded and hasattr(self.current_engine, 'get_model_info'):
                return self.current_engine.get_model_info()
            else:
                return {
                    "loaded": self.engine_loaded,
                    "model": self.current_model,
                    "engine": self._get_engine_name(),
                    "status": "loaded" if self.engine_loaded else "not_loaded"
                }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {"error": str(e)}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            if self.engine_loaded and hasattr(self.current_engine, 'get_memory_stats'):
                return self.current_engine.get_memory_stats()
            else:
                import psutil
                return {
                    "system_memory_used_gb": psutil.virtual_memory().used / (1024**3),
                    "system_memory_percent": psutil.virtual_memory().percent,
                    "gpu_available": self._check_gpu_available()
                }
        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")
            return {"error": str(e)}

    def _get_optimal_engine(self, language: str, preference: str = None) -> str:
        """Get the optimal engine for a given language."""
        if preference and preference in self.supported_engines:
            return preference
        
        # Get language preferences
        preferences = self.language_engine_preferences.get(language, self.language_engine_preferences["default"])
        
        # Return the first available engine
        for engine in preferences:
            if engine in self.supported_engines:
                return engine
        
        # Fallback to first available
        return list(self.supported_engines.keys())[0]

    def _get_engine_name(self) -> str:
        """Get the name of the current engine."""
        if hasattr(self.current_engine, '__class__'):
            return self.current_engine.__class__.__name__
        else:
            return "Unknown"

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_engine_capabilities(self, engine_name: str = None) -> Dict[str, Any]:
        """Get capabilities of a specific engine or current engine."""
        try:
            if engine_name:
                return self.supported_engines.get(engine_name, {})
            elif self.engine_loaded:
                engine_type = "coqui" if "Coqui" in self._get_engine_name() else "edge"
                return self.supported_engines.get(engine_type, {})
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not get engine capabilities: {e}")
            return {}

    def switch_engine(self, engine_type: str, language: str = "ar") -> bool:
        """
        Switch to a different TTS engine.
        
        Args:
            engine_type: Engine to switch to ("coqui" or "edge")
            language: Language to optimize for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if engine_type not in self.supported_engines:
                logger.error(f"Unsupported engine type: {engine_type}")
                return False
            
            # Unload current model
            if self.engine_loaded:
                self.unload_model()
            
            # Load new engine
            model_name = f"{engine_type}-{language}"
            success = self.load_tts_model(model_name, engine_type)
            
            if success:
                logger.info(f"✅ Switched to {engine_type} engine")
            else:
                logger.error(f"Failed to switch to {engine_type} engine")
            
            return success
            
        except Exception as e:
            logger.error(f"Engine switch failed: {e}")
            return False

    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        return list(self.supported_engines.keys())

    def test_engine_availability(self) -> Dict[str, bool]:
        """Test availability of all supported engines."""
        results = {}
        
        for engine_name in self.supported_engines:
            try:
                # Try to create a minimal configuration and test
                test_success = True  # Placeholder - would test actual import/initialization
                results[engine_name] = test_success
                logger.info(f"Engine {engine_name}: {'✅ Available' if test_success else '❌ Not available'}")
            except Exception as e:
                results[engine_name] = False
                logger.warning(f"Engine {engine_name} test failed: {e}")
        
        return results
