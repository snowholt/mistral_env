"""
Edge TTS Text-to-Speech engine for BeautyAI Framework.
Uses Microsoft Edge TTS for high-quality multilingual TTS generation.
This is compatible with Python 3.12+ .
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    logger.warning("edge-tts library not available. Install with: pip install edge-tts")
    EDGE_TTS_AVAILABLE = False


class EdgeTTSEngine(ModelInterface):
    """
    Edge TTS Text-to-Speech engine using Microsoft Edge TTS.
    
    Features:
    - Multilingual support (Arabic, English, and many more)
    - High-quality neural synthesis
    - Python 3.12+ compatible
    - No GPU required
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the Edge TTS engine with a model configuration."""
        if not EDGE_TTS_AVAILABLE:
            raise ImportError(
                "edge-tts library is required. Install with: pip install edge-tts"
            )
        
        self.config = model_config
        self.model = None
        
        # Language to voice mapping for Edge TTS
        self.voice_mapping = {
            "en": "en-US-AriaNeural",      # English (US) - Female
            "ar": "ar-SA-ZariyahNeural",   # Arabic (Saudi Arabia) - Female
            "es": "es-ES-ElviraNeural",    # Spanish (Spain) - Female
            "fr": "fr-FR-DeniseNeural",    # French (France) - Female
            "de": "de-DE-KatjaNeural",     # German (Germany) - Female
            "it": "it-IT-ElsaNeural",      # Italian (Italy) - Female
            "pt": "pt-BR-FranciscaNeural", # Portuguese (Brazil) - Female
            "pl": "pl-PL-ZofiaNeural",     # Polish (Poland) - Female
            "tr": "tr-TR-EmelNeural",      # Turkish (Turkey) - Female
            "ru": "ru-RU-SvetlanaNeural",  # Russian (Russia) - Female
            "nl": "nl-NL-ColetteNeural",   # Dutch (Netherlands) - Female
            "cs": "cs-CZ-VlastaNeural",    # Czech (Czech Republic) - Female
            "zh": "zh-CN-XiaoxiaoNeural",  # Chinese (China) - Female
            "ja": "ja-JP-NanamiNeural",    # Japanese (Japan) - Female
            "hu": "hu-HU-NoemiNeural",     # Hungarian (Hungary) - Female
            "ko": "ko-KR-SunHiNeural",     # Korean (Korea) - Female
        }
        
        # Alternative male voices
        self.male_voice_mapping = {
            "en": "en-US-ChristopherNeural",
            "ar": "ar-SA-HamedNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
            "it": "it-IT-DiegoNeural",
            "pt": "pt-BR-AntonioNeural",
            "pl": "pl-PL-MarekNeural",
            "tr": "tr-TR-AhmetNeural",
            "ru": "ru-RU-DmitryNeural",
            "nl": "nl-NL-MaartenNeural",
            "cs": "cs-CZ-AntoninNeural",
            "zh": "zh-CN-YunxiNeural",
            "ja": "ja-JP-KeitaNeural",
            "hu": "hu-HU-TamasNeural",
            "ko": "ko-KR-InJoonNeural",
        }

    def load_model(self) -> None:
        """Load the Edge TTS model (no actual loading required)."""
        logger.info("Edge TTS model ready - no loading required")
        logger.info("Available languages: " + ", ".join(self.voice_mapping.keys()))

    def unload_model(self) -> None:
        """Unload the model (no action required for Edge TTS)."""
        logger.info("Edge TTS model unloaded (no action required)")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate speech from text."""
        return self.text_to_speech(prompt, **kwargs)

    def text_to_speech(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        output_path: Optional[str] = None,
        gender: str = "female",
        **kwargs
    ) -> str:
        """
        Convert text to speech using Edge TTS.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific voice to use (overrides language mapping)
            output_path: Path to save the audio file (optional)
            gender: Voice gender ("female" or "male")
            **kwargs: Additional parameters (ignored for Edge TTS compatibility)
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Select voice
            if speaker_voice is None:
                if gender.lower() == "male" and language in self.male_voice_mapping:
                    voice = self.male_voice_mapping[language]
                else:
                    voice = self.voice_mapping.get(language, "en-US-AriaNeural")
            else:
                voice = speaker_voice
            
            # Create output path if not provided
            if output_path is None:
                tests_dir = Path(__file__).parent.parent.parent / "voice_tests"
                tests_dir.mkdir(exist_ok=True)
                output_path = tests_dir / f"edge_tts_{language}_{hash(text) % 10000}.wav"
                output_path = str(output_path)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Generating Edge TTS for language: {language}, voice: {voice}")
            
            # Generate TTS using asyncio
            asyncio.run(self._generate_tts_async(text, voice, output_path))
            
            logger.info(f"Edge TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during Edge TTS generation: {e}")
            raise

    async def _generate_tts_async(self, text: str, voice: str, output_path: str):
        """Generate TTS asynchronously using Edge TTS."""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers for the specified language."""
        if language and language in self.voice_mapping:
            return [self.voice_mapping[language], self.male_voice_mapping.get(language, "")]
        return list(self.voice_mapping.values())

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.voice_mapping.keys())

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
        
        # Calculate characters per second
        chars_per_second = len(prompt) / generation_time if generation_time > 0 else 0
        
        return {
            "generation_time": generation_time,
            "characters_per_second": chars_per_second,
            "audio_file": result_path,
            "engine": "edge-tts"
        }

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Not applicable for TTS engine."""
        raise NotImplementedError("Chat streaming not supported for TTS engine")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "memory_used_mb": 0.0,  # Edge TTS doesn't use significant memory
            "gpu_memory_used_mb": 0.0,  # Edge TTS doesn't use GPU
        }

    def is_model_loaded(self) -> bool:
        """Check if model is loaded (always True for Edge TTS)."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Microsoft Edge TTS",
            "type": "neural_tts",
            "languages": self.get_supported_languages(),
            "voices": len(self.voice_mapping),
            "gpu_required": False,
            "python_compatibility": "3.12+"
        }
