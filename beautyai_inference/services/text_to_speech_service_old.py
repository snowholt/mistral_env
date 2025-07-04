"""
Text-to-Speech Service for BeautyAI Framework.

Handles TTS processing using OuteTTS model for high-quality multilingual speech synthesis.
Supports Arabic and English with emotion and speaker control via GGUF/LlamaCpp backend.
"""
import logging
import tempfile
import os
import time
from typing import Dict, Any, Optional, BinaryIO, List
from pathlib import Path
import io

from .base.base_service import BaseService
from ..config.config_manager import AppConfig, ModelConfig
from ..inference_engines.oute_tts_engine import OuteTTSEngine
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class TextToSpeechService(BaseService):
    """Service for text-to-speech conversion using OuteTTS models."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.oute_tts_engine = None
        self.loaded_model_name = None
        
    def load_tts_model(self, model_name: str = "oute-tts-1b") -> bool:
        """
        Load a TTS model for speech synthesis.
        
        Args:
            model_name: Name of the TTS model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get model configuration
            app_config = AppConfig()
            app_config.models_file = "beautyai_inference/config/model_registry.json"
            app_config.load_model_registry()
            
            model_config = app_config.model_registry.get_model(model_name)
            if not model_config:
                logger.error(f"Model configuration for '{model_name}' not found.")
                return False
            
            # Load TTS engine
            logger.info(f"Loading TTS model: {model_config.model_id}")
            self.oute_tts_engine = OuteTTSEngine(model_config)
            self.oute_tts_engine.load_model()
            
            self.loaded_model_name = model_name
            logger.info(f"TTS model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model '{model_name}': {e}")
            return False
    
    def text_to_speech(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            emotion: Emotion/style for the voice
            speed: Speech speed multiplier
            output_path: Path to save the audio file
            
        Returns:
            str: Path to the generated audio file, or None if failed
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return None
            
            logger.info(f"Converting text to speech: '{text[:50]}...' (language: {language})")
            
            output_file = self.oute_tts_engine.text_to_speech(
                text=text,
                language=language,
                speaker_voice=speaker_voice,
                emotion=emotion,
                speed=speed,
                output_path=output_path
            )
            
            logger.info(f"TTS conversion successful: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"TTS conversion failed: {e}")
            return None
    
    def text_to_speech_bytes(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        audio_format: str = "wav"
    ) -> Optional[bytes]:
        """
        Convert text to speech and return as bytes.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            emotion: Emotion/style for the voice
            speed: Speech speed multiplier
            audio_format: Output audio format
            
        Returns:
            bytes: Audio data as bytes, or None if failed
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return None
            
            logger.info(f"Converting text to speech bytes: '{text[:50]}...' (language: {language})")
            
            audio_bytes = self.oute_tts_engine.text_to_speech_bytes(
                text=text,
                language=language,
                speaker_voice=speaker_voice,
                emotion=emotion,
                speed=speed,
                audio_format=audio_format
            )
            
            logger.info(f"TTS bytes conversion successful: {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"TTS bytes conversion failed: {e}")
            return None
    
    def text_to_speech_stream(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> Optional[io.BytesIO]:
        """
        Convert text to speech and return as stream.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific speaker voice to use
            emotion: Emotion/style for the voice
            speed: Speech speed multiplier
            
        Returns:
            io.BytesIO: Audio stream, or None if failed
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return None
            
            audio_stream = self.oute_tts_engine.text_to_speech_stream(
                text=text,
                language=language,
                speaker_voice=speaker_voice,
                emotion=emotion,
                speed=speed
            )
            
            return audio_stream
            
        except Exception as e:
            logger.error(f"TTS stream conversion failed: {e}")
            return None
    
    def is_model_loaded(self) -> bool:
        """Check if a TTS model is currently loaded."""
        return self.oute_tts_engine is not None and self.oute_tts_engine.is_model_loaded()
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded TTS model."""
        return self.loaded_model_name
    
    def unload_model(self) -> None:
        """Unload the current TTS model to free memory."""
        try:
            if self.oute_tts_engine:
                self.oute_tts_engine.unload_model()
                self.oute_tts_engine = None
            
            self.loaded_model_name = None
            logger.info("TTS model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading TTS model: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_supported_languages()
        return ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"]
    
    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers for the specified language."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_available_speakers(language)
        return []
    
    def validate_language(self, language: str) -> bool:
        """Validate if a language is supported."""
        return language in self.get_supported_languages()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for the TTS service."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_memory_stats()
        return {"service": "not_loaded"}
    
    def benchmark_tts(
        self, 
        text: str, 
        language: str = "en",
        **kwargs
    ) -> Dict[str, Any]:
        """Run a TTS benchmark."""
        try:
            if not self.oute_tts_engine:
                return {"error": "TTS model not loaded"}
            
            return self.oute_tts_engine.benchmark(text, language=language, **kwargs)
            
        except Exception as e:
            logger.error(f"TTS benchmark failed: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded TTS model."""
        if self.oute_tts_engine:
            return self.oute_tts_engine.get_model_info()
        return {
            "model_name": None,
            "model_id": None,
            "engine_type": "oute_tts",
            "is_loaded": False,
            "supported_languages": self.get_supported_languages()
        }
    
    def create_arabic_speaker_profile(
        self, 
        audio_file_path: str, 
        speaker_name: str = "custom_arabic_female"
    ) -> Optional[str]:
        """
        Create a custom Arabic speaker profile from an audio sample.
        
        Args:
            audio_file_path: Path to Arabic audio sample (WAV format recommended)
            speaker_name: Name for the custom speaker profile
            
        Returns:
            str: Speaker profile path if successful, None if failed
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return None
            
            logger.info(f"Creating Arabic speaker profile from: {audio_file_path}")
            
            profile_path = self.oute_tts_engine.create_custom_arabic_speaker(
                audio_file_path=audio_file_path,
                speaker_name=speaker_name
            )
            
            logger.info(f"Arabic speaker profile created: {profile_path}")
            return profile_path
            
        except Exception as e:
            logger.error(f"Failed to create Arabic speaker profile: {e}")
            return None
    
    def get_discovered_speakers(self) -> Dict[str, Any]:
        """Get all discovered and custom speakers."""
        try:
            if not self.oute_tts_engine:
                return {}
            
            return {
                "discovered": self.oute_tts_engine.discovered_speakers,
                "available_by_language": self.oute_tts_engine.available_speakers,
                "arabic_speakers": [
                    speaker for speaker_name, speaker in self.oute_tts_engine.discovered_speakers.items() 
                    if "ar" in speaker_name.lower() or "arabic" in speaker_name.lower()
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get discovered speakers: {e}")
            return {}
    
    def create_arabic_speaker_profile(
        self, 
        audio_file_path: str, 
        speaker_name: str = "arabic_female", 
        transcript: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a custom Arabic speaker profile from an audio file.
        
        Args:
            audio_file_path: Path to Arabic audio sample (WAV format recommended)
            speaker_name: Name for the custom speaker profile
            transcript: Optional transcript of the audio
            
        Returns:
            str: Path to the created speaker profile, or None if failed
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return None
                
            logger.info(f"Creating Arabic speaker profile: {speaker_name}")
            
            profile_path = self.oute_tts_engine.create_arabic_speaker_profile(
                audio_file_path=audio_file_path,
                speaker_name=speaker_name,
                transcript=transcript
            )
            
            logger.info(f"✅ Arabic speaker profile created: {profile_path}")
            return profile_path
            
        except Exception as e:
            logger.error(f"Failed to create Arabic speaker profile: {e}")
            return None
    
    def setup_default_arabic_speakers(self, female_audio_path: Optional[str] = None, 
                                    male_audio_path: Optional[str] = None) -> Dict[str, str]:
        """
        Setup default Arabic speakers for the BeautyAI platform.
        
        Args:
            female_audio_path: Path to female Arabic audio sample
            male_audio_path: Path to male Arabic audio sample
            
        Returns:
            Dict mapping speaker types to profile paths
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return {}
                
            logger.info("Setting up default Arabic speakers...")
            
            created_speakers = self.oute_tts_engine.setup_default_arabic_speakers(
                female_audio_path=female_audio_path,
                male_audio_path=male_audio_path
            )
            
            logger.info(f"✅ Arabic speakers setup completed: {created_speakers}")
            return created_speakers
            
        except Exception as e:
            logger.error(f"Failed to setup Arabic speakers: {e}")
            return {}
    
    def get_arabic_speakers(self) -> Dict[str, str]:
        """Get all available Arabic speaker profiles."""
        try:
            if not self.oute_tts_engine:
                logger.warning("TTS model not loaded.")
                return {}
                
            return self.oute_tts_engine.get_arabic_speakers()
            
        except Exception as e:
            logger.error(f"Failed to get Arabic speakers: {e}")
            return {}
    
    def register_arabic_speaker_profile(self, speaker_name: str, profile_path: str, gender: str = "female") -> bool:
        """
        Register a new custom Arabic speaker profile with the TTS service.
        
        Args:
            speaker_name: Name for the custom speaker
            profile_path: Path to the speaker profile JSON file
            gender: Speaker gender ("female" or "male")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.oute_tts_engine:
                logger.error("TTS model not loaded. Call load_tts_model() first.")
                return False
            
            success = self.oute_tts_engine.register_custom_arabic_speaker(
                speaker_name, profile_path, gender
            )
            
            if success:
                logger.info(f"✅ Arabic speaker registered: {speaker_name} ({gender})")
                return True
            else:
                logger.error(f"❌ Failed to register Arabic speaker: {speaker_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering Arabic speaker: {e}")
            return False
    
    def get_available_arabic_speakers(self) -> Dict[str, Any]:
        """
        Get information about available Arabic speakers.
        
        Returns:
            Dict containing available Arabic speakers and their details
        """
        try:
            if not self.oute_tts_engine:
                return {"error": "TTS model not loaded"}
            
            arabic_speakers = {}
            
            # Get custom speakers
            for name, path in self.oute_tts_engine.custom_speakers.items():
                if 'arabic' in name.lower():
                    arabic_speakers[name] = {
                        "path": path,
                        "type": "custom",
                        "exists": Path(path).exists() if path.endswith('.json') else True
                    }
            
            # Get Arabic speaker mapping
            arabic_mapping = {}
            for key, value in self.oute_tts_engine.arabic_speaker_mapping.items():
                if key.startswith('ar-'):
                    arabic_mapping[key] = value
            
            return {
                "custom_speakers": arabic_speakers,
                "arabic_mapping": arabic_mapping,
                "available_languages": list(self.oute_tts_engine.available_speakers.keys()),
                "arabic_available": "ar" in self.oute_tts_engine.available_speakers
            }
            
        except Exception as e:
            logger.error(f"Error getting Arabic speakers: {e}")
            return {"error": str(e)}
    
    def test_arabic_speaker(self, speaker_name: str, test_text: str = None) -> Optional[str]:
        """
        Test an Arabic speaker with a sample text.
        
        Args:
            speaker_name: Name of the speaker to test
            test_text: Optional custom test text
            
        Returns:
            str: Path to the test audio file, or None if failed
        """
        try:
            if not test_text:
                test_text = "مرحباً بك في عيادة الجمال. هذا اختبار للصوت العربي."
            
            # Create test output directory
            test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            import time
            timestamp = int(time.time())
            test_output = test_dir / f"test_{speaker_name}_{timestamp}.wav"
            
            # Generate speech
            result = self.text_to_speech(
                text=test_text,
                language="ar",
                speaker_voice=speaker_name,
                output_path=str(test_output)
            )
            
            if result and Path(result).exists():
                logger.info(f"✅ Arabic speaker test successful: {result}")
                return result
            else:
                logger.error(f"❌ Arabic speaker test failed for: {speaker_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing Arabic speaker: {e}")
            return None
