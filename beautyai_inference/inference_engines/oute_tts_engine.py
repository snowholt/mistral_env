"""
OuteTTS Engine for BeautyAI Framework.
Implements text-to-speech using the OuteTTS library with neural speech synthesis.
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
        self.custom_speakers = {}  # Store custom Arabic speakers
        
        # Configuration for OuteTTS
        self.model_version = outetts.Models.VERSION_1_0_SIZE_1B
        self.backend = outetts.Backend.LLAMACPP
        self.quantization = outetts.LlamaCppQuantization.FP16
        
        # Speaker configurations (based on actual OuteTTS model capabilities)
        # Note: OuteTTS only has English speakers but can synthesize multiple languages
        # Based on GitHub repository: https://github.com/edwko/OuteTTS
        self.available_speakers = {
            "en": {
                "female": "EN-FEMALE-1-NEUTRAL",
                "male": "en_male_1", 
                "neutral": "EN-FEMALE-1-NEUTRAL"
            },
            "ar": {
                "female": "arabic_female_premium_19s",  # Use our new premium Arabic speaker
                "male": "arabic_male_custom",           # Will use custom Arabic speaker  
                "neutral": "arabic_female_premium_19s"  # Default to premium female
            },
            "es": {
                "female": "EN-FEMALE-1-NEUTRAL",  # Use English speaker for Spanish text
                "male": "en_male_1",             # Use English speaker for Spanish text
                "neutral": "EN-FEMALE-1-NEUTRAL"
            },
            "fr": {
                "female": "EN-FEMALE-1-NEUTRAL",  # Use English speaker for French text
                "male": "en_male_1",             # Use English speaker for French text
                "neutral": "EN-FEMALE-1-NEUTRAL"
            },
            "de": {
                "female": "EN-FEMALE-1-NEUTRAL",  # Use English speaker for German text
                "male": "en_male_1",             # Use English speaker for German text
                "neutral": "EN-FEMALE-1-NEUTRAL"
            }
        }
        
        # Discovered available speakers (will be populated after model loading)
        self.discovered_speakers = {}
        
        # Paths for custom speaker profiles
        self.speakers_dir = Path("/home/lumi/beautyai/voice_tests/custom_speakers")
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Discover available speakers
            logger.info("Discovering available speakers...")
            self.discovered_speakers = self._discover_actual_speakers()
            
            # Load a default speaker
            logger.info("Loading default speaker...")
            self.current_speaker = self.interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
            
            # Load custom Arabic speaker profiles
            logger.info("Loading custom Arabic speaker profiles...")
            self.load_custom_arabic_speakers()
            
            # Create Arabic speaker profiles if they don't exist
            self._create_arabic_speaker_profiles()
            
            self.model_loaded = True
            logger.info("✅ OuteTTS model loaded successfully")
            logger.info(f"✅ Available speakers: {list(self.discovered_speakers.keys())}")
            
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
        
        # First, check for custom Arabic speakers if language is Arabic
        if language == "ar":
            # Check if we have custom Arabic speakers available
            if self.custom_speakers:
                for speaker_name, speaker_path in self.custom_speakers.items():
                    if speaker_voice.lower() in speaker_name.lower():
                        logger.info(f"Using custom Arabic speaker: {speaker_path}")
                        return speaker_path
                        
            # Check available speakers for custom profiles
            if language in self.available_speakers:
                speaker_dict = self.available_speakers[language]
                mapped_speaker = speaker_dict.get(speaker_voice, speaker_dict.get("female"))
                if mapped_speaker and mapped_speaker.endswith('.json'):
                    # Verify the file actually exists
                    import os
                    if os.path.exists(mapped_speaker):
                        logger.info(f"Using custom Arabic profile: {mapped_speaker}")
                        return mapped_speaker
                    else:
                        logger.warning(f"Custom Arabic profile {mapped_speaker} not found, falling back to default")
        
        # Check discovered speakers (default speakers)
        if self.discovered_speakers:
            discovered_speaker = self.discovered_speakers.get(speaker_voice.lower())
            if discovered_speaker:
                logger.info(f"Using discovered speaker: {discovered_speaker}")
                return discovered_speaker
                
            # Try finding any available speaker
            if self.discovered_speakers.values():
                first_available = list(self.discovered_speakers.values())[0]
                logger.info(f"Using first available speaker: {first_available}")
                return first_available
        
        # If speaker_voice looks like an actual speaker ID, verify it exists
        if speaker_voice and any(speaker_voice.upper().startswith(prefix) for prefix in ["AR-", "EN-", "ES-", "FR-", "DE-"]):
            # Convert to correct format and check if it exists
            test_speakers = [speaker_voice, speaker_voice.upper(), speaker_voice.lower()]
            for test_speaker in test_speakers:
                if test_speaker in self.discovered_speakers.values():
                    return test_speaker
            
            # If not found, log warning and fallback
            logger.warning(f"Speaker {speaker_voice} not found in {list(self.discovered_speakers.keys())}")
            
        # Use available speakers mapping
        if language in self.available_speakers:
            speaker_dict = self.available_speakers[language]
            mapped_speaker = speaker_dict.get(speaker_voice, speaker_dict.get("female", "EN-FEMALE-1-NEUTRAL"))
            logger.info(f"Using mapped speaker for {language}-{speaker_voice}: {mapped_speaker}")
            return mapped_speaker
        else:
            # Fallback to English
            fallback_speaker = self.available_speakers["en"].get(speaker_voice, "EN-FEMALE-1-NEUTRAL")
            logger.info(f"Using fallback speaker: {fallback_speaker}")
            return fallback_speaker

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
            
            # Load speaker (handle both default speakers and custom profiles)
            if speaker_id.endswith('.json'):
                # Custom speaker profile
                logger.info(f"Loading custom speaker profile: {speaker_id}")
                speaker = self.interface.load_speaker(speaker_id)
            else:
                # Default speaker
                logger.info(f"Loading default speaker: {speaker_id}")
                speaker = self.interface.load_default_speaker(speaker_id)
            
            # Generate speech with language-optimized parameters
            logger.info(f"Generating speech for text: '{text[:50]}...' (language: {language})")
            
            # Optimize sampler config for Arabic language
            if language == "ar":
                # Arabic-optimized parameters for better accuracy
                sampler_config = outetts.SamplerConfig(
                    temperature=0.2,          # Much lower for Arabic accuracy
                    top_p=0.75,              # Better control for Arabic morphology
                    top_k=25,                # Lower for more consistent Arabic pronunciation
                    repetition_penalty=1.02, # Minimal to avoid breaking Arabic words
                    repetition_range=32,     # Shorter for Arabic word structure
                    min_p=0.02              # Lower threshold for Arabic phonemes
                )
                generation_type = outetts.GenerationType.SENTENCE  # Better for Arabic sentences
                max_length = 12288  # Higher for longer Arabic sentences
            else:
                # Default parameters for other languages
                sampler_config = outetts.SamplerConfig(
                    temperature=0.4,
                    repetition_penalty=1.1,
                    repetition_range=64,
                    top_k=40,
                    top_p=0.9,
                    min_p=0.05
                )
                generation_type = outetts.GenerationType.CHUNKED
                max_length = 8192
            
            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=generation_type,
                    speaker=speaker,
                    sampler_config=sampler_config,
                    max_length=max_length,
                    language=language if language != "en" else None  # Explicit language for non-English
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
            
            # Generate speech with optimized Arabic parameters
            if language == "ar":
                sampler_config = outetts.SamplerConfig(
                    temperature=0.2,      # Lower for Arabic accuracy
                    top_p=0.75,          # Better control
                    top_k=25,            # More consistent
                    repetition_penalty=1.02,
                    min_p=0.02
                )
                generation_type = outetts.GenerationType.SENTENCE
            else:
                sampler_config = outetts.SamplerConfig(
                    temperature=0.4,
                    top_p=0.9,
                    top_k=50
                )
                generation_type = outetts.GenerationType.CHUNKED
            
            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=generation_type,
                    speaker=speaker,
                    sampler_config=sampler_config,
                    language=language if language != "en" else None
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

    def identify_available_speakers(self) -> Dict[str, List[str]]:
        """Identify actual available speakers from the OuteTTS model."""
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
        
        return {"available": list(self.discovered_speakers.keys())}

    def _discover_actual_speakers(self) -> Dict[str, str]:
        """Discover actual available speakers from the OuteTTS model."""
        discovered = {}
        
        try:
            # Test common OuteTTS speaker patterns based on GitHub repository
            # From: https://github.com/edwko/OuteTTS
            test_speakers = [
                "EN-FEMALE-1-NEUTRAL",
                "en_female_1", 
                "en_female_2",
                "en_male_1",
                "en_male_2", 
                "en_male_3",
                "en_male_4"
            ]
            
            for speaker_id in test_speakers:
                try:
                    # Try to load the speaker to verify it exists
                    test_speaker = self.interface.load_default_speaker(speaker_id)
                    if test_speaker:
                        # Map to simple keys for easy access
                        if "female" in speaker_id.lower():
                            key = f"female_{len([k for k in discovered.keys() if 'female' in k]) + 1}"
                        else:
                            key = f"male_{len([k for k in discovered.keys() if 'male' in k]) + 1}"
                            
                        discovered[key] = speaker_id
                        discovered[speaker_id.lower().replace("-", "_")] = speaker_id
                        logger.info(f"✅ Found working speaker: {speaker_id}")
                        
                except Exception as e:
                    logger.debug(f"Speaker {speaker_id} not available: {e}")
                    
            # Add default mappings if no speakers found
            if not discovered:
                logger.warning("No speakers discovered, using defaults")
                discovered = {
                    "female": "EN-FEMALE-1-NEUTRAL",
                    "male": "EN-FEMALE-1-NEUTRAL",  # Fallback to female if no male
                    "en_female_1_neutral": "EN-FEMALE-1-NEUTRAL"
                }
                        
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover speakers: {e}")
            # Fallback
            return {
                "female": "EN-FEMALE-1-NEUTRAL",
                "male": "EN-FEMALE-1-NEUTRAL"
            }

    def _create_arabic_speaker_profiles(self) -> None:
        """Create Arabic speaker profiles from existing English speakers."""
        try:
            # For now, we'll map Arabic to the best available English speakers
            # Later, when Arabic audio samples are provided, we can create custom profiles
            logger.info("Setting up Arabic speaker profiles...")
            
            # Map Arabic speakers to available English speakers
            if "female" in self.discovered_speakers:
                self.discovered_speakers["ar_female"] = self.discovered_speakers["female"]
                self.discovered_speakers["arabic_female"] = self.discovered_speakers["female"]
                
            if "male" in self.discovered_speakers:
                self.discovered_speakers["ar_male"] = self.discovered_speakers["male"]
                self.discovered_speakers["arabic_male"] = self.discovered_speakers["male"]
            
            # Update available speakers with discovered ones
            for lang in self.available_speakers:
                if "female" in self.discovered_speakers:
                    self.available_speakers[lang]["female"] = self.discovered_speakers["female"]
                if "male" in self.discovered_speakers:
                    self.available_speakers[lang]["male"] = self.discovered_speakers["male"]
                    
            logger.info("✅ Arabic speaker profiles created")
                    
        except Exception as e:
            logger.error(f"Failed to create Arabic speaker profiles: {e}")

    def create_custom_arabic_speaker(self, audio_file_path: str, speaker_name: str = "custom_arabic") -> str:
        """Create a custom Arabic speaker profile from an audio file.
        
        Args:
            audio_file_path: Path to Arabic audio sample (WAV format recommended)
            speaker_name: Name for the custom speaker profile
            
        Returns:
            str: Speaker ID for the created profile
        """
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"Creating custom Arabic speaker from: {audio_file_path}")
            
            # Verify audio file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Create speaker profile from audio
            custom_speaker = self.interface.create_speaker(audio_file_path)
            
            # Save the speaker profile
            speaker_profile_path = f"/home/lumi/beautyai/voice_tests/{speaker_name}_profile.json"
            self.interface.save_speaker(custom_speaker, speaker_profile_path)
            
            # Add to discovered speakers
            self.discovered_speakers[speaker_name] = speaker_profile_path
            self.discovered_speakers[f"{speaker_name}_female"] = speaker_profile_path
            
            # Update Arabic speakers mapping
            self.available_speakers["ar"]["female"] = speaker_profile_path
            self.available_speakers["ar"]["neutral"] = speaker_profile_path
            
            logger.info(f"✅ Custom Arabic speaker '{speaker_name}' created: {speaker_profile_path}")
            return speaker_profile_path
            
        except Exception as e:
            logger.error(f"Failed to create custom Arabic speaker: {e}")
            raise RuntimeError(f"Failed to create custom Arabic speaker: {e}")

    def create_arabic_speaker_profile(self, audio_file_path: str, speaker_name: str = "arabic_female", 
                                    transcript: Optional[str] = None) -> str:
        """
        Create a custom Arabic speaker profile from an audio file.
        
        Args:
            audio_file_path: Path to Arabic audio sample (WAV format recommended)
            speaker_name: Name for the custom speaker profile
            transcript: Optional transcript of the audio (will auto-transcribe if None)
            
        Returns:
            str: Path to the created speaker profile JSON file
        """
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"🎤 Creating Arabic speaker profile: {speaker_name}")
            logger.info(f"📁 Audio file: {audio_file_path}")
            
            # Verify audio file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Create speaker profile from audio
            if transcript:
                logger.info(f"📝 Using provided transcript: {transcript[:50]}...")
                custom_speaker = self.interface.create_speaker(
                    audio_path=audio_file_path,
                    transcript=transcript,
                    whisper_model="turbo",
                    whisper_device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                logger.info("🎯 Auto-transcribing audio for speaker profile...")
                custom_speaker = self.interface.create_speaker(audio_file_path)
            
            # Save the speaker profile
            speaker_profile_path = self.speakers_dir / f"{speaker_name}_profile.json"
            self.interface.save_speaker(custom_speaker, str(speaker_profile_path))
            
            # Store in custom speakers registry
            self.custom_speakers[speaker_name] = str(speaker_profile_path)
            
            # Update Arabic speakers mapping to use custom profile
            if "female" in speaker_name.lower():
                self.available_speakers["ar"]["female"] = str(speaker_profile_path)
                self.available_speakers["ar"]["neutral"] = str(speaker_profile_path)
            elif "male" in speaker_name.lower():
                self.available_speakers["ar"]["male"] = str(speaker_profile_path)
            
            logger.info(f"✅ Arabic speaker profile created: {speaker_profile_path}")
            return str(speaker_profile_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create Arabic speaker profile: {e}")
            raise RuntimeError(f"Failed to create Arabic speaker profile: {e}")

    def load_custom_speakers(self, speaker_profile_path: str):
        """
        Load a custom speaker profile from JSON file.
        
        Args:
            speaker_profile_path: Path to the speaker profile JSON file
            
        Returns:
            Speaker object ready for use in generation
        """
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
            
        try:
            if not os.path.exists(speaker_profile_path):
                raise FileNotFoundError(f"Speaker profile not found: {speaker_profile_path}")
                
            logger.info(f"📥 Loading custom speaker profile: {speaker_profile_path}")
            speaker = self.interface.load_speaker(speaker_profile_path)
            return speaker
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom speaker: {e}")
            raise RuntimeError(f"Failed to load custom speaker: {e}")

    def load_custom_arabic_speakers(self):
        """Load custom Arabic speaker profiles from saved JSON files."""
        try:
            # Check for Arabic speaker profiles directory
            profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
            if not profiles_dir.exists():
                logger.info("Arabic speaker profiles directory not found, creating...")
                profiles_dir.mkdir(parents=True, exist_ok=True)
                return
            
            # Look for speaker mapping file
            mapping_file = profiles_dir / "speaker_mapping.json"
            if mapping_file.exists():
                import json
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                    
                    if "arabic_speakers" in mapping_data:
                        self.arabic_speaker_mapping.update(mapping_data["arabic_speakers"])
                        logger.info(f"✅ Loaded Arabic speaker mapping: {mapping_data['arabic_speakers']}")
                    
                    if "profiles" in mapping_data:
                        profiles = mapping_data["profiles"]
                        if profiles.get("female") and Path(profiles["female"]).exists():
                            self.custom_speakers["arabic_female_beautyai"] = profiles["female"]
                            self.arabic_speaker_mapping["ar-female"] = profiles["female"]
                            logger.info(f"✅ Loaded female Arabic profile: {profiles['female']}")
                        
                        if profiles.get("male") and Path(profiles["male"]).exists():
                            self.custom_speakers["arabic_male_beautyai"] = profiles["male"]
                            self.arabic_speaker_mapping["ar-male"] = profiles["male"]
                            logger.info(f"✅ Loaded male Arabic profile: {profiles['male']}")
                    
                except Exception as e:
                    logger.warning(f"Could not load speaker mapping file: {e}")
            
            # Look for individual profile files
            profile_patterns = [
                ("arabic_female_premium_19s.json", "ar-female", "arabic_female_premium_19s"),
                ("arabic_female_beautyai.json", "ar-female", "arabic_female_beautyai"),
                ("arabic_male_beautyai.json", "ar-male", "arabic_male_beautyai"),
                ("arabic_female_custom.json", "ar-female", "arabic_female_custom"),
                ("arabic_male_custom.json", "ar-male", "arabic_male_custom")
            ]
            
            for filename, mapping_key, custom_key in profile_patterns:
                profile_path = profiles_dir / filename
                if profile_path.exists():
                    self.custom_speakers[custom_key] = str(profile_path)
                    self.arabic_speaker_mapping[mapping_key] = str(profile_path)
                    logger.info(f"✅ Found Arabic speaker profile: {profile_path}")
            
            # Update available speakers for Arabic
            if self.arabic_speaker_mapping.get("ar-female"):
                self.available_speakers["ar"]["female"] = self.arabic_speaker_mapping["ar-female"]
            if self.arabic_speaker_mapping.get("ar-male"):
                self.available_speakers["ar"]["male"] = self.arabic_speaker_mapping["ar-male"]
            
            logger.info(f"✅ Custom Arabic speakers loaded: {len([k for k in self.custom_speakers.keys() if 'arabic' in k])}")
            
        except Exception as e:
            logger.warning(f"Could not load custom Arabic speakers: {e}")

    def get_arabic_speakers(self) -> Dict[str, str]:
        """Get all available Arabic speaker profiles."""
        arabic_speakers = {}
        
        # Add custom speakers
        for name, path in self.custom_speakers.items():
            if "arabic" in name.lower() or "ar" in name.lower():
                arabic_speakers[name] = path
                
        # Add configured Arabic speakers
        for voice_type, speaker_id in self.available_speakers.get("ar", {}).items():
            if speaker_id.endswith('.json'):
                arabic_speakers[f"ar_{voice_type}"] = speaker_id
                
        return arabic_speakers

    def test_arabic_speaker(self, speaker_profile_path: str, test_text: str = "مرحبا، هذا اختبار للصوت العربي") -> str:
        """
        Test an Arabic speaker profile by generating speech.
        
        Args:
            speaker_profile_path: Path to the speaker profile JSON file
            test_text: Arabic text to synthesize for testing
            
        Returns:
            str: Path to the generated test audio file
        """
        if not self.model_loaded:
            raise RuntimeError("OuteTTS model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"🧪 Testing Arabic speaker: {speaker_profile_path}")
            
            # Load the custom speaker
            speaker = self.load_custom_speaker(speaker_profile_path)
            
            # Generate test speech
            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=test_text,
                    generation_type=outetts.GenerationType.CHUNKED,
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=50
                    ),
                )
            )
            
            # Save test output
            test_output_path = self.speakers_dir / f"test_arabic_speaker_{int(time.time())}.wav"
            output.save(str(test_output_path))
            
            logger.info(f"✅ Arabic speaker test completed: {test_output_path}")
            return str(test_output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to test Arabic speaker: {e}")
            raise RuntimeError(f"Failed to test Arabic speaker: {e}")

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
        created_speakers = {}
        
        try:
            logger.info("🎭 Setting up default Arabic speakers for BeautyAI...")
            
            # Create female Arabic speaker if audio provided
            if female_audio_path and os.path.exists(female_audio_path):
                female_profile = self.create_arabic_speaker_profile(
                    audio_file_path=female_audio_path,
                    speaker_name="beautyai_arabic_female"
                )
                created_speakers["female"] = female_profile
                logger.info("✅ Female Arabic speaker created")
            
            # Create male Arabic speaker if audio provided  
            if male_audio_path and os.path.exists(male_audio_path):
                male_profile = self.create_arabic_speaker_profile(
                    audio_file_path=male_audio_path,
                    speaker_name="beautyai_arabic_male"
                )
                created_speakers["male"] = male_profile
                logger.info("✅ Male Arabic speaker created")
            
            # Update the default mappings
            if created_speakers:
                logger.info("🔧 Updating Arabic speaker mappings...")
                if "female" in created_speakers:
                    self.available_speakers["ar"]["female"] = created_speakers["female"]
                    self.available_speakers["ar"]["neutral"] = created_speakers["female"]
                if "male" in created_speakers:
                    self.available_speakers["ar"]["male"] = created_speakers["male"]
                    
                logger.info("✅ Default Arabic speakers setup completed")
            else:
                logger.warning("⚠️ No Arabic audio files provided, using fallback English speakers")
                
            return created_speakers
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Arabic speakers: {e}")
            return {}

    def register_custom_arabic_speaker(self, speaker_name: str, profile_path: str, gender: str = "female") -> bool:
        """
        Register a new custom Arabic speaker profile.
        
        Args:
            speaker_name: Name for the custom speaker
            profile_path: Path to the speaker profile JSON file
            gender: Speaker gender ("female" or "male")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            profile_path = Path(profile_path)
            if not profile_path.exists():
                logger.error(f"Speaker profile file not found: {profile_path}")
                return False
            
            # Add to custom speakers
            self.custom_speakers[speaker_name] = str(profile_path)
            
            # Update Arabic speaker mapping
            mapping_key = f"ar-{gender}"
            self.arabic_speaker_mapping[mapping_key] = str(profile_path)
            
            # Update available speakers
            self.available_speakers["ar"][gender] = str(profile_path)
            self.available_speakers["ar"]["neutral"] = str(profile_path)  # Default to this speaker
            
            logger.info(f"✅ Registered custom Arabic speaker: {speaker_name} ({gender})")
            logger.info(f"✅ Profile path: {profile_path}")
            
            # Save the mapping for persistence
            self._save_speaker_mapping()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register custom Arabic speaker: {e}")
            return False
    
    def _save_speaker_mapping(self):
        """Save current speaker mapping to persistent storage."""
        try:
            profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
            profiles_dir.mkdir(parents=True, exist_ok=True)
            
            mapping_file = profiles_dir / "speaker_mapping.json"
            
            import json
            mapping_data = {
                "arabic_speakers": self.arabic_speaker_mapping,
                "custom_speakers": self.custom_speakers,
                "created_at": str(datetime.now()),
                "profiles": {}
            }
            
            # Extract profile paths for easy access
            for key, value in self.arabic_speaker_mapping.items():
                if key in ["ar-female", "ar-male"]:
                    gender = key.split("-")[1]
                    mapping_data["profiles"][gender] = value
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Speaker mapping saved to: {mapping_file}")
            
        except Exception as e:
            logger.warning(f"Could not save speaker mapping: {e}")
