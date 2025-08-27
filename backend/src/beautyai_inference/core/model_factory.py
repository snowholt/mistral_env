"""
Factory for creating model instances based on configuration.
"""
import logging
from typing import Optional, Any

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig
from ..inference_engines.transformers_engine import TransformersEngine
from ..inference_engines.vllm_engine import VLLMEngine

try:
    from ..inference_engines.llamacpp_engine import LlamaCppEngine
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    from ..inference_engines.edge_tts_engine import EdgeTTSEngine
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Whisper engines
try:
    from ..services.voice.transcription.whisper_large_v3_turbo_engine import WhisperLargeV3TurboEngine
    from ..services.voice.transcription.whisper_large_v3_engine import WhisperLargeV3Engine
    from ..services.voice.transcription.whisper_arabic_turbo_engine import WhisperArabicTurboEngine
    WHISPER_ENGINES_AVAILABLE = True
except ImportError:
    WHISPER_ENGINES_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_config: ModelConfig) -> ModelInterface:
        """Create a model instance based on the provided configuration."""
        engine_type = model_config.engine_type.lower()
        model_id = model_config.model_id.lower()
        
        # Detect model architecture type for special handling
        model_architecture = "causal_lm"  # Default assumption
        
        # Check for sequence-to-sequence models
        if "t5" in model_id or "byt5" in model_id or "bart" in model_id or "pegasus" in model_id:
            model_architecture = "seq2seq_lm"
            logger.info(f"Detected a sequence-to-sequence model: {model_config.model_id}")
            
        # Update model config with detected architecture
        model_config.model_architecture = model_architecture
        
        # Special case for Mistral3 models
        if "mistral" in model_id and "3" in model_id:
            logger.info(f"Detected a Mistral3 model: {model_config.model_id}")
            logger.info("These models work best with vLLM. Attempting to use VLLMEngine.")
            try:
                return VLLMEngine(model_config)
            except ImportError:
                logger.warning("vLLM not available for Mistral3 model, trying transformers engine")
                return TransformersEngine(model_config)
                
        # Note: Removed hardcoded Qwen3 override to respect registry engine configuration
        
        # Normal cases
        if engine_type == "transformers":
            logger.info(f"Creating TransformersEngine for model: {model_config.model_id}")
            return TransformersEngine(model_config)
        
        elif engine_type == "vllm":
            # vLLM only supports causal LMs, so warn if trying to use with seq2seq
            if model_architecture == "seq2seq_lm":
                logger.warning(f"vLLM does not support sequence-to-sequence models like {model_config.model_id}. Falling back to TransformersEngine.")
                return TransformersEngine(model_config)
                
            logger.info(f"Creating VLLMEngine for model: {model_config.model_id}")
            try:
                return VLLMEngine(model_config)
            except ImportError:
                logger.warning("vLLM not available, falling back to TransformersEngine")
                return TransformersEngine(model_config)
        
        elif engine_type == "llama.cpp":
            logger.info(f"Creating LlamaCppEngine for model: {model_config.model_id}")
            
            if not LLAMACPP_AVAILABLE:
                logger.error("llama-cpp-python not available. Install with: pip install llama-cpp-python[server]")
                logger.warning("Falling back to TransformersEngine with 4-bit quantization")
                # Modify config for 4-bit fallback
                model_config.engine_type = "transformers"
                model_config.quantization = "4bit"
                return TransformersEngine(model_config)
            
            try:
                return LlamaCppEngine(model_config)
            except Exception as e:
                logger.error(f"Failed to create LlamaCppEngine: {e}")
                logger.warning("Falling back to TransformersEngine with 4-bit quantization")
                model_config.engine_type = "transformers"
                model_config.quantization = "4bit"
                return TransformersEngine(model_config)
        
        elif engine_type == "edge_tts":
            logger.info(f"Creating EdgeTTSEngine for model: {model_config.model_id}")
            
            if not EDGE_TTS_AVAILABLE:
                logger.error("Edge TTS library not available. Install with: pip install edge-tts")
                raise ImportError("Edge TTS library is required but not installed")
            
            try:
                return EdgeTTSEngine(model_config)
            except Exception as e:
                logger.error(f"Failed to create EdgeTTSEngine: {e}")
                raise RuntimeError(f"Failed to create EdgeTTSEngine: {e}")
        
        else:
            logger.warning(f"Unknown engine type: {engine_type}, using TransformersEngine")
            return TransformersEngine(model_config)
    
    @staticmethod
    def create_whisper_model(whisper_config) -> Optional[Any]:
        """
        Create a Whisper model instance based on the provided configuration.
        
        Args:
            whisper_config: WhisperModelConfig instance
            
        Returns:
            Whisper engine instance or None if creation fails
        """
        if not WHISPER_ENGINES_AVAILABLE:
            logger.error("Whisper engines not available. Check imports.")
            return None
            
        engine_type = whisper_config.engine_type.lower()
        
        try:
            if engine_type == "whisper_large_v3_turbo":
                logger.info(f"Creating WhisperLargeV3TurboEngine for model: {whisper_config.model_id}")
                return WhisperLargeV3TurboEngine()
            
            elif engine_type == "whisper_large_v3":
                logger.info(f"Creating WhisperLargeV3Engine for model: {whisper_config.model_id}")
                return WhisperLargeV3Engine()
            
            elif engine_type == "whisper_arabic_turbo":
                logger.info(f"Creating WhisperArabicTurboEngine for model: {whisper_config.model_id}")
                return WhisperArabicTurboEngine()
            
            elif engine_type == "whisper_finetuned_arabic":
                logger.info(f"Creating WhisperFinetunedArabicEngine for model: {whisper_config.model_id}")
                from ..services.voice.transcription.whisper_finetuned_arabic_engine import WhisperFinetunedArabicEngine
                return WhisperFinetunedArabicEngine()
            
            else:
                logger.warning(f"Unknown Whisper engine type: {engine_type}, using turbo as fallback")
                return WhisperLargeV3TurboEngine()
                
        except Exception as e:
            logger.error(f"Failed to create Whisper engine for {engine_type}: {e}")
            return None
    
    @staticmethod
    def get_available_whisper_engines() -> dict:
        """
        Get a list of available Whisper engines.
        
        Returns:
            Dictionary of available engine types and their descriptions
        """
        if not WHISPER_ENGINES_AVAILABLE:
            return {}
            
        return {
            "whisper_large_v3_turbo": "Speed-optimized Whisper Large v3 Turbo (4x faster)",
            "whisper_large_v3": "Maximum accuracy Whisper Large v3",
            "whisper_arabic_turbo": "Arabic-specialized Whisper Turbo",
        }
    
    @staticmethod
    def create_whisper_from_voice_registry(model_name: Optional[str] = None, language: str = "auto") -> Optional[Any]:
        """
        Create a Whisper model instance from voice registry configuration.
        
        Args:
            model_name: Optional model name from voice registry
            language: Target language for engine selection
            
        Returns:
            Whisper engine instance or None if creation fails
        """
        try:
            from ..config.whisper_model_config import WhisperModelConfig
            from ..config.voice_config_loader import get_voice_config
            
            voice_config = get_voice_config()
            whisper_config = WhisperModelConfig.from_voice_registry(voice_config, model_name)
            
            return ModelFactory.create_whisper_model(whisper_config)
            
        except Exception as e:
            logger.error(f"Failed to create Whisper model from voice registry: {e}")
            return None
