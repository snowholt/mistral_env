"""
Whisper Model Configuration for BeautyAI Framework.

This module provides configuration classes specifically for Whisper model management,
including settings for different Whisper engines and their optimization parameters.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class WhisperModelConfig:
    """Configuration for Whisper model instances in ModelManager."""
    
    name: str  # Registry name (e.g., 'whisper-large-v3-turbo')
    model_id: str  # Hugging Face model ID
    engine_type: str  # 'whisper_large_v3', 'whisper_large_v3_turbo', 'whisper_arabic_turbo'
    device: str = "cuda:0"
    torch_dtype: str = "float16"
    quantization: Optional[str] = None
    language: str = "auto"
    
    # Engine-specific optimizations
    optimization_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default optimization config based on engine type."""
        if self.optimization_config is None:
            self.optimization_config = self._get_default_optimization_config()
    
    def _get_default_optimization_config(self) -> Dict[str, Any]:
        """Get default optimization settings for each engine type."""
        configs = {
            "whisper_large_v3_turbo": {
                "use_torch_compile": True,
                "static_cache": True,
                "warmup_steps": 2,
                "use_sdpa": True,
                "batch_size": 16,
            },
            "whisper_large_v3": {
                "use_torch_compile": False,
                "static_cache": False,
                "warmup_steps": 0,
                "use_flash_attention": True,
                "batch_size": 8,
            },
            "whisper_arabic_turbo": {
                "use_torch_compile": True,
                "static_cache": True,
                "warmup_steps": 2,
                "use_sdpa": True,
                "batch_size": 12,
            }
        }
        
        return configs.get(self.engine_type, {})
    
    @classmethod
    def from_voice_registry(cls, voice_config, model_name: Optional[str] = None) -> "WhisperModelConfig":
        """Create WhisperModelConfig from voice registry configuration."""
        if model_name is None:
            # Use default STT model from registry
            stt_config = voice_config.get_stt_model_config()
            model_name = voice_config._config["default_models"]["stt"]
        else:
            # Get specific model config
            model_config = voice_config._config["models"].get(model_name)
            if not model_config:
                raise ValueError(f"Model '{model_name}' not found in voice registry")
            stt_config = type('Config', (), model_config)()
            stt_config.model_id = model_config["model_id"]
            stt_config.engine_type = model_config["engine_type"]
        
        return cls(
            name=model_name,
            model_id=stt_config.model_id,
            engine_type=stt_config.engine_type,
            device="cuda:0" if voice_config._config.get("device") == "cuda" else "cpu",
            torch_dtype="float16" if voice_config._config.get("device") == "cuda" else "float32",
        )


@dataclass
class WhisperEngineSpec:
    """Specification for Whisper engine types and their capabilities."""
    
    engine_type: str
    description: str
    model_size: str
    speed_multiplier: float
    memory_usage_gb: float
    supports_torch_compile: bool
    supports_languages: list[str]
    
    @classmethod
    def get_available_engines(cls) -> Dict[str, "WhisperEngineSpec"]:
        """Get specifications for all available Whisper engines."""
        return {
            "whisper_large_v3_turbo": cls(
                engine_type="whisper_large_v3_turbo",
                description="Speed-optimized Whisper Large v3 Turbo (4x faster)",
                model_size="809M parameters",
                speed_multiplier=4.0,
                memory_usage_gb=2.5,
                supports_torch_compile=True,
                supports_languages=["ar", "en", "auto"]
            ),
            "whisper_large_v3": cls(
                engine_type="whisper_large_v3",
                description="Maximum accuracy Whisper Large v3",
                model_size="1.55B parameters",
                speed_multiplier=1.0,
                memory_usage_gb=4.0,
                supports_torch_compile=False,
                supports_languages=["ar", "en", "auto"]
            ),
            "whisper_arabic_turbo": cls(
                engine_type="whisper_arabic_turbo",
                description="Arabic-specialized Whisper Turbo",
                model_size="809M parameters (Arabic fine-tuned)",
                speed_multiplier=3.5,
                memory_usage_gb=2.8,
                supports_torch_compile=True,
                supports_languages=["ar", "auto"]
            )
        }
    
    @classmethod
    def get_recommended_engine(cls, language: str = "auto", priority: str = "speed") -> str:
        """Get recommended engine based on language and priority."""
        engines = cls.get_available_engines()
        
        if language == "ar" and priority == "accuracy":
            return "whisper_arabic_turbo"
        elif language == "ar":
            return "whisper_arabic_turbo"
        elif priority == "speed":
            return "whisper_large_v3_turbo"
        elif priority == "accuracy":
            return "whisper_large_v3"
        else:
            return "whisper_large_v3_turbo"  # Default