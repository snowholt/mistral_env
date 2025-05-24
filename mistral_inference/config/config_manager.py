"""
Configuration management for the package.
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    engine_type: str = "transformers"  # 'transformers' or 'vllm'
    quantization: Optional[str] = "4bit"  # '4bit', '8bit', 'awq', 'squeezellm', 'none', or None
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    gpu_memory_utilization: float = 0.9  # For vLLM
    tensor_parallel_size: int = 1  # For vLLM


@dataclass
class AppConfig:
    """Application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache_dir: Optional[str] = None
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create a configuration from a dictionary."""
        model_dict = config_dict.get("model", {})
        model_config = ModelConfig(**model_dict)
        
        return cls(
            model=model_config,
            cache_dir=config_dict.get("cache_dir"),
            log_level=config_dict.get("log_level", "INFO"),
        )
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "AppConfig":
        """Load configuration from a JSON file."""
        try:
            with open(path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return cls()  # Return default config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "model": {
                "model_id": self.model.model_id,
                "engine_type": self.model.engine_type,
                "quantization": self.model.quantization,
                "dtype": self.model.dtype,
                "max_new_tokens": self.model.max_new_tokens,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "do_sample": self.model.do_sample,
                "gpu_memory_utilization": self.model.gpu_memory_utilization,
                "tensor_parallel_size": self.model.tensor_parallel_size,
            },
            "cache_dir": self.cache_dir,
            "log_level": self.log_level,
        }
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
