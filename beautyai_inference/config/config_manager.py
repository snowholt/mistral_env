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
    model_id: str = "Qwen/Qwen3-14B"
    engine_type: str = "transformers"  # 'transformers' or 'vllm'
    quantization: Optional[str] = "4bit"  # '4bit', '8bit', 'awq', 'squeezellm', 'none', or None
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    gpu_memory_utilization: float = 0.9  # For vLLM
    tensor_parallel_size: int = 1  # For vLLM
    name: str = "default"  # Friendly name for the model configuration
    description: Optional[str] = None  # Optional description of the model configuration
    model_architecture: str = "causal_lm"  # 'causal_lm' or 'seq2seq_lm'
    model_filename: Optional[str] = None  # Specific filename for GGUF models
    documentation: Optional[Dict[str, str]] = None  # Documentation for the model configuration
    custom_generation_params: Optional[Dict[str, Any]] = None  # Custom generation parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model configuration to a dictionary."""
        return {
            "model_id": self.model_id,
            "engine_type": self.engine_type,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "name": self.name,
            "description": self.description,
            "model_architecture": self.model_architecture,
            "model_filename": self.model_filename,
            "documentation": self.documentation,
            "custom_generation_params": self.custom_generation_params,
        }


@dataclass
class ModelRegistry:
    """Registry for multiple model configurations."""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    default_model: str = "default"
    
    def add_model(self, model_config: ModelConfig) -> None:
        """Add a model configuration to the registry."""
        self.models[model_config.name] = model_config
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self.models.get(name)
    
    def remove_model(self, name: str, clear_cache: bool = False) -> bool:
        """Remove a model configuration by name and optionally clear its cache."""
        if name in self.models:
            model_config = self.models[name]
            
            # Clear cache if requested
            if clear_cache:
                try:
                    from ..core.model_manager import ModelManager
                    model_manager = ModelManager()
                    model_manager.clear_model_cache(model_config.model_id)
                except Exception as e:
                    logger.warning(f"Failed to clear cache for model '{name}': {e}")
            
            del self.models[name]
            # If we remove the default model, set a new default if possible
            if name == self.default_model and self.models:
                self.default_model = next(iter(self.models.keys()))
            return True
        return False
    
    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        return self.models.get(self.default_model)
    
    def set_default_model(self, name: str) -> bool:
        """Set the default model by name."""
        if name in self.models:
            self.default_model = name
            return True
        return False
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to a dictionary."""
        return {
            "default_model": self.default_model,
            "models": {name: model.to_dict() for name, model in self.models.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRegistry":
        """Create registry from a dictionary."""
        registry = cls()
        registry.default_model = data.get("default_model", "default")
        
        models_dict = data.get("models", {})
        for name, model_data in models_dict.items():
            # Ensure the name in the model matches the key
            model_data_copy = model_data.copy()
            model_data_copy["name"] = name
            model_config = ModelConfig(**model_data_copy)
            registry.models[name] = model_config
            
        return registry
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "ModelRegistry":
        """Load registry from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading model registry from {path}: {e}")
            # Create a default registry with at least one model
            registry = cls()
            registry.add_model(ModelConfig(name="default"))
            return registry
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save registry to a JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry to {path}: {e}")


@dataclass
class AppConfig:
    """Application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache_dir: Optional[str] = None
    log_level: str = "INFO"
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    models_file: str = "model_registry.json"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create a configuration from a dictionary."""
        model_dict = config_dict.get("model", {})
        model_config = ModelConfig(**model_dict)
        
        # Load model registry if provided
        model_registry = ModelRegistry()
        if "model_registry" in config_dict:
            model_registry = ModelRegistry.from_dict(config_dict["model_registry"])
        
        return cls(
            model=model_config,
            cache_dir=config_dict.get("cache_dir"),
            log_level=config_dict.get("log_level", "INFO"),
            model_registry=model_registry,
            models_file=config_dict.get("models_file", "model_registry.json"),
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
                "name": self.model.name,
                "description": self.model.description,
                "model_architecture": self.model.model_architecture,
            },
            "cache_dir": self.cache_dir,
            "log_level": self.log_level,
            "models_file": self.models_file,
        }
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
    
    def load_model_registry(self) -> None:
        """Load model registry from file."""
        registry_path = Path(self.models_file)
        if registry_path.exists():
            self.model_registry = ModelRegistry.load_from_file(registry_path)
        else:
            # Create a new registry with the current model
            self.model_registry = ModelRegistry()
            self.model_registry.add_model(self.model)
            self.save_model_registry()
    
    def save_model_registry(self) -> None:
        """Save model registry to file."""
        registry_path = Path(self.models_file)
        self.model_registry.save_to_file(registry_path)
    
    def add_model_config(self, model_config: ModelConfig, set_as_default: bool = False) -> None:
        """Add a model configuration to the registry."""
        self.model_registry.add_model(model_config)
        if set_as_default:
            self.model_registry.set_default_model(model_config.name)
            self.model = model_config  # Update current model
        self.save_model_registry()
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model configuration."""
        model_config = self.model_registry.get_model(model_name)
        if model_config:
            self.model = model_config
            return True
        return False
