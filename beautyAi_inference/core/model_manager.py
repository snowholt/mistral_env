"""
Singleton manager for tracking and controlling loaded models.
"""
import threading
import logging
import torch
import gc
import shutil
from pathlib import Path
from typing import Dict, Optional, List

from .model_interface import ModelInterface
from .model_factory import ModelFactory
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class to manage loaded models."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._loaded_models = {}  # Dict[str, ModelInterface]
            return cls._instance
    
    def load_model(self, model_config: ModelConfig) -> ModelInterface:
        """Load a model if not already loaded."""
        model_name = model_config.name
        
        with self._lock:
            if model_name in self._loaded_models:
                logger.info(f"Model '{model_name}' already loaded")
                return self._loaded_models[model_name]
                
            logger.info(f"Loading model '{model_name}'")
            model = ModelFactory.create_model(model_config)
            model.load_model()
            self._loaded_models[model_name] = model
            return model
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            if model_name not in self._loaded_models:
                logger.warning(f"Model '{model_name}' not loaded")
                return False
            
            logger.info(f"Unloading model '{model_name}'")
            # Delete model reference
            del self._loaded_models[model_name]
            # Force garbage collection
            gc.collect()
            # Clear CUDA cache
            clear_gpu_memory()
            return True
    
    def unload_all_models(self) -> None:
        """Unload all loaded models."""
        with self._lock:
            model_names = list(self._loaded_models.keys())
            for model_name in model_names:
                self.unload_model(model_name)
    
    def get_loaded_model(self, model_name: str) -> Optional[ModelInterface]:
        """Get a loaded model by name."""
        with self._lock:
            return self._loaded_models.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models."""
        with self._lock:
            return list(self._loaded_models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        with self._lock:
            return model_name in self._loaded_models
    
    def clear_model_cache(self, model_id: str) -> bool:
        """Clear model cache from disk."""
        try:
            # Get the Hugging Face cache directory
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Find all directories that match the model pattern
            model_patterns = [
                f"models--{model_id.replace('/', '--')}",
                f"models--{model_id.replace('/', '--')}*"
            ]
            
            removed_count = 0
            for pattern in model_patterns:
                for model_cache_dir in cache_dir.glob(pattern):
                    if model_cache_dir.is_dir():
                        logger.info(f"Removing cache directory: {model_cache_dir}")
                        shutil.rmtree(model_cache_dir)
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleared {removed_count} cache directories for model '{model_id}'")
                return True
            else:
                logger.warning(f"No cache found for model '{model_id}'")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing cache for model '{model_id}': {e}")
            return False
