"""
Singleton manager for tracking and controlling loaded models.
"""
import threading
import logging
import torch
import gc
import shutil
import json
import os
import tempfile
import time
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
    _persistence_file = None
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._loaded_models = {}  # Dict[str, ModelInterface]
                cls._instance._initialize_persistence()
            return cls._instance
    
    def _initialize_persistence(self):
        """Initialize persistence file for CLI state management."""
        # Use a temporary directory for persistence state
        temp_dir = Path(tempfile.gettempdir()) / "beautyai"
        temp_dir.mkdir(exist_ok=True)
        self._persistence_file = temp_dir / "loaded_models.json"
        
        # Load existing state if available
        self._load_persistence_state()
    
    def _load_persistence_state(self):
        """Load model state from persistence file."""
        try:
            if self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                
                # Check if state is recent (within last hour) and validate models
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp < 3600:  # 1 hour
                    loaded_models = data.get('loaded_models', {})
                    
                    # For CLI persistence, we only track model names that were loaded
                    # The actual model instances can't be serialized, so we'll recreate them
                    logger.info(f"Found {len(loaded_models)} models in persistence state")
                    for model_name, model_info in loaded_models.items():
                        logger.info(f"  - {model_name}: {model_info.get('model_id', 'unknown')}")
                else:
                    logger.info("Persistence state is stale, clearing")
                    self._clear_persistence_state()
        except Exception as e:
            logger.warning(f"Error loading persistence state: {e}")
            self._clear_persistence_state()
    
    def _save_persistence_state(self):
        """Save current model state to persistence file."""
        try:
            # Create persistence data
            persistence_data = {
                'timestamp': time.time(),
                'loaded_models': {}
            }
            
            # Save model information (not actual instances)
            for model_name, model_instance in self._loaded_models.items():
                try:
                    # Get model info from the instance
                    model_info = {
                        'model_id': getattr(model_instance, 'model_id', 'unknown'),
                        'engine': getattr(model_instance, 'engine_type', 'unknown'),
                        'loaded_at': time.time()
                    }
                    persistence_data['loaded_models'][model_name] = model_info
                except Exception as e:
                    logger.warning(f"Error saving info for model {model_name}: {e}")
            
            # Write to file atomically
            temp_file = self._persistence_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(persistence_data, f, indent=2)
            
            # Atomic move
            temp_file.replace(self._persistence_file)
            
        except Exception as e:
            logger.warning(f"Error saving persistence state: {e}")
    
    def _clear_persistence_state(self):
        """Clear persistence state file."""
        try:
            if self._persistence_file and self._persistence_file.exists():
                self._persistence_file.unlink()
        except Exception as e:
            logger.warning(f"Error clearing persistence state: {e}")
    
    def get_persistence_info(self) -> Dict[str, any]:
        """Get information about persisted models without loading them."""
        try:
            if self._persistence_file and self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                return data.get('loaded_models', {})
        except Exception as e:
            logger.warning(f"Error reading persistence info: {e}")
        return {}
    
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
            
            # Update persistence state
            self._save_persistence_state()
            
            return model

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            # Check both in-memory and persistence state
            in_memory = model_name in self._loaded_models
            in_persistence = model_name in self.get_persistence_info()
            
            if not in_memory and not in_persistence:
                logger.warning(f"Model '{model_name}' not loaded")
                return False
            
            logger.info(f"Unloading model '{model_name}'")
            
            # Remove from memory if present
            if in_memory:
                del self._loaded_models[model_name]
                
            # Force garbage collection and clear GPU memory
            gc.collect()
            clear_gpu_memory()
            
            # Update persistence state - remove from persistence
            self._remove_from_persistence(model_name)
            
            return True
    
    def unload_all_models(self) -> bool:
        """Unload all loaded models."""
        with self._lock:
            # Get models from both memory and persistence
            memory_models = list(self._loaded_models.keys())
            persistence_models = list(self.get_persistence_info().keys())
            all_models = list(set(memory_models + persistence_models))
            
            if not all_models:
                logger.info("No models to unload")
                return True
            
            logger.info(f"Unloading {len(all_models)} models")
            
            # Clear memory models
            for model_name in memory_models:
                if model_name in self._loaded_models:
                    del self._loaded_models[model_name]
            
            # Force garbage collection and clear GPU memory
            gc.collect()
            clear_gpu_memory()
            
            # Clear persistence state since no models are loaded
            self._clear_persistence_state()
            
            return True
    
    def _remove_from_persistence(self, model_name: str):
        """Remove a specific model from persistence state."""
        try:
            if self._persistence_file and self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                
                loaded_models = data.get('loaded_models', {})
                if model_name in loaded_models:
                    del loaded_models[model_name]
                    
                    # Update the data
                    data['loaded_models'] = loaded_models
                    data['timestamp'] = time.time()
                    
                    # Write back to file
                    temp_file = self._persistence_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Atomic move
                    temp_file.replace(self._persistence_file)
                    
                    logger.info(f"Removed '{model_name}' from persistence state")
                
        except Exception as e:
            logger.warning(f"Error removing model from persistence: {e}")

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
    
    def get_loaded_model(self, model_name: str) -> Optional[ModelInterface]:
        """Get a loaded model by name."""
        with self._lock:
            # For persistence scenarios, if model not in memory but in persistence,
            # return None to trigger reloading
            if model_name not in self._loaded_models:
                persistence_info = self.get_persistence_info()
                if model_name in persistence_info:
                    logger.info(f"Model '{model_name}' exists in persistence but not in memory. May need reloading.")
                    return None
            
            return self._loaded_models.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models."""
        with self._lock:
            # For CLI commands, return persisted model list if no models in memory
            if not self._loaded_models:
                persistence_info = self.get_persistence_info()
                return list(persistence_info.keys())
            
            return list(self._loaded_models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        with self._lock:
            # Check both memory and persistence state
            if model_name in self._loaded_models:
                return True
            
            # Check persistence state for CLI commands
            persistence_info = self.get_persistence_info()
            return model_name in persistence_info
