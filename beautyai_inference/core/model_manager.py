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
from typing import Dict, Optional, List, Any
from threading import Timer

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
                cls._instance._model_timers = {}  # Dict[str, Timer] - auto-unload timers
                cls._instance._model_last_used = {}  # Dict[str, float] - last access timestamps
                cls._instance._auto_unload_minutes = 60  # Default 60 minutes keep-alive
                cls._instance._initialize_persistence()
            return cls._instance
    
    def _initialize_persistence(self):
        """Initialize persistence file for CLI state management."""
        # Use config directory for persistence state - more organized
        config_dir = Path(__file__).parent.parent / "config"
        self._persistence_file = config_dir / "loaded_models_state.json"
        
        # Load existing state if available
        self._load_persistence_state()
    
    def _load_persistence_state(self):
        """Load model state from persistence file."""
        try:
            if self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                # Check if state is recent (within last 24 hours)
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp < 86400:  # 24 hours
                    loaded_models = data.get('loaded_models', {})
                    if loaded_models:
                        logger.info(f"Found {len(loaded_models)} models in recent persistence state")
                        for model_name, model_info in loaded_models.items():
                            logger.info(f"  - {model_name}: {model_info.get('model_id', 'unknown')} (not loaded in memory)")
                        logger.info("Note: Persistence tracks previous session state, actual models must be reloaded")
                    else:
                        logger.debug("No models in persistence state")
                else:
                    logger.debug("Persistence state is stale (>24h), clearing")
                    self._clear_persistence_state()
        except Exception as e:
            logger.warning(f"Error loading persistence state: {e}")
            self._clear_persistence_state()
    def get_cross_process_model_state(self) -> dict:
        """
        Returns a dict with two keys:
        - 'in_memory': models loaded in this process
        - 'persisted': models found in persistence file (may not be loaded in memory)
        """
        in_memory = list(self._loaded_models.keys())
        persisted = []
        try:
            if self._persistence_file and self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                persisted = list(data.get('loaded_models', {}).keys())
        except Exception as e:
            logger.warning(f"Error reading persistence info: {e}")
        return {
            'in_memory': in_memory,
            'persisted': persisted
        }
    def print_cross_process_model_state(self):
        """
        Print a clear summary of model state for both in-memory and persisted models.
        """
        state = self.get_cross_process_model_state()
        print("\nModel State Summary:")
        print(f"  Models loaded in this process: {state['in_memory'] if state['in_memory'] else 'None'}")
        print(f"  Models in persistence (previously loaded, not in memory): {state['persisted'] if state['persisted'] else 'None'}")
        if state['persisted'] and not state['in_memory']:
            print("  ⚠️  Models listed in persistence are NOT loaded in memory. Use 'beautyai system load <model>' to reload.")
        print()
    
    def _verify_persisted_models(self, persisted_models: Dict[str, Any]):
        """Verify if persisted models are actually still loaded in memory by checking GPU/system memory."""
        try:
            # Import here to avoid circular imports
            from ..utils.memory_utils import get_gpu_memory_stats
            import psutil
            
            # Get current memory usage
            gpu_stats = get_gpu_memory_stats()
            
            # If GPU memory is being used significantly, models might still be loaded
            if gpu_stats and len(gpu_stats) > 0:
                gpu_memory_used = gpu_stats[0].get('memory_used_mb', 0)
                
                # If GPU has significant memory usage (>1GB), assume models are loaded
                if gpu_memory_used > 1000:  # 1GB threshold
                    logger.info(f"GPU memory usage detected ({gpu_memory_used:.0f}MB), models may still be loaded from previous session")
                    logger.info("Note: Cannot restore model instances across processes. Use 'beautyai system load' to reload if needed.")
                else:
                    logger.debug(f"Low GPU memory usage ({gpu_memory_used:.0f}MB), models likely not loaded")
            else:
                logger.debug("Could not check GPU memory usage")
                
        except Exception as e:
            logger.debug(f"Could not verify persisted models: {e}")
    
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
                    # Get model info from the instance's config
                    config = getattr(model_instance, 'config', None)
                    model_info = {
                        'model_id': config.model_id if config else getattr(model_instance, 'model_id', 'unknown'),
                        'engine': config.engine_type if config else getattr(model_instance, 'engine_type', 'unknown'),
                        'loaded_at': time.time(),
                        'status': 'loaded'
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
            
            # Start or reset the keep-alive timer for the model
            self._start_model_timer(model_name)
            
            return model

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            # Check only in-memory models
            in_memory = model_name in self._loaded_models
            
            if not in_memory:
                logger.warning(f"Model '{model_name}' not loaded in memory")
                return False
            
            logger.info(f"Unloading model '{model_name}'")
            
            # Get the model instance before removing it
            model_instance = self._loaded_models.get(model_name)
            
            # Call the model's unload method if available
            if model_instance and hasattr(model_instance, 'unload_model'):
                try:
                    model_instance.unload_model()
                except Exception as e:
                    logger.error(f"Error calling model's unload method: {e}")
            
            # Remove from memory
            del self._loaded_models[model_name]
                
            # Force garbage collection and clear GPU memory
            import gc
            gc.collect()
            clear_gpu_memory()
            
            # Update persistence state - remove from persistence
            self._remove_from_persistence(model_name)
            
            # Stop the keep-alive timer for the model
            self._stop_model_timer(model_name)
            
            return True
    
    def unload_all_models(self) -> bool:
        """Unload all loaded models."""
        with self._lock:
            # Get models only from memory
            memory_models = list(self._loaded_models.keys())
            
            if not memory_models:
                logger.info("No models to unload")
                return True
            
            logger.info(f"Unloading {len(memory_models)} models")
            
            # Clear memory models one by one, calling their unload methods
            for model_name in memory_models:
                if model_name in self._loaded_models:
                    model_instance = self._loaded_models[model_name]
                    
                    # Call the model's unload method if available
                    if model_instance and hasattr(model_instance, 'unload_model'):
                        try:
                            model_instance.unload_model()
                        except Exception as e:
                            logger.error(f"Error calling unload method for {model_name}: {e}")
                    
                    del self._loaded_models[model_name]
            
            # Force garbage collection and clear GPU memory
            gc.collect()
            clear_gpu_memory()
            
            # Clear persistence state since no models are loaded
            self._clear_persistence_state()
            
            # Stop all keep-alive timers
            self._stop_all_model_timers()
            
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
            # Only return models that are actually loaded in memory
            # Don't check persistence state for get_loaded_model to avoid confusion
            return self._loaded_models.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models that are actually in memory."""
        with self._lock:
            # Only return models that are actually loaded in memory and working
            # Check if each model is still functional
            working_models = []
            models_to_remove = []
            
            for model_name, model_instance in self._loaded_models.items():
                try:
                    # Quick check if model is still accessible
                    if hasattr(model_instance, 'model') and model_instance.model is not None:
                        working_models.append(model_name)
                    else:
                        logger.warning(f"Model '{model_name}' in memory but not functional, will remove")
                        models_to_remove.append(model_name)
                except Exception as e:
                    logger.warning(f"Model '{model_name}' check failed: {e}, will remove")
                    models_to_remove.append(model_name)
            
            # Clean up non-functional models
            for model_name in models_to_remove:
                if model_name in self._loaded_models:
                    del self._loaded_models[model_name]
                    self._remove_from_persistence(model_name)
            
            # Update persistence if we removed any models
            if models_to_remove:
                self._save_persistence_state()
            
            return working_models
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is actually loaded in memory and functional."""
        with self._lock:
            # Check if model is in memory and functional
            if model_name not in self._loaded_models:
                return False
            
            try:
                model_instance = self._loaded_models[model_name]
                # Verify the model is actually loaded and functional
                if hasattr(model_instance, 'model') and model_instance.model is not None:
                    return True
                else:
                    # Model is in our dict but not actually loaded, clean up
                    logger.warning(f"Model '{model_name}' was in memory registry but not functional, removing")
                    del self._loaded_models[model_name]
                    return False
            except Exception as e:
                logger.warning(f"Error checking model '{model_name}': {e}, removing from memory registry")
                if model_name in self._loaded_models:
                    del self._loaded_models[model_name]
                return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model (both loaded and persistence state)."""
        with self._lock:
            info = {}
            
            # Check if actually loaded in memory
            if model_name in self._loaded_models:
                model_instance = self._loaded_models[model_name]
                try:
                    if hasattr(model_instance, 'model') and model_instance.model is not None:
                        config = getattr(model_instance, 'config', None)
                        info = {
                            'name': model_name,
                            'status': 'loaded_in_memory',
                            'model_id': config.model_id if config else 'unknown',
                            'engine_type': config.engine_type if config else 'unknown',
                            'device': str(getattr(model_instance.model, 'device', 'unknown')) if hasattr(model_instance, 'model') else 'unknown'
                        }
                    else:
                        info = {'name': model_name, 'status': 'registered_but_not_functional'}
                except Exception as e:
                    info = {'name': model_name, 'status': f'error: {e}'}
            
            # Check persistence state
            persistence_info = self.get_persistence_info()
            if model_name in persistence_info:
                if not info:  # Not in memory
                    info = {
                        'name': model_name,
                        'status': 'in_persistence_only',
                        'model_id': persistence_info[model_name].get('model_id', 'unknown'),
                        'engine_type': persistence_info[model_name].get('engine', 'unknown'),
                        'last_loaded': persistence_info[model_name].get('loaded_at', 'unknown')
                    }
                else:
                    # Add persistence info to memory info
                    info['persistence_state'] = persistence_info[model_name]
            
            return info if info else None
    
    def _start_model_timer(self, model_name: str):
        """Start or reset the keep-alive timer for a model."""
        self._stop_model_timer(model_name)  # Ensure no duplicate timers
        timer = Timer(self._auto_unload_minutes * 60, self._auto_unload_model, args=[model_name])
        timer.start()
        self._model_timers[model_name] = timer
        logger.info(f"Started keep-alive timer for model '{model_name}' (will unload after {self._auto_unload_minutes} minutes of inactivity)")
    
    def _stop_model_timer(self, model_name: str):
        """Stop the keep-alive timer for a model if running."""
        timer = self._model_timers.pop(model_name, None)
        if timer is not None:
            timer.cancel()
            logger.info(f"Stopped keep-alive timer for model '{model_name}'")
    
    def _stop_all_model_timers(self):
        """Stop all keep-alive timers."""
        for model_name in list(self._model_timers.keys()):
            self._stop_model_timer(model_name)
        logger.info("Stopped all keep-alive timers for loaded models")
    
    def _auto_unload_model(self, model_name: str):
        """Automatically unload a model after a period of inactivity."""
        with self._lock:
            if model_name in self._loaded_models:
                logger.info(f"Automatically unloading model '{model_name}' due to inactivity")
                self.unload_model(model_name)
            else:
                logger.debug(f"Model '{model_name}' not found in loaded models during auto-unload check")
