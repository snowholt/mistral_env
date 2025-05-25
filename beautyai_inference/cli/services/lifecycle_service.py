"""
Lifecycle service for unified CLI.
"""
import logging
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ...config.config_manager import AppConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_memory_stats

logger = logging.getLogger(__name__)


class LifecycleService(BaseService):
    """Service for managing model lifecycle operations."""
    
    def __init__(self):
        super().__init__()
        self.app_config: Optional[AppConfig] = None
        self.model_manager = ModelManager()
    
    def load_model(self, args):
        """Load a model into memory."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.get_models()
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
        
        model_config = models[name]
        
        try:
            print(f"Loading model '{name}' ({model_config.model_id})...")
            self.model_manager.load_model(name, model_config)
            print(f"Model '{name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            logger.exception(f"Failed to load model {name}")
            return 1
    
    def unload_model(self, args):
        """Unload a model from memory."""
        name = args.name
        
        if not self.model_manager.is_model_loaded(name):
            print(f"Error: Model '{name}' is not loaded.")
            return 1
        
        try:
            print(f"Unloading model '{name}'...")
            self.model_manager.unload_model(name)
            print(f"Model '{name}' unloaded successfully.")
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
            logger.exception(f"Failed to unload model {name}")
            return 1
    
    def unload_all_models(self, args):
        """Unload all models from memory."""
        try:
            loaded_models = self.model_manager.get_loaded_models()
            if not loaded_models:
                print("No models are currently loaded.")
                return
                
            print("Unloading all models...")
            self.model_manager.unload_all_models()
            print("All models unloaded successfully.")
        except Exception as e:
            print(f"Error unloading models: {str(e)}")
            logger.exception("Failed to unload all models")
            return 1
    
    def list_loaded_models(self, args):
        """List all loaded models."""
        loaded_models = self.model_manager.get_loaded_models()
        
        if not loaded_models:
            print("No models are currently loaded.")
            return
            
        print("\n{:<15} {:<30} {:<15} {:<10}".format(
            "MODEL NAME", "MODEL ID", "ENGINE", "QUANT"))
        print("-" * 70)
        
        for name, model in loaded_models.items():
            print("{:<15} {:<30} {:<15} {:<10}".format(
                name, 
                model.config.model_id, 
                model.config.engine_type, 
                model.config.quantization or "none"
            ))
        print()
    
    def show_status(self, args):
        """Show memory usage and system status."""
        loaded_models = self.model_manager.get_loaded_models()
        
        print("\n=== System Status ===")
        
        # Get GPU memory stats
        memory_stats = get_gpu_memory_stats()
        for i, stats in enumerate(memory_stats):
            print(f"\nGPU {i}: {stats['name']}")
            print(f"  Memory Used:     {stats['memory_used_mb']:.2f} MB / {stats['memory_total_mb']:.2f} MB "
                  f"({stats['memory_used_percent']:.1f}%)")
            print(f"  Memory Free:     {stats['memory_free_mb']:.2f} MB")
            print(f"  Utilization:     {stats['gpu_utilization']:.1f}%")
        
        # Show loaded models
        print("\n=== Loaded Models ===")
        if loaded_models:
            for name, model in loaded_models.items():
                print(f"  - {name} ({model.config.model_id})")
        else:
            print("  No models currently loaded.")
        print()
    
    def clear_cache(self, args):
        """Clear model cache from disk."""
        self._load_config(args)
        
        name = args.name
        models = self.app_config.get_models()
        
        if name not in models:
            print(f"Error: Model '{name}' not found in registry.")
            return 1
        
        model_config = models[name]
        
        try:
            print(f"Clearing cache for model '{name}' ({model_config.model_id})...")
            self.model_manager.clear_model_cache(model_config.model_id)
            print(f"Cache for model '{name}' cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            logger.exception(f"Failed to clear cache for model {name}")
            return 1
    
    def _load_config(self, args):
        """Load the configuration."""
        config_file = getattr(args, "config", None)
        models_file = getattr(args, "models_file", None)
        
        self.app_config = AppConfig(
            config_file=config_file,
            models_file=models_file
        )
