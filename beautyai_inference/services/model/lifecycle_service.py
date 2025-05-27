"""
Model lifecycle service for loading and unloading operations.

This service handles:
- Loading models into memory
- Unloading models from memory
- Listing loaded models
- Memory status reporting
- Cache management operations
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from ..base.base_service import BaseService
from ...config.config_manager import ModelConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_memory_stats, format_size

logger = logging.getLogger(__name__)


class ModelLifecycleService(BaseService):
    """Service for managing model lifecycle operations."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
    
    def load_model(self, model_config: ModelConfig, show_progress: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Load a model into memory.
        
        Args:
            model_config: The model configuration to load
            show_progress: Whether to show loading progress
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if already loaded
            if self.model_manager.is_model_loaded(model_config.name):
                logger.info(f"Model '{model_config.name}' is already loaded")
                return True, None
            
            # Show memory status before loading
            if show_progress:
                logger.info(f"Loading model '{model_config.name}'...")
                self._print_memory_status()
            
            # Estimate memory requirements
            estimated_memory = self._estimate_memory_requirements(model_config)
            available_memory = self._check_available_memory()
            
            if estimated_memory > available_memory:
                error_msg = (f"Insufficient memory to load model. "
                           f"Estimated: {format_size(estimated_memory)}, "
                           f"Available: {format_size(available_memory)}")
                logger.warning(error_msg)
                return False, error_msg
            
            # Load the model with progress bar
            if show_progress:
                with tqdm(total=100, desc=f"Loading {model_config.name}", unit="%") as pbar:
                    pbar.update(20)  # Configuration loaded
                    
                    model_instance = self.model_manager.load_model(model_config)
                    pbar.update(80)  # Model loaded
                    
                    if model_instance:
                        pbar.update(100)
                        logger.info(f"✅ Model '{model_config.name}' loaded successfully")
                        self._print_memory_status()
                        return True, None
                    else:
                        error_msg = f"Failed to load model '{model_config.name}'"
                        logger.error(error_msg)
                        return False, error_msg
            else:
                model_instance = self.model_manager.load_model(model_config)
                if model_instance:
                    logger.info(f"Model '{model_config.name}' loaded successfully")
                    return True, None
                else:
                    error_msg = f"Failed to load model '{model_config.name}'"
                    logger.error(error_msg)
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Error loading model '{model_config.name}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def unload_model(self, model_name: str, show_progress: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            show_progress: Whether to show unloading progress
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not self.model_manager.is_model_loaded(model_name):
                logger.warning(f"Model '{model_name}' is not loaded")
                return False, f"Model '{model_name}' is not loaded"
            
            if show_progress:
                logger.info(f"Unloading model '{model_name}'...")
                self._print_memory_status()
            
            success = self.model_manager.unload_model(model_name)
            
            if success:
                logger.info(f"✅ Model '{model_name}' unloaded successfully")
                if show_progress:
                    self._print_memory_status()
                return True, None
            else:
                error_msg = f"Failed to unload model '{model_name}'"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error unloading model '{model_name}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def unload_all_models(self, show_progress: bool = True) -> Tuple[bool, List[str]]:
        """
        Unload all models from memory.
        
        Args:
            show_progress: Whether to show unloading progress
            
        Returns:
            Tuple of (all_successful, list_of_errors)
        """
        try:
            loaded_models = self.model_manager.list_loaded_models()
            
            if not loaded_models:
                logger.info("No models are currently loaded")
                return True, []
            
            if show_progress:
                logger.info(f"Unloading {len(loaded_models)} models...")
                self._print_memory_status()
            
            success = self.model_manager.unload_all_models()
            
            if success:
                logger.info(f"✅ All models unloaded successfully")
                if show_progress:
                    self._print_memory_status()
                return True, []
            else:
                error_msg = "Failed to unload all models"
                logger.error(error_msg)
                return False, [error_msg]
                
        except Exception as e:
            error_msg = f"Error unloading all models: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List all currently loaded models and show cross-process state.
        Returns:
            List of loaded model information
        """
        try:
            # Print cross-process state summary for user clarity
            self.model_manager.print_cross_process_model_state()
            loaded_models = self.model_manager.list_loaded_models()
            models_info = []
            for model_name in loaded_models:
                model_instance = self.model_manager.get_loaded_model(model_name)
                model_info = {
                    'name': model_name,
                    'status': 'loaded' if model_instance else 'error',
                    'engine': getattr(getattr(model_instance, 'config', None), 'engine_type', 'unknown') if model_instance else 'unknown',
                    'model_id': getattr(getattr(model_instance, 'config', None), 'model_id', 'unknown') if model_instance else 'unknown'
                }
                models_info.append(model_info)
            return models_info
        except Exception as e:
            logger.error(f"Error listing loaded models: {str(e)}")
            return []
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status and loaded models.
        
        Returns:
            Dictionary with memory status information
        """
        try:
            gpu_stats = get_gpu_memory_stats()
            loaded_models = self.model_manager.list_loaded_models()
            
            status = {
                'loaded_models_count': len(loaded_models),
                'loaded_models': loaded_models,
                'gpu_memory': gpu_stats,
                'has_gpu': gpu_stats is not None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting memory status: {str(e)}")
            return {
                'loaded_models_count': 0,
                'loaded_models': [],
                'gpu_memory': None,
                'has_gpu': False,
                'error': str(e)
            }
    
    def clear_model_cache(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """
        Clear cache for a specific model.
        
        Args:
            model_id: The model ID to clear cache for
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            success = self.model_manager.clear_model_cache(model_id)
            
            if success:
                logger.info(f"✅ Cache cleared for model '{model_id}'")
                return True, None
            else:
                error_msg = f"Failed to clear cache for model '{model_id}'"
                logger.warning(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error clearing cache for model '{model_id}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if model is loaded, False otherwise
        """
        try:
            return self.model_manager.is_model_loaded(model_name)
        except Exception as e:
            logger.error(f"Error checking if model '{model_name}' is loaded: {str(e)}")
            return False
    
    def _estimate_memory_requirements(self, model_config: ModelConfig) -> int:
        """
        Estimate memory requirements for a model.
        
        Args:
            model_config: The model configuration
            
        Returns:
            Estimated memory in bytes
        """
        try:
            # Basic estimation based on model type and quantization
            # This is a simplified estimation - could be enhanced with actual model inspection
            
            base_memory = 2 * 1024 * 1024 * 1024  # 2GB base
            
            # Adjust based on quantization
            if model_config.quantization == '4bit':
                multiplier = 0.5
            elif model_config.quantization == '8bit':
                multiplier = 0.75
            else:
                multiplier = 1.0
            
            # Rough estimation based on model ID (could be enhanced)
            if 'large' in model_config.model_id.lower():
                base_memory *= 4
            elif 'medium' in model_config.model_id.lower():
                base_memory *= 2
            elif 'small' in model_config.model_id.lower():
                base_memory *= 0.5
            
            return int(base_memory * multiplier)
            
        except Exception as e:
            logger.warning(f"Could not estimate memory for '{model_config.name}': {e}")
            return 4 * 1024 * 1024 * 1024  # 4GB fallback
    
    def _check_available_memory(self) -> int:
        """
        Check available GPU memory.
        
        Returns:
            Available memory in bytes
        """
        try:
            gpu_stats = get_gpu_memory_stats()
            if gpu_stats and len(gpu_stats) > 0:
                # Use the first GPU's available memory
                return gpu_stats[0].get('memory_free', 0)
            else:
                # Fallback to a conservative estimate
                return 8 * 1024 * 1024 * 1024  # 8GB fallback
                
        except Exception as e:
            logger.warning(f"Could not check available memory: {e}")
            return 8 * 1024 * 1024 * 1024  # 8GB fallback
    
    def _print_memory_status(self) -> None:
        """Print current memory status to console."""
        try:
            status = self.get_memory_status()
            
            print(f"\n📊 Memory Status:")
            print(f"   Loaded models: {status['loaded_models_count']}")
            
            if status['has_gpu'] and status['gpu_memory']:
                for i, gpu in enumerate(status['gpu_memory']):
                    total = gpu.get('memory_total', 0)
                    used = gpu.get('memory_used', 0)
                    free = gpu.get('memory_free', 0)
                    
                    if total > 0:
                        usage_percent = (used / total) * 100
                        print(f"   GPU {i}: {format_size(used)}/{format_size(total)} ({usage_percent:.1f}% used)")
                    else:
                        print(f"   GPU {i}: Status unavailable")
            else:
                print("   GPU: Not available or not detected")
            
            print()
            
        except Exception as e:
            logger.warning(f"Could not print memory status: {e}")
            print("📊 Memory status unavailable\n")
