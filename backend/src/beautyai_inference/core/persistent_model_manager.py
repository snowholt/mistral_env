"""
Persistent Model Manager for BeautyAI Voice Service

This manager extends the existing ModelManager to provide persistent model preloading
capabilities for 24/7 voice service operations. It ensures models are loaded on
server startup and kept warm for instant response times.

Features:
- Singleton pattern for persistent model management
- Automatic model preloading on server startup
- Thread-safe access to persistent model instances
- Memory monitoring and cleanup methods
- Graceful fallback to existing ModelManager

Author: BeautyAI Framework
Date: 2024-09-11
"""

import asyncio
import logging
import time
import threading
import gc
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .model_manager import ModelManager
from ..config.config_manager import ModelConfig
from ..utils.memory_utils import clear_gpu_memory, get_gpu_memory_stats

logger = logging.getLogger(__name__)


class PersistentModelManager:
    """
    Singleton class for persistent model management with preloading capabilities.
    
    This manager extends ModelManager to provide:
    - Automatic model preloading on startup
    - Persistent instances for voice services
    - Memory monitoring and optimization
    - Thread-safe access patterns
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialization_lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PersistentModelManager, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance._preloaded_models = {}
                cls._instance._preload_config = None
                cls._instance._startup_time = None
                cls._instance._model_manager = ModelManager()
            return cls._instance
    
    def __init__(self):
        """Initialize the persistent model manager."""
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self._preloaded_models = {}
        self._preload_config = None
        self._startup_time = None
        self._model_manager = ModelManager()
        self._memory_thresholds = {
            'max_gpu_memory_mb': 20000,  # 20GB max GPU memory
            'warning_threshold_mb': 16000,  # 16GB warning threshold
            'min_free_memory_mb': 4000   # 4GB minimum free memory
        }
        self._initialized = False
        
        self.logger.info("PersistentModelManager instance created")
    
    async def preload_models(self) -> bool:
        """
        Preload models defined in configuration on server startup.
        
        Returns:
            bool: True if all models loaded successfully, False if any failed
        """
        with self._initialization_lock:
            if self._initialized:
                self.logger.info("Models already preloaded, skipping initialization")
                return True
            
            try:
                self.logger.info("🚀 Starting persistent model preloading...")
                start_time = time.time()
                
                # Load preload configuration
                await self._load_preload_config()
                
                if not self._preload_config or not self._preload_config.get('preload_on_startup', False):
                    self.logger.info("Model preloading disabled in configuration")
                    self._initialized = True
                    return True
                
                # Check memory before loading models
                memory_check = await self._check_memory_availability()
                if not memory_check['sufficient_memory']:
                    self.logger.error(f"Insufficient memory for model preloading: {memory_check}")
                    return False
                
                # Preload each configured model
                models_config = self._preload_config.get('models', {})
                success_count = 0
                total_count = len(models_config)
                
                for model_type, model_config in models_config.items():
                    try:
                        self.logger.info(f"Preloading {model_type} model...")
                        success = await self._preload_single_model(model_type, model_config)
                        if success:
                            success_count += 1
                            self.logger.info(f"✅ {model_type} model preloaded successfully")
                        else:
                            self.logger.error(f"❌ Failed to preload {model_type} model")
                    except Exception as e:
                        self.logger.error(f"❌ Error preloading {model_type} model: {e}")
                
                # Update initialization status
                self._startup_time = time.time() - start_time
                self._initialized = True
                
                if success_count == total_count:
                    self.logger.info(f"🎉 All {total_count} models preloaded successfully in {self._startup_time:.2f}s")
                    return True
                else:
                    self.logger.warning(f"⚠️ Only {success_count}/{total_count} models preloaded successfully")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Critical error during model preloading: {e}")
                return False
    
    async def _load_preload_config(self):
        """Load preload configuration from file."""
        try:
            config_dir = Path(__file__).parent.parent / "config"
            config_file = config_dir / "preload_config.json"
            
            if not config_file.exists():
                self.logger.warning(f"Preload config file not found: {config_file}")
                # Create default configuration
                await self._create_default_preload_config(config_file)
            
            import json
            with open(config_file, 'r') as f:
                self._preload_config = json.load(f)
            
            self.logger.info(f"Loaded preload configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading preload configuration: {e}")
            # Use fallback configuration
            self._preload_config = await self._get_fallback_config()
    
    async def _create_default_preload_config(self, config_file: Path):
        """Create default preload configuration file."""
        try:
            default_config = await self._get_fallback_config()
            
            import json
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            self.logger.info(f"Created default preload configuration: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating default preload config: {e}")
    
    async def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration when config file is not available."""
        return {
            "preload_on_startup": True,
            "models": {
                "whisper": {
                    "model_id": "whisper-large-v3-turbo",
                    "device": "cuda",
                    "compute_type": "float16",
                    "priority": 1
                },
                "llm": {
                    "model_path": "qwen3-unsloth-q4ks",
                    "context_size": 4096,
                    "priority": 2
                }
            },
            "memory_thresholds": {
                "max_gpu_memory_mb": 20000,
                "warning_threshold_mb": 16000,
                "min_free_memory_mb": 4000
            }
        }
    
    async def _check_memory_availability(self) -> Dict[str, Any]:
        """Check if sufficient memory is available for model loading."""
        try:
            gpu_stats = get_gpu_memory_stats()
            
            if not gpu_stats:
                return {
                    'sufficient_memory': False,
                    'reason': 'No GPU information available'
                }
            
            gpu_info = gpu_stats[0]  # Use first GPU
            free_memory_mb = gpu_info.get('memory_free_mb', 0)
            total_memory_mb = gpu_info.get('memory_total_mb', 0)
            used_memory_mb = gpu_info.get('memory_used_mb', 0)
            
            min_required = self._memory_thresholds['min_free_memory_mb']
            
            if free_memory_mb < min_required:
                return {
                    'sufficient_memory': False,
                    'reason': f'Insufficient free GPU memory: {free_memory_mb}MB < {min_required}MB required',
                    'free_memory_mb': free_memory_mb,
                    'total_memory_mb': total_memory_mb,
                    'used_memory_mb': used_memory_mb
                }
            
            return {
                'sufficient_memory': True,
                'free_memory_mb': free_memory_mb,
                'total_memory_mb': total_memory_mb,
                'used_memory_mb': used_memory_mb
            }
            
        except Exception as e:
            self.logger.error(f"Error checking memory availability: {e}")
            return {
                'sufficient_memory': False,
                'reason': f'Memory check failed: {e}'
            }
    
    async def _preload_single_model(self, model_type: str, model_config: Dict[str, Any]) -> bool:
        """
        Preload a single model based on its type and configuration.
        
        Args:
            model_type: Type of model ('whisper', 'llm', 'tts')
            model_config: Model configuration dictionary
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if model_type == "whisper":
                return await self._preload_whisper_model(model_config)
            elif model_type == "llm":
                return await self._preload_llm_model(model_config)
            elif model_type == "tts":
                return await self._preload_tts_model(model_config)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading {model_type} model: {e}")
            return False
    
    async def _preload_whisper_model(self, config: Dict[str, Any]) -> bool:
        """Preload Whisper model using existing ModelManager."""
        try:
            # Use existing ModelManager's get_streaming_whisper method
            whisper_engine = self._model_manager.get_streaming_whisper(
                model_name=config.get('model_id'),
                language="auto"
            )
            
            if whisper_engine:
                self._preloaded_models['whisper'] = whisper_engine
                self.logger.info(f"Whisper model preloaded: {config.get('model_id')}")
                return True
            else:
                self.logger.error("Failed to preload Whisper model")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading Whisper model: {e}")
            return False
    
    async def _preload_llm_model(self, config: Dict[str, Any]) -> bool:
        """Preload LLM model using existing ModelManager."""
        try:
            from ..config.config_manager import ModelConfig
            
            # Use the fastest model from config or fallback to qwen3-unsloth-q4ks
            fastest_model_name = config.get('model_path', 'qwen3-unsloth-q4ks')
            
            # Create model config for the fastest model
            model_config = ModelConfig(
                name=fastest_model_name,
                model_id=fastest_model_name,  # Use model name as ID for registry lookup
                engine_type="llama.cpp"  # Correct engine type for GGUF models
            )
            
            # Load model using ModelManager
            model_instance = self._model_manager.load_model(model_config)
            if model_instance:
                self._preloaded_models['llm'] = model_instance
                self.logger.info(f"LLM model preloaded: {fastest_model_name}")
                return True
            else:
                self.logger.error("Failed to load LLM model instance")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading LLM model: {e}")
            return False
    
    async def _preload_tts_model(self, config: Dict[str, Any]) -> bool:
        """Preload TTS model (Edge TTS is always available)."""
        try:
            # Edge TTS doesn't need preloading, just mark as available
            tts_engine = self._model_manager.get_tts_engine()
            if tts_engine:
                self._preloaded_models['tts'] = tts_engine
                self.logger.info("TTS engine ready (Edge TTS)")
                return True
            else:
                self.logger.error("Failed to get TTS engine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading TTS model: {e}")
            return False
    
    def get_whisper_model(self) -> Optional[Any]:
        """
        Get persistent Whisper model instance.
        
        Returns:
            Persistent Whisper model instance or None if not loaded
        """
        if 'whisper' in self._preloaded_models:
            return self._preloaded_models['whisper']
        
        # Fallback to ModelManager if not preloaded
        self.logger.warning("Whisper model not preloaded, using ModelManager fallback")
        return self._model_manager.get_streaming_whisper()
    
    def get_llm_model(self) -> Optional[Any]:
        """
        Get persistent LLM model instance.
        
        Returns:
            Persistent LLM model instance or None if not loaded
        """
        if 'llm' in self._preloaded_models:
            return self._preloaded_models['llm']
        
        # Fallback to ModelManager
        self.logger.warning("LLM model not preloaded, using ModelManager fallback")
        loaded_models = self._model_manager.list_loaded_models()
        if loaded_models:
            return self._model_manager.get_loaded_model(loaded_models[0])
        return None
    
    def get_tts_engine(self) -> Optional[Any]:
        """
        Get persistent TTS engine instance.
        
        Returns:
            Persistent TTS engine instance or None if not available
        """
        if 'tts' in self._preloaded_models:
            return self._preloaded_models['tts']
        
        # Fallback to ModelManager
        return self._model_manager.get_tts_engine()
    
    def is_initialized(self) -> bool:
        """Check if persistent models are initialized."""
        return self._initialized
    
    def check_models_ready(self) -> Dict[str, bool]:
        """
        Check readiness status of all preloaded models.
        
        Returns:
            Dictionary with model readiness status
        """
        return {
            'whisper': 'whisper' in self._preloaded_models,
            'llm': 'llm' in self._preloaded_models,
            'tts': 'tts' in self._preloaded_models,
            'all_ready': all(
                model_type in self._preloaded_models 
                for model_type in ['whisper', 'llm', 'tts']
            )
        }
    
    async def monitor_memory(self) -> Dict[str, Any]:
        """
        Monitor memory usage and model status.
        
        Returns:
            Dictionary with memory and model monitoring data
        """
        try:
            # Get GPU memory stats
            gpu_stats = get_gpu_memory_stats()
            memory_info = gpu_stats[0] if gpu_stats else {}
            
            # Get model status
            model_status = self.check_models_ready()
            
            # Check memory thresholds
            used_memory = memory_info.get('memory_used_mb', 0)
            warning_threshold = self._memory_thresholds['warning_threshold_mb']
            max_threshold = self._memory_thresholds['max_gpu_memory_mb']
            
            memory_warning = used_memory > warning_threshold
            memory_critical = used_memory > max_threshold
            
            monitoring_data = {
                'timestamp': time.time(),
                'memory_info': memory_info,
                'model_status': model_status,
                'memory_warnings': {
                    'warning_level': memory_warning,
                    'critical_level': memory_critical,
                    'used_memory_mb': used_memory,
                    'warning_threshold_mb': warning_threshold,
                    'max_threshold_mb': max_threshold
                },
                'startup_info': {
                    'initialized': self._initialized,
                    'startup_time_seconds': self._startup_time,
                    'preloaded_models_count': len(self._preloaded_models)
                }
            }
            
            # Log warnings if needed
            if memory_critical:
                self.logger.error(f"CRITICAL: GPU memory usage exceeds maximum threshold: {used_memory}MB > {max_threshold}MB")
            elif memory_warning:
                self.logger.warning(f"WARNING: GPU memory usage exceeds warning threshold: {used_memory}MB > {warning_threshold}MB")
            
            return monitoring_data
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def cleanup_models(self) -> bool:
        """
        Gracefully cleanup preloaded models for shutdown.
        
        Returns:
            bool: True if cleanup successful
        """
        try:
            self.logger.info("🛑 Cleaning up preloaded models...")
            
            # Cleanup each preloaded model
            for model_type, model_instance in self._preloaded_models.items():
                try:
                    if hasattr(model_instance, 'cleanup'):
                        model_instance.cleanup()
                    elif hasattr(model_instance, 'unload_model'):
                        model_instance.unload_model()
                    
                    self.logger.info(f"Cleaned up {model_type} model")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {model_type} model: {e}")
            
            # Clear preloaded models registry
            self._preloaded_models.clear()
            
            # Force garbage collection
            gc.collect()
            clear_gpu_memory()
            
            # Reset initialization status
            self._initialized = False
            
            self.logger.info("✅ Preloaded models cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during models cleanup: {e}")
            return False
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Get initialization and performance statistics."""
        return {
            'initialized': self._initialized,
            'startup_time_seconds': self._startup_time,
            'preloaded_models': list(self._preloaded_models.keys()),
            'preloaded_models_count': len(self._preloaded_models),
            'memory_thresholds': self._memory_thresholds,
            'configuration_loaded': self._preload_config is not None
        }


# Global instance for singleton access
_persistent_model_manager = None


def get_persistent_model_manager() -> PersistentModelManager:
    """
    Get the global persistent model manager instance.
    
    Returns:
        PersistentModelManager: Global singleton instance
    """
    global _persistent_model_manager
    
    if _persistent_model_manager is None:
        _persistent_model_manager = PersistentModelManager()
    
    return _persistent_model_manager


async def initialize_persistent_models() -> bool:
    """
    Initialize persistent models for production use.
    
    Returns:
        bool: True if initialization successful
    """
    manager = get_persistent_model_manager()
    return await manager.preload_models()


async def cleanup_persistent_models() -> bool:
    """
    Cleanup persistent models for shutdown.
    
    Returns:
        bool: True if cleanup successful
    """
    manager = get_persistent_model_manager()
    return await manager.cleanup_models()