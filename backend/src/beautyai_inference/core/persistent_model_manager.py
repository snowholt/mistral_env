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
import os
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
                # LLM pool configuration
                cls._instance._llm_pool_size = 1
                cls._instance._llm_instance_counter = 0  # For round-robin selection
                cls._instance._llm_instances = {}  # Dict of instance_id -> model
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
        # LLM pool configuration - read from environment
        self._llm_pool_size = int(os.getenv('LLM_POOL_SIZE', '1'))
        self._llm_instance_counter = 0  # For round-robin selection
        self._llm_instances = {}  # Dict of instance_id -> model instance
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
            if model_type in ("whisper", "stt"):
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
        """Preload Whisper model using ModelManager."""
        try:
            model_id = config.get('model_id', 'whisper-byne-arabic')
            self.logger.info(f"Preloading Whisper model: {model_id}")
            
            # Use ModelManager's get_streaming_whisper with proper model name
            whisper_engine = self._model_manager.get_streaming_whisper(
                model_name=model_id,
                language="ar"  # Arabic for Genius AI model
            )
            
            if whisper_engine:
                self._preloaded_models['whisper'] = whisper_engine
                self._preloaded_models['stt'] = whisper_engine  # Alias for compatibility
                self.logger.info(f"✅ Whisper model preloaded: {model_id}")
                return True
            else:
                self.logger.error(f"Failed to preload Whisper model: {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading Whisper model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    async def _preload_llm_model(self, config: Dict[str, Any]) -> bool:
        """Preload LLM model(s) using ModelManager registry.
        
        Supports loading multiple instances based on LLM_POOL_SIZE environment variable
        or 'instances' field in preload config. Each instance is stored as 'llm:0', 'llm:1', etc.
        """
        try:
            from ..config.config_manager import AppConfig, ModelRegistry
            from pathlib import Path
            
            # Get model ID from config
            model_id = config.get('model_id', 'qwen3-unsloth-q4ks')
            
            # Determine number of instances to load:
            # Priority: config file 'instances' > LLM_POOL_SIZE env var > default (1)
            instances_to_load = config.get('instances', self._llm_pool_size)
            instances_to_load = max(1, min(instances_to_load, 4))  # Clamp between 1-4
            
            self.logger.info(f"🔢 Preloading {instances_to_load} LLM instance(s): {model_id}")
            self.logger.info(f"   (LLM_POOL_SIZE env: {os.getenv('LLM_POOL_SIZE', 'not set')}, config instances: {config.get('instances', 'not set')})")
            
            # Load the model registry
            config_dir = Path(__file__).parent.parent / "config"
            registry_file = config_dir / "model_registry.json"
            
            if not registry_file.exists():
                self.logger.error(f"Model registry file not found: {registry_file}")
                return False
            
            model_registry = ModelRegistry.load_from_file(registry_file)
            base_model_config = model_registry.get_model(model_id)
            
            if not base_model_config:
                self.logger.error(f"Model '{model_id}' not found in registry")
                return False
            
            self.logger.info(f"Loaded registry config for {model_id}: engine={base_model_config.engine_type}")
            
            # Load requested number of instances
            loaded_count = 0
            for i in range(instances_to_load):
                instance_name = f"llm:{i}"
                self.logger.info(f"   Loading instance {i+1}/{instances_to_load}: {instance_name}")
                
                try:
                    # Create a unique model config for each instance
                    from copy import deepcopy
                    instance_config = deepcopy(base_model_config)
                    instance_config.name = instance_name
                    
                    # Load model using ModelManager
                    model_instance = self._model_manager.load_model(instance_config)
                    
                    if model_instance:
                        self._llm_instances[i] = model_instance
                        self._preloaded_models[instance_name] = model_instance
                        loaded_count += 1
                        self.logger.info(f"   ✅ LLM instance {i} loaded successfully")
                    else:
                        self.logger.error(f"   ❌ Failed to load LLM instance {i}")
                        
                except Exception as instance_error:
                    self.logger.error(f"   ❌ Error loading LLM instance {i}: {instance_error}")
            
            # Store primary instance as 'llm' for backward compatibility
            if 0 in self._llm_instances:
                self._preloaded_models['llm'] = self._llm_instances[0]
            
            # Update pool size to actual loaded count
            self._llm_pool_size = loaded_count
            
            if loaded_count > 0:
                self.logger.info(f"✅ LLM pool ready: {loaded_count}/{instances_to_load} instances loaded")
                return True
            else:
                self.logger.error(f"❌ Failed to load any LLM instances")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading LLM model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    async def _preload_tts_model(self, config: Dict[str, Any]) -> bool:
        """Preload TTS model (Edge TTS or XTTS)."""
        try:
            model_id = config.get('model_id', 'edge-tts')
            engine_type = config.get('engine_type', 'edge_tts')
            self.logger.info(f"Preloading TTS model: {model_id} ({engine_type})")
            
            # Get TTS engine
            tts_engine = self._model_manager.get_tts_engine(model_name=model_id)
            
            if tts_engine:
                self._preloaded_models['tts'] = tts_engine
                self.logger.info(f"✅ TTS model preloaded: {model_id} ({engine_type})")
                return True
            else:
                self.logger.error(f"Failed to preload TTS model: {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading TTS model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_whisper_model(self, language: Optional[str] = None) -> Optional[Any]:
        """
        Get persistent Whisper model instance.
        
        Args:
            language: Optional language code to request specific model capabilities
        
        Returns:
            Persistent Whisper model instance or None if not loaded
        """
        # Check if we have a preloaded model
        if 'whisper' in self._preloaded_models:
            model = self._preloaded_models['whisper']
            
            # If language is specified and NOT Arabic, check if current model is Arabic-only
            if language and language.lower() not in ('ar', 'arabic', 'auto'):
                # Check if the loaded model is the Genius Arabic one
                is_arabic_model = False
                if hasattr(model, '_get_engine_name'):
                    engine_name = model._get_engine_name()
                    if engine_name in ('whisper_genius_arabic', 'whisper_finetuned_arabic'):
                        is_arabic_model = True
                
                if is_arabic_model:
                    self.logger.info(f"Requested language '{language}' but preloaded model is Arabic-only. Fetching multilingual model via ModelManager.")
                    # Fallback to ModelManager to get a suitable model (e.g. turbo)
                    return self._model_manager.get_streaming_whisper(language=language)
            
            return model
        
        # Fallback to ModelManager if not preloaded
        self.logger.warning("Whisper model not preloaded, using ModelManager fallback")
        whisper_engine = self._model_manager.get_streaming_whisper(language=language)
        
        # Only cache if it's the default/Arabic one to avoid polluting preloaded cache with temp models
        if whisper_engine and (not language or language.lower() in ('ar', 'arabic', 'auto')):
            # Cache for subsequent callers so we do not reload per connection
            self._preloaded_models['whisper'] = whisper_engine
            self.logger.info("Cached Whisper engine obtained via fallback for reuse")
            
        return whisper_engine

    async def ensure_whisper_loaded(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ) -> bool:
        """Ensure the persistent Whisper model is loaded once (typically for WebRTC)."""
        if 'whisper' in self._preloaded_models:
            return True

        resolved_model_id = model_id or os.getenv('WEBRTC_WHISPER_MODEL_ID')
        resolved_device = device or os.getenv('WEBRTC_WHISPER_DEVICE', 'cuda')
        resolved_compute = compute_type or os.getenv('WEBRTC_WHISPER_COMPUTE', 'float16')

        with self._initialization_lock:
            if 'whisper' in self._preloaded_models:
                return True

            preload_config = {
                'model_id': resolved_model_id,
                'device': resolved_device,
                'compute_type': resolved_compute
            }

            self.logger.info(
                "Ensuring persistent Whisper model is loaded (model_id=%s, device=%s, compute=%s)",
                preload_config['model_id'],
                preload_config['device'],
                preload_config['compute_type']
            )

            success = await self._preload_whisper_model(preload_config)
            if success:
                self.logger.info("Persistent Whisper model ready for reuse")
            else:
                self.logger.warning("Failed to preload Whisper model via ensure_whisper_loaded")
            return success
    
    def get_llm_model(self, instance_id: Optional[int] = None) -> Optional[Any]:
        """
        Get persistent LLM model instance.
        
        Args:
            instance_id: Specific instance ID (0, 1, 2, ...) or None for round-robin selection
        
        Returns:
            Persistent LLM model instance or None if not loaded
        """
        # If specific instance requested
        if instance_id is not None:
            if instance_id in self._llm_instances:
                return self._llm_instances[instance_id]
            # Fallback to instance 0 if requested instance doesn't exist
            if 0 in self._llm_instances:
                self.logger.warning(f"LLM instance {instance_id} not available, using instance 0")
                return self._llm_instances[0]
        
        # Round-robin selection when multiple instances available
        if self._llm_instances:
            if len(self._llm_instances) > 1:
                # Round-robin: cycle through available instances
                instance_id = self._llm_instance_counter % len(self._llm_instances)
                self._llm_instance_counter += 1
                self.logger.debug(f"🔄 Round-robin LLM selection: instance {instance_id} (counter: {self._llm_instance_counter})")
                return self._llm_instances[instance_id]
            else:
                return self._llm_instances[0]
        
        # Legacy fallback: check 'llm' key in preloaded_models
        if 'llm' in self._preloaded_models:
            return self._preloaded_models['llm']
        
        # Fallback to ModelManager - try to load LLM from registry
        self.logger.warning("LLM model not preloaded, using ModelManager fallback")
        try:
            # Try to load an actual LLM model from voice config registry
            from ..config.voice_config_loader import get_voice_config
            voice_config = get_voice_config()
            
            # Get default LLM from registry or use fallback
            default_llm = voice_config._config.get("default_models", {}).get("llm", "qwen3-chat")
            llm_config = voice_config._config.get("models", {}).get(default_llm)
            
            if llm_config:
                engine_type = llm_config.get("engine_type", "llama.cpp")
                model_path = llm_config.get("model_path", llm_config.get("model_id"))
                
                if engine_type == "llama.cpp":
                    # Correct import path - llamacpp_engine is directly under inference_engines
                    from ..inference_engines.llamacpp_engine import LlamaCppEngine
                    from ..config.config_manager import ModelConfig
                    
                    config = ModelConfig(
                        name=default_llm,
                        model_id=model_path,
                        engine_type=engine_type
                    )
                    engine = LlamaCppEngine(config)
                    engine.load_model()
                    
                    # Store for reuse as instance 0
                    self._llm_instances[0] = engine
                    self._preloaded_models['llm'] = engine
                    self._preloaded_models['llm:0'] = engine
                    self._llm_pool_size = 1
                    self.logger.info(f"✅ LLM model loaded on-demand: {default_llm}")
                    return engine
        except Exception as e:
            self.logger.error(f"Failed to load LLM on-demand: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return None
    
    def get_llm_pool_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM pool.
        
        Returns:
            Dictionary with pool size, loaded instances, and selection counter
        """
        return {
            'configured_pool_size': int(os.getenv('LLM_POOL_SIZE', '1')),
            'actual_pool_size': self._llm_pool_size,
            'loaded_instances': list(self._llm_instances.keys()),
            'round_robin_counter': self._llm_instance_counter,
            'instances_info': {
                i: {
                    'name': f'llm:{i}',
                    'loaded': i in self._llm_instances,
                    'model_id': getattr(self._llm_instances.get(i), 'model_id', 'unknown') if i in self._llm_instances else None
                }
                for i in range(self._llm_pool_size)
            }
        }
    
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
        llm_ready = len(self._llm_instances) > 0 or 'llm' in self._preloaded_models
        return {
            'whisper': 'whisper' in self._preloaded_models,
            'llm': llm_ready,
            'llm_pool_size': self._llm_pool_size,
            'llm_instances_loaded': len(self._llm_instances),
            'tts': 'tts' in self._preloaded_models,
            'all_ready': all([
                'whisper' in self._preloaded_models,
                llm_ready,
                'tts' in self._preloaded_models
            ])
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
            
            # Cleanup LLM instances first
            for instance_id, model_instance in self._llm_instances.items():
                try:
                    if hasattr(model_instance, 'cleanup'):
                        model_instance.cleanup()
                    elif hasattr(model_instance, 'unload_model'):
                        model_instance.unload_model()
                    self.logger.info(f"Cleaned up LLM instance {instance_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up LLM instance {instance_id}: {e}")
            
            # Clear LLM instances registry
            self._llm_instances.clear()
            self._llm_pool_size = 0
            self._llm_instance_counter = 0
            
            # Cleanup each preloaded model
            for model_type, model_instance in self._preloaded_models.items():
                # Skip llm:X entries as they're already cleaned up above
                if model_type.startswith('llm:') or model_type == 'llm':
                    continue
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
            'llm_pool': {
                'configured_size': int(os.getenv('LLM_POOL_SIZE', '1')),
                'actual_size': self._llm_pool_size,
                'instances_loaded': list(self._llm_instances.keys()),
                'round_robin_counter': self._llm_instance_counter
            },
            'memory_thresholds': self._memory_thresholds,
            'configuration_loaded': self._preload_config is not None
        }
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a specific model is currently loaded.
        
        Args:
            model_id: Model identifier (e.g., 'stt', 'llm', 'tts', 'whisper')
            
        Returns:
            bool: True if model is loaded
        """
        # Normalize common aliases
        normalized_id = model_id.lower()
        alias_map = {
            'stt': 'whisper',
            'speech-to-text': 'whisper',
            'transcription': 'whisper',
        }
        normalized_id = alias_map.get(normalized_id, normalized_id)
        
        # Check direct match
        if normalized_id in self._preloaded_models:
            return True
        
        # Check LLM instances
        if normalized_id.startswith('llm'):
            if normalized_id == 'llm':
                return len(self._llm_instances) > 0
            # Check specific instance like 'llm:0'
            if ':' in normalized_id:
                try:
                    instance_num = int(normalized_id.split(':')[1])
                    return instance_num in self._llm_instances
                except (ValueError, IndexError):
                    pass
        
        return False
    
    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a specific model by ID.
        
        Args:
            model_id: Model identifier (e.g., 'stt', 'llm', 'tts', 'whisper')
            
        Returns:
            bool: True if model was unloaded successfully
        """
        try:
            # Normalize common aliases
            normalized_id = model_id.lower()
            alias_map = {
                'stt': 'whisper',
                'speech-to-text': 'whisper',
                'transcription': 'whisper',
            }
            actual_id = alias_map.get(normalized_id, normalized_id)
            
            self.logger.info(f"Unloading model: {model_id} (normalized: {actual_id})")
            
            # Handle LLM unloading (all instances)
            if actual_id == 'llm':
                for instance_id, model_instance in list(self._llm_instances.items()):
                    try:
                        if hasattr(model_instance, 'cleanup'):
                            model_instance.cleanup()
                        elif hasattr(model_instance, 'unload_model'):
                            model_instance.unload_model()
                        self.logger.info(f"  Unloaded LLM instance {instance_id}")
                    except Exception as e:
                        self.logger.error(f"  Error unloading LLM instance {instance_id}: {e}")
                
                # Clear LLM tracking
                self._llm_instances.clear()
                self._llm_pool_size = 0
                self._llm_instance_counter = 0
                
                # Remove from preloaded_models
                for key in list(self._preloaded_models.keys()):
                    if key == 'llm' or key.startswith('llm:'):
                        del self._preloaded_models[key]
                
                gc.collect()
                clear_gpu_memory()
                self.logger.info(f"✅ All LLM instances unloaded")
                return True
            
            # Handle specific LLM instance
            if actual_id.startswith('llm:'):
                try:
                    instance_num = int(actual_id.split(':')[1])
                    if instance_num in self._llm_instances:
                        model_instance = self._llm_instances[instance_num]
                        if hasattr(model_instance, 'cleanup'):
                            model_instance.cleanup()
                        elif hasattr(model_instance, 'unload_model'):
                            model_instance.unload_model()
                        
                        del self._llm_instances[instance_num]
                        if actual_id in self._preloaded_models:
                            del self._preloaded_models[actual_id]
                        
                        self._llm_pool_size = len(self._llm_instances)
                        gc.collect()
                        clear_gpu_memory()
                        self.logger.info(f"✅ LLM instance {instance_num} unloaded")
                        return True
                except (ValueError, IndexError):
                    pass
                return False
            
            # Handle other models (whisper, tts)
            if actual_id in self._preloaded_models:
                model_instance = self._preloaded_models[actual_id]
                
                try:
                    if hasattr(model_instance, 'cleanup'):
                        model_instance.cleanup()
                    elif hasattr(model_instance, 'unload_model'):
                        model_instance.unload_model()
                    elif hasattr(model_instance, 'model') and hasattr(model_instance.model, 'unload'):
                        model_instance.model.unload()
                except Exception as e:
                    self.logger.warning(f"Cleanup method failed for {actual_id}: {e}")
                
                del self._preloaded_models[actual_id]
                
                # Also remove aliases
                if actual_id == 'whisper':
                    self._preloaded_models.pop('stt', None)
                elif actual_id == 'stt':
                    self._preloaded_models.pop('whisper', None)
                
                gc.collect()
                clear_gpu_memory()
                self.logger.info(f"✅ Model {actual_id} unloaded")
                return True
            
            self.logger.warning(f"Model {model_id} not found in preloaded models")
            return False
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_loaded_models_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all loaded models.
        
        Returns:
            List of dictionaries with model info (id, type, device, memory_mb estimate)
        """
        models_info = []
        
        # Check Whisper/STT
        if 'whisper' in self._preloaded_models:
            model = self._preloaded_models['whisper']
            models_info.append({
                'id': 'stt',
                'name': 'Whisper STT',
                'type': 'speech-to-text',
                'device': getattr(model, 'device', 'cuda'),
                'model_id': getattr(model, 'model_id', 'whisper-byne-arabic'),
                'estimated_vram_mb': 3000,  # ~3GB for Whisper large
                'can_unload': True
            })
        
        # Check LLM instances
        for instance_id, model in self._llm_instances.items():
            config = getattr(model, 'config', None)
            models_info.append({
                'id': f'llm:{instance_id}' if instance_id > 0 else 'llm',
                'name': f'LLM Instance {instance_id}',
                'type': 'large-language-model',
                'device': 'cuda',
                'model_id': getattr(config, 'model_id', 'qwen3-unsloth') if config else 'unknown',
                'estimated_vram_mb': 8000,  # ~8GB for Q4 14B
                'can_unload': True
            })
        
        # Check TTS
        if 'tts' in self._preloaded_models:
            model = self._preloaded_models['tts']
            engine_type = 'edge_tts'
            device = 'cpu'
            vram = 0
            
            if hasattr(model, 'config'):
                engine_type = getattr(model.config, 'engine_type', 'edge_tts')
            
            # Check if GPU TTS
            if 'saudi' in str(type(model)).lower() or 'xtts' in str(type(model)).lower():
                device = 'cuda'
                vram = 4000
            
            models_info.append({
                'id': 'tts',
                'name': 'Text-to-Speech',
                'type': 'text-to-speech',
                'device': device,
                'model_id': getattr(model, 'model_id', engine_type),
                'estimated_vram_mb': vram,
                'can_unload': True
            })
        
        return models_info


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