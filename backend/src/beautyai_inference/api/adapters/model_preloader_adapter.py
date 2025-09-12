"""
Model Preloader API Adapter.

Provides FastAPI dependencies and utilities for accessing preloaded models
in the BeautyAI inference framework. This adapter enables efficient access
to persistent models for voice and chat services.

Author: BeautyAI Framework
Date: 2024-09-11
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Depends, HTTPException, status

from .base_adapter import APIServiceAdapter
from ..models import APIResponse, ErrorResponse
from ..errors import APIError, ValidationError
from ...core.persistent_model_manager import (
    PersistentModelManager,
    get_persistent_model_manager,
    initialize_persistent_models
)

logger = logging.getLogger(__name__)


class ModelPreloaderAdapter:
    """
    FastAPI adapter for accessing preloaded models.
    
    This adapter provides FastAPI dependencies and utilities for:
    - Accessing persistent Whisper models
    - Accessing persistent LLM models  
    - Accessing persistent TTS engines
    - Checking model readiness and health
    - Managing model initialization lifecycle
    """
    
    def __init__(self):
        """Initialize the model preloader adapter."""
        self.logger = logging.getLogger(__name__)
        self._initialization_lock = asyncio.Lock()
        self._initialization_attempted = False
    
    async def get_persistent_manager(self) -> PersistentModelManager:
        """
        Get the persistent model manager instance.
        
        Returns:
            PersistentModelManager: Singleton instance
        """
        return get_persistent_model_manager()
    
    async def ensure_models_initialized(self, manager: Optional[PersistentModelManager] = None) -> bool:
        """
        Ensure models are initialized.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            bool: True if models are initialized
            
        Raises:
            HTTPException: If model initialization fails
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        async with self._initialization_lock:
            if not manager.is_initialized() and not self._initialization_attempted:
                self.logger.info("Initializing persistent models on first request...")
                self._initialization_attempted = True
                
                success = await manager.preload_models()
                if not success:
                    self.logger.error("Failed to initialize persistent models")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Model initialization failed. Service temporarily unavailable."
                    )
        
        if not manager.is_initialized():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not initialized. Service temporarily unavailable."
            )
        
        return True
    
    async def get_preloaded_whisper(self, manager: Optional[PersistentModelManager] = None) -> Any:
        """
        Get preloaded Whisper model.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            Whisper model instance
            
        Raises:
            HTTPException: If Whisper model is not available
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        await self.ensure_models_initialized(manager)
        
        whisper_model = manager.get_whisper_model()
        if whisper_model is None:
            self.logger.error("Whisper model not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Whisper model not available. Please try again later."
            )
        
        return whisper_model
    
    async def get_preloaded_llm(self, manager: Optional[PersistentModelManager] = None) -> Any:
        """
        Get preloaded LLM model.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            LLM model instance
            
        Raises:
            HTTPException: If LLM model is not available
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        await self.ensure_models_initialized(manager)
        
        llm_model = manager.get_llm_model()
        if llm_model is None:
            self.logger.error("LLM model not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM model not available. Please try again later."
            )
        
        return llm_model
    
    async def get_preloaded_tts(self, manager: Optional[PersistentModelManager] = None) -> Any:
        """
        Get preloaded TTS engine.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            TTS engine instance
            
        Raises:
            HTTPException: If TTS engine is not available
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        await self.ensure_models_initialized(manager)
        
        tts_engine = manager.get_tts_engine()
        if tts_engine is None:
            self.logger.error("TTS engine not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS engine not available. Please try again later."
            )
        
        return tts_engine
    
    async def check_models_ready(self, manager: Optional[PersistentModelManager] = None) -> Dict[str, bool]:
        """
        Check model readiness status.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            Dictionary with model readiness status
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        return manager.check_models_ready()
    
    async def get_model_health_status(self, manager: Optional[PersistentModelManager] = None) -> Dict[str, Any]:
        """
        Get comprehensive model health status for monitoring.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            Dictionary with comprehensive health information
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        try:
            # Get basic readiness status
            readiness = manager.check_models_ready()
            
            # Get memory monitoring data
            memory_info = await manager.monitor_memory()
            
            # Get initialization stats
            init_stats = manager.get_initialization_stats()
            
            # Combine all health information
            health_status = {
                'timestamp': time.time(),
                'service_status': 'healthy' if readiness['all_ready'] else 'degraded',
                'models_ready': readiness,
                'memory_monitoring': memory_info,
                'initialization_stats': init_stats,
                'api_adapter_info': {
                    'initialization_attempted': self._initialization_attempted,
                    'dependencies_available': True
                }
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error getting model health status: {e}")
            return {
                'timestamp': time.time(),
                'service_status': 'error',
                'error': str(e),
                'models_ready': {'all_ready': False},
                'api_adapter_info': {
                    'initialization_attempted': self._initialization_attempted,
                    'error_occurred': True
                }
            }
    
    async def warmup_models(self, manager: Optional[PersistentModelManager] = None) -> Dict[str, Any]:
        """
        Perform model warmup operations for optimal performance.
        
        Args:
            manager: Persistent model manager instance (optional)
            
        Returns:
            Dictionary with warmup results
        """
        if manager is None:
            manager = await self.get_persistent_manager()
            
        try:
            warmup_start = time.time()
            results = {}
            
            # Warm up Whisper model with dummy audio
            whisper_model = manager.get_whisper_model()
            if whisper_model:
                try:
                    # Create minimal dummy audio for warmup
                    dummy_audio = b'\x00' * 1024  # 1KB of silence
                    # Note: This might fail with dummy data, but it warms up the model
                    warmup_whisper_start = time.time()
                    try:
                        whisper_model.transcribe_audio_bytes(dummy_audio, audio_format="wav")
                    except:
                        pass  # Expected to fail with dummy data
                    results['whisper_warmup_ms'] = (time.time() - warmup_whisper_start) * 1000
                except Exception as e:
                    results['whisper_warmup_error'] = str(e)
            
            # Warm up LLM model with dummy prompt
            llm_model = manager.get_llm_model()
            if llm_model:
                try:
                    warmup_llm_start = time.time()
                    # Generate a minimal response to warm up the model
                    if hasattr(llm_model, 'generate'):
                        try:
                            llm_model.generate("Hello", max_new_tokens=1)
                        except:
                            pass  # Expected to potentially fail
                    results['llm_warmup_ms'] = (time.time() - warmup_llm_start) * 1000
                except Exception as e:
                    results['llm_warmup_error'] = str(e)
            
            # TTS doesn't need warmup (Edge TTS is always ready)
            results['tts_warmup_ms'] = 0.1  # Minimal time
            
            total_warmup_time = (time.time() - warmup_start) * 1000
            
            return {
                'success': True,
                'total_warmup_time_ms': total_warmup_time,
                'individual_results': results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error during model warmup: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


# Global adapter instance
_model_preloader_adapter = ModelPreloaderAdapter()


# Dependency functions for FastAPI
async def get_preloaded_whisper() -> Any:
    """FastAPI dependency to get preloaded Whisper model."""
    manager = get_persistent_model_manager()
    await _model_preloader_adapter.ensure_models_initialized(manager)
    return manager.get_whisper_model()


async def get_preloaded_llm() -> Any:
    """FastAPI dependency to get preloaded LLM model."""
    manager = get_persistent_model_manager()
    await _model_preloader_adapter.ensure_models_initialized(manager)
    return manager.get_llm_model()


async def get_preloaded_tts() -> Any:
    """FastAPI dependency to get preloaded TTS engine."""
    manager = get_persistent_model_manager()
    await _model_preloader_adapter.ensure_models_initialized(manager)
    return manager.get_tts_engine()


async def check_models_ready() -> Dict[str, bool]:
    """FastAPI dependency to check model readiness."""
    manager = get_persistent_model_manager()
    return manager.check_models_ready()


async def ensure_models_initialized() -> bool:
    """FastAPI dependency to ensure models are initialized."""
    manager = get_persistent_model_manager()
    return await _model_preloader_adapter.ensure_models_initialized(manager)


async def get_model_health_status() -> Dict[str, Any]:
    """FastAPI dependency to get comprehensive model health status."""
    manager = get_persistent_model_manager()
    return await _model_preloader_adapter.get_model_health_status(manager)


async def warmup_models() -> Dict[str, Any]:
    """FastAPI dependency to perform model warmup operations."""
    manager = get_persistent_model_manager()
    return await _model_preloader_adapter.warmup_models(manager)


# Startup and shutdown events
async def startup_initialize_models():
    """Initialize models on FastAPI startup."""
    try:
        logger.info("🚀 Initializing persistent models on API startup...")
        success = await initialize_persistent_models()
        if success:
            logger.info("✅ Persistent models initialized successfully on startup")
        else:
            logger.warning("⚠️ Some models failed to initialize on startup, will initialize on first request")
    except Exception as e:
        logger.error(f"❌ Error initializing models on startup: {e}")


async def shutdown_cleanup_models():
    """Cleanup models on FastAPI shutdown."""
    try:
        logger.info("🛑 Cleaning up persistent models on API shutdown...")
        from ...core.persistent_model_manager import cleanup_persistent_models
        success = await cleanup_persistent_models()
        if success:
            logger.info("✅ Persistent models cleanup completed")
        else:
            logger.warning("⚠️ Some issues occurred during model cleanup")
    except Exception as e:
        logger.error(f"❌ Error cleaning up models on shutdown: {e}")