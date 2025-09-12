"""
Voice Service Manager for 24/7 Persistent Model Loading

This manager ensures all voice models (STT, TTS, LLM) are pre-loaded and 
kept warm on GPU for instant response in production environments.

Author: BeautyAI Framework
Date: 2025-01-31
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceServiceManager:
    """
    Enhanced manager for persistent voice service with 24/7 model availability.
    
    This service integrates with PersistentModelManager for optimal performance:
    - STT (Whisper): Pre-loaded via PersistentModelManager for instant transcription
    - LLM (Chat): Pre-loaded via PersistentModelManager for instant responses
    - TTS (Edge): Always available, no pre-loading needed
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.simple_voice_service = None
        self.persistent_model_manager = None
        self.is_initialized = False
        self.initialization_time = None
        self.health_stats = {
            "uptime_seconds": 0,
            "models_loaded": {},
            "last_health_check": None
        }
        
    async def initialize_persistent_service(self) -> bool:
        """
        Initialize all voice models for persistent 24/7 operation using PersistentModelManager.
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            self.logger.info("🚀 Initializing BeautyAI Voice Service Manager with PersistentModelManager")
            start_time = time.time()
            
            # Initialize PersistentModelManager first
            from beautyai_inference.core.persistent_model_manager import PersistentModelManager
            self.persistent_model_manager = PersistentModelManager.get_instance()
            
            # Preload models if not already loaded
            preload_success = await self.persistent_model_manager.preload_models()
            if not preload_success:
                self.logger.warning("⚠️ Some models failed to preload, continuing with fallback")
            
            # Initialize the SimpleVoiceService with persistent models
            from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
            self.simple_voice_service = SimpleVoiceService()
            
            # Initialize with pre-loading (this loads STT, LLM, and tests TTS)
            await self.simple_voice_service.initialize()
            
            self.initialization_time = time.time() - start_time
            self.is_initialized = True
            
            # Perform initial health check
            health_status = await self.health_check()
            
            if health_status["all_models_ready"]:
                self.logger.info(f"✅ Voice Service Manager initialized successfully in {self.initialization_time:.2f}s")
                self.logger.info(f"📊 Models loaded: {health_status['models_status']}")
                return True
            else:
                self.logger.error("❌ Some models failed to load during initialization")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Voice Service Manager: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all voice models including PersistentModelManager.
        
        Returns:
            Dict containing health status of all components
        """
        health_status = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - (self.initialization_time or time.time()),
            "all_models_ready": False,
            "models_status": {},
            "persistent_models_status": {},
            "performance_metrics": {}
        }
        
        try:
            if not self.is_initialized:
                health_status["error"] = "Service not initialized"
                return health_status
            
            # Check PersistentModelManager status
            if self.persistent_model_manager:
                persistent_status = await self.persistent_model_manager.get_model_status()
                health_status["persistent_models_status"] = persistent_status
                
                # Check individual persistent models
                whisper_model = await self.persistent_model_manager.get_whisper_model()
                llm_model = await self.persistent_model_manager.get_llm_model()
                
                health_status["models_status"]["STT_Persistent"] = "loaded" if whisper_model else "not_loaded"
                health_status["models_status"]["LLM_Persistent"] = "loaded" if llm_model else "not_loaded"
            else:
                health_status["persistent_models_status"] = {"error": "PersistentModelManager not initialized"}
            
            # Check legacy service models for fallback
            if self.simple_voice_service:
                # Check STT Model
                if hasattr(self.simple_voice_service, 'transcription_service') and self.simple_voice_service.transcription_service:
                    health_status["models_status"]["STT_Legacy"] = "loaded"
                else:
                    health_status["models_status"]["STT_Legacy"] = "not_loaded"
                
                # Check LLM Model
                if hasattr(self.simple_voice_service, 'chat_service') and self.simple_voice_service.chat_service:
                    if self.simple_voice_service.chat_service._default_model_loaded:
                        health_status["models_status"]["LLM_Legacy"] = f"loaded ({self.simple_voice_service.chat_service._default_model_name})"
                    else:
                        # Check if any model is loaded
                        loaded_models = list(self.simple_voice_service.chat_service.model_manager._loaded_models.keys())
                        if loaded_models:
                            health_status["models_status"]["LLM_Legacy"] = f"loaded ({loaded_models[0]})"
                        else:
                            health_status["models_status"]["LLM_Legacy"] = "not_loaded"
                else:
                    health_status["models_status"]["LLM_Legacy"] = "not_loaded"
            
            # Check TTS (Edge TTS is always available)
            health_status["models_status"]["TTS"] = "ready (Edge TTS)"
            
            # Overall health - check if either persistent or legacy models are available
            stt_ready = (health_status["models_status"].get("STT_Persistent") == "loaded" or 
                        health_status["models_status"].get("STT_Legacy") == "loaded")
            llm_ready = (health_status["models_status"].get("LLM_Persistent") == "loaded" or 
                        "loaded" in health_status["models_status"].get("LLM_Legacy", ""))
            tts_ready = health_status["models_status"].get("TTS") == "ready (Edge TTS)"
            
            health_status["all_models_ready"] = stt_ready and llm_ready and tts_ready
            
            # Add memory monitoring if available
            if self.persistent_model_manager:
                memory_stats = await self.persistent_model_manager.get_memory_stats()
                health_status["memory_stats"] = memory_stats
            
            self.health_stats = health_status
            self.logger.info(f"🔍 Health check completed: {health_status['all_models_ready']}")
            
        except Exception as e:
            health_status["error"] = str(e)
            self.logger.error(f"❌ Health check failed: {e}")
        
        return health_status
    
    async def process_voice_request(self, audio_data: bytes, language: str = None, gender: str = "female") -> Dict[str, Any]:
        """
        Process voice request using persistent models.
        
        Args:
            audio_data: Raw audio bytes
            language: Target language (ar/en) or None for auto-detect
            gender: Voice gender preference
            
        Returns:
            Dict with processing results
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Service not initialized. Call initialize_persistent_service() first."
            }
        
        try:
            start_time = time.time()
            
            # Use the persistent service for processing
            result = await self.simple_voice_service.process_voice_message(
                audio_data=audio_data,
                language=language,
                gender=gender
            )
            
            # Add service manager metrics
            result["service_manager_metrics"] = {
                "total_processing_time": time.time() - start_time,
                "service_uptime": time.time() - (self.initialization_time or time.time()),
                "models_persistent": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing voice request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics including PersistentModelManager data."""
        stats = {
            "service_info": {
                "initialized": self.is_initialized,
                "initialization_time_seconds": self.initialization_time,
                "uptime_seconds": time.time() - (self.initialization_time or time.time()) if self.initialization_time else 0
            },
            "latest_health_check": self.health_stats,
            "persistent_model_stats": await self.get_persistent_model_stats(),
            "performance_profile": {
                "persistent_models": True,
                "gpu_optimized": True,
                "production_ready": self.is_initialized and self.health_stats.get("all_models_ready", False),
                "uses_persistent_model_manager": self.persistent_model_manager is not None
            }
        }
        
        return stats
    
    async def restart_failed_models(self) -> bool:
        """Restart any failed models while keeping the service running, using PersistentModelManager."""
        try:
            self.logger.info("🔄 Attempting to restart failed models...")
            
            health = await self.health_check()
            restart_needed = False
            
            # Check and restart persistent models if needed
            if self.persistent_model_manager:
                persistent_status = health.get("persistent_models_status", {})
                
                # Restart Whisper if needed
                if health["models_status"].get("STT_Persistent") in ["not_loaded", "error"]:
                    self.logger.info("🔄 Restarting Whisper model via PersistentModelManager...")
                    whisper_success = await self.persistent_model_manager.reload_whisper_model()
                    if whisper_success:
                        self.logger.info("✅ Whisper model restarted successfully")
                    else:
                        self.logger.warning("⚠️ Failed to restart Whisper model")
                    restart_needed = True
                
                # Restart LLM if needed
                if health["models_status"].get("LLM_Persistent") in ["not_loaded", "error"]:
                    self.logger.info("🔄 Restarting LLM model via PersistentModelManager...")
                    llm_success = await self.persistent_model_manager.reload_llm_model()
                    if llm_success:
                        self.logger.info("✅ LLM model restarted successfully")
                    else:
                        self.logger.warning("⚠️ Failed to restart LLM model")
                    restart_needed = True
            
            # Restart legacy models if persistent models failed
            if health["models_status"].get("STT_Legacy") in ["not_loaded", "error"]:
                self.logger.info("🔄 Restarting STT legacy model...")
                if hasattr(self.simple_voice_service, 'transcription_service'):
                    self.simple_voice_service.transcription_service = None
                restart_needed = True
            
            if health["models_status"].get("LLM_Legacy") in ["not_loaded", "error"]:
                self.logger.info("🔄 Restarting LLM legacy model...")
                if hasattr(self.simple_voice_service, 'chat_service') and self.simple_voice_service.chat_service:
                    success = await self.simple_voice_service.chat_service.load_default_model_from_config()
                    if not success:
                        self.logger.warning("Failed to restart default model, service will use fallbacks")
                restart_needed = True
            
            # Re-check health if any restarts were needed
            if restart_needed:
                new_health = await self.health_check()
                return new_health["all_models_ready"]
            else:
                self.logger.info("✅ No model restarts needed")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error restarting models: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the service manager including PersistentModelManager."""
        try:
            self.logger.info("🛑 Shutting down Voice Service Manager...")
            
            # Shutdown simple voice service
            if self.simple_voice_service:
                await self.simple_voice_service.cleanup()
            
            # Cleanup PersistentModelManager
            if self.persistent_model_manager:
                await self.persistent_model_manager.cleanup()
                self.logger.info("✅ PersistentModelManager cleanup complete")
            
            self.is_initialized = False
            self.logger.info("✅ Voice Service Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    async def get_persistent_model_stats(self) -> Dict[str, Any]:
        """Get statistics from PersistentModelManager."""
        if not self.persistent_model_manager:
            return {"error": "PersistentModelManager not initialized"}
        
        try:
            return {
                "model_status": await self.persistent_model_manager.get_model_status(),
                "memory_stats": await self.persistent_model_manager.get_memory_stats(),
                "preload_config": self.persistent_model_manager.preload_config,
                "models_ready": await self.persistent_model_manager.are_models_ready()
            }
        except Exception as e:
            self.logger.error(f"Error getting persistent model stats: {e}")
            return {"error": str(e)}


# Global service manager instance for production use
_voice_service_manager = None


async def get_voice_service_manager() -> VoiceServiceManager:
    """Get or create the global voice service manager instance."""
    global _voice_service_manager
    
    if _voice_service_manager is None:
        _voice_service_manager = VoiceServiceManager()
        await _voice_service_manager.initialize_persistent_service()
    
    return _voice_service_manager


async def init_production_voice_service() -> bool:
    """Initialize the production voice service for 24/7 operation."""
    manager = await get_voice_service_manager()
    return manager.is_initialized
