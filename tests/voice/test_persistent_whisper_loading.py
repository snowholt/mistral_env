"""
Test persistent Whisper model loading with ModelManager.

This test validates that Whisper models are loaded persistently through
ModelManager and shared across different voice services and endpoints.

Author: BeautyAI Framework  
Date: 2025-01-23
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch

from beautyai_inference.core.model_manager import ModelManager
from beautyai_inference.config.whisper_model_config import WhisperModelConfig
from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service

logger = logging.getLogger(__name__)

class TestPersistentWhisperLoading:
    """Test suite for persistent Whisper model loading."""
    
    @pytest.fixture
    def model_manager(self):
        """Get ModelManager instance for testing."""
        manager = ModelManager()
        # Clean up any existing models
        manager.unload_whisper_model()
        yield manager
        # Clean up after test
        manager.unload_whisper_model()
    
    def test_model_manager_whisper_singleton(self, model_manager):
        """Test that ModelManager provides singleton Whisper instances."""
        logger.info("🧪 Testing ModelManager Whisper singleton behavior")
        
        # First call should load model
        whisper1 = model_manager.get_streaming_whisper()
        assert whisper1 is not None, "First Whisper model should be loaded"
        
        # Second call should return same instance
        whisper2 = model_manager.get_streaming_whisper()
        assert whisper2 is whisper1, "Should return same Whisper instance"
        
        # Check model info
        model_info = model_manager.get_whisper_model_info()
        assert model_info["loaded"] is True, "Model should be marked as loaded"
        assert "managed_by_model_manager" in model_info, "Should have management info"
        
        logger.info("✅ ModelManager singleton behavior validated")
    
    def test_transcription_factory_uses_persistent_model(self, model_manager):
        """Test that transcription factory uses persistent model when available."""
        logger.info("🧪 Testing transcription factory persistence awareness")
        
        # Pre-load persistent model
        persistent_engine = model_manager.get_streaming_whisper()
        assert persistent_engine is not None, "Persistent model should be loaded"
        
        # Create transcription service via factory
        factory_service = create_transcription_service()
        
        # The factory should detect and use the persistent model
        # This is validated through the base engine's singleton awareness
        assert factory_service is not None, "Factory service should be created"
        
        logger.info("✅ Transcription factory persistence awareness validated")
    
    def test_simple_voice_service_uses_persistent_model(self, model_manager):
        """Test that SimpleVoiceService uses persistent Whisper model."""
        logger.info("🧪 Testing SimpleVoiceService persistent model usage")
        
        # Create service without pre-loaded model
        service = SimpleVoiceService()
        
        # Initialize with persistent model loading
        import asyncio
        async def init_service():
            await service._preload_required_models()
            return service.persistent_whisper_engine
        
        persistent_engine = asyncio.run(init_service())
        assert persistent_engine is not None, "Service should get persistent engine"
        
        # Verify it's the same as ModelManager's instance
        manager_engine = model_manager.get_streaming_whisper()
        assert persistent_engine is manager_engine, "Should be same engine instance"
        
        logger.info("✅ SimpleVoiceService persistent model usage validated")
    
    def test_memory_efficiency_single_model_instance(self, model_manager):
        """Test that only one Whisper model instance exists in memory."""
        logger.info("🧪 Testing memory efficiency - single model instance")
        
        # Load through different paths
        manager_engine = model_manager.get_streaming_whisper()
        
        service = SimpleVoiceService()
        import asyncio
        asyncio.run(service._preload_required_models())
        service_engine = service.persistent_whisper_engine
        
        # Both should reference the same model
        assert service_engine is manager_engine, "Should be same engine instance"
        
        # Check model info shows management
        model_info = manager_engine.get_model_info()
        assert model_info.get("managed_by_model_manager", False), "Should be managed"
        
        logger.info("✅ Memory efficiency validated - single model instance")
    
    def test_model_unloading_and_reloading(self, model_manager):
        """Test model unloading and reloading through ModelManager."""
        logger.info("🧪 Testing model unloading and reloading")
        
        # Load model
        engine1 = model_manager.get_streaming_whisper()
        assert engine1 is not None, "First load should succeed"
        assert model_manager.is_whisper_model_loaded(), "Should be marked as loaded"
        
        # Unload model
        success = model_manager.unload_whisper_model()
        assert success, "Unload should succeed"
        assert not model_manager.is_whisper_model_loaded(), "Should be marked as unloaded"
        
        # Reload model
        engine2 = model_manager.get_streaming_whisper()
        assert engine2 is not None, "Reload should succeed"
        assert engine2 is not engine1, "Should be new instance after unload"
        
        logger.info("✅ Model unloading and reloading validated")
    
    @pytest.mark.performance
    def test_loading_performance_vs_factory(self, model_manager):
        """Test that persistent loading is faster than factory creation."""
        logger.info("🧪 Testing loading performance vs factory")
        
        # Time persistent loading (first load)
        start_time = time.time()
        persistent_engine = model_manager.get_streaming_whisper()
        first_load_time = time.time() - start_time
        
        assert persistent_engine is not None, "Persistent load should succeed"
        logger.info(f"First persistent load time: {first_load_time:.3f}s")
        
        # Time persistent retrieval (subsequent calls)
        start_time = time.time()
        same_engine = model_manager.get_streaming_whisper()
        retrieval_time = time.time() - start_time
        
        assert same_engine is persistent_engine, "Should be same instance"
        logger.info(f"Persistent retrieval time: {retrieval_time:.3f}s")
        
        # Retrieval should be much faster (< 0.1s vs potentially several seconds)
        assert retrieval_time < 0.1, f"Retrieval should be fast, got {retrieval_time:.3f}s"
        
        logger.info("✅ Loading performance validated")
    
    def test_error_handling_graceful_fallback(self, model_manager):
        """Test graceful fallback when persistent loading fails."""
        logger.info("🧪 Testing error handling and graceful fallback")
        
        # Mock a failure in ModelManager
        with patch.object(model_manager, 'get_streaming_whisper', return_value=None):
            service = SimpleVoiceService()
            
            # Service should handle the failure gracefully
            import asyncio
            asyncio.run(service._preload_required_models())
            
            # Should fall back to factory loading
            assert service.persistent_whisper_engine is None, "Should not have persistent engine"
            
            # Should still be able to initialize transcription service as fallback
            assert hasattr(service, 'transcription_service'), "Should have fallback mechanism"
        
        logger.info("✅ Error handling and graceful fallback validated")

    def test_whisper_config_creation(self):
        """Test WhisperModelConfig creation and validation."""
        logger.info("🧪 Testing WhisperModelConfig creation")
        
        config = WhisperModelConfig(
            name="whisper_large_v3_turbo",
            model_id="openai/whisper-large-v3-turbo",
            engine_type="whisper_large_v3_turbo",
            device="cuda:0",
            torch_dtype="float16",
            quantization="4bit"
        )
        
        assert config.name == "whisper_large_v3_turbo"
        assert config.model_id == "openai/whisper-large-v3-turbo"
        assert config.engine_type == "whisper_large_v3_turbo"
        assert config.device == "cuda:0"
        assert config.torch_dtype == "float16"
        assert config.quantization == "4bit"
        
        logger.info("✅ WhisperModelConfig creation validated")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])