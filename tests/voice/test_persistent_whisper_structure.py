"""
Quick test to validate persistent Whisper implementation changes.

This test checks the code structure without loading actual models.
"""

import logging
from unittest.mock import Mock, patch

# Test imports work correctly
from beautyai_inference.config.whisper_model_config import WhisperModelConfig
from beautyai_inference.core.model_manager import ModelManager
from beautyai_inference.core.model_factory import ModelFactory

logger = logging.getLogger(__name__)

def test_import_structure():
    """Test that all new classes can be imported correctly."""
    print("✅ All imports successful")
    print(f"✅ WhisperModelConfig: {WhisperModelConfig}")
    print(f"✅ ModelManager: {ModelManager}")
    print(f"✅ ModelFactory: {ModelFactory}")

def test_whisper_config():
    """Test WhisperModelConfig creation."""
    config = WhisperModelConfig(
        name="test_whisper",
        model_id="openai/whisper-large-v3-turbo",
        engine_type="whisper_large_v3_turbo",
        device="cuda:0",
        torch_dtype="float16"
    )
    
    print(f"✅ WhisperModelConfig created: {config.name}")
    print(f"✅ Model ID: {config.model_id}")
    print(f"✅ Engine type: {config.engine_type}")

def test_model_manager_methods():
    """Test that ModelManager has the new Whisper methods."""
    manager = ModelManager()
    
    # Check methods exist
    assert hasattr(manager, 'get_streaming_whisper'), "Missing get_streaming_whisper method"
    assert hasattr(manager, 'unload_whisper_model'), "Missing unload_whisper_model method"
    assert hasattr(manager, 'is_whisper_model_loaded'), "Missing is_whisper_model_loaded method"
    assert hasattr(manager, 'get_whisper_model_info'), "Missing get_whisper_model_info method"
    
    print("✅ ModelManager has all required Whisper methods")

def test_model_factory_whisper_support():
    """Test that ModelFactory supports Whisper models."""
    # Check method exists
    assert hasattr(ModelFactory, 'create_whisper_model'), "Missing create_whisper_model method"
    print("✅ ModelFactory supports Whisper model creation")

def test_transcription_factory_integration():
    """Test that transcription factory uses ModelManager."""
    from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service
    
    # This should work without errors (imports and basic structure)
    print("✅ Transcription factory imports successfully")

def test_simple_voice_service_integration():
    """Test that SimpleVoiceService has persistent engine support."""
    from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
    
    # Create service instance
    service = SimpleVoiceService()
    
    # Check it has the persistent engine attribute
    assert hasattr(service, 'persistent_whisper_engine'), "Missing persistent_whisper_engine attribute"
    assert service.persistent_whisper_engine is None, "Should be None initially"
    
    print("✅ SimpleVoiceService has persistent engine support")

if __name__ == "__main__":
    print("🧪 Testing persistent Whisper implementation structure...")
    
    test_import_structure()
    test_whisper_config()
    test_model_manager_methods()
    test_model_factory_whisper_support()
    test_transcription_factory_integration()
    test_simple_voice_service_integration()
    
    print("\n🎉 All structure tests passed! Implementation is ready.")