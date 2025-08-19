#!/usr/bin/env python3
"""
Whisper Engines Validation Script

Tests the new specialized Whisper engines to ensure they load properly
and can be instantiated through the factory.
"""

import sys
import os
import logging

# Add the backend src to path
sys.path.insert(0, '/home/lumi/beautyai/backend/src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_engine_imports():
    """Test that all engines can be imported successfully."""
    try:
        from beautyai_inference.services.voice.transcription import (
            WhisperLargeV3Engine,
            WhisperLargeV3TurboEngine,
            WhisperArabicTurboEngine,
            create_transcription_service,
            get_available_engines,
            validate_engine_availability
        )
        logger.info("✅ All engine imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_engine_instantiation():
    """Test that engines can be instantiated without model loading."""
    engines = {
        "WhisperLargeV3Engine": None,
        "WhisperLargeV3TurboEngine": None, 
        "WhisperArabicTurboEngine": None
    }
    
    try:
        from beautyai_inference.services.voice.transcription import (
            WhisperLargeV3Engine,
            WhisperLargeV3TurboEngine,
            WhisperArabicTurboEngine
        )
        
        engines["WhisperLargeV3Engine"] = WhisperLargeV3Engine()
        logger.info("✅ WhisperLargeV3Engine instantiated")
        
        engines["WhisperLargeV3TurboEngine"] = WhisperLargeV3TurboEngine()
        logger.info("✅ WhisperLargeV3TurboEngine instantiated")
        
        engines["WhisperArabicTurboEngine"] = WhisperArabicTurboEngine()
        logger.info("✅ WhisperArabicTurboEngine instantiated")
        
        # Test basic methods
        for name, engine in engines.items():
            if engine:
                assert hasattr(engine, 'load_whisper_model'), f"{name} missing load_whisper_model"
                assert hasattr(engine, 'transcribe_audio_bytes'), f"{name} missing transcribe_audio_bytes"
                assert hasattr(engine, 'is_model_loaded'), f"{name} missing is_model_loaded"
                assert hasattr(engine, 'get_model_info'), f"{name} missing get_model_info"
                assert hasattr(engine, 'cleanup'), f"{name} missing cleanup"
                
                # Test method calls without model loaded
                assert engine.is_model_loaded() == False, f"{name} should not have model loaded initially"
                info = engine.get_model_info()
                assert info["loaded"] == False, f"{name} model_info should show loaded=False"
                
                logger.info(f"✅ {name} interface validation passed")
        
        # Cleanup
        for engine in engines.values():
            if engine:
                engine.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine instantiation failed: {e}")
        return False

def test_factory_function():
    """Test the factory function and engine selection."""
    try:
        from beautyai_inference.services.voice.transcription import (
            create_transcription_service,
            get_available_engines,
            validate_engine_availability
        )
        
        # Test available engines function
        available = get_available_engines()
        logger.info(f"Available engines: {list(available.keys())}")
        assert len(available) == 3, f"Expected 3 engines, got {len(available)}"
        
        # Test engine availability validation
        availability = validate_engine_availability()
        logger.info(f"Engine availability: {availability}")
        
        # Test factory creation (should use default from registry)
        service = create_transcription_service()
        logger.info(f"✅ Factory created service: {type(service).__name__}")
        
        # Test service interface
        assert hasattr(service, 'load_whisper_model'), "Service missing load_whisper_model"
        assert hasattr(service, 'transcribe_audio_bytes'), "Service missing transcribe_audio_bytes"
        assert service.is_model_loaded() == False, "Service should not have model loaded initially"
        
        info = service.get_model_info()
        logger.info(f"Service info: {info}")
        
        # Cleanup
        service.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory function test failed: {e}")
        return False

def test_voice_config_integration():
    """Test integration with voice configuration."""
    try:
        from beautyai_inference.config.voice_config_loader import get_voice_config
        
        voice_config = get_voice_config()
        logger.info(f"✅ Voice config loaded successfully")
        
        # Test STT config
        stt_config = voice_config.get_stt_model_config()
        logger.info(f"STT config - Model: {stt_config.model_id}, Engine: {stt_config.engine_type}")
        
        # Test model registry
        models = voice_config._config.get("models", {})
        logger.info(f"Available models in registry: {list(models.keys())}")
        
        expected_models = ["whisper-large-v3", "whisper-large-v3-turbo", "whisper-arabic-turbo"]
        for model in expected_models:
            if model in models:
                logger.info(f"✅ Found model in registry: {model}")
            else:
                logger.warning(f"⚠️ Model not found in registry: {model}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice config integration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("🚀 Starting Whisper Engines Validation")
    
    tests = [
        ("Engine Imports", test_engine_imports),
        ("Engine Instantiation", test_engine_instantiation),
        ("Factory Function", test_factory_function),
        ("Voice Config Integration", test_voice_config_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Testing: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n📊 VALIDATION SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:25} {status}")
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Engines are ready!")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED - Check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())