#!/usr/bin/env python3
"""
Test script to verify OuteTTS functionality.
Tests the new OuteTTS engine implementation in the BeautyAI framework.
"""

import sys
import os
import logging
import json

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_oute_tts():
    """Test OuteTTS functionality."""
    print("🎙️ Testing OuteTTS Engine")
    print("="*60)
    
    try:
        from beautyai_inference.config.config_manager import ModelConfig
        from beautyai_inference.inference_engines.oute_tts_engine import OuteTTSEngine
        
        # Create model config for OuteTTS
        model_config = ModelConfig(
            name="oute-tts-1b",
            model_id="OuteAI/Llama-OuteTTS-1.0-1B-GGUF",
            engine_type="oute_tts",
            quantization="Q4_K_M",
            dtype="float16",
            max_new_tokens=None,
            model_filename="model.gguf"
        )
        
        # Initialize engine
        print("📥 Initializing OuteTTS engine...")
        engine = OuteTTSEngine(model_config)
        print(f"✅ Engine initialized. Model loaded: {engine.model_loaded}")
        
        # Load model
        print("📥 Loading OuteTTS model...")
        engine.load_model()
        print("✅ Model loaded successfully")
        
        # Test text-to-speech with different languages and speakers
        test_cases = [
            ("Hello, this is a test using OuteTTS neural speech synthesis.", "en", "female"),
            ("Hello, this is a test with a male voice using OuteTTS.", "en", "male"),
            ("مرحبا، هذا اختبار باستخدام محرك OuteTTS للتحويل النصي إلى صوت.", "ar", "female"),
            ("Bonjour, ceci est un test avec OuteTTS pour la synthèse vocale.", "fr", "female"),
            ("Hola, esta es una prueba con OuteTTS para síntesis de voz.", "es", "female"),
        ]
        
        created_files = []
        
        for i, (text, language, gender) in enumerate(test_cases):
            print(f"\n🔊 Test {i+1}: Generating OuteTTS for {language} ({gender})")
            print(f"Text: '{text}'")
            
            try:
                # Use a specific output path
                output_path = f"/home/lumi/beautyai/voice_tests/oute_tts_{language}_{gender}_{i+1}.wav"
                
                result_path = engine.text_to_speech(
                    text=text,
                    language=language,
                    output_path=output_path,
                    speaker_voice=gender,
                    emotion="neutral",
                    speed=1.0
                )
                
                if result_path and os.path.exists(result_path):
                    file_size = os.path.getsize(result_path)
                    print(f"✅ OuteTTS successful: {result_path}")
                    print(f"📁 File size: {file_size} bytes")
                    created_files.append(result_path)
                else:
                    print(f"❌ OuteTTS failed: No output file created")
                    
            except Exception as e:
                print(f"❌ OuteTTS failed: {e}")
        
        # Test text_to_speech_bytes method
        print(f"\n🔊 Testing text_to_speech_bytes method...")
        try:
            audio_bytes = engine.text_to_speech_bytes(
                "Testing bytes output from OuteTTS engine.",
                language="en",
                speaker_voice="female"
            )
            print(f"✅ Bytes method successful: {len(audio_bytes)} bytes returned")
        except Exception as e:
            print(f"❌ Bytes method failed: {e}")
        
        # Test benchmark
        print(f"\n🚀 Running benchmark...")
        try:
            benchmark_result = engine.benchmark("This is a benchmark test for OuteTTS performance and quality.")
            print(f"✅ Benchmark completed:")
            print(f"   Generation time: {benchmark_result['generation_time']:.2f}s")
            print(f"   Characters/second: {benchmark_result['characters_per_second']:.1f}")
            print(f"   Engine: {benchmark_result['engine']}")
            print(f"   Success: {benchmark_result['success']}")
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
        
        # Test available speakers
        print(f"\n🎤 Testing available speakers...")
        try:
            speakers = engine.get_available_speakers("en")
            print(f"✅ Available speakers for English: {len(speakers)}")
            for speaker in speakers[:5]:  # Show first 5
                print(f"   - {speaker}")
        except Exception as e:
            print(f"❌ Get speakers failed: {e}")
        
        # Test supported languages
        print(f"\n🌍 Testing supported languages...")
        try:
            languages = engine.get_supported_languages()
            print(f"✅ Supported languages: {len(languages)}")
            print(f"   Languages: {', '.join(languages)}")
        except Exception as e:
            print(f"❌ Get languages failed: {e}")
        
        # Show model info
        print(f"\n📋 Model Information:")
        try:
            model_info = engine.get_model_info()
            print(f"   Name: {model_info['name']}")
            print(f"   Engine: {model_info['engine_type']}")
            print(f"   Languages: {len(model_info['supported_languages'])} supported")
            print(f"   GPU Required: {model_info['gpu_required']}")
            print(f"   Python Compatibility: {model_info['python_compatibility']}")
            print(f"   Model Loaded: {model_info['loaded']}")
        except Exception as e:
            print(f"❌ Model info failed: {e}")
        
        # Test memory stats
        print(f"\n💾 Memory Statistics:")
        try:
            memory_stats = engine.get_memory_stats()
            print(f"   System memory used: {memory_stats.get('system_memory_used_gb', 0):.1f} GB")
            if 'gpu_memory_used_gb' in memory_stats:
                print(f"   GPU memory used: {memory_stats['gpu_memory_used_gb']:.1f} GB")
        except Exception as e:
            print(f"❌ Memory stats failed: {e}")
        
        print(f"\n" + "="*60)
        print("📊 RESULTS SUMMARY")
        print("="*60)
        print(f"✅ Successfully created {len(created_files)} audio files:")
        
        for file_path in created_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  📁 {file_path} ({file_size} bytes)")
            else:
                print(f"  ❌ {file_path} (file missing!)")
        
        return len(created_files) > 0
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_oute_tts_service():
    """Test OuteTTS through the TTS service."""
    print("\n🔧 Testing OuteTTS through TTS Service")
    print("="*60)
    
    try:
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Initialize TTS service
        print("📥 Initializing TTS service...")
        tts_service = TextToSpeechService()
        
        # Load the OuteTTS model
        print("📥 Loading OuteTTS model...")
        tts_service.load_tts_model("oute-tts-1b")
        
        # Test text-to-speech
        test_text = "Testing OuteTTS through the BeautyAI TTS service framework."
        print(f"🔊 Converting text: '{test_text}'")
        
        result = tts_service.text_to_speech(
            text=test_text,
            language="en",
            speaker_voice="female",
            output_path="/home/lumi/beautyai/voice_tests/service_test_oute_tts.wav"
        )
        
        if result and os.path.exists(result):
            file_size = os.path.getsize(result)
            print(f"✅ Service test successful:")
            print(f"   Output file: {result}")
            print(f"   File size: {file_size} bytes")
            print(f"   Model used: oute-tts-1b")
            print(f"   Language: en")
            return True
        else:
            print(f"❌ Service test failed: No output file created")
            return False
            
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_registry():
    """Test OuteTTS model in the registry."""
    print("\n📋 Testing OuteTTS Model Registry")
    print("="*60)
    
    try:
        from beautyai_inference.config.config_manager import AppConfig
        
        # Load model registry
        app_config = AppConfig()
        app_config.models_file = "beautyai_inference/config/model_registry.json"
        app_config.load_model_registry()
        
        # Check OuteTTS model
        oute_model = app_config.model_registry.get_model("oute-tts-1b")
        
        if oute_model:
            print("✅ OuteTTS model found in registry:")
            print(f"   Model ID: {oute_model.model_id}")
            print(f"   Engine Type: {oute_model.engine_type}")
            print(f"   Quantization: {oute_model.quantization}")
            print(f"   Description: {oute_model.description}")
            return True
        else:
            print("❌ OuteTTS model not found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎙️ BeautyAI OuteTTS Test Suite")
    print("Testing OuteTTS neural speech synthesis (GGUF format)")
    print("="*80)
    
    # Run all tests
    results = []
    
    # Test 1: OuteTTS Engine
    results.append(test_oute_tts())
    
    # Test 2: TTS Service
    results.append(test_oute_tts_service())
    
    # Test 3: Model Registry
    results.append(test_model_registry())
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print("🎯 OuteTTS is successfully integrated into BeautyAI framework!")
        print("💡 Check the /home/lumi/beautyai/voice_tests directory for output files.")
        print("🚀 OuteTTS provides high-quality neural speech synthesis with GGUF optimization!")
    else:
        print("❌ SOME TESTS FAILED")
        print("💡 Check the errors above for details.")
        print("🔧 You may need to install additional dependencies or check the model configuration.")

if __name__ == "__main__":
    main()
