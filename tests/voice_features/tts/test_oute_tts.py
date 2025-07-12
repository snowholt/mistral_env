#!/usr/bin/env python3
"""
Test script to verify OuteTTS functionality.
Tests the new OuteTTS engine implementation in the BeautyAI framework.
"""

import sys
import os
import logging
import json
import time

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
        
        # First, identify available speakers for Arabic
        print("🎤 Identifying available Arabic speakers...")
        try:
            available_speakers = engine.identify_available_speakers()
            arabic_speakers = available_speakers.get("ar", ["AR-FEMALE-1-NEUTRAL"])
            print(f"✅ Found {len(arabic_speakers)} Arabic speakers:")
            for speaker in arabic_speakers:
                print(f"   - {speaker}")
        except Exception as e:
            print(f"⚠️ Could not identify speakers, using defaults: {e}")
            arabic_speakers = ["AR-FEMALE-1-NEUTRAL", "AR-MALE-1-NEUTRAL"]
        
        # Arabic beauty clinic test cases - 5 specific scenarios
        beauty_clinic_test_cases = [
            ("أهلاً وسهلاً بك في عيادة الجمال. كيف يمكنني مساعدتك اليوم؟", "ar", arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"),
            ("لديك موعد لعلاج البشرة في الساعة الثالثة. هل تريد تأكيد الموعد؟", "ar", arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"),
            ("نوصي بكريم الترطيب اليومي لنوع بشرتك. يجب استخدامه صباحاً ومساءً.", "ar", arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"),
            ("عملية تنظيف البشرة تستغرق ساعة واحدة. ستشعر بتحسن كبير بعدها.", "ar", arabic_speakers[1] if len(arabic_speakers) > 1 else arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"),
            ("شكراً لزيارة عيادتنا. نتطلع لرؤيتك في الموعد القادم للمتابعة.", "ar", arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"),
        ]
        
        # Use only Arabic beauty clinic test cases
        test_cases = beauty_clinic_test_cases
        
        created_files = []
        performance_results = []  # Track performance metrics
        
        for i, (text, language, speaker_id) in enumerate(test_cases):
            print(f"\n🔊 Beauty Clinic Test {i+1}: Generating Arabic TTS")
            print(f"Text: '{text}'")
            print(f"Speaker: {speaker_id}")
            
            try:
                # Measure performance
                start_time = time.time()
                
                # Use a specific output path for beauty clinic scenarios
                output_path = f"/home/lumi/beautyai/voice_tests/beauty_clinic_ar_{i+1}.wav"
                
                # Use the actual speaker ID instead of gender mapping
                result_path = engine.text_to_speech(
                    text=text,
                    language=language,
                    output_path=output_path,
                    speaker_voice=speaker_id,  # Pass the actual speaker ID
                    emotion="neutral",
                    speed=1.0
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                chars_per_second = len(text) / generation_time if generation_time > 0 else 0
                
                performance_data = {
                    "test_number": i + 1,
                    "scenario": f"beauty_clinic_{i+1}",
                    "language": language,
                    "speaker_id": speaker_id,
                    "text_length": len(text),
                    "generation_time": generation_time,
                    "chars_per_second": chars_per_second,
                    "success": False,
                    "file_size": 0
                }
                
                if result_path and os.path.exists(result_path):
                    file_size = os.path.getsize(result_path)
                    performance_data["success"] = True
                    performance_data["file_size"] = file_size
                    
                    print(f"✅ OuteTTS successful: {result_path}")
                    print(f"📁 File size: {file_size} bytes")
                    print(f"⏱️ Generation time: {generation_time:.2f}s")
                    print(f"🚀 Speed: {chars_per_second:.1f} chars/sec")
                    created_files.append(result_path)
                else:
                    print(f"❌ OuteTTS failed: No output file created")
                
                performance_results.append(performance_data)
                    
            except Exception as e:
                print(f"❌ OuteTTS failed: {e}")
                # Add failed result to performance tracking
                performance_results.append({
                    "test_number": i + 1,
                    "scenario": f"beauty_clinic_{i+1}",
                    "language": language,
                    "speaker_id": speaker_id,
                    "text_length": len(text),
                    "generation_time": 0,
                    "chars_per_second": 0,
                    "success": False,
                    "file_size": 0,
                    "error": str(e)
                })
        
        # Test text_to_speech_bytes method with Arabic
        print(f"\n🔊 Testing text_to_speech_bytes method (Arabic)...")
        try:
            audio_bytes = engine.text_to_speech_bytes(
                "اختبار إخراج البايتات من محرك OuteTTS.",
                language="ar",
                speaker_voice=arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"
            )
            print(f"✅ Bytes method successful: {len(audio_bytes)} bytes returned")
        except Exception as e:
            print(f"❌ Bytes method failed: {e}")
        
        # Test benchmark with Arabic
        print(f"\n🚀 Running benchmark (Arabic)...")
        try:
            benchmark_result = engine.benchmark(
                "هذا اختبار أداء لتقنية OuteTTS لقياس الجودة والسرعة.",
                language="ar",
                speaker_voice=arabic_speakers[0] if arabic_speakers else "AR-FEMALE-1-NEUTRAL"
            )
            print(f"✅ Benchmark completed:")
            print(f"   Generation time: {benchmark_result['generation_time']:.2f}s")
            print(f"   Characters/second: {benchmark_result['characters_per_second']:.1f}")
            print(f"   Engine: {benchmark_result['engine']}")
            print(f"   Success: {benchmark_result['success']}")
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
        
        # Test available speakers for Arabic
        print(f"\n🎤 Testing available speakers (Arabic)...")
        try:
            speakers = engine.get_available_speakers("ar")
            print(f"✅ Available speakers for Arabic: {len(speakers)}")
            for speaker in speakers:
                print(f"   - {speaker}")
        except Exception as e:
            print(f"❌ Get speakers failed: {e}")
        
        # Test supported languages
        print(f"\n🌍 Testing supported languages...")
        try:
            languages = engine.get_supported_languages()
            arabic_supported = "ar" in languages
            print(f"✅ Arabic supported: {arabic_supported}")
            print(f"   Total languages: {len(languages)}")
            print(f"   Arabic position: {languages.index('ar') + 1 if arabic_supported else 'Not found'}")
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
        
        # Performance Analysis for Beauty Clinic Scenarios
        print(f"\n📈 PERFORMANCE ANALYSIS (Arabic Beauty Clinic)")
        print("="*60)
        
        successful_tests = [r for r in performance_results if r["success"]]
        
        if successful_tests:
            avg_speed = sum(r["chars_per_second"] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r["generation_time"] for r in successful_tests) / len(successful_tests)
            total_chars = sum(r["text_length"] for r in successful_tests)
            total_time = sum(r["generation_time"] for r in successful_tests)
            
            print(f"🎙️ Arabic Beauty Clinic TTS Performance:")
            print(f"   Average Speed: {avg_speed:.1f} chars/sec")
            print(f"   Average Generation Time: {avg_time:.2f}s")
            print(f"   Total Characters Processed: {total_chars}")
            print(f"   Total Generation Time: {total_time:.2f}s")
            print(f"   Success Rate: {len(successful_tests)}/{len(performance_results)} ({len(successful_tests)/len(performance_results)*100:.1f}%)")
            
            # Show individual test performance
            print(f"\n📊 Individual Beauty Clinic Scenario Performance:")
            for result in successful_tests:
                scenario = result.get("scenario", f"test_{result['test_number']}")
                print(f"   {scenario}: {result['chars_per_second']:.1f} chars/sec ({result['generation_time']:.2f}s)")
                
        else:
            print("❌ No successful tests to analyze")
        
        print(f"\n📁 Generated Beauty Clinic Audio Files:")
        for file_path in created_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  🎵 {os.path.basename(file_path)} ({file_size} bytes)")
            else:
                print(f"  ❌ {os.path.basename(file_path)} (missing!)")
        
        # Save performance data to JSON
        performance_file = "/home/lumi/beautyai/voice_tests/oute_tts_performance.json"
        try:
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Performance data saved to: {performance_file}")
        except Exception as e:
            print(f"❌ Failed to save performance data: {e}")
        
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
        
        # Test text-to-speech with Arabic beauty clinic scenario
        test_text = "أهلاً وسهلاً بكم في عيادة الجمال. كيف يمكنني مساعدتكم اليوم؟"
        print(f"🔊 Converting Arabic text: '{test_text}'")
        
        result = tts_service.text_to_speech(
            text=test_text,
            language="ar",
            speaker_voice="female",
            output_path="/home/lumi/beautyai/voice_tests/service_test_beauty_clinic_ar.wav"
        )
        
        if result and os.path.exists(result):
            file_size = os.path.getsize(result)
            print(f"✅ Service test successful:")
            print(f"   Output file: {result}")
            print(f"   File size: {file_size} bytes")
            print(f"   Model used: oute-tts-1b")
            print(f"   Language: Arabic (ar)")
            print(f"   Scenario: Beauty clinic welcome")
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
    """Main test function for Arabic beauty clinic TTS scenarios."""
    print("🎙️ BeautyAI OuteTTS Arabic Beauty Clinic Test Suite")
    print("Testing OuteTTS neural speech synthesis for Arabic beauty clinic scenarios")
    print("="*80)
    
    # Run all tests
    results = []
    
    # Test 1: OuteTTS Engine (Arabic Beauty Clinic)
    results.append(test_oute_tts())
    
    # Test 2: TTS Service (Arabic)
    results.append(test_oute_tts_service())
    
    # Test 3: Model Registry
    results.append(test_model_registry())
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - ARABIC BEAUTY CLINIC TTS")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL ARABIC BEAUTY CLINIC TESTS PASSED!")
        print("🎯 OuteTTS is successfully integrated for Arabic beauty clinic scenarios!")
        print("💄 Perfect for beauty clinic customer service applications!")
        print("🎵 Check the /home/lumi/beautyai/voice_tests directory for beauty clinic audio files:")
        print("   • beauty_clinic_ar_*.wav (Arabic beauty clinic scenarios)")
        print("   • service_test_beauty_clinic_ar.wav (Service integration test)")
        print("🚀 OuteTTS provides high-quality Arabic neural speech synthesis!")
    else:
        print("❌ SOME ARABIC BEAUTY CLINIC TESTS FAILED")
        print("💡 Check the errors above for details.")
        print("🔧 You may need to install OuteTTS: pip install outetts")
        print("📋 Or check the Arabic speaker configurations.")

if __name__ == "__main__":
    main()
