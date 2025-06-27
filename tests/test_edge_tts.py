#!/usr/bin/env python3
"""
Test script to verify Edge TTS functionality.
"""

import sys
import os
import logging
import time
import json

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_edge_tts():
    """Test Edge TTS functionality."""
    print("🎙️ Testing Edge TTS Engine")
    print("="*60)
    
    try:
        from beautyai_inference.config.config_manager import ModelConfig
        from beautyai_inference.inference_engines.edge_tts_engine import EdgeTTSEngine
        
        # Create model config for Edge TTS
        model_config = ModelConfig(
            name="edge-tts",
            model_id="microsoft/edge-tts",
            engine_type="edge_tts",
            quantization=None,
            dtype=None,
            max_new_tokens=None
        )
        
        # Initialize engine
        print("📥 Initializing Edge TTS engine...")
        engine = EdgeTTSEngine(model_config)
        print(f"✅ Engine initialized. Mock mode: {engine.mock_mode}")
        
        # Load model
        print("📥 Loading Edge TTS model...")
        engine.load_model()
        print("✅ Model loaded successfully")
        
        # Arabic-focused test cases for comparison with OuteTTS
        arabic_test_cases = [
            ("مرحبا، كيف حالك اليوم؟ أتمنى أن تكون بخير.", "ar", "female"),
            ("مرحبا، كيف حالك اليوم؟ أتمنى أن تكون بخير.", "ar", "male"),
            ("هذا اختبار لجودة الصوت العربي باستخدام تقنية OuteTTS المتقدمة.", "ar", "female"),
            ("هذا اختبار لجودة الصوت العربي باستخدام تقنية OuteTTS المتقدمة.", "ar", "male"),
            ("أهلا وسهلا بكم في إطار عمل BeautyAI للذكاء الاصطناعي.", "ar", "female"),
            ("أهلا وسهلا بكم في إطار عمل BeautyAI للذكاء الاصطناعي.", "ar", "male"),
            ("الذكاء الاصطناعي يساعد في تطوير التقنيات الحديثة.", "ar", "female"),
            ("الذكاء الاصطناعي يساعد في تطوير التقنيات الحديثة.", "ar", "male"),
        ]
        
        # Additional multilingual test cases
        general_test_cases = [
            ("Hello, this is a test using Microsoft Edge TTS.", "en", "female"),
            ("Hello, this is a test with a male voice.", "en", "male"),
            ("Bonjour, ceci est un test avec Edge TTS.", "fr", "female"),
        ]
        
        # Combine all test cases
        test_cases = arabic_test_cases + general_test_cases
        
        created_files = []
        performance_results = []  # Track performance metrics
        
        for i, (text, language, gender) in enumerate(test_cases):
            print(f"\n🔊 Test {i+1}: Generating Edge TTS for {language} ({gender})")
            print(f"Text: '{text}'")
            
            try:
                # Measure performance
                start_time = time.time()
                
                # Use a specific output path
                output_path = f"/home/lumi/beautyai/voice_tests/edge_tts_{language}_{gender}_{i+1}.wav"
                
                result_path = engine.text_to_speech(
                    text=text,
                    language=language,
                    output_path=output_path,
                    gender=gender
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                chars_per_second = len(text) / generation_time if generation_time > 0 else 0
                
                # Store performance data
                performance_data = {
                    "test_number": i + 1,
                    "language": language,
                    "gender": gender,
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
                    
                    print(f"✅ Edge TTS successful: {result_path}")
                    print(f"📁 File size: {file_size} bytes")
                    print(f"⏱️ Generation time: {generation_time:.2f}s")
                    print(f"🚀 Speed: {chars_per_second:.1f} chars/sec")
                    created_files.append(result_path)
                else:
                    print(f"❌ Edge TTS failed: No output file created")
                
                performance_results.append(performance_data)
                    
            except Exception as e:
                print(f"❌ Edge TTS failed: {e}")
                # Add failed result to performance tracking
                performance_results.append({
                    "test_number": i + 1,
                    "language": language,
                    "gender": gender,
                    "text_length": len(text),
                    "generation_time": 0,
                    "chars_per_second": 0,
                    "success": False,
                    "file_size": 0,
                    "error": str(e)
                })
        
        # Test benchmark
        print(f"\n🚀 Running benchmark...")
        try:
            benchmark_result = engine.benchmark("This is a benchmark test for Edge TTS performance.")
            print(f"✅ Benchmark completed:")
            print(f"   Generation time: {benchmark_result['generation_time']:.2f}s")
            print(f"   Characters/second: {benchmark_result['characters_per_second']:.1f}")
            print(f"   Engine: {benchmark_result['engine']}")
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
        
        # Show model info
        print(f"\n📋 Model Information:")
        try:
            model_info = engine.get_model_info()
            print(f"   Name: {model_info['name']}")
            print(f"   Type: {model_info['type']}")
            print(f"   Languages: {len(model_info['languages'])} supported")
            print(f"   GPU Required: {model_info['gpu_required']}")
            print(f"   Python Compatibility: {model_info['python_compatibility']}")
        except Exception as e:
            print(f"❌ Model info failed: {e}")
        
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
        
        # Performance Analysis
        print(f"\n📈 PERFORMANCE ANALYSIS (Edge TTS)")
        print("="*60)
        
        successful_tests = [r for r in performance_results if r["success"]]
        arabic_tests = [r for r in successful_tests if r["language"] == "ar"]
        english_tests = [r for r in successful_tests if r["language"] == "en"]
        
        if successful_tests:
            avg_speed = sum(r["chars_per_second"] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r["generation_time"] for r in successful_tests) / len(successful_tests)
            print(f"🚀 Overall Performance:")
            print(f"   Average Speed: {avg_speed:.1f} chars/sec")
            print(f"   Average Generation Time: {avg_time:.2f}s")
            print(f"   Success Rate: {len(successful_tests)}/{len(performance_results)} ({len(successful_tests)/len(performance_results)*100:.1f}%)")
        
        if arabic_tests:
            ar_avg_speed = sum(r["chars_per_second"] for r in arabic_tests) / len(arabic_tests)
            ar_avg_time = sum(r["generation_time"] for r in arabic_tests) / len(arabic_tests)
            print(f"\n🇸🇦 Arabic Performance:")
            print(f"   Average Speed: {ar_avg_speed:.1f} chars/sec")
            print(f"   Average Generation Time: {ar_avg_time:.2f}s")
            print(f"   Arabic Tests: {len(arabic_tests)}")
        
        if english_tests:
            en_avg_speed = sum(r["chars_per_second"] for r in english_tests) / len(english_tests)
            en_avg_time = sum(r["generation_time"] for r in english_tests) / len(english_tests)
            print(f"\n🇺🇸 English Performance:")
            print(f"   Average Speed: {en_avg_speed:.1f} chars/sec")
            print(f"   Average Generation Time: {en_avg_time:.2f}s")
            print(f"   English Tests: {len(english_tests)}")
        
        # Save performance data to JSON
        performance_file = "/home/lumi/beautyai/voice_tests/edge_tts_performance.json"
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

def main():
    """Main test function."""
    print("🎙️ BeautyAI Edge TTS Test")
    print("Testing Edge TTS functionality (Python 3.12 compatible)")
    print("="*80)
    
    success = test_edge_tts()
    
    print("\n" + "="*80)
    if success:
        print("✅ EDGE TTS TEST COMPLETED SUCCESSFULLY!")
        print("💡 Check the /home/lumi/beautyai/voice_tests directory for output files.")
        print("🎯 Edge TTS provides real speech synthesis with high quality voices!")
    else:
        print("❌ TEST FAILED")
        print("💡 Check the errors above for details.")

if __name__ == "__main__":
    main()
