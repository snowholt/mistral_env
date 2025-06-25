#!/usr/bin/env python3
"""
Test script to verify Edge TTS functionality.
"""

import sys
import os
import logging

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
        
        # Test text-to-speech with different languages and genders
        test_cases = [
            ("Hello, this is a test using Microsoft Edge TTS.", "en", "female"),
            ("Hello, this is a test with a male voice.", "en", "male"),
            ("مرحبا، هذا اختبار باستخدام مايكروسوفت Edge TTS.", "ar", "female"),
            ("Bonjour, ceci est un test avec Edge TTS.", "fr", "female"),
        ]
        
        created_files = []
        
        for i, (text, language, gender) in enumerate(test_cases):
            print(f"\n🔊 Test {i+1}: Generating Edge TTS for {language} ({gender})")
            print(f"Text: '{text}'")
            
            try:
                # Use a specific output path
                output_path = f"/home/lumi/beautyai/voice_tests/edge_tts_{language}_{gender}_{i+1}.wav"
                
                result_path = engine.text_to_speech(
                    text=text,
                    language=language,
                    output_path=output_path,
                    gender=gender
                )
                
                if result_path and os.path.exists(result_path):
                    file_size = os.path.getsize(result_path)
                    print(f"✅ Edge TTS successful: {result_path}")
                    print(f"📁 File size: {file_size} bytes")
                    created_files.append(result_path)
                else:
                    print(f"❌ Edge TTS failed: No output file created")
                    
            except Exception as e:
                print(f"❌ Edge TTS failed: {e}")
        
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
