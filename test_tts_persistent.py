#!/usr/bin/env python3
"""
Test script to verify TTS functionality and keep output files for inspection.
"""

import sys
import os
import logging

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tts_with_persistent_files():
    """Test TTS functionality and keep the output files."""
    print("🔧 Testing TTS with Persistent Output Files")
    print("="*60)
    
    try:
        from beautyai_inference.config.config_manager import AppConfig, ModelConfig
        from beautyai_inference.inference_engines.xtts_engine import XTTSEngine
        
        # Create mock model config
        model_config = ModelConfig(
            name="xtts-v2",
            model_id="coqui/XTTS-v2",
            engine_type="xtts",
            quantization=None,
            dtype="float16",
            max_new_tokens=None
        )
        
        # Initialize engine
        print("📥 Initializing XTTS engine...")
        engine = XTTSEngine(model_config)
        print(f"✅ Engine initialized. Mock mode: {engine.mock_mode}")
        
        # Load model
        print("📥 Loading XTTS model...")
        engine.load_model()
        print("✅ Model loaded successfully")
        
        # Test text-to-speech with persistent files
        test_texts = [
            ("Hello, this is a test that will keep the output file.", "en"),
            ("مرحبا، هذا اختبار سيحتفظ بملف الإخراج.", "ar")
        ]
        
        created_files = []
        
        for i, (text, language) in enumerate(test_texts):
            print(f"\n🔊 Test {i+1}: Generating TTS for {language}")
            print(f"Text: '{text}'")
            
            try:
                # Use a specific output path to keep the file
                output_path = f"/home/lumi/beautyai/voice_tests/tts_{language}_{i+1}.wav"
                
                result_path = engine.text_to_speech(
                    text=text,
                    language=language,
                    output_path=output_path,
                    emotion="neutral",
                    speed=1.0
                )
                
                if result_path and os.path.exists(result_path):
                    file_size = os.path.getsize(result_path)
                    print(f"✅ TTS successful: {result_path}")
                    print(f"📁 File size: {file_size} bytes")
                    created_files.append(result_path)
                else:
                    print(f"❌ TTS failed: No output file created")
                    
            except Exception as e:
                print(f"❌ TTS failed: {e}")
        
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
        return False

def main():
    """Main test function."""
    print("🎙️ BeautyAI TTS Persistent File Test")
    print("Testing TTS functionality and keeping output files")
    print("="*80)
    
    success = test_tts_with_persistent_files()
    
    print("\n" + "="*80)
    if success:
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("💡 Check the /home/lumi/beautyai/voice_tests directory for output files.")
    else:
        print("❌ TEST FAILED")
        print("💡 Check the errors above for details.")

if __name__ == "__main__":
    main()
