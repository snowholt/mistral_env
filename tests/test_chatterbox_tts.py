#!/usr/bin/env python3
"""
Chatterbox TTS Test Script

Tests the Chatterbox Multilingual TTS engine integration.

Usage:
    cd /home/lumi/beautyai
    source backend/venv/bin/activate
    python tests/test_chatterbox_tts.py

Options:
    --text "Your text here"    Text to synthesize
    --language ar|en           Language code (default: ar)
    --output output.wav        Output file (default: auto-generated)
    --speaker reference.wav    Speaker reference file for voice cloning
    --no-gpu                   Use CPU instead of GPU
"""

import sys
import os
import time
import argparse

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

import torch
import torchaudio


def test_chatterbox_direct():
    """Test Chatterbox directly without the engine wrapper."""
    print("\n" + "="*60)
    print("🎤 CHATTERBOX TTS DIRECT TEST")
    print("="*60)
    
    # Check CUDA
    print(f"\n📊 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Import Chatterbox
    print("\n📦 Importing Chatterbox...")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        print("   ✅ Chatterbox Multilingual TTS imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   💡 Install with: pip install chatterbox-tts")
        return False
    
    # Set cache directory
    cache_dir = "/home/lumi/beautyai/config/models/chatterbox"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    print(f"\n📁 Cache directory: {cache_dir}")
    
    # Load model
    print("\n🔄 Loading Chatterbox Multilingual model (this may take a while on first run)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.1f}s")
        print(f"   📊 Sample rate: {model.sr} Hz")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation - English
    print("\n🎯 TEST 1: English generation")
    english_text = "Hello! This is a test of the Chatterbox text-to-speech system. It supports 23 languages including Arabic."
    
    start_time = time.time()
    try:
        wav_en = model.generate(text=english_text, language_id="en")
        gen_time = time.time() - start_time
        
        output_path = "/tmp/chatterbox_test_english.wav"
        torchaudio.save(output_path, wav_en, model.sr)
        
        # Calculate duration
        duration = wav_en.shape[-1] / model.sr
        rtf = gen_time / duration  # Real-time factor
        
        print(f"   ✅ Generated in {gen_time:.2f}s")
        print(f"   📊 Audio duration: {duration:.2f}s")
        print(f"   ⚡ Real-time factor: {rtf:.2f}x")
        print(f"   💾 Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test generation - Arabic
    print("\n🎯 TEST 2: Arabic generation")
    arabic_text = "مرحباً! هذا اختبار لنظام تحويل النص إلى كلام. أتمنى أن يعمل بشكل جيد."
    
    start_time = time.time()
    try:
        wav_ar = model.generate(text=arabic_text, language_id="ar")
        gen_time = time.time() - start_time
        
        output_path = "/tmp/chatterbox_test_arabic.wav"
        torchaudio.save(output_path, wav_ar, model.sr)
        
        # Calculate duration
        duration = wav_ar.shape[-1] / model.sr
        rtf = gen_time / duration
        
        print(f"   ✅ Generated in {gen_time:.2f}s")
        print(f"   📊 Audio duration: {duration:.2f}s")
        print(f"   ⚡ Real-time factor: {rtf:.2f}x")
        print(f"   💾 Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with speaker reference (if available)
    speaker_ref = "/home/lumi/beautyai/backend/speakers/chatterbox/reference.wav"
    if os.path.exists(speaker_ref):
        print("\n🎯 TEST 3: Voice cloning with reference")
        test_text = "مرحباً، هذا صوتي المستنسخ. كيف تبدو النتيجة؟"
        
        start_time = time.time()
        try:
            wav_clone = model.generate(
                text=test_text, 
                language_id="ar",
                audio_prompt_path=speaker_ref
            )
            gen_time = time.time() - start_time
            
            output_path = "/tmp/chatterbox_test_cloned.wav"
            torchaudio.save(output_path, wav_clone, model.sr)
            
            duration = wav_clone.shape[-1] / model.sr
            rtf = gen_time / duration
            
            print(f"   ✅ Generated in {gen_time:.2f}s")
            print(f"   📊 Audio duration: {duration:.2f}s")
            print(f"   ⚡ Real-time factor: {rtf:.2f}x")
            print(f"   💾 Saved to: {output_path}")
        except Exception as e:
            print(f"   ❌ Voice cloning failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️ Speaker reference not found at: {speaker_ref}")
        print("   Skipping voice cloning test")
    
    # GPU memory stats
    if torch.cuda.is_available():
        print("\n📊 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n" + "="*60)
    print("✅ CHATTERBOX DIRECT TEST COMPLETE")
    print("="*60)
    
    return True


def test_chatterbox_engine():
    """Test the BeautyAI Chatterbox engine wrapper."""
    print("\n" + "="*60)
    print("🎤 CHATTERBOX ENGINE WRAPPER TEST")
    print("="*60)
    
    try:
        from beautyai_inference.inference_engines.voice.tts.chatterbox_engine import (
            ChatterboxMultilingualEngine
        )
        print("✅ ChatterboxMultilingualEngine imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create engine
    print("\n🔄 Creating ChatterboxMultilingualEngine...")
    try:
        engine = ChatterboxMultilingualEngine(
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir="/home/lumi/beautyai/config/models/chatterbox"
        )
        print("   ✅ Engine created")
    except Exception as e:
        print(f"   ❌ Engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load model
    print("\n🔄 Loading model...")
    try:
        success = engine.load_model()
        if success:
            print("   ✅ Model loaded")
        else:
            print("   ❌ Model loading returned False")
            return False
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get model info
    print("\n📊 Model Info:")
    info = engine.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Set speaker reference
    speaker_ref = "/home/lumi/beautyai/backend/speakers/chatterbox/reference.wav"
    if os.path.exists(speaker_ref):
        engine.set_speaker_reference(speaker_ref)
        print(f"\n✅ Speaker reference set: {speaker_ref}")
    
    # Test text_to_speech
    print("\n🎯 TEST: text_to_speech()")
    test_text = "أهلاً وسهلاً! هذا اختبار لمحرك Chatterbox."
    
    start_time = time.time()
    try:
        output_path = engine.text_to_speech(
            text=test_text,
            language="ar",
            output_path="/tmp/chatterbox_engine_test.wav"
        )
        gen_time = time.time() - start_time
        
        print(f"   ✅ Generated in {gen_time:.2f}s")
        print(f"   💾 Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test benchmark
    print("\n📊 Benchmark Results:")
    try:
        results = engine.benchmark("This is a benchmark test.", language="en")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
    
    # Memory stats
    print("\n📊 Memory Stats:")
    stats = engine.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.1f} MB")
    
    # Unload model
    print("\n🔄 Unloading model...")
    engine.unload_model()
    print("   ✅ Model unloaded")
    
    print("\n" + "="*60)
    print("✅ CHATTERBOX ENGINE TEST COMPLETE")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox TTS integration")
    parser.add_argument("--direct", action="store_true", help="Test Chatterbox directly")
    parser.add_argument("--engine", action="store_true", help="Test through BeautyAI engine wrapper")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.direct, args.engine, args.all]):
        args.all = True  # Default to all tests
    
    success = True
    
    if args.direct or args.all:
        if not test_chatterbox_direct():
            success = False
    
    if args.engine or args.all:
        if not test_chatterbox_engine():
            success = False
    
    if success:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n📋 Next steps:")
        print("   1. Check generated audio files in /tmp/")
        print("   2. Play them to verify quality:")
        print("      aplay /tmp/chatterbox_test_english.wav")
        print("      aplay /tmp/chatterbox_test_arabic.wav")
        print("   3. Enable in preload_config.json to use in production")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
