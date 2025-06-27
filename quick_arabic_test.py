#!/usr/bin/env python3
"""
Quick Arabic Speaker Test - Simple version for immediate testing.
Tests one sample from each category to verify functionality.
"""

import sys
import os
import time
from pathlib import Path

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

def quick_arabic_test():
    """Quick test of Arabic speakers with sample sentences."""
    
    print("🚀 Quick Arabic Speaker Test")
    print("="*50)
    
    # Create test directory
    test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample test sentences
    test_sentences = {
        "long": "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً.",
        "medium": "مرحباً بكم في عيادة الجمال للعناية بالبشرة المتخصصة",  # 10 words
        "short": "مرحباً وأهلاً بكم"  # 3 words
    }
    
    try:
        # Import OuteTTS
        import outetts
        
        print("📥 Initializing OuteTTS...")
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
            )
        )
        
        # Test with default English speaker first (to verify OuteTTS works)
        print("\n🔊 Testing with default English speaker...")
        default_speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
        
        # Test Arabic text with English speaker
        test_output = interface.generate(
            config=outetts.GenerationConfig(
                text=test_sentences["short"],
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=default_speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.4,
                    top_p=0.9,
                    top_k=50
                ),
            )
        )
        
        test_file = test_dir / "quick_test_english_speaker_arabic_text.wav"
        test_output.save(str(test_file))
        
        if test_file.exists():
            print(f"✅ OuteTTS working: {test_file}")
        else:
            print("❌ OuteTTS test failed")
            return False
        
        # Now try with custom Arabic speakers if they exist
        arabic_speakers = [
            ("female", "/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/arabic_female_beautyai.json"),
            ("male", "/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/arabic_male_beautyai.json")
        ]
        
        for gender, profile_path in arabic_speakers:
            if os.path.exists(profile_path):
                print(f"\n🎤 Testing {gender} Arabic speaker...")
                try:
                    speaker = interface.load_speaker(profile_path)
                    
                    # Test each sentence type
                    for sentence_type, text in test_sentences.items():
                        print(f"   Testing {sentence_type}: '{text[:50]}...'")
                        
                        start_time = time.time()
                        output = interface.generate(
                            config=outetts.GenerationConfig(
                                text=text,
                                generation_type=outetts.GenerationType.CHUNKED,
                                speaker=speaker,
                                sampler_config=outetts.SamplerConfig(
                                    temperature=0.4,
                                    top_p=0.9,
                                    top_k=50
                                ),
                            )
                        )
                        generation_time = time.time() - start_time
                        
                        output_file = test_dir / f"quick_{gender}_{sentence_type}.wav"
                        output.save(str(output_file))
                        
                        if output_file.exists():
                            file_size = output_file.stat().st_size
                            chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
                            print(f"   ✅ {sentence_type}: {output_file.name} ({file_size:,} bytes, {generation_time:.2f}s, {chars_per_sec:.1f} chars/sec)")
                        else:
                            print(f"   ❌ {sentence_type}: Failed to generate")
                            
                except Exception as e:
                    print(f"   ❌ Failed to test {gender} speaker: {e}")
            else:
                print(f"\n⚠️ {gender.capitalize()} Arabic speaker profile not found: {profile_path}")
                print("💡 Run create_arabic_speaker_profiles.py first")
        
        print(f"\n📁 Quick test files saved in: {test_dir}")
        return True
        
    except ImportError:
        print("❌ OuteTTS library not available. Install with: pip install outetts")
        return False
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def quick_beautyai_test():
    """Quick test using BeautyAI TTS service."""
    
    print("\n🏥 Quick BeautyAI TTS Service Test")
    print("="*50)
    
    try:
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Create test directory
        test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS service
        print("📥 Initializing BeautyAI TTS service...")
        tts_service = TextToSpeechService()
        
        # Load OuteTTS model
        print("📥 Loading OuteTTS model...")
        success = tts_service.load_tts_model("oute-tts-1b")
        
        if not success:
            print("❌ Failed to load OuteTTS model")
            return False
        
        print("✅ OuteTTS model loaded successfully")
        
        # Test sentences
        test_sentences = [
            ("short", "مرحباً وأهلاً بكم"),
            ("medium", "مرحباً بكم في عيادة الجمال للعناية بالبشرة المتخصصة"),
            ("long", "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة")
        ]
        
        # Test both male and female voices
        for speaker_voice in ["female", "male"]:
            print(f"\n🎤 Testing {speaker_voice} voice:")
            
            for sentence_type, text in test_sentences:
                print(f"   Testing {sentence_type}: '{text[:50]}...'")
                
                start_time = time.time()
                output_path = test_dir / f"beautyai_{speaker_voice}_{sentence_type}.wav"
                
                result = tts_service.text_to_speech(
                    text=text,
                    language="ar",
                    speaker_voice=speaker_voice,
                    output_path=str(output_path)
                )
                
                generation_time = time.time() - start_time
                
                if result and os.path.exists(result):
                    file_size = os.path.getsize(result)
                    chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
                    print(f"   ✅ {sentence_type}: {output_path.name} ({file_size:,} bytes, {generation_time:.2f}s, {chars_per_sec:.1f} chars/sec)")
                else:
                    print(f"   ❌ {sentence_type}: Failed to generate")
        
        print(f"\n📁 BeautyAI test files saved in: {test_dir}")
        return True
        
    except ImportError as e:
        print(f"❌ BeautyAI TTS service not available: {e}")
        return False
    except Exception as e:
        print(f"❌ BeautyAI test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Arabic Speaker Test - Multiple Options")
    print("="*60)
    
    # Test 1: BeautyAI TTS Service (Recommended)
    print("\n1️⃣ Testing BeautyAI TTS Service:")
    beautyai_success = quick_beautyai_test()
    
    # Test 2: Direct OuteTTS (Alternative)
    print("\n2️⃣ Testing Direct OuteTTS:")
    outetts_success = quick_arabic_test()
    
    # Summary
    print("\n📋 Test Summary:")
    print("="*60)
    if beautyai_success:
        print("✅ BeautyAI TTS Service: PASSED")
    else:
        print("❌ BeautyAI TTS Service: FAILED")
        
    if outetts_success:
        print("✅ Direct OuteTTS: PASSED")
    else:
        print("❌ Direct OuteTTS: FAILED")
    
    if beautyai_success or outetts_success:
        print("\n🎉 At least one test method succeeded!")
        print("💡 Use BeautyAI TTS Service for production applications")
    else:
        print("\n⚠️ All tests failed. Check dependencies and configurations.")
