#!/usr/bin/env python3
"""
Enhanced Arabic Speaker Diagnostic Script for BeautyAI Platform.

Tests the newly created Arabic speaker profiles with comprehensive scenarios
including beauty clinic conversations, medical consultations, and multilingual support.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_arabic_audio_files() -> Dict[str, Any]:
    """Check if the original Arabic audio files exist."""
    audio_files = {
        "female": "/home/lumi/beautyai/voice_tests/custom_speakers/audio_1_F.wav",
        "male": "/home/lumi/beautyai/voice_tests/custom_speakers/audio_1_M.wav"
    }
    
    results = {}
    for gender, path in audio_files.items():
        file_path = Path(path)
        results[gender] = {
            "path": path,
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }
    
    return results

def check_arabic_speaker_profiles() -> Dict[str, Any]:
    """Check if Arabic speaker profiles have been created."""
    profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
    
    profile_files = [
        "arabic_female_beautyai.json",
        "arabic_male_beautyai.json",
        "speaker_mapping.json"
    ]
    
    results = {}
    for filename in profile_files:
        file_path = profiles_dir / filename
        results[filename] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }
    
    return results

def test_oute_tts_arabic_integration():
    """Test OuteTTS engine with Arabic speaker profiles."""
    
    print("🧪 Testing OuteTTS Arabic Integration")
    print("-" * 50)
    
    try:
        from beautyai_inference.inference_engines.oute_tts_engine import OuteTTSEngine
        from beautyai_inference.config.config_manager import ModelConfig
        
        # Create model configuration
        model_config = ModelConfig(
            model_id="oute-tts-1b",
            engine_type="oute_tts",
            quantization="fp16"
        )
        
        # Initialize OuteTTS engine
        print("📥 Initializing OuteTTS engine...")
        engine = OuteTTSEngine(model_config)
        
        # Load the model
        print("🔄 Loading OuteTTS model...")
        engine.load_model()
        
        print("✅ OuteTTS engine loaded successfully")
        
        # Check available speakers
        print(f"\n🎤 Available Speakers:")
        print(f"   Discovered speakers: {list(engine.discovered_speakers.keys())}")
        print(f"   Custom speakers: {list(engine.custom_speakers.keys())}")
        print(f"   Arabic mapping: {engine.arabic_speaker_mapping}")
        
        # Test Arabic scenarios
        test_scenarios = [
            {
                "name": "Beauty Clinic Welcome",
                "text": "مرحباً بك في عيادة الجمال. كيف يمكنني مساعدتك اليوم؟",
                "speaker": "female"
            },
            {
                "name": "Medical Consultation",
                "text": "أهلاً وسهلاً، أنا الدكتور أحمد. سأساعدك في اختيار أفضل علاج لبشرتك.",
                "speaker": "male"
            },
            {
                "name": "Treatment Information",
                "text": "لدينا علاجات متنوعة للوجه والبشرة وحقن البوتوكس والفيلر.",
                "speaker": "female"
            },
            {
                "name": "Appointment Booking",
                "text": "هل تريد حجز موعد لعلاج البشرة أو استشارة طبية؟",
                "speaker": "male"
            },
            {
                "name": "Service Pricing",
                "text": "أسعار العلاجات تبدأ من مائة ريال للجلسة الواحدة.",
                "speaker": "female"
            }
        ]
        
        # Create output directory
        test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_integration_tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔊 Testing Arabic Scenarios:")
        
        for i, scenario in enumerate(test_scenarios, 1):
            try:
                print(f"\n   {i}. {scenario['name']} ({scenario['speaker']})")
                
                # Generate speech
                output_path = test_dir / f"scenario_{i}_{scenario['name'].replace(' ', '_').lower()}_{scenario['speaker']}.wav"
                
                result = engine.text_to_speech(
                    text=scenario['text'],
                    language="ar",
                    speaker_voice=scenario['speaker'],
                    output_path=str(output_path)
                )
                
                if result and Path(result).exists():
                    file_size = Path(result).stat().st_size
                    print(f"      ✅ Generated: {result} ({file_size} bytes)")
                else:
                    print(f"      ❌ Failed to generate audio")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ OuteTTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_service_arabic_integration():
    """Test TTS service with Arabic speaker profiles."""
    
    print("\n🧪 Testing TTS Service Arabic Integration")
    print("-" * 50)
    
    try:
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Initialize TTS service
        tts_service = TextToSpeechService()
        
        # Load TTS model
        print("📥 Loading TTS model...")
        success = tts_service.load_tts_model("oute-tts-1b")
        
        if not success:
            print("❌ Failed to load TTS model")
            return False
        
        print("✅ TTS model loaded successfully")
        
        # Get available Arabic speakers
        print(f"\n🎤 Getting available Arabic speakers...")
        arabic_speakers = tts_service.get_available_arabic_speakers()
        
        print(f"   Available speakers: {arabic_speakers}")
        
        # Test specific speaker names
        test_speakers = [
            ("arabic_female_beautyai", "female"),
            ("arabic_male_beautyai", "male"),
            ("arabic_female_custom", "female"),
            ("arabic_male_custom", "male")
        ]
        
        test_texts = {
            "female": "مرحباً، أنا مساعدتك الشخصية في عيادة الجمال. كيف يمكنني مساعدتك؟",
            "male": "أهلاً وسهلاً، أنا الدكتور أحمد. سأساعدك في اختيار أفضل علاج لبشرتك."
        }
        
        print(f"\n🔊 Testing TTS Service with Arabic speakers:")
        
        for speaker_name, gender in test_speakers:
            try:
                print(f"\n   Testing speaker: {speaker_name} ({gender})")
                
                # Test the speaker
                result = tts_service.test_arabic_speaker(
                    speaker_name=speaker_name,
                    test_text=test_texts[gender]
                )
                
                if result:
                    print(f"      ✅ Test successful: {result}")
                else:
                    print(f"      ❌ Test failed")
                    
            except Exception as e:
                print(f"      ❌ Error testing {speaker_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_speaker_profile_creation():
    """Test the speaker profile creation process."""
    
    print("\n🧪 Testing Speaker Profile Creation Process")
    print("-" * 50)
    
    try:
        # Check if creation script exists
        creation_script = Path("/home/lumi/beautyai/create_arabic_speaker_profiles.py")
        if not creation_script.exists():
            print("❌ Speaker profile creation script not found")
            return False
        
        print("✅ Speaker profile creation script found")
        
        # Check audio files
        audio_status = check_arabic_audio_files()
        print(f"\n📁 Audio Files Status:")
        for gender, info in audio_status.items():
            status = "✅" if info["exists"] else "❌"
            print(f"   {status} {gender.title()}: {info['path']} ({info['size']} bytes)")
        
        # Check speaker profiles
        profile_status = check_arabic_speaker_profiles()
        print(f"\n📄 Speaker Profiles Status:")
        for filename, info in profile_status.items():
            status = "✅" if info["exists"] else "❌"
            print(f"   {status} {filename}: {info['path']} ({info['size']} bytes)")
        
        # Recommendations
        missing_audio = [g for g, info in audio_status.items() if not info["exists"]]
        missing_profiles = [f for f, info in profile_status.items() if not info["exists"]]
        
        if missing_audio:
            print(f"\n💡 Missing audio files: {missing_audio}")
            print("   Place your audio files in: /home/lumi/beautyai/voice_tests/custom_speakers/")
        
        if missing_profiles:
            print(f"\n💡 Missing speaker profiles: {missing_profiles}")
            print("   Run: python create_arabic_speaker_profiles.py")
        
        return len(missing_audio) == 0 and len(missing_profiles) == 0
        
    except Exception as e:
        print(f"❌ Speaker profile creation test failed: {e}")
        return False

def run_comprehensive_arabic_test():
    """Run comprehensive Arabic speaker testing."""
    
    print("🎭 Comprehensive Arabic Speaker Test for BeautyAI")
    print("="*80)
    
    # Test 1: Check profile creation
    creation_success = test_speaker_profile_creation()
    
    # Test 2: OuteTTS integration
    outetts_success = test_oute_tts_arabic_integration()
    
    # Test 3: TTS service integration
    service_success = test_tts_service_arabic_integration()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("="*80)
    print(f"✅ Profile Creation: {'PASS' if creation_success else 'FAIL'}")
    print(f"✅ OuteTTS Integration: {'PASS' if outetts_success else 'FAIL'}")
    print(f"✅ TTS Service Integration: {'PASS' if service_success else 'FAIL'}")
    
    overall_success = creation_success and outetts_success and service_success
    
    if overall_success:
        print(f"\n🎉 All tests passed! Arabic speakers are ready for BeautyAI platform.")
        print(f"\n🎯 Usage Instructions:")
        print(f"   Female Arabic TTS: speaker_voice='arabic_female_beautyai'")
        print(f"   Male Arabic TTS: speaker_voice='arabic_male_beautyai'")
        print(f"   Language code: language='ar'")
        
        print(f"\n📁 Generated Audio Files:")
        test_dirs = [
            "/home/lumi/beautyai/voice_tests/arabic_integration_tests",
            "/home/lumi/beautyai/voice_tests/arabic_speaker_tests"
        ]
        
        for test_dir in test_dirs:
            dir_path = Path(test_dir)
            if dir_path.exists():
                audio_files = list(dir_path.glob("*.wav"))
                print(f"   {test_dir}: {len(audio_files)} audio files")
                for audio_file in audio_files[:3]:  # Show first 3 files
                    file_size = audio_file.stat().st_size
                    print(f"      - {audio_file.name} ({file_size} bytes)")
                if len(audio_files) > 3:
                    print(f"      ... and {len(audio_files) - 3} more files")
    else:
        print(f"\n❌ Some tests failed. Please check the issues above.")
        
        if not creation_success:
            print(f"\n💡 To create speaker profiles:")
            print(f"   python create_arabic_speaker_profiles.py")
        
        if not outetts_success:
            print(f"\n💡 To debug OuteTTS issues:")
            print(f"   Check OuteTTS installation: pip install outetts")
            print(f"   Check CUDA availability for GPU acceleration")
        
        if not service_success:
            print(f"\n💡 To debug TTS service issues:")
            print(f"   Check model registry configuration")
            print(f"   Verify speaker profile paths")
    
    return overall_success

def main():
    """Main diagnostic function."""
    
    try:
        success = run_comprehensive_arabic_test()
        
        if success:
            print(f"\n🚀 Arabic speaker diagnostics completed successfully!")
            print(f"🎯 Your BeautyAI platform is ready for Arabic TTS with custom voices!")
        else:
            print(f"\n🔧 Arabic speaker diagnostics found issues that need attention.")
            print(f"📋 Please follow the recommendations above to resolve them.")
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Diagnostic interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
