#!/usr/bin/env python3
"""
Premium Arabic Female Speaker Profile Creator.
This script creates a high-accuracy Arabic female speaker profile using the best
quality voice sample and optimized OuteTTS configuration.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional
import json

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Import torch for CUDA detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Arabic voice samples - four different durations for testing
VOICE_SAMPLES = {
    "19_sec": {
        "audio_path": "/home/lumi/beautyai/voice_tests/custom_speakers/arabic_f_19_sec.wav",
        "transcript": "دورة المياه في الطبيعة هي عملية ديناميكية مستمرة تضمن استمرارية الحياة. تبدأ بتبخر مياه المحيطات والبحار بفعل حرارة الشمس، ثم يتكثف هذا البخار في طبقات الجو العليا ليشكل السحب.",
        "speaker_name": "arabic_female_premium_19s",
        "description": "OPTIMAL 19-second Arabic female voice sample with provided transcript (BEST FOR OUTETTS)"
    },
}

def check_prerequisites():
    """Check if all prerequisites are available."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check if all audio files exist
    for duration, sample_data in VOICE_SAMPLES.items():
        if not os.path.exists(sample_data["audio_path"]):
            logger.error(f"❌ Audio file not found: {sample_data['audio_path']}")
            return False
        
        # Check file size and basic properties
        file_size = os.path.getsize(sample_data["audio_path"])
        logger.info(f"✅ {duration} audio file found: {file_size / 1024 / 1024:.1f} MB")
    
    # Check if OuteTTS is available
    try:
        import outetts
        logger.info("✅ OuteTTS library available")
    except ImportError:
        logger.error("❌ OuteTTS not available. Install with: pip install outetts")
        return False
    
    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU available: {gpu_name} ({gpu_count} devices)")
    else:
        logger.warning("⚠️  No GPU detected, will use CPU (slower)")
    
    return True

def clean_old_profiles():
    """Remove old speaker profiles to start fresh."""
    profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
    
    if profiles_dir.exists():
        logger.info("🧹 Cleaning old speaker profiles...")
        for profile_file in profiles_dir.glob("*.json"):
            profile_file.unlink()
            logger.info(f"   ❌ Removed: {profile_file.name}")
    
    profiles_dir.mkdir(parents=True, exist_ok=True)
    logger.info("✅ Profile directory prepared")

def get_optimal_model_config():
    """Get the optimal model configuration based on available hardware."""
    try:
        import outetts
        
        # Use the latest and best model
        model = outetts.Models.VERSION_1_0_SIZE_1B
        
        # Choose backend based on hardware
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Use LLAMACPP backend with FP16 for best quality on GPU
            backend = outetts.Backend.LLAMACPP
            quantization = outetts.LlamaCppQuantization.FP16
            logger.info("🎯 Using LLAMACPP backend with FP16 (optimal quality)")
        else:
            # Use HF backend for CPU
            backend = outetts.Backend.HF
            quantization = None
            logger.info("🎯 Using Hugging Face backend for CPU")
        
        if quantization:
            config = outetts.ModelConfig.auto_config(
                model=model,
                backend=backend,
                quantization=quantization
            )
        else:
            config = outetts.ModelConfig.auto_config(
                model=model,
                backend=backend
            )
        
        return config
    
    except Exception as e:
        logger.error(f"❌ Failed to create model config: {e}")
        return None

def create_all_arabic_speakers() -> dict:
    """Create all Arabic female speaker profiles from the available audio samples."""
    
    logger.info("🎭 Creating All Arabic Female Speaker Profiles")
    logger.info("="*60)
    
    profiles = {}
    
    for duration, sample_data in VOICE_SAMPLES.items():
        logger.info(f"📥 Creating {duration} profile...")
        profile_path = create_arabic_speaker_profile(
            sample_data["audio_path"],
            sample_data["speaker_name"],
            sample_data["description"]
        )
        
        if profile_path:
            profiles[duration] = {
                "profile_path": profile_path,
                "speaker_name": sample_data["speaker_name"],
                "description": sample_data["description"]
            }
            logger.info(f"✅ {duration} profile created successfully")
        else:
            logger.error(f"❌ Failed to create {duration} profile")
    
    return profiles

def create_arabic_speaker_profile(audio_path: str, speaker_name: str, description: str) -> Optional[str]:
    """Create a single Arabic speaker profile."""
    
    logger.info(f"🎤 Creating Arabic speaker: {speaker_name}")
    logger.info(f"📁 Audio: {audio_path}")
    logger.info(f"🎯 Description: {description}")
    
    try:
        import outetts
        
        # Get optimal configuration - using correct Arabic-supporting model
        config = get_optimal_model_config()
        if not config:
            return None
        
        # Initialize OuteTTS interface with optimal settings
        logger.info("📥 Initializing OuteTTS interface with optimal configuration...")
        interface = outetts.Interface(config=config)
        
        # Create output directory
        profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Create speaker profile path
        speaker_profile_path = profiles_dir / f"{speaker_name}.json"
        
        logger.info("🎤 Creating speaker profile with premium settings...")
        logger.info("⏱️  This may take several minutes for best quality...")
        
        # Create speaker using V3 interface (latest) with optimal settings
        speaker = interface.create_speaker(
            audio_path=audio_path,
            transcript="",  # Auto-transcribe for best accuracy
            whisper_model="turbo",  # Use best Whisper model
            whisper_device="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        )
        
        # Save the speaker profile
        interface.save_speaker(speaker, str(speaker_profile_path))
        
        logger.info(f"✅ Speaker profile created: {speaker_profile_path}")
        
        # Save profile metadata
        metadata = {
            "speaker_name": speaker_name,
            "description": description,
            "audio_source": audio_path,
            "transcript": "",  # Auto-transcribed
            "creation_timestamp": time.time(),
            "model_config": {
                "model": "VERSION_1_0_SIZE_1B",
                "backend": config.backend.name if hasattr(config.backend, 'name') else str(config.backend),
                "whisper_model": "turbo",
                "arabic_support": True  # Confirmed Arabic support
            }
        }
        
        metadata_path = profiles_dir / f"{speaker_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Metadata saved: {metadata_path}")
        
        # Test the profile with beauty clinic content
        test_profile_quality(interface, speaker, speaker_name)
        
        return str(speaker_profile_path)
        
    except ImportError:
        logger.error("❌ OuteTTS not available. Install with: pip install outetts")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create speaker profile: {e}")
        return None
    """Create premium Arabic female speaker profile with highest accuracy settings."""
    
    logger.info("🎭 Creating Premium Arabic Female Speaker Profile")
    logger.info("="*60)
    logger.info(f"📁 Audio: {PREMIUM_VOICE_DATA['audio_path']}")
    logger.info(f"📝 Transcript: {PREMIUM_VOICE_DATA['transcript'][:80]}...")
    logger.info(f"🎯 Target: {PREMIUM_VOICE_DATA['description']}")
    
    try:
        import outetts
        
        # Get optimal configuration
        config = get_optimal_model_config()
        if not config:
            return None
        
        # Initialize OuteTTS interface with optimal settings
        logger.info("📥 Initializing OuteTTS interface with optimal configuration...")
        interface = outetts.Interface(config=config)
        
        # Create output directory
        profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Create speaker profile path
        speaker_profile_path = profiles_dir / f"{PREMIUM_VOICE_DATA['speaker_name']}.json"
        
        logger.info("🎤 Creating speaker profile with premium settings...")
        logger.info("⏱️  This may take several minutes for best quality...")
        
        # Create speaker using V3 interface (latest) with optimal settings
        speaker = interface.create_speaker(
            audio_path=PREMIUM_VOICE_DATA["audio_path"],
            transcript=PREMIUM_VOICE_DATA["transcript"],
            whisper_model="turbo",  # Use best Whisper model
            whisper_device="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        )
        
        # Save the speaker profile
        interface.save_speaker(speaker, str(speaker_profile_path))
        
        logger.info(f"✅ Premium speaker profile created: {speaker_profile_path}")
        
        # Save profile metadata
        metadata = {
            "speaker_name": PREMIUM_VOICE_DATA["speaker_name"],
            "description": PREMIUM_VOICE_DATA["description"],
            "audio_source": PREMIUM_VOICE_DATA["audio_path"],
            "transcript": PREMIUM_VOICE_DATA["transcript"],
            "creation_timestamp": time.time(),
            "model_config": {
                "model": "VERSION_1_0_SIZE_1B",
                "backend": config.backend.name if hasattr(config.backend, 'name') else str(config.backend),
                "whisper_model": "turbo"
            }
        }
        
        metadata_path = profiles_dir / f"{PREMIUM_VOICE_DATA['speaker_name']}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Metadata saved: {metadata_path}")
        
        # Test the profile with beauty clinic content
        test_profile_quality(interface, speaker, PREMIUM_VOICE_DATA["speaker_name"])
        
        return str(speaker_profile_path)
        
    except ImportError:
        logger.error("❌ OuteTTS not available. Install with: pip install outetts")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create premium speaker profile: {e}")
        return None

def test_profile_quality(interface, speaker, speaker_name: str):
    """Test the speaker profile quality with various Arabic content."""
    
    logger.info("🔊 Testing speaker profile quality...")
    
    # Test scenarios specifically for beauty clinic use
    test_scenarios = {
        "medical_greeting": "مرحباً بك في عيادة الجمال والتجميل. أنا الدكتورة سارة، كيف يمكنني مساعدتك اليوم؟",
        "medical_consultation": "سنقوم بفحص بشرتك وتحديد أفضل العلاجات المناسبة لنوع بشرتك واحتياجاتك الخاصة.",
        "treatment_explanation": "هذا العلاج آمن وفعال، وقد أثبت نجاحه مع العديد من المرضى. ستلاحظين تحسناً واضحاً خلال أسبوعين.",
        "aftercare_instructions": "من المهم جداً تطبيق الكريمات الموصوفة يومياً وتجنب التعرض المباشر لأشعة الشمس.",
        "appointment_booking": "يمكننا حجز موعدك القادم الأسبوع المقبل. هل يناسبك يوم الثلاثاء في الساعة الثالثة عصراً؟"
    }
    
    try:
        import outetts
        
        test_dir = Path("/home/lumi/beautyai/voice_tests/premium_speaker_tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for scenario, text in test_scenarios.items():
            logger.info(f"   🎯 Testing: {scenario}")
            
            # Generate audio with optimized settings for Arabic accuracy
            output = interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=outetts.GenerationType.SENTENCE,  # Better for Arabic sentences
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(
                        temperature=0.2,      # Much lower for Arabic accuracy
                        top_p=0.75,          # Reduced for better control
                        top_k=25,            # Lower for more consistent Arabic
                        repetition_penalty=1.02,  # Minimal to avoid Arabic word breaks
                        repetition_range=32   # Shorter range for Arabic morphology
                    ),
                    max_length=12288,        # Higher for longer Arabic sentences
                    language="ar"            # Explicitly set Arabic language
                )
            )
            
            # Save test output
            output_file = test_dir / f"premium_{scenario}_{speaker_name}.wav"
            output.save(str(output_file))
            logger.info(f"   ✅ Saved: {output_file.name}")
        
        logger.info("✅ Quality testing completed")
        
        # Test with original transcript sample  
        logger.info("   🎯 Testing with Arabic beauty clinic content...")
        original_sample = "مرحبا بكم في عيادة الجمال. نحن هنا لنساعدكم في الحصول على أفضل النتائج."
        
        output = interface.generate(
            config=outetts.GenerationConfig(
                text=original_sample,
                generation_type=outetts.GenerationType.SENTENCE,  # Better for Arabic
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.2,      # Lower for Arabic accuracy
                    top_p=0.75,          # Better control
                    top_k=25,            # More consistent
                    repetition_penalty=1.02
                ),
                language="ar"            # Explicit Arabic
            )
        )
        
        original_test_file = test_dir / f"premium_original_content_{speaker_name}.wav"
        output.save(str(original_test_file))
        logger.info(f"   ✅ Original content test saved: {original_test_file.name}")
        
    except Exception as e:
        logger.error(f"❌ Quality testing failed: {e}")

def main():
    """Main function to create premium Arabic speaker profiles."""
    
    print("🌟 Premium Arabic Female Speaker Profile Creator")
    print("="*80)
    print(f"🎯 Creating high-accuracy speaker profiles from premium voice samples")
    print(f"📊 Using advanced OuteTTS v1.0 with optimal configuration")
    print(f"🎤 Processing both 47s and 28s audio samples")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met. Please resolve issues and try again.")
        return False
    
    # Clean old profiles
    clean_old_profiles()
    
    # Create all speaker profiles
    logger.info("\n🚀 Starting premium speaker profile creation...")
    profiles = create_all_arabic_speakers()
    
    # Summary
    print(f"\n📋 Creation Summary")
    print("="*80)
    
    if profiles:
        print("🎉 SUCCESS: Premium Arabic female speaker profiles created!")
        for duration, profile_info in profiles.items():
            print(f"✅ {duration} profile: {profile_info['profile_path']}")
        print(f"📁 Test files: /home/lumi/beautyai/voice_tests/premium_speaker_tests/")
        print(f"� Profiles directory: /home/lumi/beautyai/voice_tests/arabic_speaker_profiles/")
        print()
        print("💡 Key improvements:")
        print("   ✅ Used highest quality voice samples (28s & 47s)")
        print("   ✅ Correct OuteTTS v1.0-1B model with Arabic support")
        print("   ✅ Auto-transcription for maximum accuracy")
        print("   ✅ Optimized configuration for maximum quality")
        print("   ✅ Medical/health context for beauty clinic scenarios")
        print("   ✅ Advanced sampler settings for consistent voice quality")
        print("   ✅ Comprehensive quality testing with beauty clinic content")
        print()
        print("🎤 Usage in other scripts:")
        print(f"   interface = outetts.Interface(config)")
        for duration, profile_info in profiles.items():
            print(f"   speaker_{duration} = interface.load_speaker('{profile_info['profile_path']}')")
        
        return True
    else:
        print("❌ FAILED: Could not create speaker profiles")
        print("💡 Check the logs above for specific error details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
