#!/usr/bin/env python3
"""
Recreate Arabic Speaker Profiles with Correct Transcriptions.
This script properly creates speaker profiles using the actual transcriptions
from the provided Arabic audio files.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Import torch for CUDA detection
try:
    import torch
except ImportError:
    torch = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def create_arabic_speaker_with_transcript(
    audio_path: str, 
    transcript: str, 
    speaker_name: str,
    gender: str
) -> Optional[str]:
    """Create Arabic speaker profile with correct transcription."""
    
    logger.info(f"🎭 Creating {gender} Arabic speaker: {speaker_name}")
    logger.info(f"📁 Audio: {audio_path}")
    logger.info(f"📝 Transcript: {transcript[:50]}...")
    
    try:
        # Import OuteTTS
        import outetts
        
        # Initialize OuteTTS interface with optimal settings
        logger.info("📥 Initializing OuteTTS interface with optimal configuration...")
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,  # Latest and best model
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16  # Best quality
            )
        )
        
        # Verify audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create output directory
        profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Create speaker profile path
        speaker_profile_path = profiles_dir / f"{speaker_name}.json"
        
        logger.info("🎤 Creating speaker profile with provided transcript...")
        logger.info("⏱️  This may take a few minutes...")
        
        # Create speaker using the correct transcript
        speaker = interface.create_speaker(
            audio_path=audio_path,
            transcript=transcript,
            whisper_model="turbo",  # Use Whisper for transcription verification
            whisper_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Save the speaker profile
        interface.save_speaker(speaker, str(speaker_profile_path))
        
        logger.info(f"✅ Speaker profile created: {speaker_profile_path}")
        
        # Test the profile with a sample from the original transcript
        test_sample = transcript.split()[:10]  # First 10 words
        test_text = " ".join(test_sample)
        
        logger.info(f"🔊 Testing speaker with original content: '{test_text}'")
        
        # Generate test audio with optimal settings
        test_output = interface.generate(
            config=outetts.GenerationConfig(
                text=test_text,
                generation_type=outetts.GenerationType.CHUNKED,  # Best for single texts
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.3,  # Lower for more consistent voice
                    top_p=0.85,      # Balanced control
                    top_k=40,        # Focused selection
                    repetition_penalty=1.05  # Prevent repetition
                ),
                max_length=8192
            )
        )
        
        # Save test output
        test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / f"transcript_test_{gender}_{speaker_name}.wav"
        test_output.save(str(test_file))
        
        logger.info(f"✅ Test audio saved: {test_file}")
        
        return str(speaker_profile_path)
        
    except ImportError:
        logger.error("❌ OuteTTS not available. Install with: pip install outetts")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create {gender} speaker profile: {e}")
        return None

def test_beauty_clinic_scenarios(female_profile: str, male_profile: str):
    """Test both profiles with beauty clinic scenarios."""
    
    logger.info("🏥 Testing beauty clinic scenarios...")
    
    # Beauty clinic scenarios
    beauty_scenarios = {
        "female_greeting": "مرحباً بك في عيادة الجمال. كيف يمكنني مساعدتك اليوم؟",
        "female_services": "لدينا علاجات متنوعة للوجه والبشرة وحقن البوتوكس.",
        "male_greeting": "أهلاً وسهلاً، أنا الدكتور أحمد. سأساعدك في اختيار العلاج المناسب.",
        "male_consultation": "يمكننا مناقشة خيارات العلاج المختلفة بناءً على احتياجاتك."
    }
    
    try:
        import outetts
        
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
            )
        )
        
        test_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
        
        # Test female scenarios
        if female_profile and os.path.exists(female_profile):
            logger.info("👩 Testing female speaker scenarios...")
            female_speaker = interface.load_speaker(female_profile)
            
            for scenario, text in beauty_scenarios.items():
                if scenario.startswith("female_"):
                    logger.info(f"   🎯 {scenario}: {text[:30]}...")
                    
                    output = interface.generate(
                        config=outetts.GenerationConfig(
                            text=text,
                            generation_type=outetts.GenerationType.CHUNKED,  # Best for single texts
                            speaker=female_speaker,
                            sampler_config=outetts.SamplerConfig(
                                temperature=0.3,  # Lower for consistency
                                top_p=0.85,      # Better control
                                top_k=40,        # Focused selection
                                repetition_penalty=1.05
                            ),
                            max_length=8192
                        )
                    )
                    
                    output_file = test_dir / f"beauty_clinic_{scenario}.wav"
                    output.save(str(output_file))
                    logger.info(f"   ✅ Saved: {output_file.name}")
        
        # Test male scenarios only if male profile exists
        if male_profile and os.path.exists(male_profile):
            logger.info("👨 Testing male speaker scenarios...")
            male_speaker = interface.load_speaker(male_profile)
            
            for scenario, text in beauty_scenarios.items():
                if scenario.startswith("male_"):
                    logger.info(f"   🎯 {scenario}: {text[:30]}...")
                    
                    output = interface.generate(
                        config=outetts.GenerationConfig(
                            text=text,
                            generation_type=outetts.GenerationType.CHUNKED,
                            speaker=male_speaker,
                            sampler_config=outetts.SamplerConfig(
                                temperature=0.4,
                                top_p=0.9,
                                top_k=50
                            ),
                        )
                    )
                    
                    output_file = test_dir / f"beauty_clinic_{scenario}.wav"
                    output.save(str(output_file))
                    logger.info(f"   ✅ Saved: {output_file.name}")
        else:
            logger.info("👨 Skipping male speaker scenarios (no male profile)")
        
        logger.info("✅ Beauty clinic scenario testing completed")
        
    except Exception as e:
        logger.error(f"❌ Beauty clinic testing failed: {e}")

def main():
    """Main function to recreate Arabic speaker profiles."""
    
    print("🔄 Creating Premium Arabic Female Speaker Profile")
    print("="*80)
    
    # Check if torch is available for CUDA detection
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, will use CPU for Whisper")
    
    # Clean old profiles
    clean_old_profiles()
    
    # Use premium female voice from egy2k dataset (highest quality)
    audio_data = {
        "female": {
            "path": "/home/lumi/beautyai/voice_tests/custom_speakers/arabic_female_premium.wav",
            "transcript": "والله يكنت أتمنى أحكي لك قصة مثيرة جدا عن طريقة اكتشاف الأدوية المتطورة جدا لعلاج السمنة في الطب الحديث ونقعد كده نشوف الكرياتيبتي وشغلها عامل إزاي وإزاي قدر المتخصصين يقلفوا التركيبة الكيميائية",
            "speaker_name": "arabic_female_premium",
            "description": "Premium Egyptian Arabic female voice with medical/health expertise"
        }
    }
    
    # Verify audio files exist
    for gender, data in audio_data.items():
        if not os.path.exists(data["path"]):
            logger.error(f"❌ {gender.capitalize()} audio file not found: {data['path']}")
            logger.info("💡 Premium voice sample should be available in custom_speakers directory")
            return False
    
    created_profiles = {}
    
    # Create female speaker profile with premium quality
    logger.info(f"\n🎭 Creating premium Arabic female speaker profile...")
    data = audio_data["female"]
    
    profile_path = create_arabic_speaker_with_transcript(
        audio_path=data["path"],
        transcript=data["transcript"],
        speaker_name=data["speaker_name"],
        gender="female"
    )
    
    if profile_path:
        created_profiles["female"] = profile_path
        logger.info(f"✅ Premium female profile created successfully")
    else:
        logger.error(f"❌ Failed to create premium female profile")
    
    # Test with beauty clinic scenarios (female only)
    if len(created_profiles) > 0:
        logger.info("\n🏥 Testing with beauty clinic scenarios...")
        test_beauty_clinic_scenarios(
            female_profile=created_profiles.get("female"),
            male_profile=None  # No male profile needed
        )
    
    # Summary
    print(f"\n📋 Recreation Summary")
    print("="*80)
    
    if len(created_profiles) == 1:
        print("🎉 SUCCESS: Premium Arabic female speaker profile created successfully!")
        print(f"👩 Female profile: {created_profiles['female']}")
        print(f" Test files saved in: /home/lumi/beautyai/voice_tests/arabic_speaker_tests/")
        print("\n💡 Key improvements:")
        print("   ✅ Used premium voice sample from egy2k dataset")
        print("   ✅ Medical/health context matching beauty clinic scenarios")
        print("   ✅ Focused on single high-quality female voice")
        print("   ✅ Better voice quality and accuracy expected")
        print("   ✅ Tested with beauty clinic scenarios")
        return True
    else:
        print("❌ FAILED: Could not create premium female speaker profile")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
