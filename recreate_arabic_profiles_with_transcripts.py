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
        
        # Initialize OuteTTS interface
        logger.info("📥 Initializing OuteTTS interface...")
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
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
        
        # Generate test audio
        test_output = interface.generate(
            config=outetts.GenerationConfig(
                text=test_text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.4,
                    top_p=0.9,
                    top_k=50
                ),
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
                            generation_type=outetts.GenerationType.CHUNKED,
                            speaker=female_speaker,
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
        
        # Test male scenarios
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
        
        logger.info("✅ Beauty clinic scenario testing completed")
        
    except Exception as e:
        logger.error(f"❌ Beauty clinic testing failed: {e}")

def main():
    """Main function to recreate Arabic speaker profiles."""
    
    print("🔄 Recreating Arabic Speaker Profiles with Correct Transcriptions")
    print("="*80)
    
    # Check if torch is available for CUDA detection
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, will use CPU for Whisper")
    
    # Clean old profiles
    clean_old_profiles()
    
    # Audio files and their correct transcriptions
    audio_data = {
        "female": {
            "path": "/home/lumi/beautyai/voice_tests/trimmed_audio/arabic_female_15s.wav",
            "transcript": "يعني هي في البدايات كانت ايوه توجع، بس الان لا، بالعكس يعني صار الشعور اجمل بكثير لما الناس يشاركونا الصور حقتهم، كيف اهدوها لاحد، لشخص عزيز او كيف موجوده عندهم في البيت، يعني انا",
            "speaker_name": "arabic_female_corrected"
        },
        "male": {
            "path": "/home/lumi/beautyai/voice_tests/trimmed_audio/arabic_male_15s.wav",
            "transcript": "وفيها خير ولكن لكل مجتهد نصيب، الرزق راح يجيك ان شاء الله بس انت ما تقعد في البيت. فانا اتمنى من المرشد السياحي تعقيبا على سؤالك، انه هو يكون زي المراسل التلفزيوني. ان يكون امين وحريص على نقل المعلومه لما يقولها للشخص",
            "speaker_name": "arabic_male_corrected"
        }
    }
    
    # Verify audio files exist
    for gender, data in audio_data.items():
        if not os.path.exists(data["path"]):
            logger.error(f"❌ {gender.capitalize()} audio file not found: {data['path']}")
            logger.info("💡 Run trim_audio_samples.py first")
            return False
    
    created_profiles = {}
    
    # Create speaker profiles with correct transcriptions
    for gender, data in audio_data.items():
        logger.info(f"\n🎭 Creating {gender} Arabic speaker profile...")
        
        profile_path = create_arabic_speaker_with_transcript(
            audio_path=data["path"],
            transcript=data["transcript"],
            speaker_name=data["speaker_name"],
            gender=gender
        )
        
        if profile_path:
            created_profiles[gender] = profile_path
            logger.info(f"✅ {gender.capitalize()} profile created successfully")
        else:
            logger.error(f"❌ Failed to create {gender} profile")
    
    # Test with beauty clinic scenarios
    if len(created_profiles) > 0:
        logger.info("\n🏥 Testing with beauty clinic scenarios...")
        test_beauty_clinic_scenarios(
            female_profile=created_profiles.get("female"),
            male_profile=created_profiles.get("male")
        )
    
    # Summary
    print(f"\n📋 Recreation Summary")
    print("="*80)
    
    if len(created_profiles) == 2:
        print("🎉 SUCCESS: Both Arabic speaker profiles recreated successfully!")
        print(f"👩 Female profile: {created_profiles['female']}")
        print(f"👨 Male profile: {created_profiles['male']}")
        print(f"📁 Test files saved in: /home/lumi/beautyai/voice_tests/arabic_speaker_tests/")
        print("\n💡 Key improvements:")
        print("   ✅ Used correct transcriptions from your audio files")
        print("   ✅ Profiles now match the actual voice content")
        print("   ✅ Better voice quality and accuracy expected")
        print("   ✅ Tested with beauty clinic scenarios")
        return True
    else:
        print("❌ PARTIAL SUCCESS: Some profiles failed to create")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
