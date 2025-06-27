#!/usr/bin/env python3
"""
Arabic Speaker Setup Script for BeautyAI Framework.
Creates custom Arabic speaker profiles from provided audio files.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_arabic_speakers(female_audio_path: str = None, male_audio_path: str = None):
    """
    Setup Arabic speaker profiles for the BeautyAI platform.
    
    Args:
        female_audio_path: Path to female Arabic audio sample
        male_audio_path: Path to male Arabic audio sample
    """
    print("🎭 Setting up Arabic Speaker Profiles for BeautyAI")
    print("="*60)
    
    try:
        # Import BeautyAI components
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
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
        
        created_speakers = {}
        
        # Create female Arabic speaker
        if female_audio_path and os.path.exists(female_audio_path):
            print(f"\n🎤 Creating female Arabic speaker from: {female_audio_path}")
            
            female_profile = tts_service.create_arabic_speaker_profile(
                audio_file_path=female_audio_path,
                speaker_name="beautyai_arabic_female",
                transcript=None  # Will auto-transcribe
            )
            
            if female_profile:
                created_speakers["female"] = female_profile
                print(f"✅ Female Arabic speaker created: {female_profile}")
                
                # Test the female speaker
                test_output = tts_service.test_arabic_speaker(
                    speaker_profile_path=female_profile,
                    test_text="مرحبا بكم في منصة BeautyAI للذكاء الاصطناعي والجمال"
                )
                if test_output:
                    print(f"✅ Female speaker test successful: {test_output}")
            else:
                print("❌ Failed to create female Arabic speaker")
        else:
            print(f"⚠️ Female audio file not found: {female_audio_path}")
        
        # Create male Arabic speaker
        if male_audio_path and os.path.exists(male_audio_path):
            print(f"\n🎤 Creating male Arabic speaker from: {male_audio_path}")
            
            male_profile = tts_service.create_arabic_speaker_profile(
                audio_file_path=male_audio_path,
                speaker_name="beautyai_arabic_male",
                transcript=None  # Will auto-transcribe
            )
            
            if male_profile:
                created_speakers["male"] = male_profile
                print(f"✅ Male Arabic speaker created: {male_profile}")
                
                # Test the male speaker
                test_output = tts_service.test_arabic_speaker(
                    speaker_profile_path=male_profile,
                    test_text="أهلا وسهلا، هذا اختبار للصوت الذكوري العربي في منصة BeautyAI"
                )
                if test_output:
                    print(f"✅ Male speaker test successful: {test_output}")
            else:
                print("❌ Failed to create male Arabic speaker")
        else:
            print(f"⚠️ Male audio file not found: {male_audio_path}")
        
        # Setup as default speakers
        if created_speakers:
            print(f"\n🔧 Setting up as default Arabic speakers...")
            default_speakers = tts_service.setup_default_arabic_speakers(
                female_audio_path=female_audio_path if "female" in created_speakers else None,
                male_audio_path=male_audio_path if "male" in created_speakers else None
            )
            
            print(f"✅ Default Arabic speakers configured: {default_speakers}")
            
            # Test the complete setup
            print(f"\n🧪 Testing complete Arabic speaker setup...")
            test_text = "مرحبا، تم إعداد الأصوات العربية بنجاح في منصة BeautyAI للذكاء الاصطناعي"
            
            for gender in ["female", "male"]:
                if gender in created_speakers:
                    output_path = f"/home/lumi/beautyai/voice_tests/final_test_arabic_{gender}.wav"
                    result = tts_service.text_to_speech(
                        text=test_text,
                        language="ar",
                        speaker_voice=gender,
                        output_path=output_path
                    )
                    
                    if result:
                        print(f"✅ Final {gender} speaker test successful: {result}")
                    else:
                        print(f"❌ Final {gender} speaker test failed")
        
        print(f"\n" + "="*60)
        print("🎊 ARABIC SPEAKER SETUP COMPLETE")
        print("="*60)
        print(f"✅ Created speakers: {list(created_speakers.keys())}")
        print("📁 Speaker profiles saved in: /home/lumi/beautyai/voice_tests/custom_speakers/")
        print("🎤 Test audio files saved in: /home/lumi/beautyai/voice_tests/")
        print("\n🚀 Your BeautyAI platform now supports high-quality Arabic speech synthesis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Arabic speaker setup failed: {e}")
        logger.exception("Full error details:")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup Arabic Speaker Profiles for BeautyAI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup both female and male speakers
  python setup_arabic_speakers.py --female /path/to/arabic_female.wav --male /path/to/arabic_male.wav
  
  # Setup only female speaker
  python setup_arabic_speakers.py --female /path/to/arabic_female.wav
  
  # Setup only male speaker
  python setup_arabic_speakers.py --male /path/to/arabic_male.wav

Audio Requirements:
  - WAV format recommended (MP3, OGG also supported)
  - Clear Arabic speech, 5-30 seconds long
  - Single speaker per file
  - Good audio quality (16kHz+ sample rate)
        """
    )
    
    parser.add_argument("--female", "-f", 
                        help="Path to female Arabic audio file")
    parser.add_argument("--male", "-m", 
                        help="Path to male Arabic audio file")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run diagnostic tests, don't create speakers")
    
    args = parser.parse_args()
    
    if args.test_only:
        print("🧪 Running diagnostic tests only...")
        # Import and run the diagnostic script
        try:
            import diagnose_outetts_speakers_enhanced
            # The main function will run the diagnostics
        except Exception as e:
            print(f"❌ Diagnostic test failed: {e}")
        return
    
    if not args.female and not args.male:
        print("❌ Error: At least one audio file (--female or --male) must be provided")
        print("Use --help for more information")
        return
    
    # Validate audio files
    audio_files = []
    if args.female:
        if os.path.exists(args.female):
            print(f"✅ Female audio file found: {args.female}")
            audio_files.append(("female", args.female))
        else:
            print(f"❌ Female audio file not found: {args.female}")
            return
    
    if args.male:
        if os.path.exists(args.male):
            print(f"✅ Male audio file found: {args.male}")
            audio_files.append(("male", args.male))
        else:
            print(f"❌ Male audio file not found: {args.male}")
            return
    
    print(f"\n🎯 Setting up {len(audio_files)} Arabic speaker(s)...")
    
    # Setup Arabic speakers
    success = setup_arabic_speakers(
        female_audio_path=args.female,
        male_audio_path=args.male
    )
    
    if success:
        print("\n🎉 Arabic speaker setup completed successfully!")
        print("Your BeautyAI platform is now ready for Arabic TTS!")
    else:
        print("\n💥 Arabic speaker setup failed!")
        print("Check the error messages above and try again.")


if __name__ == "__main__":
    main()
