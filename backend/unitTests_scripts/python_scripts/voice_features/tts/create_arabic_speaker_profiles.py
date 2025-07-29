#!/usr/bin/env python3
"""
Arabic Speaker Profile Creator for BeautyAI Platform.

Creates custom Arabic speaker profiles from provided audio samples using OuteTTS.
This script uses the provided audio_1_F.wav (female) and audio_1_M.wav (male) 
to create high-quality Arabic speaker profiles for the BeautyAI TTS system.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_audio_file(audio_path: str) -> Dict[str, Any]:
    """Analyze audio file properties."""
    try:
        import soundfile as sf
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Read audio file
        data, samplerate = sf.read(audio_path)
        duration = len(data) / samplerate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        file_size = audio_file.stat().st_size
        
        return {
            "path": str(audio_file),
            "sample_rate": samplerate,
            "duration": duration,
            "channels": channels,
            "file_size": file_size,
            "quality": "Optimal" if samplerate >= 22050 else "Good" if samplerate >= 16000 else "Low"
        }
        
    except ImportError:
        logger.warning("soundfile not available, install with: pip install soundfile")
        return {"error": "soundfile library not available"}
    except Exception as e:
        return {"error": f"Could not analyze audio: {e}"}

def convert_audio_to_optimal_format(input_path: str, output_path: str) -> bool:
    """Convert audio to optimal format for OuteTTS (22kHz, mono, 16-bit)."""
    try:
        import subprocess
        
        # FFmpeg command for optimal conversion
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", "22050",           # 22kHz sample rate
            "-ac", "1",               # Mono channel
            "-sample_fmt", "s16",     # 16-bit depth
            "-af", "highpass=f=80,lowpass=f=8000",  # Clean frequency range
            "-y",                     # Overwrite output
            output_path
        ]
        
        logger.info(f"Converting {input_path} to optimal format...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if Path(output_path).exists():
            logger.info(f"✅ Audio converted successfully: {output_path}")
            return True
        else:
            logger.error("❌ Conversion failed - output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg conversion failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found. Install with: sudo apt install ffmpeg")
        return False
    except Exception as e:
        logger.error(f"❌ Conversion error: {e}")
        return False

def create_arabic_speaker_profile(audio_path: str, speaker_name: str, gender: str) -> Optional[str]:
    """Create a custom Arabic speaker profile from audio file."""
    
    logger.info(f"🎭 Creating Arabic speaker profile: {speaker_name}")
    logger.info(f"📁 Audio file: {audio_path}")
    
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
        
        logger.info("✅ OuteTTS interface initialized")
        
        # Create output directory for speaker profiles
        profiles_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles")
        profiles_dir.mkdir(exist_ok=True)
        
        # Create speaker profile path
        speaker_profile_path = profiles_dir / f"{speaker_name}.json"
        
        # Create custom speaker from audio sample
        logger.info(f"🎤 Creating speaker profile from audio sample...")
        logger.info("⏱️  This may take a few minutes for processing...")
        
        speaker = interface.create_speaker(str(audio_path))
        
        # Save the speaker profile
        interface.save_speaker(speaker, str(speaker_profile_path))
        
        logger.info(f"✅ Arabic speaker profile created successfully!")
        logger.info(f"📄 Profile saved to: {speaker_profile_path}")
        
        # Test the new speaker with Arabic text
        logger.info(f"\n🔊 Testing new Arabic speaker: {speaker_name}")
        
        # Arabic test texts for different scenarios
        test_texts = {
            "greeting": "مرحباً بك في عيادة الجمال. كيف يمكنني مساعدتك اليوم؟",
            "services": "لدينا علاجات متنوعة للوجه والبشرة وحقن البوتوكس.",
            "appointment": "هل تريد حجز موعد لعلاج البشرة أو استشارة طبية؟"
        }
        
        # Create test outputs directory
        test_outputs_dir = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_tests")
        test_outputs_dir.mkdir(exist_ok=True)
        
        # Test each scenario
        for scenario, text in test_texts.items():
            try:
                logger.info(f"🎯 Testing scenario: {scenario}")
                
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
                
                # Save test output
                test_output_path = test_outputs_dir / f"{speaker_name}_{scenario}_test.wav"
                output.save(str(test_output_path))
                
                if test_output_path.exists():
                    file_size = test_output_path.stat().st_size
                    logger.info(f"✅ Test successful: {test_output_path} ({file_size} bytes)")
                else:
                    logger.warning(f"⚠️ Test file not created: {test_output_path}")
                    
            except Exception as e:
                logger.error(f"❌ Test scenario '{scenario}' failed: {e}")
        
        return str(speaker_profile_path)
        
    except ImportError:
        logger.error("❌ OuteTTS library not available. Install with: pip install outetts")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create Arabic speaker profile: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_oute_tts_engine_with_profiles(female_profile: str, male_profile: str):
    """Update the OuteTTS engine with the new Arabic speaker profiles."""
    
    logger.info("🔧 Updating OuteTTS engine with new Arabic speaker profiles...")
    
    try:
        from beautyai_inference.inference_engines.oute_tts_engine import OuteTTSEngine
        from beautyai_inference.config.config_manager import ModelConfig
        
        # Create a test engine instance to update the mapping
        model_config = ModelConfig(
            model_id="oute-tts-1b",
            engine_type="oute_tts",
            quantization="fp16"
        )
        
        # This will create the profiles directory and mapping
        engine = OuteTTSEngine(model_config)
        
        # Update the Arabic speaker mapping with new profiles
        if female_profile and Path(female_profile).exists():
            engine.arabic_speaker_mapping["ar-female"] = female_profile
            logger.info(f"✅ Female Arabic speaker registered: {female_profile}")
        
        if male_profile and Path(male_profile).exists():
            engine.arabic_speaker_mapping["ar-male"] = male_profile
            logger.info(f"✅ Male Arabic speaker registered: {male_profile}")
        
        # Save the updated mapping to a persistent file
        mapping_file = Path("/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/speaker_mapping.json")
        
        import json
        mapping_data = {
            "arabic_speakers": engine.arabic_speaker_mapping,
            "created_at": str(Path().absolute()),
            "profiles": {
                "female": female_profile if female_profile else None,
                "male": male_profile if male_profile else None
            }
        }
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Speaker mapping saved to: {mapping_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update OuteTTS engine: {e}")
        return False

def test_integration_with_tts_service(female_profile: str, male_profile: str):
    """Test integration with the BeautyAI TTS service."""
    
    logger.info("🧪 Testing integration with BeautyAI TTS service...")
    
    try:
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Initialize TTS service
        tts_service = TextToSpeechService()
        
        # Load the OuteTTS model
        success = tts_service.load_tts_model("oute-tts-1b")
        if not success:
            logger.error("❌ Failed to load TTS model")
            return False
        
        # Register custom speakers if engine is available
        if tts_service.oute_tts_engine:
            if female_profile and Path(female_profile).exists():
                tts_service.oute_tts_engine.custom_speakers["arabic_female_custom"] = female_profile
                
            if male_profile and Path(male_profile).exists():
                tts_service.oute_tts_engine.custom_speakers["arabic_male_custom"] = male_profile
        
        # Test with female speaker
        if female_profile:
            logger.info("🎤 Testing female Arabic speaker...")
            female_output = tts_service.text_to_speech(
                text="مرحباً، أنا مساعدتك الشخصية في عيادة الجمال. كيف يمكنني مساعدتك؟",
                language="ar",
                speaker_voice="arabic_female_custom",
                output_path="/home/lumi/beautyai/voice_tests/arabic_speaker_tests/beautyai_female_integration_test.wav"
            )
            
            if female_output and Path(female_output).exists():
                logger.info(f"✅ Female integration test successful: {female_output}")
            else:
                logger.warning("⚠️ Female integration test failed")
        
        # Test with male speaker
        if male_profile:
            logger.info("🎤 Testing male Arabic speaker...")
            male_output = tts_service.text_to_speech(
                text="أهلاً وسهلاً، أنا الدكتور أحمد. سأساعدك في اختيار أفضل علاج لبشرتك.",
                language="ar",
                speaker_voice="arabic_male_custom",
                output_path="/home/lumi/beautyai/voice_tests/arabic_speaker_tests/beautyai_male_integration_test.wav"
            )
            
            if male_output and Path(male_output).exists():
                logger.info(f"✅ Male integration test successful: {male_output}")
            else:
                logger.warning("⚠️ Male integration test failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to create Arabic speaker profiles."""
    
    print("🎭 Arabic Speaker Profile Creator for BeautyAI Platform")
    print("="*80)
    
    # Audio file paths
    female_audio = "/home/lumi/beautyai/trimmed_audio/arabic_female_15s.wav"
    male_audio = "/home/lumi/beautyai/trimmed_audio/arabic_male_15s.wav"
    
    # Check if audio files exist
    if not Path(female_audio).exists():
        logger.error(f"❌ Female audio file not found: {female_audio}")
        return False
    
    if not Path(male_audio).exists():
        logger.error(f"❌ Male audio file not found: {male_audio}")
        return False
    
    # Analyze audio files
    print("\n📊 Analyzing Audio Files")
    print("-" * 40)
    
    female_analysis = analyze_audio_file(female_audio)
    male_analysis = analyze_audio_file(male_audio)
    
    print(f"🎤 Female Audio Analysis:")
    for key, value in female_analysis.items():
        if key == "duration":
            print(f"   {key}: {value:.2f} seconds")
        elif key == "file_size":
            print(f"   {key}: {value:,} bytes")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n🎤 Male Audio Analysis:")
    for key, value in male_analysis.items():
        if key == "duration":
            print(f"   {key}: {value:.2f} seconds")
        elif key == "file_size":
            print(f"   {key}: {value:,} bytes")
        else:
            print(f"   {key}: {value}")
    
    # Convert audio files to optimal format if needed
    optimized_dir = Path("/home/lumi/beautyai/voice_tests/optimized_speakers")
    optimized_dir.mkdir(exist_ok=True)
    
    female_optimized = str(optimized_dir / "arabic_female_22khz.wav")
    male_optimized = str(optimized_dir / "arabic_male_22khz.wav")
    
    print(f"\n🔄 Converting Audio to Optimal Format")
    print("-" * 40)
    
    female_converted = convert_audio_to_optimal_format(female_audio, female_optimized)
    male_converted = convert_audio_to_optimal_format(male_audio, male_optimized)
    
    # Use optimized files if conversion was successful
    final_female_audio = female_optimized if female_converted else female_audio
    final_male_audio = male_optimized if male_converted else male_audio
    
    # Create speaker profiles
    print(f"\n🎭 Creating Arabic Speaker Profiles")
    print("-" * 40)
    
    female_profile = create_arabic_speaker_profile(
        final_female_audio, 
        "arabic_female_beautyai", 
        "female"
    )
    
    male_profile = create_arabic_speaker_profile(
        final_male_audio, 
        "arabic_male_beautyai", 
        "male"
    )
    
    # Update engine configuration
    if female_profile or male_profile:
        print(f"\n🔧 Updating BeautyAI Configuration")
        print("-" * 40)
        update_oute_tts_engine_with_profiles(female_profile, male_profile)
        
        # Test integration
        print(f"\n🧪 Testing Integration")
        print("-" * 40)
        test_integration_with_tts_service(female_profile, male_profile)
    
    # Summary
    print(f"\n📋 Summary")
    print("="*80)
    
    if female_profile:
        print(f"✅ Female Arabic speaker profile created: {female_profile}")
    else:
        print("❌ Female Arabic speaker profile creation failed")
    
    if male_profile:
        print(f"✅ Male Arabic speaker profile created: {male_profile}")
    else:
        print("❌ Male Arabic speaker profile creation failed")
    
    if female_profile or male_profile:
        print(f"\n🎯 Next Steps:")
        print(f"   1. Use 'arabic_female_beautyai' for female Arabic TTS")
        print(f"   2. Use 'arabic_male_beautyai' for male Arabic TTS")
        print(f"   3. Test with: python diagnose_outetts_speakers_enhanced.py")
        print(f"   4. Profiles saved in: /home/lumi/beautyai/voice_tests/arabic_speaker_profiles/")
        
        return True
    else:
        print(f"\n❌ No speaker profiles were created successfully")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Arabic speaker profile creation completed successfully!")
    else:
        print(f"\n💥 Arabic speaker profile creation failed")
        sys.exit(1)
