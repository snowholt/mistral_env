#!/usr/bin/env python3
"""
Enhanced OuteTTS Speaker Diagnostic Tool for BeautyAI Framework.
Tests speaker discovery, Arabic text synthesis, and custom speaker profile creation.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def discover_outetts_speakers():
    """Discover all available speakers in OuteTTS."""
    print("🔍 Discovering OuteTTS Available Speakers")
    print("="*60)
    
    try:
        import outetts
        print("✅ OuteTTS library imported successfully")
        
        # Initialize the model
        print("📥 Initializing OuteTTS model...")
        model_config = outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.LLAMACPP,
            quantization=outetts.LlamaCppQuantization.FP16
        )
        
        interface = outetts.Interface(config=model_config)
        print("✅ OuteTTS interface initialized")
        
        # Test patterns based on repository documentation
        print("\n🧪 Testing speakers from repository documentation...")
        
        # Known speakers from OuteTTS GitHub repository
        documented_speakers = [
            "EN-FEMALE-1-NEUTRAL",
            "en_female_1", "en_female_2",
            "en_male_1", "en_male_2", "en_male_3", "en_male_4"
        ]
        
        working_speakers = []
        
        for speaker_id in documented_speakers:
            try:
                print(f"🔍 Testing documented speaker: {speaker_id}")
                speaker = interface.load_default_speaker(speaker_id)
                
                # Try a simple generation to verify the speaker works
                output = interface.generate(
                    config=outetts.GenerationConfig(
                        text="Test",
                        generation_type=outetts.GenerationType.CHUNKED,
                        speaker=speaker,
                        sampler_config=outetts.SamplerConfig(
                            temperature=0.4,
                            top_p=0.9,
                            top_k=50
                        ),
                    )
                )
                
                working_speakers.append(speaker_id)
                print(f"✅ WORKING: {speaker_id}")
                
            except Exception as e:
                print(f"❌ FAILED: {speaker_id} - {e}")
        
        print(f"\n" + "="*60)
        print("📊 DISCOVERY RESULTS")
        print("="*60)
        print(f"✅ Working speakers found: {len(working_speakers)}")
        
        if working_speakers:
            print("\n🎤 CONFIRMED WORKING SPEAKERS:")
            for i, speaker in enumerate(working_speakers, 1):
                print(f"   {i}. {speaker}")
            
            # Test Arabic text with working speakers
            print(f"\n🇸🇦 Testing Arabic text with working speakers...")
            arabic_test_text = "مرحبا، هذا اختبار للغة العربية في منصة BeautyAI للذكاء الاصطناعي"
            
            # Create voice_tests directory
            voice_tests_dir = Path("/home/lumi/beautyai/voice_tests")
            voice_tests_dir.mkdir(exist_ok=True)
            
            for speaker_id in working_speakers[:2]:  # Test first 2 working speakers
                try:
                    print(f"\n🔊 Testing Arabic with speaker: {speaker_id}")
                    speaker = interface.load_default_speaker(speaker_id)
                    
                    # Generate Arabic speech
                    output = interface.generate(
                        config=outetts.GenerationConfig(
                            text=arabic_test_text,
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
                    test_file = voice_tests_dir / f"arabic_test_{speaker_id.replace('-', '_').replace('.', '_')}.wav"
                    output.save(str(test_file))
                    
                    if test_file.exists():
                        file_size = test_file.stat().st_size
                        print(f"✅ ARABIC TEST SUCCESS: {speaker_id}")
                        print(f"   File: {test_file} ({file_size} bytes)")
                    else:
                        print(f"❌ ARABIC TEST FAILED: {speaker_id} - no file created")
                        
                except Exception as e:
                    print(f"❌ ARABIC TEST FAILED: {speaker_id} - {e}")
        
        else:
            print("\n❌ NO WORKING SPEAKERS FOUND!")
            print("This indicates a fundamental issue with OuteTTS speaker discovery.")

        # Test custom speaker creation capability
        print(f"\n🎭 Testing custom speaker creation capability...")
        try:
            print("🔍 Checking if interface.create_speaker method is available...")
            if hasattr(interface, 'create_speaker'):
                print("✅ create_speaker method is available")
                print("✅ Custom Arabic speaker profiles can be created when audio files are provided")
                
                # Test if we have any sample audio files to create a speaker
                sample_audio_files = [
                    "/home/lumi/beautyai/voice_tests/edge_tts_ar_female_1.wav",
                    "/home/lumi/beautyai/voice_tests/edge_tts_ar_male_2.wav"
                ]
                
                for audio_file in sample_audio_files:
                    if os.path.exists(audio_file):
                        print(f"🎯 Found sample audio for custom speaker creation: {audio_file}")
                        # We could create a custom speaker here if needed
                        break
                else:
                    print("💡 No sample audio files found for custom speaker creation")
                    print("   Provide Arabic audio files to create custom speakers")
            else:
                print("❌ create_speaker method not available")
                
        except Exception as e:
            print(f"❌ Custom speaker test failed: {e}")
        
    except ImportError:
        print("❌ OuteTTS library not available. Install with: pip install outetts")
    except Exception as e:
        print(f"❌ Unexpected error during speaker discovery: {e}")


def test_beautyai_integration():
    """Test the integration with BeautyAI TTS service."""
    print(f"\n" + "="*60)
    print("🧪 TESTING BEAUTYAI INTEGRATION")
    print("="*60)
    
    try:
        # Import BeautyAI components
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Initialize TTS service
        print("📥 Initializing BeautyAI TTS service...")
        tts_service = TextToSpeechService()
        
        # Load OuteTTS model
        print("📥 Loading OuteTTS model via BeautyAI...")
        success = tts_service.load_tts_model("oute-tts-1b")
        
        if success:
            print("✅ BeautyAI TTS service loaded successfully")
            
            # Test Arabic text generation with default speakers
            print("\n🎤 Testing Arabic text generation...")
            arabic_text = "مرحبا، هذا اختبار لخدمة التحويل من النص إلى الصوت في منصة BeautyAI"
            
            output_path = tts_service.text_to_speech(
                text=arabic_text,
                language="ar",
                speaker_voice="female",
                output_path="/home/lumi/beautyai/voice_tests/beautyai_arabic_test.wav"
            )
            
            if output_path:
                print(f"✅ BeautyAI Arabic test successful: {output_path}")
            else:
                print("❌ BeautyAI Arabic test failed")
                
            # Test getting available Arabic speakers
            print("\n🎭 Checking available Arabic speakers...")
            arabic_speakers = tts_service.get_arabic_speakers()
            print(f"Available Arabic speakers: {arabic_speakers}")
            
        else:
            print("❌ Failed to load BeautyAI TTS service")
            
    except Exception as e:
        print(f"❌ BeautyAI integration test failed: {e}")


def demonstrate_arabic_speaker_creation():
    """Demonstrate how to create Arabic speaker profiles."""
    print(f"\n" + "="*60)
    print("🎭 ARABIC SPEAKER PROFILE CREATION DEMO")
    print("="*60)
    
    print("📋 To create custom Arabic speaker profiles, follow these steps:")
    print("\n1. 📁 Prepare Arabic audio files:")
    print("   - Female Arabic speaker: /path/to/arabic_female.wav")
    print("   - Male Arabic speaker: /path/to/arabic_male.wav")
    print("   - Audio should be clear, 5-30 seconds long")
    print("   - WAV format recommended")
    
    print("\n2. 🔧 Use the BeautyAI TTS service:")
    print("```python")
    print("from beautyai_inference.services.text_to_speech_service import TextToSpeechService")
    print("")
    print("# Initialize service")
    print("tts_service = TextToSpeechService()")
    print("tts_service.load_tts_model('oute-tts-1b')")
    print("")
    print("# Create custom Arabic speakers")
    print("female_profile = tts_service.create_arabic_speaker_profile(")
    print("    audio_file_path='/path/to/arabic_female.wav',")
    print("    speaker_name='beautyai_arabic_female'")
    print(")")
    print("")
    print("male_profile = tts_service.create_arabic_speaker_profile(")
    print("    audio_file_path='/path/to/arabic_male.wav',")
    print("    speaker_name='beautyai_arabic_male'")
    print(")")
    print("")
    print("# Setup as default Arabic speakers")
    print("tts_service.setup_default_arabic_speakers(")
    print("    female_audio_path='/path/to/arabic_female.wav',")
    print("    male_audio_path='/path/to/arabic_male.wav'")
    print(")")
    print("```")
    
    print("\n3. 🎤 Use Arabic speakers for synthesis:")
    print("```python")
    print("# Generate Arabic speech with custom speaker")
    print("output_path = tts_service.text_to_speech(")
    print("    text='مرحبا بكم في منصة BeautyAI',")
    print("    language='ar',")
    print("    speaker_voice='female',")
    print("    output_path='arabic_speech.wav'")
    print(")")
    print("```")


if __name__ == "__main__":
    print("🚀 Enhanced OuteTTS Speaker Diagnostic Tool for BeautyAI")
    print("="*80)
    
    # Run speaker discovery
    discover_outetts_speakers()
    
    # Test BeautyAI integration
    test_beautyai_integration()
    
    # Demonstrate Arabic speaker creation
    demonstrate_arabic_speaker_creation()
    
    print(f"\n" + "="*80)
    print("📋 DIAGNOSIS COMPLETE")
    print("="*80)
    print("✅ Next steps:")
    print("1. If speakers were discovered, the OuteTTS engine should work")
    print("2. To create custom Arabic speakers, provide Arabic audio files")
    print("3. Use the BeautyAI TTS service for production Arabic speech synthesis")
    print("4. For the BeautyAI platform, provide high-quality Arabic audio samples")
    print("="*80)
