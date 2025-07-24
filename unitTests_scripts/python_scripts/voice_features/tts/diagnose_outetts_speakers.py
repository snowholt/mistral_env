#!/usr/bin/env python3
"""
Diagnostic script to discover actual OuteTTS speakers and test their functionality.
This will help us fix the speaker ID mismatch issue.
"""

import sys
import os
import logging

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
        
        # Try to discover available speakers
        print("\n🎤 Attempting to discover speakers...")
        
        # Method 1: Try to access speaker list directly
        try:
            if hasattr(interface, 'list_speakers'):
                speakers = interface.list_speakers()
                print(f"✅ Found speakers via list_speakers(): {speakers}")
            elif hasattr(interface, 'get_speakers'):
                speakers = interface.get_speakers()
                print(f"✅ Found speakers via get_speakers(): {speakers}")
            elif hasattr(interface, 'speakers'):
                speakers = interface.speakers
                print(f"✅ Found speakers via speakers attribute: {speakers}")
            else:
                print("❌ No direct speaker listing method found")
        except Exception as e:
            print(f"❌ Failed to get speaker list: {e}")
        
        # Method 2: Try common speaker patterns
        print("\n🧪 Testing common speaker patterns...")
        
        # Test patterns for different languages and genders
        test_patterns = [
            # English speakers
            "en-female-1-neutral", "en-male-1-neutral",
            "EN-FEMALE-1-NEUTRAL", "EN-MALE-1-NEUTRAL",
            "en_female_1_neutral", "en_male_1_neutral",
            "english-female-1", "english-male-1",
            
            # Arabic speakers
            "ar-female-1-neutral", "ar-male-1-neutral", 
            "AR-FEMALE-1-NEUTRAL", "AR-MALE-1-NEUTRAL",
            "ar_female_1_neutral", "ar_male_1_neutral",
            "arabic-female-1", "arabic-male-1",
            
            # Common generic patterns
            "female-1", "male-1", "neutral-1",
            "speaker-1", "speaker-2", "speaker-3",
            "default", "female", "male"
        ]
        
        working_speakers = []
        
        for speaker_id in test_patterns:
            try:
                print(f"🔍 Testing speaker: {speaker_id}")
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
            arabic_test_text = "مرحبا، هذا اختبار للغة العربية"
            
            for speaker_id in working_speakers[:3]:  # Test first 3 working speakers
                try:
                    print(f"\n🔊 Testing Arabic with speaker: {speaker_id}")
                    speaker = interface.load_default_speaker(speaker_id)
                    
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
                    
                    # Save test file
                    test_file = f"/home/lumi/beautyai/voice_tests/speaker_test_{speaker_id.replace('-', '_').lower()}.wav"
                    output.save(test_file)
                    
                    if os.path.exists(test_file):
                        file_size = os.path.getsize(test_file)
                        print(f"✅ Arabic generation successful with {speaker_id}")
                        print(f"   File: {test_file} ({file_size} bytes)")
                    else:
                        print(f"❌ Arabic generation failed - no file created")
                        
                except Exception as e:
                    print(f"❌ Arabic test failed with {speaker_id}: {e}")
        
        else:
            print("❌ No working speakers found!")
            print("💡 This suggests the OuteTTS model may have different speaker IDs")
            print("💡 Or there might be an issue with the model loading")
        
        # Method 3: Try to inspect the model itself
        print(f"\n🔍 Inspecting OuteTTS model internals...")
        try:
            # Try to access model attributes
            if hasattr(interface, 'model'):
                model = interface.model
                print(f"✅ Model object found: {type(model)}")
                
                # Look for speaker-related attributes
                for attr in dir(model):
                    if 'speaker' in attr.lower():
                        print(f"   Model attribute: {attr}")
            
            if hasattr(interface, 'config'):
                config = interface.config
                print(f"✅ Config object found: {type(config)}")
                
        except Exception as e:
            print(f"❌ Model inspection failed: {e}")
        
        return working_speakers
        
    except ImportError:
        print("❌ OuteTTS library not available. Install with: pip install outetts")
        return []
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_with_discovered_speakers(working_speakers):
    """Test the actual BeautyAI engine with discovered speakers."""
    if not working_speakers:
        print("❌ No working speakers to test")
        return
    
    print(f"\n🧪 Testing BeautyAI OuteTTS Engine with discovered speakers")
    print("="*60)
    
    try:
        from beautyai_inference.config.config_manager import ModelConfig
        from beautyai_inference.inference_engines.oute_tts_engine import OuteTTSEngine
        
        # Create model config
        model_config = ModelConfig(
            name="oute-tts-1b",
            model_id="OuteAI/Llama-OuteTTS-1.0-1B-GGUF",
            engine_type="oute_tts",
            quantization="Q4_K_M",
            dtype="float16",
            max_new_tokens=None,
            model_filename="model.gguf"
        )
        
        # Initialize engine
        engine = OuteTTSEngine(model_config)
        engine.load_model()
        
        # Test with each working speaker
        arabic_text = "أهلاً وسهلاً بكم في عيادة الجمال"
        
        for speaker_id in working_speakers[:3]:  # Test first 3
            print(f"\n🔊 Testing with speaker: {speaker_id}")
            
            try:
                output_path = f"/home/lumi/beautyai/voice_tests/engine_test_{speaker_id.replace('-', '_').lower()}.wav"
                
                # Use the speaker ID directly
                result = engine.text_to_speech(
                    text=arabic_text,
                    language="ar",
                    output_path=output_path,
                    speaker_voice=speaker_id  # Use the actual working speaker ID
                )
                
                if result and os.path.exists(result):
                    file_size = os.path.getsize(result)
                    print(f"✅ Engine test successful with {speaker_id}")
                    print(f"   File: {result} ({file_size} bytes)")
                else:
                    print(f"❌ Engine test failed with {speaker_id}")
                    
            except Exception as e:
                print(f"❌ Engine test failed with {speaker_id}: {e}")
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")

def main():
    """Main diagnostic function."""
    print("🔍 OuteTTS Speaker Discovery & Diagnostic Tool")
    print("BeautyAI Framework - Troubleshooting Speaker Issues")
    print("="*80)
    
    # Discover working speakers
    working_speakers = discover_outetts_speakers()
    
    # Test with discovered speakers
    if working_speakers:
        test_with_discovered_speakers(working_speakers)
        
        print(f"\n" + "="*80)
        print("📋 SUMMARY & RECOMMENDATIONS")
        print("="*80)
        print(f"✅ Working speakers discovered: {len(working_speakers)}")
        print(f"🎤 Speakers to use in BeautyAI:")
        
        for speaker in working_speakers:
            print(f"   - {speaker}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"1. Update OuteTTS engine speaker configurations")
        print(f"2. Use these working speaker IDs instead of hardcoded ones")
        print(f"3. Re-run the Arabic beauty clinic tests")
        
        # Generate updated speaker configuration
        print(f"\n🔧 SUGGESTED SPEAKER CONFIGURATION UPDATE:")
        print("```python")
        print("self.available_speakers = {")
        
        # Try to categorize speakers by language/gender
        for speaker in working_speakers:
            if 'en' in speaker.lower() or speaker == working_speakers[0]:
                print(f'    "en": {{')
                print(f'        "female": "{speaker}",')
                print(f'        "male": "{speaker}",')  # Use same for now
                print(f'        "neutral": "{speaker}"')
                print(f'    }},')
                break
        
        print(f'    "ar": {{')
        # Use the first working speaker for Arabic
        first_speaker = working_speakers[0]
        print(f'        "female": "{first_speaker}",')
        print(f'        "male": "{first_speaker}",')
        print(f'        "neutral": "{first_speaker}"')
        print(f'    }}')
        print("}")
        print("```")
        
    else:
        print(f"\n❌ No working speakers found!")
        print(f"💡 Possible issues:")
        print(f"   - OuteTTS model not properly loaded")
        print(f"   - Different speaker naming convention")
        print(f"   - Model version incompatibility")
        print(f"   - Installation issues")

if __name__ == "__main__":
    main()
