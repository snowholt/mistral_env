#!/usr/bin/env python3
"""
Standalone TTS Test Script for BeautyAI Framework.

This script tests the Text-to-Speech functionality step by step to ensure
the TTS models work properly before proceeding to voice-to-voice and WebSocket testing.

Usage:
    python test_tts_standalone.py
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TTSStandaloneTest:
    """Comprehensive standalone TTS testing."""
    
    def __init__(self):
        self.output_dir = Path("/home/lumi/beautyai/tests/websocket/tts_test_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test cases for different scenarios
        self.test_cases = [
            {
                "name": "arabic_beauty_greeting",
                "text": "مرحباً بكم في عيادة الجمال. كيف يمكنني مساعدتكم اليوم؟",
                "language": "ar",
                "speaker": "female",
                "description": "Arabic beauty clinic greeting"
            },
            {
                "name": "arabic_medical_consultation",
                "text": "أهلاً وسهلاً، أنا الدكتور أحمد. سأساعدك في اختيار أفضل علاج لبشرتك.",
                "language": "ar",
                "speaker": "male",
                "description": "Arabic medical consultation"
            },
            {
                "name": "english_beauty_greeting",
                "text": "Welcome to our beauty clinic. How can I help you today?",
                "language": "en",
                "speaker": "female",
                "description": "English beauty clinic greeting"
            },
            {
                "name": "english_medical_info",
                "text": "Hello, I'm Dr. Ahmed. I'll help you choose the best treatment for your skin.",
                "language": "en",
                "speaker": "male",
                "description": "English medical information"
            },
            {
                "name": "arabic_botox_info",
                "text": "البوتوكس علاج آمن وفعال لتقليل التجاعيد. فترة التعافي عادة ما تكون قصيرة.",
                "language": "ar",
                "speaker": "female",
                "description": "Arabic Botox information"
            },
            {
                "name": "english_botox_info",
                "text": "Botox is a safe and effective treatment for reducing wrinkles. Recovery time is usually short.",
                "language": "en",
                "speaker": "female",
                "description": "English Botox information"
            }
        ]
    
    def test_tts_service_initialization(self):
        """Test TTS service initialization."""
        logger.info("🔍 Testing TTS Service Initialization")
        logger.info("="*50)
        
        try:
            tts_service = TextToSpeechService()
            logger.info("✅ TTS Service initialized successfully")
            return tts_service
        except Exception as e:
            logger.error(f"❌ TTS Service initialization failed: {e}")
            return None
    
    def test_model_loading(self, tts_service):
        """Test different TTS model loading."""
        logger.info("\n🔍 Testing TTS Model Loading")
        logger.info("="*50)
        
        models_to_test = [
            "coqui-tts-arabic",
            "edge-tts-arabic",
            "coqui-arabic"
        ]
        
        successful_models = []
        
        for model_name in models_to_test:
            logger.info(f"📥 Testing model: {model_name}")
            try:
                start_time = time.time()
                success = tts_service.load_tts_model(model_name)
                load_time = time.time() - start_time
                
                if success:
                    logger.info(f"✅ {model_name} loaded successfully in {load_time:.2f}s")
                    successful_models.append(model_name)
                else:
                    logger.warning(f"⚠️ {model_name} failed to load")
                    
            except Exception as e:
                logger.error(f"❌ {model_name} loading error: {e}")
        
        logger.info(f"\n📊 Successful models: {successful_models}")
        return successful_models
    
    def test_tts_generation(self, tts_service, model_name):
        """Test TTS generation with a specific model."""
        logger.info(f"\n🔍 Testing TTS Generation with {model_name}")
        logger.info("="*50)
        
        # Load the model first
        logger.info(f"📥 Loading {model_name}...")
        success = tts_service.load_tts_model(model_name)
        if not success:
            logger.error(f"❌ Failed to load {model_name}")
            return False
        
        logger.info(f"✅ {model_name} loaded successfully")
        
        results = []
        
        for test_case in self.test_cases:
            logger.info(f"\n🎯 Testing: {test_case['name']}")
            logger.info(f"📝 Text: {test_case['text'][:50]}...")
            logger.info(f"🌍 Language: {test_case['language']}, Speaker: {test_case['speaker']}")
            
            try:
                # Generate output path
                output_file = self.output_dir / f"{model_name.replace('-', '_')}_{test_case['name']}.wav"
                
                # Measure generation time
                start_time = time.time()
                
                result_path = tts_service.text_to_speech(
                    text=test_case['text'],
                    language=test_case['language'],
                    speaker_voice=test_case['speaker'],
                    output_path=str(output_file)
                )
                
                generation_time = time.time() - start_time
                
                if result_path and Path(result_path).exists():
                    file_size = Path(result_path).stat().st_size
                    chars_per_sec = len(test_case['text']) / generation_time if generation_time > 0 else 0
                    
                    logger.info(f"✅ Generation successful: {output_file.name}")
                    logger.info(f"   📁 File size: {file_size:,} bytes")
                    logger.info(f"   ⏱️ Generation time: {generation_time:.2f}s")
                    logger.info(f"   🚀 Speed: {chars_per_sec:.1f} chars/sec")
                    
                    results.append({
                        "test_case": test_case['name'],
                        "success": True,
                        "file_path": str(result_path),
                        "file_size": file_size,
                        "generation_time": generation_time,
                        "chars_per_sec": chars_per_sec
                    })
                else:
                    logger.error(f"❌ Generation failed: No output file created")
                    results.append({
                        "test_case": test_case['name'],
                        "success": False,
                        "error": "No output file created"
                    })
                    
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                results.append({
                    "test_case": test_case['name'],
                    "success": False,
                    "error": str(e)
                })
        
        # Summary
        successful = [r for r in results if r.get('success', False)]
        logger.info(f"\n📊 TTS Generation Summary for {model_name}:")
        logger.info(f"   Total tests: {len(results)}")
        logger.info(f"   Successful: {len(successful)}")
        logger.info(f"   Failed: {len(results) - len(successful)}")
        
        if successful:
            avg_time = sum(r['generation_time'] for r in successful) / len(successful)
            avg_speed = sum(r['chars_per_sec'] for r in successful) / len(successful)
            logger.info(f"   Average generation time: {avg_time:.2f}s")
            logger.info(f"   Average speed: {avg_speed:.1f} chars/sec")
        
        return len(successful) > 0
    
    def test_language_specific_generation(self, tts_service):
        """Test language-specific TTS generation."""
        logger.info("\n🔍 Testing Language-Specific Generation")
        logger.info("="*50)
        
        # Test Arabic specifically
        logger.info("🇸🇦 Testing Arabic TTS...")
        arabic_test = {
            "text": "هذا اختبار لمحرك التحويل من النص إلى الصوت باللغة العربية",
            "language": "ar",
            "speaker": "female"
        }
        
        output_file = self.output_dir / "arabic_specific_test.wav"
        
        try:
            result = tts_service.text_to_speech(
                text=arabic_test['text'],
                language=arabic_test['language'],
                speaker_voice=arabic_test['speaker'],
                output_path=str(output_file)
            )
            
            if result and Path(result).exists():
                file_size = Path(result).stat().st_size
                logger.info(f"✅ Arabic TTS successful: {file_size:,} bytes")
            else:
                logger.error("❌ Arabic TTS failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Arabic TTS error: {e}")
            return False
        
        # Test English specifically
        logger.info("\n🇺🇸 Testing English TTS...")
        english_test = {
            "text": "This is a test of the text-to-speech engine in English language",
            "language": "en",
            "speaker": "female"
        }
        
        output_file = self.output_dir / "english_specific_test.wav"
        
        try:
            result = tts_service.text_to_speech(
                text=english_test['text'],
                language=english_test['language'],
                speaker_voice=english_test['speaker'],
                output_path=str(output_file)
            )
            
            if result and Path(result).exists():
                file_size = Path(result).stat().st_size
                logger.info(f"✅ English TTS successful: {file_size:,} bytes")
            else:
                logger.error("❌ English TTS failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ English TTS error: {e}")
            return False
        
        return True
    
    def test_voice_variants(self, tts_service):
        """Test different voice variants (male/female)."""
        logger.info("\n🔍 Testing Voice Variants")
        logger.info("="*50)
        
        test_text = "مرحباً، هذا اختبار لأصوات مختلفة"
        
        voices = ["female", "male"]
        results = {}
        
        for voice in voices:
            logger.info(f"🎤 Testing {voice} voice...")
            output_file = self.output_dir / f"voice_test_{voice}.wav"
            
            try:
                result = tts_service.text_to_speech(
                    text=test_text,
                    language="ar",
                    speaker_voice=voice,
                    output_path=str(output_file)
                )
                
                if result and Path(result).exists():
                    file_size = Path(result).stat().st_size
                    logger.info(f"✅ {voice} voice successful: {file_size:,} bytes")
                    results[voice] = True
                else:
                    logger.error(f"❌ {voice} voice failed")
                    results[voice] = False
                    
            except Exception as e:
                logger.error(f"❌ {voice} voice error: {e}")
                results[voice] = False
        
        return all(results.values())
    
    def run_comprehensive_test(self):
        """Run comprehensive TTS testing."""
        logger.info("🚀 Starting Comprehensive TTS Testing")
        logger.info("="*80)
        
        # Step 1: Initialize TTS service
        tts_service = self.test_tts_service_initialization()
        if not tts_service:
            logger.error("❌ Cannot continue without TTS service")
            return False
        
        # Step 2: Test model loading
        successful_models = self.test_model_loading(tts_service)
        if not successful_models:
            logger.error("❌ No TTS models could be loaded")
            return False
        
        # Step 3: Test generation with each successful model
        working_models = []
        for model in successful_models:
            logger.info(f"\n🔧 Testing generation with {model}")
            if self.test_tts_generation(tts_service, model):
                working_models.append(model)
                logger.info(f"✅ {model} generation works")
                break  # Use the first working model for subsequent tests
            else:
                logger.warning(f"⚠️ {model} generation failed")
        
        if not working_models:
            logger.error("❌ No models can generate audio")
            return False
        
        # Step 4: Test language-specific generation
        logger.info(f"\n🌍 Using {working_models[0]} for language tests")
        tts_service.load_tts_model(working_models[0])
        
        if not self.test_language_specific_generation(tts_service):
            logger.error("❌ Language-specific generation failed")
            return False
        
        # Step 5: Test voice variants
        if not self.test_voice_variants(tts_service):
            logger.error("❌ Voice variant testing failed")
            return False
        
        # Final summary
        logger.info("\n🎯 TTS TESTING SUMMARY")
        logger.info("="*50)
        logger.info(f"✅ TTS Service: Working")
        logger.info(f"✅ Models Available: {len(successful_models)}")
        logger.info(f"✅ Models Working: {len(working_models)}")
        logger.info(f"✅ Languages: Arabic and English")
        logger.info(f"✅ Voice Variants: Male and Female")
        logger.info(f"📁 Test outputs: {self.output_dir}")
        
        logger.info("\n🎉 TTS testing completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("   1. ✅ TTS is working - proceed to voice-to-voice testing")
        logger.info("   2. Test full voice-to-voice pipeline")
        logger.info("   3. Test WebSocket endpoints")
        
        return True


def main():
    """Main test function."""
    tester = TTSStandaloneTest()
    
    try:
        success = tester.run_comprehensive_test()
        
        if success:
            logger.info("\n✅ All TTS tests passed! Ready for voice-to-voice testing.")
            return True
        else:
            logger.error("\n❌ TTS tests failed. Fix TTS issues before proceeding.")
            return False
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ TTS testing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ TTS testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
