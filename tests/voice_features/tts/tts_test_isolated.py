#!/usr/bin/env python3
"""
Isolated TTS (Text-to-Speech) Test Script

This script tests TTS functionality independently to ensure the TTS models
work properly before testing the full voice-to-voice pipeline.

Tests:
1. TTS Service initialization
2. Arabic text generation
3. English text generation  
4. Different voice types
5. Model loading verification
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService
from beautyai_inference.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TTSIsolatedTester:
    """Isolated TTS testing class."""
    
    def __init__(self):
        self.tts_service = TextToSpeechService()
        self.model_manager = ModelManager()
        self.test_results = []
        
        # Test output directory
        self.output_dir = Path("/home/lumi/beautyai/tests/tts_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test texts
        self.test_texts = {
            "arabic": {
                "text": "مرحبا، أنا مساعد ذكي متخصص في مجال التجميل والطب التجميلي. كيف يمكنني مساعدتك اليوم؟",
                "language": "ar",
                "description": "Arabic greeting about beauty and cosmetic medicine"
            },
            "english": {
                "text": "Hello, I am an AI assistant specialized in beauty and cosmetic medicine. How can I help you today?",
                "language": "en", 
                "description": "English greeting about beauty and cosmetic medicine"
            },
            "botox_arabic": {
                "text": "البوتوكس هو علاج تجميلي شائع يستخدم لتقليل التجاعيد وخطوط التعبير في الوجه.",
                "language": "ar",
                "description": "Arabic text about Botox treatment"
            },
            "botox_english": {
                "text": "Botox is a popular cosmetic treatment used to reduce wrinkles and expression lines on the face.",
                "language": "en",
                "description": "English text about Botox treatment"
            }
        }
    
    def test_tts_service_initialization(self) -> bool:
        """Test TTS service initialization."""
        logger.info("🔧 TESTING TTS SERVICE INITIALIZATION")
        logger.info("="*50)
        
        try:
            # Test service creation
            logger.info("✅ TTS service created successfully")
            
            # Test available engines
            if hasattr(self.tts_service, 'supported_engines'):
                logger.info(f"📋 Supported engines: {list(self.tts_service.supported_engines.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TTS service initialization failed: {e}")
            return False
    
    def test_model_loading(self, model_name: str = "coqui-tts-arabic") -> bool:
        """Test TTS model loading."""
        logger.info(f"\n🤖 TESTING MODEL LOADING: {model_name}")
        logger.info("="*50)
        
        try:
            # Test model loading
            logger.info(f"🔄 Loading TTS model: {model_name}")
            result = self.tts_service.load_tts_model(model_name)
            
            if result:
                logger.info("✅ TTS model loaded successfully")
                
                # Check model status
                if hasattr(self.tts_service, 'engine_loaded'):
                    logger.info(f"📊 Engine loaded: {self.tts_service.engine_loaded}")
                if hasattr(self.tts_service, 'current_engine'):
                    logger.info(f"🎯 Current engine: {self.tts_service.current_engine}")
                if hasattr(self.tts_service, 'current_model'):
                    logger.info(f"📋 Current model: {self.tts_service.current_model}")
                
                return True
            else:
                logger.error("❌ TTS model loading failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            return False
    
    def test_tts_generation(self, test_name: str, test_config: dict) -> dict:
        """Test TTS generation for specific text."""
        logger.info(f"\n🎵 TESTING TTS GENERATION: {test_name}")
        logger.info("="*50)
        
        result = {
            "test_name": test_name,
            "success": False,
            "error": None,
            "output_file": None,
            "generation_time": 0.0,
            "audio_size": 0
        }
        
        try:
            text = test_config["text"]
            language = test_config["language"]
            description = test_config["description"]
            
            logger.info(f"📝 Text: {text[:100]}...")
            logger.info(f"🌍 Language: {language}")
            logger.info(f"📖 Description: {description}")
            
            # Generate output filename
            output_file = self.output_dir / f"tts_test_{test_name}_{int(time.time())}.wav"
            
            # Test TTS generation
            start_time = time.time()
            
            tts_result_path = self.tts_service.text_to_speech(
                text=text,
                language=language,
                speaker_voice="female",
                output_path=str(output_file)
            )
            
            # Convert path result to dict format for consistency
            tts_result = {
                "success": tts_result_path is not None,
                "audio_path": tts_result_path or str(output_file)
            }
            
            generation_time = time.time() - start_time
            result["generation_time"] = generation_time
            
            logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
            
            if tts_result.get("success", False):
                audio_path = tts_result.get("audio_path", output_file)
                
                if Path(audio_path).exists():
                    audio_size = Path(audio_path).stat().st_size
                    result["success"] = True
                    result["output_file"] = str(audio_path)
                    result["audio_size"] = audio_size
                    
                    logger.info(f"✅ TTS generation successful")
                    logger.info(f"📁 Output file: {audio_path}")
                    logger.info(f"📏 Audio size: {audio_size} bytes")
                else:
                    result["error"] = f"Audio file not found: {audio_path}"
                    logger.error(f"❌ {result['error']}")
            else:
                result["error"] = tts_result.get("error", "TTS generation failed")
                logger.error(f"❌ TTS generation failed: {result['error']}")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ TTS generation error: {e}")
        
        return result
    
    def test_different_voice_types(self) -> list:
        """Test different voice types."""
        logger.info(f"\n🎭 TESTING DIFFERENT VOICE TYPES")
        logger.info("="*50)
        
        voice_results = []
        voice_types = ["female", "male", "neutral"]
        test_text = "This is a test of different voice types in English."
        
        for voice_type in voice_types:
            logger.info(f"\n🔊 Testing voice type: {voice_type}")
            
            try:
                output_file = self.output_dir / f"voice_test_{voice_type}_{int(time.time())}.wav"
                
                start_time = time.time()
                result_path = self.tts_service.text_to_speech(
                    text=test_text,
                    language="en",
                    speaker_voice=voice_type,
                    output_path=str(output_file)
                )
                generation_time = time.time() - start_time
                
                voice_result = {
                    "voice_type": voice_type,
                    "success": result_path is not None,
                    "generation_time": generation_time,
                    "output_file": result_path or str(output_file),
                    "error": None if result_path else "TTS generation failed"
                }
                
                if voice_result["success"] and Path(voice_result["output_file"]).exists():
                    audio_size = Path(voice_result["output_file"]).stat().st_size
                    voice_result["audio_size"] = audio_size
                    logger.info(f"✅ {voice_type} voice: {audio_size} bytes ({generation_time:.2f}s)")
                else:
                    logger.error(f"❌ {voice_type} voice failed: {voice_result['error']}")
                
                voice_results.append(voice_result)
                
            except Exception as e:
                logger.error(f"❌ Voice type {voice_type} error: {e}")
                voice_results.append({
                    "voice_type": voice_type,
                    "success": False,
                    "error": str(e)
                })
        
        return voice_results
    
    def run_comprehensive_tts_tests(self) -> dict:
        """Run all TTS tests."""
        logger.info("🚀 STARTING COMPREHENSIVE TTS TESTS")
        logger.info("="*60)
        
        all_results = {
            "initialization": False,
            "model_loading": False,
            "text_generation": [],
            "voice_types": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
        
        # Test 1: Service initialization
        all_results["initialization"] = self.test_tts_service_initialization()
        
        # Test 2: Model loading
        if all_results["initialization"]:
            all_results["model_loading"] = self.test_model_loading()
        
        # Test 3: Text generation tests
        if all_results["model_loading"]:
            for test_name, test_config in self.test_texts.items():
                result = self.test_tts_generation(test_name, test_config)
                all_results["text_generation"].append(result)
            
            # Test 4: Voice types
            all_results["voice_types"] = self.test_different_voice_types()
        
        # Calculate summary
        total_tests = 0
        passed_tests = 0
        
        # Count initialization and model loading
        total_tests += 2
        if all_results["initialization"]:
            passed_tests += 1
        if all_results["model_loading"]:
            passed_tests += 1
        
        # Count text generation tests
        for result in all_results["text_generation"]:
            total_tests += 1
            if result["success"]:
                passed_tests += 1
        
        # Count voice type tests
        for result in all_results["voice_types"]:
            total_tests += 1
            if result["success"]:
                passed_tests += 1
        
        all_results["summary"]["total_tests"] = total_tests
        all_results["summary"]["passed_tests"] = passed_tests
        all_results["summary"]["failed_tests"] = total_tests - passed_tests
        
        return all_results
    
    def generate_test_report(self, results: dict) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("="*60)
        report.append("🧪 TTS ISOLATED TEST REPORT")
        report.append("="*60)
        report.append(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        summary = results["summary"]
        success_rate = (summary["passed_tests"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0
        
        report.append("📊 SUMMARY:")
        report.append(f"   Total Tests: {summary['total_tests']}")
        report.append(f"   ✅ Passed: {summary['passed_tests']}")
        report.append(f"   ❌ Failed: {summary['failed_tests']}")
        report.append(f"   📈 Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Initialization
        status = "✅ PASS" if results["initialization"] else "❌ FAIL"
        report.append(f"🔧 Service Initialization: {status}")
        
        # Model Loading
        status = "✅ PASS" if results["model_loading"] else "❌ FAIL"
        report.append(f"🤖 Model Loading: {status}")
        report.append("")
        
        # Text Generation Results
        report.append("📝 TEXT GENERATION TESTS:")
        for result in results["text_generation"]:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report.append(f"   {status} {result['test_name']}")
            if result["success"]:
                report.append(f"      Time: {result['generation_time']:.2f}s")
                report.append(f"      Size: {result['audio_size']} bytes")
            else:
                report.append(f"      Error: {result['error']}")
        report.append("")
        
        # Voice Type Results
        if results["voice_types"]:
            report.append("🎭 VOICE TYPE TESTS:")
            for result in results["voice_types"]:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                report.append(f"   {status} {result['voice_type']}")
                if result["success"]:
                    report.append(f"      Time: {result['generation_time']:.2f}s")
                    report.append(f"      Size: {result.get('audio_size', 0)} bytes")
                else:
                    report.append(f"      Error: {result.get('error', 'Unknown')}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if summary["failed_tests"] == 0:
            report.append("   🎉 All tests passed! TTS system is working correctly.")
        else:
            report.append("   🔍 Issues found:")
            if not results["initialization"]:
                report.append("      - Fix TTS service initialization")
            if not results["model_loading"]:
                report.append("      - Fix TTS model loading")
            
            failed_texts = [r for r in results["text_generation"] if not r["success"]]
            if failed_texts:
                report.append("      - Fix text generation for:")
                for r in failed_texts:
                    report.append(f"        • {r['test_name']}: {r['error']}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)


def main():
    """Run TTS isolated tests."""
    try:
        # Create tester
        tester = TTSIsolatedTester()
        
        # Run tests
        results = tester.run_comprehensive_tts_tests()
        
        # Generate report
        report = tester.generate_test_report(results)
        print(report)
        
        # Save report
        report_file = Path("/home/lumi/beautyai/tests/tts_isolated_test_report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        
        # Return success if all critical tests passed
        critical_passed = results["initialization"] and results["model_loading"]
        some_generation_passed = any(r["success"] for r in results["text_generation"])
        
        return critical_passed and some_generation_passed
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
