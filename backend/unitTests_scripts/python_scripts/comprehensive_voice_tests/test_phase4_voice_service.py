#!/usr/bin/env python3
"""
Phase 4: Simple Voice-to-Voice Pipeline Test

Clean and focused test for the complete voice-to-voice pipeline:
- Voice input -> Transcription (STT)  
- Transcription -> Chat response generation
- Chat response -> Voice output (TTS)

Focus on accuracy and latency with clean response logging.
Output saved to: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add the backend src directory to Python path
backend_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService


class VoiceToVoiceTester:
    """Clean and focused voice-to-voice pipeline tester."""
    
    def __init__(self):
        self.service = None
        self.test_files_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions")
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/phase4_voice_service_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create audio output directory for TTS results
        self.audio_output_dir = self.output_dir / "generated_audio"
        self.audio_output_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.results = {
            "test_timestamp": time.time(),
            "test_summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "average_latency_seconds": 0.0
            },
            "voice_conversations": [],
            "errors": []
        }
        
        # Ground truth transcriptions for reference
        self.ground_truth = {
            "greeting.wav": "Hello, how are you today?",
            "greeting_ar.wav": "مرحبًا، كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصة بكم.",
            "botox.wav": "What is botox used for?",
            "laser_hair.wav": "How does laser hair removal work?",
            "q1.wav": "ما هو استخدام البوتوكس؟",
            "q2.wav": "كيف يعمل إزالة الشعر بالليزر؟",
            "q3.wav": "هل الحشوات الجلدية دائمة؟",
            "q4.wav": "ما هي الآثار الجانبية الشائعة للتقشير الكيميائي؟",
            "q5.wav": "هل الميزوثيرابي؟",
            "q6.wav": "كم تدوم نتائج جلسة تنظيف البشرة عادة؟",
            "q7.wav": "هل يمكن لأي شخص إجراء عملية تجميل الأنف غير الجراحية؟",
            "q8.wav": "ما هو الغرض من علاج البلازما الغنية بالصفائح الدموية PRP للبشرة؟",
            "q9.wav": "هل هناك فترة نقاهة بعد عملية شد الوجه بالخيوط؟",
            "q10.wav": "ما هي الفائدة الرئيسية لعلاج الضوء النبضي المكثف IPL؟"
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    async def setup_service(self) -> bool:
        """Initialize the SimpleVoiceService."""
        try:
            self.log("🚀 Initializing SimpleVoiceService...")
            self.service = SimpleVoiceService()
            await self.service.initialize()
            self.log("✅ SimpleVoiceService initialized successfully")
            return True
                
        except Exception as e:
            self.log(f"❌ Service initialization failed: {e}", "ERROR")
            self.results["errors"].append({
                "phase": "initialization",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            })
            return False

    async def test_voice_conversation(self, audio_file: Path) -> Dict[str, Any]:
        """Test a single voice-to-voice conversation."""
        try:
            self.log(f"🎤 Testing: {audio_file.name}")
            
            # Read audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Determine language based on filename
            if "ar" in audio_file.name or "q" in audio_file.name:
                language = "ar"
            else:
                language = "en"
            
            # Record start time for latency measurement
            start_time = time.time()
            
            # Process voice message through complete pipeline
            result = await self.service.process_voice_message(
                audio_data=audio_data,
                language=language,
                gender="female"
            )
            
            # Calculate total latency
            total_latency = time.time() - start_time
            
            # Extract results
            transcribed_text = result.get("transcribed_text", "")
            response_text = result.get("response_text", "")
            response_text_clean = result.get("response_text_clean", response_text)
            audio_file_path = result.get("audio_file_path", "")
            voice_used = result.get("voice_used", "")
            language_detected = result.get("language_detected", language)
            service_processing_time = result.get("processing_time", total_latency)
            
            # Save generated audio with descriptive name
            generated_audio_path = ""
            if audio_file_path and Path(audio_file_path).exists():
                new_audio_name = f"{audio_file.stem}_response_{language}.wav"
                generated_audio_path = self.audio_output_dir / new_audio_name
                import shutil
                shutil.copy2(audio_file_path, generated_audio_path)
            
            # Create conversation record
            conversation = {
                "input_file": audio_file.name,
                "ground_truth_text": self.ground_truth.get(audio_file.name, ""),
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "response_text_clean": response_text_clean,
                "language_detected": language_detected,
                "voice_used": voice_used,
                "latency_seconds": total_latency,
                "service_processing_time_seconds": service_processing_time,
                "generated_audio_file": str(generated_audio_path) if generated_audio_path else "",
                "success": bool(transcribed_text and response_text),
                "timestamp": time.time()
            }
            
            # Log summary
            if conversation["success"]:
                self.log(f"   ✅ Success | Latency: {total_latency:.2f}s")
                self.log(f"      📝 Transcribed: '{transcribed_text[:60]}...'")
                self.log(f"      🤖 Response: '{response_text_clean[:60]}...'")
            else:
                self.log(f"   ❌ Failed - No transcription or response")
            
            return conversation
            
        except Exception as e:
            self.log(f"❌ Error testing {audio_file.name}: {e}", "ERROR")
            error_conversation = {
                "input_file": audio_file.name,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }
            self.results["errors"].append(error_conversation)
            return error_conversation
    
    async def run_voice_to_voice_test(self):
        """Run the complete voice-to-voice test suite."""
        self.log("🎯 Starting Voice-to-Voice Pipeline Test")
        self.log("=" * 50)
        
        # Initialize service
        if not await self.setup_service():
            self.log("❌ Cannot proceed without service initialization", "ERROR")
            return
        
        # Get all audio files
        audio_files = list(self.test_files_dir.glob("*.wav"))
        if not audio_files:
            self.log("❌ No audio files found in test directory", "ERROR")
            return
        
        self.log(f"📁 Testing {len(audio_files)} voice conversations")
        
        # Test each conversation
        successful_tests = 0
        total_latency = 0.0
        
        for audio_file in sorted(audio_files):
            conversation = await self.test_voice_conversation(audio_file)
            self.results["voice_conversations"].append(conversation)
            
            if conversation.get("success", False):
                successful_tests += 1
                total_latency += conversation.get("latency_seconds", 0.0)
        
        # Calculate summary metrics
        self.results["test_summary"] = {
            "total_tests": len(audio_files),
            "successful_tests": successful_tests,
            "failed_tests": len(audio_files) - successful_tests,
            "success_rate_percent": (successful_tests / len(audio_files)) * 100 if audio_files else 0,
            "average_latency_seconds": total_latency / successful_tests if successful_tests > 0 else 0.0
        }
        
        # Save results and cleanup
        await self.save_results()
        await self.service.cleanup()
        self.print_summary()
    
    async def save_results(self):
        """Save test results to JSON file."""
        results_file = self.output_dir / f"voice_to_voice_test_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        self.log("\n📊 VOICE-TO-VOICE TEST SUMMARY")
        self.log("=" * 50)
        
        summary = self.results["test_summary"]
        self.log(f"✅ Success Rate: {summary['success_rate_percent']:.1f}%")
        self.log(f"📊 Tests: {summary['successful_tests']}/{summary['total_tests']} successful")
        self.log(f"⚡ Average Latency: {summary['average_latency_seconds']:.2f} seconds")
        
        # Latency evaluation
        avg_latency = summary['average_latency_seconds']
        if avg_latency < 2.0:
            self.log("🚀 EXCELLENT: Latency < 2.0 seconds (real-time ready)")
        elif avg_latency < 4.0:
            self.log("✅ GOOD: Latency < 4.0 seconds") 
        else:
            self.log("⚠️ SLOW: Latency > 4.0 seconds (needs optimization)")
        
        if self.results["errors"]:
            self.log(f"❌ Errors: {len(self.results['errors'])}")
        
        self.log(f"\n📁 Generated audio files: {self.audio_output_dir}")
        self.log("🔍 Review JSON results for detailed conversation logs")


async def main():
    """Main test execution."""
    print("🎤 Voice-to-Voice Pipeline Test - Phase 4 (Clean & Focused)")
    print("=" * 60)
    
    tester = VoiceToVoiceTester()
    await tester.run_voice_to_voice_test()
    
    print("\n✅ Phase 4 voice-to-voice testing complete!")
    print("📁 Check results in: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/")
    print("🔊 Check generated audio in: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/generated_audio/")


if __name__ == "__main__":
    asyncio.run(main())
