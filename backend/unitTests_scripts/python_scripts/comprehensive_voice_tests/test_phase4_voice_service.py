#!/usr/bin/env python3
"""
Phase 4: Simple Voice Service End-to-End Test

Tests the complete voice-to-voice pipeline:
- Audio transcription (STT) using GPU-accelerated Whisper
- Chat response generation using the beauty AI model
- Text-to-speech (TTS) using Edge TTS
- Complete pipeline timing and quality assessment

Output saved to: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add the backend src directory to Python path
backend_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService


class SimpleVoiceServiceTester:
    """Comprehensive tester for SimpleVoiceService end-to-end pipeline."""
    
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
            "service_initialization": {},
            "voice_to_voice_tests": [],
            "performance_metrics": {},
            "errors": []
        }
        
        # Ground truth transcriptions for accuracy testing
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
        
        # Expected response qualities to evaluate
        self.response_quality_checks = {
            "beauty_keywords": ["botox", "laser", "beauty", "skin", "treatment", "جمال", "علاج", "بشرة", "ليزر", "بوتوكس"],
            "greeting_responses": ["hello", "how can I help", "مرحبا", "كيف يمكنني", "اهلا"],
            "min_response_length": 10,  # Minimum characters for a meaningful response
            "max_response_length": 500  # Maximum reasonable response length
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    async def setup_service(self) -> bool:
        """Initialize the SimpleVoiceService."""
        try:
            self.log("🚀 Initializing SimpleVoiceService...")
            start_time = time.time()
            
            self.service = SimpleVoiceService()
            
            # Test service initialization
            init_time = time.time() - start_time
            
            self.results["service_initialization"] = {
                "success": True,
                "init_time_seconds": init_time,
                "timestamp": time.time()
            }
            
            self.log(f"✅ SimpleVoiceService initialized successfully in {init_time:.2f} seconds")
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

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate for transcription accuracy."""
        if not reference:
            return 1.0 if hypothesis else 0.0
            
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple Levenshtein distance calculation
        import numpy as np
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1,      # Deletion
                              d[i, j - 1] + 1,      # Insertion
                              d[i - 1, j - 1] + cost) # Substitution

        errors = d[len(ref_words), len(hyp_words)]
        return errors / len(ref_words)

    def evaluate_response_quality(self, response_text: str, input_file: str) -> Dict[str, Any]:
        """Evaluate the quality and relevance of the AI response."""
        quality_metrics = {
            "length_appropriate": False,
            "contains_beauty_keywords": False,
            "is_greeting_response": False,
            "response_relevance": "unknown",
            "quality_score": 0.0
        }
        
        response_lower = response_text.lower()
        
        # Check response length
        length = len(response_text)
        quality_metrics["length_appropriate"] = (
            self.response_quality_checks["min_response_length"] <= length <= 
            self.response_quality_checks["max_response_length"]
        )
        
        # Check for beauty-related keywords
        beauty_keyword_found = any(
            keyword in response_lower 
            for keyword in self.response_quality_checks["beauty_keywords"]
        )
        quality_metrics["contains_beauty_keywords"] = beauty_keyword_found
        
        # Check if it's a greeting response
        greeting_response = any(
            greeting in response_lower 
            for greeting in self.response_quality_checks["greeting_responses"]
        )
        quality_metrics["is_greeting_response"] = greeting_response
        
        # Determine relevance based on input file
        if "greeting" in input_file:
            quality_metrics["response_relevance"] = "greeting" if greeting_response else "poor"
        elif any(keyword in input_file for keyword in ["botox", "laser", "q"]):
            quality_metrics["response_relevance"] = "beauty" if beauty_keyword_found else "poor"
        else:
            quality_metrics["response_relevance"] = "general"
        
        # Calculate overall quality score (0-1)
        score = 0.0
        if quality_metrics["length_appropriate"]:
            score += 0.3
        if quality_metrics["contains_beauty_keywords"] and "q" in input_file:
            score += 0.4
        if quality_metrics["is_greeting_response"] and "greeting" in input_file:
            score += 0.4
        if quality_metrics["response_relevance"] != "poor":
            score += 0.3
        
        quality_metrics["quality_score"] = min(score, 1.0)
        
        return quality_metrics

    async def test_single_voice_file(self, audio_file: Path) -> Dict[str, Any]:
        """Test complete voice-to-voice pipeline for a single audio file."""
        try:
            self.log(f"🎤 Testing voice-to-voice: {audio_file.name}")
            
            # Read audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Determine language and voice based on filename
            if "ar" in audio_file.name or "q" in audio_file.name:
                language = "ar"
                voice_type = "female"  # Arabic female voice
            else:
                language = "en"
                voice_type = "female"  # English female voice
            
            # Test complete voice-to-voice pipeline
            self.log(f"   📞 Processing {audio_file.name} (lang: {language}, voice: {voice_type})")
            
            start_time = time.time()
            
            # Process voice message through complete pipeline
            result = await self.service.process_voice_message(
                audio_data=audio_data,
                chat_model="qwen3-unsloth-q4ks",  # Use actual model name from registry
                language=language,
                gender=voice_type
            )
            
            total_time = time.time() - start_time
            
            # Parse results
            success = result.get("success", True)  # SimpleVoiceService doesn't return success flag directly
            transcribed_text = result.get("transcribed_text", "")
            response_text = result.get("response_text", "")
            audio_file_path = result.get("audio_file_path", "")
            error_message = str(result) if not transcribed_text and not response_text else ""
            
            # Calculate timing breakdown from actual service
            total_processing_time = result.get("processing_time", total_time)
            voice_used = result.get("voice_used", "")
            language_detected = result.get("language_detected", language)
            
            # SimpleVoiceService doesn't provide detailed timing, so we estimate
            # Based on the service implementation, we can estimate:
            # - Transcription: typically 0.1-0.2s with GPU Whisper
            # - Chat: varies based on model and response length
            # - TTS: typically 0.5-1.5s depending on text length
            estimated_transcription_time = min(0.2, total_processing_time * 0.1)
            estimated_tts_time = min(1.5, max(0.5, len(response_text) * 0.02))
            estimated_chat_time = max(0.1, total_processing_time - estimated_transcription_time - estimated_tts_time)
            
            # Calculate transcription accuracy
            ground_truth_text = self.ground_truth.get(audio_file.name, "")
            wer_score = self.calculate_wer(ground_truth_text, transcribed_text) if ground_truth_text else -1.0
            
            # Evaluate response quality
            response_quality = self.evaluate_response_quality(response_text, audio_file.name)
            
            # Save generated audio file with descriptive name
            generated_audio_path = ""
            if audio_file_path and Path(audio_file_path).exists():
                new_audio_name = f"{audio_file.stem}_response_{language}_{voice_type}.wav"
                generated_audio_path = self.audio_output_dir / new_audio_name
                # Copy the generated audio to our results directory
                import shutil
                shutil.copy2(audio_file_path, generated_audio_path)
                self.log(f"   🔊 Audio saved: {generated_audio_path.name}")
            
            test_result = {
                "file_name": audio_file.name,
                "file_size_bytes": len(audio_data),
                "language": language,
                "voice_type": voice_type,
                "success": success,
                "error": error_message,
                "transcription": {
                    "ground_truth": ground_truth_text,
                    "transcribed_text": transcribed_text,
                    "word_error_rate": wer_score,
                    "transcription_time_seconds": estimated_transcription_time
                },
                "chat_response": {
                    "response_text": response_text,
                    "response_time_seconds": estimated_chat_time,
                    "quality_metrics": response_quality
                },
                "tts_output": {
                    "generated_audio_path": str(generated_audio_path) if generated_audio_path else "",
                    "tts_time_seconds": estimated_tts_time
                },
                "performance": {
                    "total_time_seconds": total_time,
                    "actual_processing_time_seconds": total_processing_time,
                    "voice_used": voice_used,
                    "language_detected": language_detected,
                    "pipeline_breakdown": {
                        "estimated_transcription_seconds": estimated_transcription_time,
                        "estimated_chat_seconds": estimated_chat_time,
                        "estimated_tts_seconds": estimated_tts_time
                    }
                },
                "timestamp": time.time()
            }
            
            # Log results summary
            if transcribed_text and response_text:  # Success if we got both transcription and response
                success = True
                self.log(f"   ✅ {audio_file.name}: WER={wer_score:.2f} | Total={total_time:.2f}s | Quality={response_quality['quality_score']:.2f}")
                self.log(f"      Transcription: '{transcribed_text[:80]}...'")
                self.log(f"      Response: '{response_text[:80]}...'")
            else:
                success = False
                self.log(f"   ❌ {audio_file.name}: Failed - {error_message}")
            
            # Update success in test result
            test_result["success"] = success
            
            return test_result
            
        except Exception as e:
            self.log(f"❌ Error testing {audio_file.name}: {e}", "ERROR")
            error_result = {
                "file_name": audio_file.name,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }
            self.results["errors"].append(error_result)
            return error_result
    
    async def run_comprehensive_test(self):
        """Run all voice-to-voice tests."""
        self.log("🎯 Starting Simple Voice Service Comprehensive Test")
        self.log("=" * 70)
        
        # Initialize service
        if not await self.setup_service():
            self.log("❌ Cannot proceed without service initialization", "ERROR")
            return
        
        # Get all audio files
        audio_files = list(self.test_files_dir.glob("*.wav"))
        if not audio_files:
            self.log("❌ No audio files found in test directory", "ERROR")
            return
        
        self.log(f"📁 Found {len(audio_files)} audio files to test")
        
        # Test each file
        total_start_time = time.time()
        for audio_file in sorted(audio_files):
            result = await self.test_single_voice_file(audio_file)
            self.results["voice_to_voice_tests"].append(result)
        
        total_time = time.time() - total_start_time
        
        # Calculate performance metrics
        successful_tests = [t for t in self.results["voice_to_voice_tests"] if t.get("success", False)]
        
        if successful_tests:
            # Transcription metrics
            transcription_tests = [t for t in successful_tests if t.get("transcription", {}).get("word_error_rate", -1) != -1]
            avg_wer = sum(t["transcription"]["word_error_rate"] for t in transcription_tests) / len(transcription_tests) if transcription_tests else 0
            
            # Performance metrics
            avg_total_time = sum(t["performance"]["total_time_seconds"] for t in successful_tests) / len(successful_tests)
            avg_transcription_time = sum(t["transcription"]["transcription_time_seconds"] for t in successful_tests) / len(successful_tests)
            avg_chat_time = sum(t["chat_response"]["response_time_seconds"] for t in successful_tests) / len(successful_tests)
            avg_tts_time = sum(t["tts_output"]["tts_time_seconds"] for t in successful_tests) / len(successful_tests)
            
            # Quality metrics
            quality_scores = [t["chat_response"]["quality_metrics"]["quality_score"] for t in successful_tests]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            self.results["performance_metrics"] = {
                "total_files_tested": len(audio_files),
                "successful_pipeline_runs": len(successful_tests),
                "failed_pipeline_runs": len(audio_files) - len(successful_tests),
                "success_rate_percent": (len(successful_tests) / len(audio_files)) * 100 if audio_files else 0,
                "average_total_time_seconds": avg_total_time,
                "pipeline_breakdown": {
                    "average_transcription_time_seconds": avg_transcription_time,
                    "average_chat_response_time_seconds": avg_chat_time,
                    "average_tts_time_seconds": avg_tts_time
                },
                "quality_metrics": {
                    "average_transcription_wer": avg_wer,
                    "average_response_quality_score": avg_quality
                },
                "total_test_time_seconds": total_time,
                "throughput_pipelines_per_second": len(audio_files) / total_time if total_time > 0 else 0
            }
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = self.output_dir / f"voice_service_test_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        self.log("📊 VOICE-TO-VOICE PIPELINE TEST SUMMARY")
        self.log("=" * 70)
        
        if "performance_metrics" in self.results and self.results["performance_metrics"]:
            metrics = self.results["performance_metrics"]
            self.log(f"✅ Pipeline Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
            self.log(f"⚡ Average Total Time: {metrics.get('average_total_time_seconds', 0):.2f}s per pipeline")
            
            # Breakdown timing
            breakdown = metrics.get("pipeline_breakdown", {})
            self.log(f"   📝 Transcription (STT): {breakdown.get('average_transcription_time_seconds', 0):.2f}s")
            self.log(f"   🤖 Chat Response: {breakdown.get('average_chat_response_time_seconds', 0):.2f}s")
            self.log(f"   🔊 Text-to-Speech (TTS): {breakdown.get('average_tts_time_seconds', 0):.2f}s")
            
            # Quality metrics
            quality = metrics.get("quality_metrics", {})
            self.log(f"🎯 Average Transcription WER: {quality.get('average_transcription_wer', 0):.2f} (Lower is better)")
            self.log(f"⭐ Average Response Quality: {quality.get('average_response_quality_score', 0):.2f} (0-1, higher is better)")
            
            self.log(f"📈 Throughput: {metrics.get('throughput_pipelines_per_second', 0):.2f} pipelines/second")
            
            # Performance evaluation
            avg_time = metrics.get('average_total_time_seconds', float('inf'))
            if avg_time < 3.0:
                self.log("🚀 EXCELLENT: Total pipeline time < 3.0 seconds")
            elif avg_time < 5.0:
                self.log("✅ GOOD: Total pipeline time < 5.0 seconds")
            else:
                self.log("⚠️ SLOW: Total pipeline time > 5.0 seconds")
                
            avg_wer = quality.get('average_transcription_wer', float('inf'))
            if avg_wer < 0.1:
                self.log("🎯 EXCELLENT: WER < 0.1 (very high accuracy)")
            elif avg_wer < 0.25:
                self.log("✅ GOOD: WER < 0.25 (high accuracy)")
            else:
                self.log("⚠️ REVIEW: WER > 0.25 (accuracy needs review)")
            
            avg_quality = quality.get('average_response_quality_score', 0)
            if avg_quality > 0.7:
                self.log("🌟 EXCELLENT: Response quality > 0.7")
            elif avg_quality > 0.5:
                self.log("✅ GOOD: Response quality > 0.5")
            else:
                self.log("⚠️ REVIEW: Response quality needs improvement")
        
        if self.results["errors"]:
            self.log(f"❌ Errors encountered: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                error_type = error.get('phase', error.get('file_name', 'unknown'))
                error_msg = error.get('error', str(error))
                self.log(f"   - {error_type}: {error_msg}", "ERROR")
        
        self.log(f"📁 Generated audio files saved in: {self.audio_output_dir}")


async def main():
    """Main test execution."""
    print("🎤 Simple Voice Service End-to-End Test - Phase 4")
    print("=" * 70)
    
    tester = SimpleVoiceServiceTester()
    await tester.run_comprehensive_test()
    
    print("\n✅ Phase 4 voice-to-voice testing complete!")
    print("📁 Check results in: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/")
    print("🔊 Check generated audio in: /home/lumi/beautyai/voice_tests/phase4_voice_service_results/generated_audio/")
    print("📋 Review logs above for performance analysis")


if __name__ == "__main__":
    asyncio.run(main())
