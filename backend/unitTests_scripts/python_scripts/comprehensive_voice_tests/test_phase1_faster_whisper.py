#!/usr/bin/env python3
"""
Phase 1: Faster-Whisper Service Test

Tests the FasterWhisperTranscriptionService for:
- Accuracy with Arabic and English audio files
- Processing speed and performance
- Model loading and initialization
- Error handling and edge cases

Output saved to: /home/lumi/beautyai/voice_tests/phase1_faster_whisper_results/
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add the backend src directory to Python path
backend_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from beautyai_inference.services.voice.transcription.transformers_whisper_service import TransformersWhisperService


class TransformersWhisperTester:
    """Comprehensive tester for TransformersWhisperService."""
    
    def __init__(self):
        self.service = None
        self.test_files_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions")
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/phase1_transformers_whisper_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.results = {
            "test_timestamp": time.time(),
            "model_loading": {},
            "transcription_tests": [],
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
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def setup_service(self) -> bool:
        """Initialize the TransformersWhisperService."""
        try:
            self.log("🚀 Initializing TransformersWhisperService...")
            start_time = time.time()
            
            self.service = TransformersWhisperService()
            
            # Test model loading
            self.log("📥 Loading whisper-turbo-arabic model...")
            model_load_start = time.time()
            success = self.service.load_whisper_model("whisper-turbo-arabic")
            model_load_time = time.time() - model_load_start
            
            self.results["model_loading"] = {
                "model_name": "whisper-turbo-arabic",
                "success": success,
                "load_time_seconds": model_load_time,
                "timestamp": time.time()
            }
            
            if success:
                self.log(f"✅ Model loaded successfully in {model_load_time:.2f} seconds")
                return True
            else:
                self.log("❌ Failed to load whisper-turbo-arabic model", "ERROR")
                return False
                
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
        """
        Calculates the Word Error Rate (WER) between a reference and hypothesis.
        WER = (Substitutions + Deletions + Insertions) / Number of words in reference
        """
        if not reference:
            return 1.0 if hypothesis else 0.0
            
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Using Levenshtein distance for simplicity
        # This is a common way to calculate WER
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

    def test_single_file(self, audio_file: Path) -> Dict[str, Any]:
        """Test transcription of a single audio file."""
        try:
            self.log(f"🎤 Testing: {audio_file.name}")
            
            # Read audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Test transcription with timing
            start_time = time.time()
            
            # Test with Arabic language preference
            transcription_ar = self.service.transcribe_audio_bytes(
                audio_bytes=audio_data,
                audio_format=audio_file.suffix[1:],  # Remove the dot
                language="ar"
            )
            
            ar_time = time.time() - start_time
            
            # Test with English language preference
            start_time = time.time()
            transcription_en = self.service.transcribe_audio_bytes(
                audio_bytes=audio_data,
                audio_format=audio_file.suffix[1:],
                language="en"
            )
            
            en_time = time.time() - start_time
            
            # Test with auto-detection
            start_time = time.time()
            transcription_auto = self.service.transcribe_audio_bytes(
                audio_bytes=audio_data,
                audio_format=audio_file.suffix[1:],
                language=None  # Auto-detect
            )
            
            auto_time = time.time() - start_time
            
            # Check accuracy against ground truth
            ground_truth_text = self.ground_truth.get(audio_file.name, "")
            wer_score = self.calculate_wer(ground_truth_text, transcription_auto) if ground_truth_text else -1.0

            result = {
                "file_name": audio_file.name,
                "file_size_bytes": len(audio_data),
                "transcriptions": {
                    "arabic_mode": transcription_ar,
                    "english_mode": transcription_en,
                    "auto_detect": transcription_auto
                },
                "ground_truth": ground_truth_text,
                "word_error_rate": wer_score,
                "processing_times": {
                    "arabic_mode_seconds": ar_time,
                    "english_mode_seconds": en_time,
                    "auto_detect_seconds": auto_time
                },
                "timestamp": time.time(),
                "success": True
            }
            
            self.log(f"✅ {audio_file.name}: WER={wer_score:.2f} | Text={transcription_auto[:50]}...")
            return result
            
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
    
    def run_comprehensive_test(self):
        """Run all transcription tests."""
        self.log("🎯 Starting Transformers-Whisper Comprehensive Test")
        self.log("=" * 60)
        
        # Initialize service
        if not self.setup_service():
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
            result = self.test_single_file(audio_file)
            self.results["transcription_tests"].append(result)
        
        total_time = time.time() - total_start_time
        
        # Calculate performance metrics
        successful_tests = [t for t in self.results["transcription_tests"] if t.get("success", False) and t.get("word_error_rate", -1) != -1]
        
        if successful_tests:
            avg_time = sum(t["processing_times"]["auto_detect_seconds"] for t in successful_tests) / len(successful_tests)
            avg_wer = sum(t["word_error_rate"] for t in successful_tests) / len(successful_tests)
            
            self.results["performance_metrics"] = {
                "total_files_tested": len(audio_files),
                "successful_transcriptions": len(successful_tests),
                "failed_transcriptions": len(audio_files) - len(successful_tests),
                "success_rate_percent": (len(successful_tests) / len(audio_files)) * 100 if audio_files else 0,
                "average_processing_time_seconds": avg_time,
                "average_word_error_rate": avg_wer,
                "total_test_time_seconds": total_time,
                "throughput_files_per_second": len(audio_files) / total_time if total_time > 0 else 0
            }
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = self.output_dir / f"transformers_whisper_test_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        self.log("📊 TEST SUMMARY")
        self.log("=" * 60)
        
        if "performance_metrics" in self.results and self.results["performance_metrics"]:
            metrics = self.results["performance_metrics"]
            self.log(f"✅ Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
            self.log(f"⚡ Average Speed: {metrics.get('average_processing_time_seconds', 0):.2f}s per file")
            self.log(f"🎯 Average Word Error Rate (WER): {metrics.get('average_word_error_rate', 0):.2f} (Lower is better)")
            self.log(f"📈 Throughput: {metrics.get('throughput_files_per_second', 0):.2f} files/second")
            
            # Performance evaluation
            avg_time = metrics.get('average_processing_time_seconds', float('inf'))
            if avg_time < 1.5:
                self.log("🚀 EXCELLENT: Processing time < 1.5 seconds")
            elif avg_time < 3.0:
                self.log("✅ GOOD: Processing time < 3.0 seconds")
            else:
                self.log("⚠️ SLOW: Processing time > 3.0 seconds")
                
            avg_wer = metrics.get('average_word_error_rate', float('inf'))
            if avg_wer < 0.1:
                self.log("🎯 EXCELLENT: WER < 0.1 (very high accuracy)")
            elif avg_wer < 0.25:
                self.log("✅ GOOD: WER < 0.25 (high accuracy)")
            else:
                self.log("⚠️ REVIEW: WER > 0.25 (accuracy needs review)")
        
        if self.results["errors"]:
            self.log(f"❌ Errors encountered: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                error_type = error.get('phase', error.get('file_name', 'unknown'))
                error_msg = error.get('error', str(error))
                self.log(f"   - {error_type}: {error_msg}", "ERROR")


def main():
    """Main test execution."""
    print("🎤 Transformers-Whisper Service Test - Phase 1 (GPU Optimized)")
    print("=" * 60)
    
    tester = TransformersWhisperTester()
    tester.run_comprehensive_test()
    
    print("\n✅ Phase 1 testing complete!")
    print("📁 Check results in: /home/lumi/beautyai/voice_tests/phase1_transformers_whisper_results/")
    print("📋 Review logs above for performance analysis")


if __name__ == "__main__":
    main()
