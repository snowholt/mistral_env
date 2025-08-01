#!/usr/bin/env python3
"""
Phase 3: Voice Generation Test

Generates voice files using Edge TTS from ground truth texts for:
- Arabic beauty/medical questions and answers
- English beauty/medical questions and answers
- Multiple voice variations for testing
- Performance benchmarking of TTS generation

Output saved to: /home/lumi/beautyai/voice_tests/phase3_generated_voices/
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add the backend src directory to Python path
backend_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from beautyai_inference.inference_engines.edge_tts_engine import EdgeTTSEngine
from beautyai_inference.config.config_manager import ModelConfig


class VoiceGenerationTester:
    """Comprehensive voice generation tester using Edge TTS."""
    
    def __init__(self):
        self.engine = None
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/phase3_generated_voices")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.results = {
            "test_timestamp": time.time(),
            "engine_loading": {},
            "voice_generation_tests": [],
            "performance_metrics": {},
            "errors": []
        }
        
        # Ground truth texts for voice generation
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
        
        # Voice configurations for different speakers
        self.voice_configs = {
            "en": [
                {"voice": "en-US-JennyNeural", "name": "jenny", "description": "Professional female English"},
                {"voice": "en-US-AriaNeural", "name": "aria", "description": "Natural female English"},
                {"voice": "en-US-DavisNeural", "name": "davis", "description": "Professional male English"}
            ],
            "ar": [
                {"voice": "ar-SA-ZariyahNeural", "name": "zariyah", "description": "Professional female Arabic"},
                {"voice": "ar-SA-HamedNeural", "name": "hamed", "description": "Professional male Arabic"},
                {"voice": "ar-AE-FatimaNeural", "name": "fatima", "description": "UAE female Arabic"}
            ]
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def setup_engine(self) -> bool:
        """Initialize the Edge TTS Engine."""
        try:
            self.log("🚀 Initializing Edge TTS Engine...")
            start_time = time.time()
            
            config = ModelConfig(model_id='microsoft/edge-tts', engine_type='edge_tts')
            self.engine = EdgeTTSEngine(config)
            
            # Test engine loading
            self.log("📥 Loading Edge TTS engine...")
            load_start = time.time()
            success = self.engine.load_model()
            load_time = time.time() - load_start
            
            self.results["engine_loading"] = {
                "engine_type": "edge_tts",
                "success": success,
                "load_time_seconds": load_time,
                "timestamp": time.time()
            }
            
            if success:
                self.log(f"✅ Engine loaded successfully in {load_time:.3f} seconds")
                return True
            else:
                self.log("❌ Failed to load Edge TTS engine", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Engine initialization failed: {e}", "ERROR")
            self.results["errors"].append({
                "phase": "initialization",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            })
            return False

    def detect_language(self, text: str) -> str:
        """Simple language detection based on Arabic characters."""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars > 0 and arabic_chars / total_chars > 0.5:
            return "ar"
        else:
            return "en"
    
    def generate_voice_file(self, text: str, filename: str, voice_config: Dict[str, str], language: str) -> Dict[str, Any]:
        """Generate a single voice file."""
        try:
            self.log(f"🎤 Generating: {filename} with {voice_config['name']} voice")
            
            # Create output filename with voice variant
            output_filename = f"{filename.replace('.wav', '')}_{voice_config['name']}.wav"
            output_path = self.output_dir / output_filename
            
            # Generate voice with timing
            start_time = time.time()
            
            audio_path = self.engine.text_to_speech(
                text=text,
                language=language,
                voice=voice_config.get("voice"),
                output_path=str(output_path)
            )
            
            generation_time = time.time() - start_time
            
            # Check if file was created successfully
            success = output_path.exists()
            file_size = output_path.stat().st_size if success else 0
            
            result = {
                "original_filename": filename,
                "generated_filename": output_filename,
                "text": text,
                "language": language,
                "voice_config": voice_config,
                "generation_time_seconds": generation_time,
                "file_size_bytes": file_size,
                "success": success,
                "output_path": str(output_path) if success else None,
                "timestamp": time.time()
            }
            
            if success:
                self.log(f"✅ Generated {output_filename} in {generation_time:.3f}s ({file_size} bytes)")
            else:
                self.log(f"❌ Failed to generate {output_filename}", "ERROR")
            
            return result
            
        except Exception as e:
            self.log(f"❌ Error generating {filename}: {e}", "ERROR")
            error_result = {
                "original_filename": filename,
                "text": text,
                "language": language,
                "voice_config": voice_config,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }
            self.results["errors"].append(error_result)
            return error_result
    
    def run_voice_generation_test(self):
        """Run comprehensive voice generation test."""
        self.log("🎯 Starting Voice Generation Test - Phase 3")
        self.log("=" * 60)
        
        # Initialize engine
        if not self.setup_engine():
            self.log("❌ Cannot proceed without engine initialization", "ERROR")
            return
        
        self.log(f"📝 Generating voices for {len(self.ground_truth)} texts")
        
        # Generate voices for each text with multiple voice variants
        total_start_time = time.time()
        total_files_generated = 0
        
        for filename, text in self.ground_truth.items():
            self.log(f"\n📄 Processing: {filename}")
            self.log(f"   Text: {text[:50]}...")
            
            # Detect language
            language = self.detect_language(text)
            self.log(f"   Language: {language}")
            
            # Get voice configurations for the detected language
            voices = self.voice_configs.get(language, [])
            if not voices:
                self.log(f"⚠️ No voice configurations for language {language}", "WARNING")
                continue
            
            # Generate voice with each voice variant
            for voice_config in voices:
                result = self.generate_voice_file(text, filename, voice_config, language)
                self.results["voice_generation_tests"].append(result)
                
                if result.get("success", False):
                    total_files_generated += 1
        
        total_time = time.time() - total_start_time
        
        # Calculate performance metrics
        successful_generations = [t for t in self.results["voice_generation_tests"] if t.get("success", False)]
        
        if successful_generations:
            avg_time = sum(t["generation_time_seconds"] for t in successful_generations) / len(successful_generations)
            total_size = sum(t["file_size_bytes"] for t in successful_generations)
            
            self.results["performance_metrics"] = {
                "total_texts_processed": len(self.ground_truth),
                "total_voice_variants": sum(len(voices) for voices in self.voice_configs.values()),
                "total_files_attempted": len(self.results["voice_generation_tests"]),
                "successful_generations": len(successful_generations),
                "failed_generations": len(self.results["voice_generation_tests"]) - len(successful_generations),
                "success_rate_percent": (len(successful_generations) / len(self.results["voice_generation_tests"])) * 100 if self.results["voice_generation_tests"] else 0,
                "average_generation_time_seconds": avg_time,
                "total_generation_time_seconds": total_time,
                "total_output_size_bytes": total_size,
                "total_output_size_mb": total_size / (1024 * 1024),
                "throughput_files_per_second": len(successful_generations) / total_time if total_time > 0 else 0
            }
        
        # Cleanup engine
        self.engine.unload_model()
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = self.output_dir / f"voice_generation_test_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        self.log("\n📊 VOICE GENERATION TEST SUMMARY")
        self.log("=" * 60)
        
        if "performance_metrics" in self.results and self.results["performance_metrics"]:
            metrics = self.results["performance_metrics"]
            self.log(f"✅ Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
            self.log(f"📝 Texts Processed: {metrics.get('total_texts_processed', 0)}")
            self.log(f"🎤 Voice Variants: {metrics.get('total_voice_variants', 0)}")
            self.log(f"📁 Files Generated: {metrics.get('successful_generations', 0)}")
            self.log(f"⚡ Average Generation Time: {metrics.get('average_generation_time_seconds', 0):.3f}s per file")
            self.log(f"💾 Total Output Size: {metrics.get('total_output_size_mb', 0):.1f} MB")
            self.log(f"📈 Throughput: {metrics.get('throughput_files_per_second', 0):.2f} files/second")
            
            # Performance evaluation
            avg_time = metrics.get('average_generation_time_seconds', float('inf'))
            if avg_time < 1.0:
                self.log("🚀 EXCELLENT: Generation time < 1.0 second per file")
            elif avg_time < 2.0:
                self.log("✅ GOOD: Generation time < 2.0 seconds per file")
            else:
                self.log("⚠️ SLOW: Generation time > 2.0 seconds per file")
        
        if self.results["errors"]:
            self.log(f"❌ Errors encountered: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                error_type = error.get('phase', error.get('original_filename', 'unknown'))
                error_msg = error.get('error', str(error))
                self.log(f"   - {error_type}: {error_msg}", "ERROR")
        
        # List generated files
        self.log(f"\n📁 Generated files location: {self.output_dir}")
        generated_files = list(self.output_dir.glob("*.wav"))
        if generated_files:
            self.log(f"🎵 Generated {len(generated_files)} audio files:")
            for file_path in sorted(generated_files):
                size_kb = file_path.stat().st_size / 1024
                self.log(f"   - {file_path.name} ({size_kb:.1f} KB)")
        else:
            self.log("⚠️ No audio files were generated")


def main():
    """Main test execution."""
    print("🎤 Voice Generation Test - Phase 3 (Edge TTS)")
    print("=" * 60)
    
    tester = VoiceGenerationTester()
    tester.run_voice_generation_test()
    
    print("\n✅ Phase 3 voice generation complete!")
    print("📁 Check generated files in: /home/lumi/beautyai/voice_tests/phase3_generated_voices/")
    print("📋 Review logs above for performance analysis")


if __name__ == "__main__":
    main()
