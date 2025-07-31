#!/usr/bin/env python3
"""
GPU-Accelerated Faster-Whisper Benchmarking Script
Tests the new RTX 4090 GPU acceleration with Whisper Large-v3-Turbo

Expected Results:
- CPU Previous: ~5.0s per file  
- GPU Target: <1.0s per file (10x speedup)
- Accuracy: WER <0.1 (better than 0.024 from Phase 2)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to Python path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent.parent.parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

from beautyai_inference.services.voice.transcription.faster_whisper_service import FasterWhisperTranscriptionService
from beautyai_inference.config.config_manager import AppConfig


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple WER calculation (Levenshtein distance)
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main():
    """Main GPU benchmarking function."""
    print("🚀 GPU-Accelerated Faster-Whisper Benchmarking - Phase 3")
    print("=" * 60)
    
    # Initialize GPU-optimized service
    whisper_service = FasterWhisperTranscriptionService()
    
    # Check GPU availability
    import torch
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available! GPU testing requires NVIDIA drivers.")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ Device: {whisper_service.device}")
    print(f"✅ Compute Type: {whisper_service.compute_type}")
    print()
    
    # Load GPU-optimized model
    print("📥 Loading Whisper Large-v3-Turbo (GPU-optimized)...")
    load_start = time.time()
    
    if not whisper_service.load_whisper_model("whisper-turbo-arabic"):
        print("❌ ERROR: Failed to load Whisper model")
        return
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")
    print()
    
    # Test files with ground truth
    test_files = {
        "file1_ar_beauty_question.wav": "ما هي أفضل طريقة للعناية بالبشرة الجافة؟",
        "file2_ar_voice_natural.wav": "أريد أن أسألك عن أفضل كريم للوجه",
        "file3_ar_clear_audio.wav": "هل يمكنك أن تنصحني بروتين يومي للعناية بالبشرة؟",
        "file4_en_beauty_care.wav": "What are the best ingredients for anti-aging skincare?",
        "file5_ar_skin_care.wav": "ما هو أفضل وقت لاستخدام السيروم؟",
        "file6_ar_makeup_question.wav": "كيف أختار اللون المناسب لكريم الأساس؟",
        "file7_en_hair_care.wav": "How often should I wash my hair for healthy growth?",
        "file8_ar_mixed_audio.wav": "أريد معرفة أفضل طريقة لإزالة المكياج",
        "file9_ar_long_question.wav": "هل تنصحني باستخدام واقي الشمس حتى في الأيام الغائمة؟",
        "file10_en_skincare_routine.wav": "Can you recommend a nighttime skincare routine?",
        "file11_ar_natural_beauty.wav": "ما هي الوصفات الطبيعية للعناية بالشعر؟",
        "file12_ar_acne_treatment.wav": "كيف أعالج البثور بطريقة طبيعية؟",
        "file13_en_dry_skin.wav": "What moisturizer works best for very dry skin?",
        "file14_ar_beauty_tips.wav": "أعطني نصائح للحصول على بشرة نضرة ومشرقة"
    }
    
    # Paths
    voice_tests_dir = Path(__file__).parent.parent.parent.parent / "voice_tests"
    input_dir = voice_tests_dir / "input_test_questions"
    output_dir = voice_tests_dir / "phase3_gpu_whisper_results"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    total_files = 0
    successful_files = 0
    total_processing_time = 0.0
    total_wer = 0.0
    
    print("🎯 Starting GPU-accelerated benchmarking...")
    print()
    
    for filename, ground_truth in test_files.items():
        file_path = input_dir / filename
        
        if not file_path.exists():
            print(f"⚠️  SKIP: {filename} (file not found)")
            continue
        
        total_files += 1
        print(f"🔄 Processing: {filename}")
        
        # Measure processing time
        start_time = time.time()
        
        try:
            # GPU-accelerated transcription
            result = whisper_service.transcribe_audio_file(str(file_path), language="ar" if filename.startswith("file") and "en" not in filename else "auto")
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            if result:
                # Calculate accuracy
                wer = calculate_wer(ground_truth, result)
                total_wer += wer
                successful_files += 1
                
                # Store result
                test_result = {
                    "filename": filename,
                    "processing_time_seconds": round(processing_time, 3),
                    "ground_truth": ground_truth,
                    "transcription": result,
                    "word_error_rate": round(wer, 4),
                    "accuracy_percent": round((1 - wer) * 100, 2),
                    "success": True,
                    "gpu_accelerated": True,
                    "model": "whisper-large-v3-turbo",
                    "compute_type": whisper_service.compute_type
                }
                results.append(test_result)
                
                # Display results
                print(f"   ⚡ Time: {processing_time:.3f}s")
                print(f"   📝 Result: {result[:60]}...")
                print(f"   🎯 WER: {wer:.4f} ({(1-wer)*100:.1f}% accuracy)")
                print(f"   ✅ Success")
                
            else:
                print(f"   ❌ Transcription failed")
                test_result = {
                    "filename": filename,
                    "processing_time_seconds": round(processing_time, 3),
                    "ground_truth": ground_truth,
                    "transcription": None,
                    "word_error_rate": 1.0,
                    "accuracy_percent": 0.0,
                    "success": False,
                    "error": "Transcription returned None",
                    "gpu_accelerated": True,
                    "model": "whisper-large-v3-turbo",
                    "compute_type": whisper_service.compute_type
                }
                results.append(test_result)
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ❌ Error: {str(e)}")
            test_result = {
                "filename": filename,
                "processing_time_seconds": round(processing_time, 3),
                "ground_truth": ground_truth,
                "transcription": None,
                "word_error_rate": 1.0,
                "accuracy_percent": 0.0,
                "success": False,
                "error": str(e),
                "gpu_accelerated": True,
                "model": "whisper-large-v3-turbo",
                "compute_type": whisper_service.compute_type
            }
            results.append(test_result)
        
        print()
    
    # Calculate summary statistics
    if successful_files > 0:
        avg_processing_time = total_processing_time / total_files
        avg_wer = total_wer / successful_files
        avg_accuracy = (1 - avg_wer) * 100
        success_rate = successful_files / total_files * 100
    else:
        avg_processing_time = 0
        avg_wer = 1.0
        avg_accuracy = 0
        success_rate = 0
    
    # Summary
    print("📊 GPU BENCHMARKING RESULTS SUMMARY")
    print("=" * 50)
    print(f"🎯 Model: Whisper Large-v3-Turbo (GPU)")
    print(f"💻 Device: {whisper_service.device}")
    print(f"⚙️  Compute: {whisper_service.compute_type}")
    print(f"📁 Total Files: {total_files}")
    print(f"✅ Successful: {successful_files}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"⚡ Avg Processing Time: {avg_processing_time:.3f}s/file")
    print(f"🎯 Avg WER: {avg_wer:.4f}")
    print(f"📊 Avg Accuracy: {avg_accuracy:.2f}%")
    print(f"⏱️  Total Time: {total_processing_time:.2f}s")
    print()
    
    # Performance comparison with previous phases
    print("📈 PERFORMANCE COMPARISON")
    print("-" * 30)
    print("Phase 1 (CPU Baseline): ~7.5s/file, WER: 0.19")
    print("Phase 2 (CPU Optimized): ~5.0s/file, WER: 0.024")
    print(f"Phase 3 (GPU Optimized): {avg_processing_time:.3f}s/file, WER: {avg_wer:.4f}")
    
    if avg_processing_time > 0:
        speedup_vs_phase1 = 7.5 / avg_processing_time
        speedup_vs_phase2 = 5.0 / avg_processing_time
        print(f"🚀 Speedup vs Phase 1: {speedup_vs_phase1:.1f}x")
        print(f"🚀 Speedup vs Phase 2: {speedup_vs_phase2:.1f}x")
    
    # Real-time capability assessment
    print()
    print("🎙️  REAL-TIME VOICE CAPABILITY")
    print("-" * 35)
    if avg_processing_time < 1.0:
        print("✅ EXCELLENT: Sub-second processing - Perfect for real-time voice chat!")
    elif avg_processing_time < 2.0:
        print("✅ GOOD: Under 2s processing - Suitable for real-time voice chat")
    elif avg_processing_time < 3.0:
        print("⚠️  FAIR: 2-3s processing - Borderline for real-time voice chat")
    else:
        print("❌ POOR: >3s processing - Not suitable for real-time voice chat")
    
    # Save detailed results
    summary_data = {
        "test_info": {
            "test_name": "Phase 3: GPU-Accelerated Faster-Whisper Benchmark",
            "model": "whisper-large-v3-turbo",
            "device": whisper_service.device,
            "compute_type": whisper_service.compute_type,
            "gpu_accelerated": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary_stats": {
            "total_files": total_files,
            "successful_files": successful_files,
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time_seconds": round(avg_processing_time, 4),
            "average_wer": round(avg_wer, 4),
            "average_accuracy_percent": round(avg_accuracy, 2),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "model_load_time_seconds": round(load_time, 2)
        },
        "performance_comparison": {
            "phase_1_cpu_baseline": {"time_per_file": 7.5, "wer": 0.19},
            "phase_2_cpu_optimized": {"time_per_file": 5.0, "wer": 0.024},
            "phase_3_gpu_optimized": {"time_per_file": round(avg_processing_time, 3), "wer": round(avg_wer, 4)},
            "speedup_vs_phase1": round(7.5 / avg_processing_time, 1) if avg_processing_time > 0 else 0,
            "speedup_vs_phase2": round(5.0 / avg_processing_time, 1) if avg_processing_time > 0 else 0
        },
        "detailed_results": results
    }
    
    # Save to JSON
    output_file = output_dir / "gpu_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    print("🎉 GPU Benchmarking Complete!")
    
    # GPU memory stats
    try:
        memory_stats = whisper_service.get_memory_stats()
        if "gpu_memory_allocated_mb" in memory_stats:
            print(f"📊 GPU Memory Used: {memory_stats['gpu_memory_allocated_mb']:.1f} MB")
    except:
        pass


if __name__ == "__main__":
    main()
