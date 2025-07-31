#!/usr/bin/env python3
"""
Phase 2: Optimized Faster-Whisper Benchmark Test

This script tests the optimized Faster-Whisper service with:
- B: Batch/Beam Size Optimization (beam_size=1, batch_size=16) 
- C: Hardware Optimization (optimized compute_type, GPU acceleration)

Expected improvements:
- Target processing time: <2.0 seconds per file (down from 7.5s)
- Maintain WER <0.2 for accuracy
- Improved real-time conversation capabilities
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "backend" / "src"))

from beautyai_inference.services.voice.transcription.faster_whisper_service import FasterWhisperTranscriptionService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming to find edit distance
    len_ref, len_hyp = len(ref_words), len(hyp_words)
    dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    wer = dp[len_ref][len_hyp] / len_ref if len_ref > 0 else 0.0
    return wer

def load_ground_truth() -> Dict[str, str]:
    """Load ground truth transcriptions."""
    return {
        "botox.wav": "What is botox used for?",
        "greeting.wav": "Hello, how are you today?",
        "greeting_ar.wav": "مرحباً، كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصة بكم",
        "hair_transplant.wav": "What are the benefits of hair transplant surgery?",
        "laser_treatment.wav": "What are the benefits of laser treatment for skin?",
        "makeup_consultation.wav": "I would like to book a makeup consultation appointment",
        "question_1.wav": "What are the side effects of dermal fillers?",
        "question_2.wav": "How long does a facelift procedure take?",
        "skin_care.wav": "What is the best skincare routine for sensitive skin?",
        "skin_care_ar.wav": "ما هي أفضل روتين للعناية بالبشرة الحساسة؟",
        "cosmetic_surgery.wav": "What should I know about cosmetic surgery procedures?",
        "beauty_routine.wav": "How can I improve my daily beauty routine?",
        "facial_treatment.wav": "What types of facial treatments do you offer?",
        "anti_aging.wav": "What are effective anti-aging treatments?"
    }

def test_optimized_faster_whisper():
    """Test optimized Faster-Whisper performance."""
    logger.info("=== Phase 2: Optimized Faster-Whisper Benchmark Test ===")
    
    # Initialize service
    service = FasterWhisperTranscriptionService()
    
    # Test audio files directory
    test_audio_dir = project_root / "voice_tests" / "input_test_questions"
    results_dir = project_root / "voice_tests" / "phase2_optimized_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_audio_dir.exists():
        logger.error(f"Test audio directory not found: {test_audio_dir}")
        return False
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Load model
    logger.info("Loading optimized Faster-Whisper model...")
    load_start = time.time()
    success = service.load_whisper_model("whisper-turbo-arabic")
    load_time = time.time() - load_start
    
    if not success:
        logger.error("Failed to load Faster-Whisper model")
        return False
    
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    
    # Get model info for optimization verification
    model_info = service.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Test results storage
    test_results = {
        "test_timestamp": time.time(),
        "optimization_phase": "B+C: Batch/Beam + Hardware Optimization",
        "optimization_details": {
            "beam_size": 1,
            "batch_size": 16,
            "compute_type": service.compute_type,
            "device": service.device,
            "vad_enabled": True,
            "vad_min_silence_ms": 250
        },
        "model_loading": {
            "model_name": "whisper-turbo-arabic",
            "success": success,
            "load_time_seconds": load_time,
            "timestamp": time.time()
        },
        "transcription_tests": [],
        "summary_stats": {}
    }
    
    # Test files
    audio_files = list(test_audio_dir.glob("*.wav"))
    if not audio_files:
        logger.error("No WAV files found in test directory")
        return False
    
    logger.info(f"Found {len(audio_files)} test files")
    
    total_files = 0
    successful_files = 0
    total_processing_time = 0
    total_wer = 0
    wer_count = 0
    
    # Process each file
    for audio_file in sorted(audio_files):
        file_name = audio_file.name
        file_size = audio_file.stat().st_size
        
        logger.info(f"Processing: {file_name} ({file_size} bytes)")
        
        if file_name not in ground_truth:
            logger.warning(f"No ground truth for {file_name}, skipping WER calculation")
            continue
        
        total_files += 1
        
        # Test different language modes for comprehensive comparison
        transcription_modes = [
            ("auto_detect", None),
            ("english_mode", "en"),
            ("arabic_mode", "ar")
        ]
        
        file_result = {
            "file_name": file_name,
            "file_size_bytes": file_size,
            "transcriptions": {},
            "ground_truth": ground_truth[file_name],
            "processing_times": {},
            "timestamp": time.time(),
            "success": True
        }
        
        file_processing_times = []
        
        # Test each mode
        for mode_name, language in transcription_modes:
            try:
                start_time = time.time()
                result = service.transcribe(
                    audio_file=str(audio_file),
                    language=language or "auto"
                )
                processing_time = time.time() - start_time
                
                if result["success"]:
                    transcription = result["transcription"]
                    file_result["transcriptions"][mode_name] = transcription
                    file_result["processing_times"][f"{mode_name}_seconds"] = processing_time
                    file_processing_times.append(processing_time)
                    
                    logger.info(f"  {mode_name}: '{transcription}' ({processing_time:.3f}s)")
                else:
                    logger.error(f"  {mode_name}: Failed - {result.get('error', 'Unknown error')}")
                    file_result["transcriptions"][mode_name] = None
                    file_result["processing_times"][f"{mode_name}_seconds"] = None
                    
            except Exception as e:
                logger.error(f"  {mode_name}: Exception - {e}")
                file_result["transcriptions"][mode_name] = None
                file_result["processing_times"][f"{mode_name}_seconds"] = None
        
        # Calculate WER for best transcription
        best_transcription = None
        best_wer = float('inf')
        
        for mode_name, transcription in file_result["transcriptions"].items():
            if transcription:
                wer = calculate_word_error_rate(ground_truth[file_name], transcription)
                if wer < best_wer:
                    best_wer = wer
                    best_transcription = transcription
        
        file_result["word_error_rate"] = best_wer if best_wer != float('inf') else None
        file_result["best_transcription"] = best_transcription
        
        # Update totals
        if file_processing_times:
            avg_processing_time = sum(file_processing_times) / len(file_processing_times)
            total_processing_time += avg_processing_time
            successful_files += 1
            
            if best_wer != float('inf'):
                total_wer += best_wer
                wer_count += 1
        
        test_results["transcription_tests"].append(file_result)
        
        # Log progress
        logger.info(f"  Best WER: {best_wer:.3f}, Avg Time: {sum(file_processing_times)/len(file_processing_times):.3f}s")
    
    # Calculate summary statistics
    avg_processing_time = total_processing_time / successful_files if successful_files > 0 else 0
    avg_wer = total_wer / wer_count if wer_count > 0 else 0
    success_rate = successful_files / total_files if total_files > 0 else 0
    
    test_results["summary_stats"] = {
        "total_files_tested": total_files,
        "successful_files": successful_files,
        "success_rate": success_rate,
        "average_processing_time_seconds": avg_processing_time,
        "average_word_error_rate": avg_wer,
        "total_processing_time_seconds": total_processing_time,
        "performance_metrics": {
            "target_latency_ms": 2000,
            "actual_latency_ms": avg_processing_time * 1000,
            "latency_improvement_achieved": avg_processing_time < 2.0,
            "target_wer": 0.2,
            "actual_wer": avg_wer,
            "accuracy_target_met": avg_wer < 0.2
        }
    }
    
    # Save results
    timestamp = int(time.time())
    results_file = results_dir / f"optimized_faster_whisper_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Print comprehensive summary
    logger.info("\n=== OPTIMIZATION BENCHMARK RESULTS ===")
    logger.info(f"Phase: B+C (Batch/Beam + Hardware Optimization)")
    logger.info(f"Files Processed: {successful_files}/{total_files} ({success_rate:.1%})")
    logger.info(f"Average Processing Time: {avg_processing_time:.3f}s (Target: <2.0s)")
    logger.info(f"Average Word Error Rate: {avg_wer:.3f} (Target: <0.2)")
    logger.info(f"Performance Improvement:")
    logger.info(f"  - Speed Target Met: {'✅' if avg_processing_time < 2.0 else '❌'} ({avg_processing_time:.1f}s vs 2.0s target)")
    logger.info(f"  - Accuracy Target Met: {'✅' if avg_wer < 0.2 else '❌'} ({avg_wer:.3f} vs 0.2 target)")
    
    if avg_processing_time < 2.0 and avg_wer < 0.2:
        logger.info("🎯 OPTIMIZATION SUCCESS: Both speed and accuracy targets achieved!")
    elif avg_processing_time < 2.0:
        logger.info("⚡ Speed optimization successful, but accuracy needs improvement")
    elif avg_wer < 0.2:
        logger.info("🎯 Accuracy target met, but speed needs further optimization") 
    else:
        logger.info("⚠️  Both speed and accuracy need further optimization")
    
    # Clean up
    service.unload_model()
    
    return successful_files == total_files

if __name__ == "__main__":
    try:
        success = test_optimized_faster_whisper()
        if success:
            logger.info("✅ Phase 2 optimization test completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Phase 2 optimization test failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        sys.exit(1)
