#!/usr/bin/env python3
"""
Comprehensive Analysis of Batch Test Results

Analyzes batch test results with proper quality metrics:
1. STT Quality: Input transcription vs Expected question
2. TTS Quality: TTS transcription vs LLM response
3. Complete pipeline validation
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    # Normalize texts
    text1_clean = text1.strip().replace(" /no_think", "").replace("/no_think", "")
    text2_clean = text2.strip()
    
    # Calculate similarity
    return SequenceMatcher(None, text1_clean, text2_clean).ratio()


def analyze_result(result: Dict) -> Dict:
    """Analyze a single test result with proper metrics."""
    analysis = {
        "file": result["file"],
        "success": result["success"],
    }
    
    if not result["success"]:
        analysis["error"] = result.get("error", "Unknown error")
        return analysis
    
    # Extract data
    input_transcription = result.get("input_transcription", "")
    expected_text = result.get("expected_text", "")
    llm_response = result.get("llm_response", "")
    tts_transcription = result.get("tts_transcription", "")
    
    # Calculate STT Quality (Input vs Expected)
    if expected_text:
        stt_quality = calculate_similarity(input_transcription, expected_text)
        analysis["stt_quality"] = {
            "score": stt_quality,
            "input": input_transcription,
            "expected": expected_text,
            "label": "EXCELLENT" if stt_quality >= 0.95 else ("GOOD" if stt_quality >= 0.85 else "LOW")
        }
    
    # Calculate TTS Quality (TTS transcription vs LLM response)
    if llm_response and tts_transcription:
        tts_quality = calculate_similarity(tts_transcription, llm_response)
        analysis["tts_quality"] = {
            "score": tts_quality,
            "tts_output": tts_transcription,
            "expected_llm": llm_response,
            "label": "EXCELLENT" if tts_quality >= 0.95 else ("GOOD" if tts_quality >= 0.85 else "LOW")
        }
    
    # Performance metrics
    analysis["performance"] = {
        "total_time": result.get("total_processing_time", 0),
        "response_time": result.get("response_time", 0),
        "validation_time": result.get("validation_time", 0),
    }
    
    return analysis


def generate_report(results_file: Path):
    """Generate comprehensive analysis report."""
    
    # Load results
    with open(results_file) as f:
        data = json.load(f)
    
    results = data["results"]
    total_tests = len(results)
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE BATCH TEST ANALYSIS")
    print("="*80)
    
    # Analyze all results
    analyses = [analyze_result(r) for r in results]
    
    # Calculate statistics
    successful = [a for a in analyses if a["success"]]
    failed = [a for a in analyses if not a["success"]]
    
    stt_scores = [a["stt_quality"]["score"] for a in successful if "stt_quality" in a]
    tts_scores = [a["tts_quality"]["score"] for a in successful if "tts_quality" in a]
    
    avg_stt = sum(stt_scores) / len(stt_scores) if stt_scores else 0
    avg_tts = sum(tts_scores) / len(tts_scores) if tts_scores else 0
    
    # Overall Summary
    print(f"\n🎯 Overall Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    print(f"  Success Rate: {len(successful)/total_tests*100:.1f}%")
    
    # STT Quality Summary
    print(f"\n🎤 STT Quality (Input Recognition):")
    print(f"  Average Score: {avg_stt*100:.1f}%")
    stt_excellent = sum(1 for a in successful if "stt_quality" in a and a["stt_quality"]["score"] >= 0.95)
    stt_good = sum(1 for a in successful if "stt_quality" in a and 0.85 <= a["stt_quality"]["score"] < 0.95)
    stt_low = sum(1 for a in successful if "stt_quality" in a and a["stt_quality"]["score"] < 0.85)
    print(f"  EXCELLENT (≥95%): {stt_excellent}")
    print(f"  GOOD (85-95%): {stt_good}")
    print(f"  LOW (<85%): {stt_low}")
    
    # TTS Quality Summary
    print(f"\n🔊 TTS Quality (Voice Pronunciation):")
    print(f"  Average Score: {avg_tts*100:.1f}%")
    tts_excellent = sum(1 for a in successful if "tts_quality" in a and a["tts_quality"]["score"] >= 0.95)
    tts_good = sum(1 for a in successful if "tts_quality" in a and 0.85 <= a["tts_quality"]["score"] < 0.95)
    tts_low = sum(1 for a in successful if "tts_quality" in a and a["tts_quality"]["score"] < 0.85)
    print(f"  EXCELLENT (≥95%): {tts_excellent}")
    print(f"  GOOD (85-95%): {tts_good}")
    print(f"  LOW (<85%): {tts_low}")
    
    # Performance Metrics
    total_times = [a["performance"]["total_time"] for a in successful if "performance" in a]
    avg_time = sum(total_times) / len(total_times) if total_times else 0
    min_time = min(total_times) if total_times else 0
    max_time = max(total_times) if total_times else 0
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"  Average Processing Time: {avg_time:.2f}s")
    print(f"  Min Time: {min_time:.2f}s")
    print(f"  Max Time: {max_time:.2f}s")
    
    # Detailed Results
    print(f"\n📋 Detailed Results:")
    print(f"{'File':<15} {'STT':<8} {'TTS':<8} {'Time':<8} {'Status'}")
    print("-" * 60)
    
    for a in analyses:
        file_name = a["file"]
        status = "✅" if a["success"] else "❌"
        
        if a["success"]:
            stt = f"{a.get('stt_quality', {}).get('score', 0)*100:.1f}%" if "stt_quality" in a else "N/A"
            tts = f"{a.get('tts_quality', {}).get('score', 0)*100:.1f}%" if "tts_quality" in a else "N/A"
            time_val = f"{a.get('performance', {}).get('total_time', 0):.2f}s"
        else:
            stt = "ERROR"
            tts = "ERROR"
            time_val = "N/A"
        
        print(f"{file_name:<15} {stt:<8} {tts:<8} {time_val:<8} {status}")
    
    # Failed Tests Detail
    if failed:
        print(f"\n❌ Failed Tests Detail:")
        for a in failed:
            print(f"  {a['file']}: {a.get('error', 'Unknown error')}")
    
    # Low Quality Tests
    low_stt = [a for a in successful if "stt_quality" in a and a["stt_quality"]["score"] < 0.85]
    if low_stt:
        print(f"\n⚠️  Low STT Quality (<85%):")
        for a in low_stt:
            print(f"  {a['file']}: {a['stt_quality']['score']*100:.1f}%")
            print(f"    Expected: {a['stt_quality']['expected']}")
            print(f"    Got:      {a['stt_quality']['input']}")
    
    low_tts = [a for a in successful if "tts_quality" in a and a["tts_quality"]["score"] < 0.85]
    if low_tts:
        print(f"\n⚠️  Low TTS Quality (<85%):")
        for a in low_tts:
            print(f"  {a['file']}: {a['tts_quality']['score']*100:.1f}%")
            print(f"    Expected: {a['tts_quality']['expected_llm']}")
            print(f"    Got:      {a['tts_quality']['tts_output']}")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Find latest results file
    results_dir = Path(__file__).parent
    results_files = sorted(results_dir.glob("batch_test_results_*.json"))
    
    if not results_files:
        print("❌ No batch test results found!")
        sys.exit(1)
    
    latest_file = results_files[-1]
    print(f"📄 Analyzing: {latest_file.name}")
    
    generate_report(latest_file)
