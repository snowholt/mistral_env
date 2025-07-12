#!/usr/bin/env python3
"""
Arabic TTS Comparison Script for BeautyAI Framework.
Compares OuteTTS vs Edge TTS performance and quality for Arabic speech synthesis.
"""

import sys
import os
import logging
import json
import time
from pathlib import Path

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_voice_tests_directory():
    """Create the voice_tests directory if it doesn't exist."""
    voice_tests_dir = Path("/home/lumi/beautyai/voice_tests")
    voice_tests_dir.mkdir(exist_ok=True)
    return voice_tests_dir

def run_edge_tts_test():
    """Run Edge TTS test and return results."""
    print("🎙️ RUNNING EDGE TTS TEST")
    print("="*80)
    
    try:
        import tests.voice_features.tts.test_edge_tts as test_edge_tts
        success = test_edge_tts.test_edge_tts()
        return success, "edge_tts_performance.json"
    except Exception as e:
        print(f"❌ Edge TTS test failed: {e}")
        return False, None

def run_oute_tts_test():
    """Run OuteTTS test and return results."""
    print("\n🎙️ RUNNING OUTE TTS TEST")
    print("="*80)
    
    try:
        import tests.voice_features.tts.test_oute_tts as test_oute_tts
        success = test_oute_tts.test_oute_tts()
        return success, "oute_tts_performance.json"
    except Exception as e:
        print(f"❌ OuteTTS test failed: {e}")
        return False, None

def load_performance_data(filename):
    """Load performance data from JSON file."""
    try:
        file_path = f"/home/lumi/beautyai/voice_tests/{filename}"
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return []

def analyze_and_compare():
    """Analyze and compare the performance data from both engines."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TTS COMPARISON ANALYSIS")
    print("="*80)
    
    # Load performance data
    edge_data = load_performance_data("edge_tts_performance.json")
    oute_data = load_performance_data("oute_tts_performance.json")
    
    if not edge_data and not oute_data:
        print("❌ No performance data available for comparison")
        return
    
    # Filter Arabic tests only
    edge_arabic = [r for r in edge_data if r["language"] == "ar" and r["success"]]
    oute_arabic = [r for r in oute_data if r["language"] == "ar" and r["success"]]
    
    print(f"🇸🇦 ARABIC VOICE SYNTHESIS COMPARISON")
    print("="*80)
    
    if edge_arabic:
        edge_avg_speed = sum(r["chars_per_second"] for r in edge_arabic) / len(edge_arabic)
        edge_avg_time = sum(r["generation_time"] for r in edge_arabic) / len(edge_arabic)
        edge_success_rate = len(edge_arabic) / len([r for r in edge_data if r["language"] == "ar"]) * 100
        
        print(f"📱 Edge TTS Arabic Performance:")
        print(f"   Average Speed: {edge_avg_speed:.1f} chars/sec")
        print(f"   Average Generation Time: {edge_avg_time:.2f}s")
        print(f"   Success Rate: {edge_success_rate:.1f}%")
        print(f"   Tests Completed: {len(edge_arabic)}")
    
    if oute_arabic:
        oute_avg_speed = sum(r["chars_per_second"] for r in oute_arabic) / len(oute_arabic)
        oute_avg_time = sum(r["generation_time"] for r in oute_arabic) / len(oute_arabic)
        oute_success_rate = len(oute_arabic) / len([r for r in oute_data if r["language"] == "ar"]) * 100
        
        print(f"\n🚀 OuteTTS Arabic Performance:")
        print(f"   Average Speed: {oute_avg_speed:.1f} chars/sec")
        print(f"   Average Generation Time: {oute_avg_time:.2f}s")
        print(f"   Success Rate: {oute_success_rate:.1f}%")
        print(f"   Tests Completed: {len(oute_arabic)}")
    
    # Direct comparison
    if edge_arabic and oute_arabic:
        print(f"\n⚖️ DIRECT COMPARISON (Arabic)")
        print("="*50)
        
        speed_winner = "OuteTTS" if oute_avg_speed > edge_avg_speed else "Edge TTS"
        speed_diff = abs(oute_avg_speed - edge_avg_speed)
        speed_percentage = (speed_diff / max(oute_avg_speed, edge_avg_speed)) * 100
        
        print(f"🏆 Speed Winner: {speed_winner}")
        print(f"   OuteTTS: {oute_avg_speed:.1f} chars/sec")
        print(f"   Edge TTS: {edge_avg_speed:.1f} chars/sec")
        print(f"   Difference: {speed_diff:.1f} chars/sec ({speed_percentage:.1f}%)")
        
        time_winner = "OuteTTS" if oute_avg_time < edge_avg_time else "Edge TTS"
        time_diff = abs(oute_avg_time - edge_avg_time)
        time_percentage = (time_diff / max(oute_avg_time, edge_avg_time)) * 100
        
        print(f"\n⏱️ Generation Time Winner: {time_winner}")
        print(f"   OuteTTS: {oute_avg_time:.2f}s")
        print(f"   Edge TTS: {edge_avg_time:.2f}s")
        print(f"   Difference: {time_diff:.2f}s ({time_percentage:.1f}%)")
    
    # Overall comparison across all languages
    print(f"\n🌍 OVERALL PERFORMANCE (All Languages)")
    print("="*50)
    
    edge_all = [r for r in edge_data if r["success"]]
    oute_all = [r for r in oute_data if r["success"]]
    
    if edge_all:
        edge_overall_speed = sum(r["chars_per_second"] for r in edge_all) / len(edge_all)
        edge_overall_time = sum(r["generation_time"] for r in edge_all) / len(edge_all)
        print(f"📱 Edge TTS Overall: {edge_overall_speed:.1f} chars/sec, {edge_overall_time:.2f}s avg")
    
    if oute_all:
        oute_overall_speed = sum(r["chars_per_second"] for r in oute_all) / len(oute_all)
        oute_overall_time = sum(r["generation_time"] for r in oute_all) / len(oute_all)
        print(f"🚀 OuteTTS Overall: {oute_overall_speed:.1f} chars/sec, {oute_overall_time:.2f}s avg")
    
    # File size analysis
    print(f"\n📁 FILE SIZE ANALYSIS")
    print("="*50)
    
    if edge_arabic:
        edge_avg_size = sum(r["file_size"] for r in edge_arabic) / len(edge_arabic)
        print(f"📱 Edge TTS Avg File Size: {edge_avg_size/1024:.1f} KB")
    
    if oute_arabic:
        oute_avg_size = sum(r["file_size"] for r in oute_arabic) / len(oute_arabic)
        print(f"🚀 OuteTTS Avg File Size: {oute_avg_size/1024:.1f} KB")
    
    # Quality assessment note
    print(f"\n🎧 QUALITY ASSESSMENT (Manual)")
    print("="*50)
    print("📝 Listen to the generated audio files to compare:")
    print("   📂 Edge TTS files: /home/lumi/beautyai/voice_tests/edge_tts_ar_*")
    print("   📂 OuteTTS files: /home/lumi/beautyai/voice_tests/oute_tts_ar_*")
    print("\n💡 Quality factors to consider:")
    print("   • Naturalness and fluency")
    print("   • Pronunciation accuracy")
    print("   • Intonation and rhythm")
    print("   • Audio clarity and artifacts")
    print("   • Speaker voice quality")
    
    # Generate comparison report
    comparison_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "edge_tts": {
            "arabic_tests": len(edge_arabic),
            "arabic_avg_speed": edge_avg_speed if edge_arabic else 0,
            "arabic_avg_time": edge_avg_time if edge_arabic else 0,
            "arabic_success_rate": edge_success_rate if edge_arabic else 0,
            "overall_tests": len(edge_all),
            "overall_avg_speed": edge_overall_speed if edge_all else 0,
            "overall_avg_time": edge_overall_time if edge_all else 0
        },
        "oute_tts": {
            "arabic_tests": len(oute_arabic),
            "arabic_avg_speed": oute_avg_speed if oute_arabic else 0,
            "arabic_avg_time": oute_avg_time if oute_arabic else 0,
            "arabic_success_rate": oute_success_rate if oute_arabic else 0,
            "overall_tests": len(oute_all),
            "overall_avg_speed": oute_overall_speed if oute_all else 0,
            "overall_avg_time": oute_overall_time if oute_all else 0
        }
    }
    
    # Save comparison report
    try:
        report_file = "/home/lumi/beautyai/voice_tests/tts_comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Comparison report saved to: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save comparison report: {e}")

def main():
    """Main function to run Arabic beauty clinic TTS comparison."""
    print("🎙️ BeautyAI Arabic Beauty Clinic TTS Comparison Suite")
    print("Testing OuteTTS vs Edge TTS for Arabic beauty clinic scenarios")
    print("="*80)
    
    # Create voice tests directory
    voice_dir = create_voice_tests_directory()
    print(f"📁 Voice tests directory: {voice_dir}")
    
    # Run both tests
    results = []
    
    # Test Edge TTS (Arabic beauty clinic scenarios)
    edge_success, edge_perf_file = run_edge_tts_test()
    results.append(("Edge TTS", edge_success))
    
    # Test OuteTTS (Arabic beauty clinic scenarios)
    oute_success, oute_perf_file = run_oute_tts_test()
    results.append(("OuteTTS", oute_success))
    
    # Analyze and compare results
    analyze_and_compare()
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏁 ARABIC BEAUTY CLINIC TTS COMPARISON SUMMARY")
    print("="*80)
    
    for engine, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{engine}: {status}")
    
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    if successful_tests == total_tests:
        print(f"\n🎉 ALL BEAUTY CLINIC TTS TESTS COMPLETED SUCCESSFULLY!")
        print(f"� Both engines tested for Arabic beauty clinic scenarios")
        print(f"🎧 Listen to the audio files to determine quality for beauty applications")
        print(f"📈 Check performance metrics above for speed comparison")
        print(f"🏆 Choose the best engine for your beauty clinic AI assistant")
    else:
        print(f"\n⚠️ {total_tests - successful_tests} out of {total_tests} engines failed")
        print(f"🔧 Check the errors above for troubleshooting")
    
    print(f"\n📂 All beauty clinic audio files are in: /home/lumi/beautyai/voice_tests/")
    print(f"🎵 Arabic beauty clinic audio files:")
    print(f"   • edge_tts_beauty_clinic_ar_*.wav (Edge TTS Arabic)")
    print(f"   • beauty_clinic_ar_*.wav (OuteTTS Arabic)")
    print(f"💡 Perfect for testing customer service voice quality!")

if __name__ == "__main__":
    main()
