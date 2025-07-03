#!/usr/bin/env python3
"""
Arabic Speaker Profile Test Script for BeautyAI Platform.

Tests premium Arabic female speaker profile with:
- Long sentences (complex medical/beauty terminology)
- Medium sentences (10 words)  
- Short sentences (3 words)

All outputs saved to voice_tests/premium_speaker_tests/
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_directories():
    """Create necessary test directories."""
    test_dir = Path("/home/lumi/beautyai/voice_tests/premium_speaker_tests")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (test_dir / "female").mkdir(exist_ok=True)
    (test_dir / "long_sentences").mkdir(exist_ok=True)
    (test_dir / "medium_sentences").mkdir(exist_ok=True)
    (test_dir / "short_sentences").mkdir(exist_ok=True)
    
    return test_dir

def get_arabic_test_sentences() -> Dict[str, List[str]]:
    """Get Arabic test sentences organized by complexity."""
    
    # Long sentences (complex beauty/medical terminology)
    long_sentences = [
        "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى.",
        "نحن متخصصون في علاجات مكافحة الشيخوخة وتجديد البشرة باستخدام حقن البوتوكس والفيلر الطبيعي والتقشير الكيميائي العميق مع ضمان السلامة والفعالية التامة تحت إشراف أطباء مختصين.",
        "تتضمن خدماتنا الطبية المتميزة علاج حب الشباب وآثاره، وإزالة التصبغات الجلدية، وشد الوجه بالخيوط الذهبية، وعلاج الندبات بالليزر التكسيري مع متابعة طبية شاملة وخطة علاجية مخصصة.",
        "يمكنكم الآن الاستفادة من استشارتنا المجانية عبر منصة BeautyAI الذكية التي تحلل حالة بشرتكم وتقترح أفضل العلاجات المناسبة باستخدام خوارزميات التعلم العميق والذكاء الاصطناعي المتطور.",
        "نفخر بتقديم أحدث تقنيات زراعة الشعر بطريقة FUE المتقدمة وعلاج الصلع الوراثي بالخلايا الجذعية والبلازما الغنية بالصفائح الدموية مع ضمان النتائج الطبيعية والدائمة بأيدي خبراء متخصصين."
    ]
    
    # Medium sentences (exactly 10 words)
    medium_sentences = [
        "مرحباً بكم في عيادة الجمال للعناية بالبشرة المتخصصة",  # 10 words
        "نقدم أفضل علاجات البوتوكس والفيلر للوجه والرقبة",  # 10 words  
        "احجزوا موعدكم اليوم للحصول على استشارة طبية مجانية",  # 10 words
        "خدماتنا تشمل تنظيف البشرة وعلاج حب الشباب المتقدم",  # 10 words
        "أطباؤنا متخصصون في جراحة التجميل وعلاج الشيخوخة المبكرة",  # 10 words
        "تقنياتنا الحديثة تضمن نتائج آمنة وفعالة لجميع المرضى",  # 10 words
        "نستخدم أحدث أجهزة الليزر لعلاج التصبغات والندبات العميقة",  # 10 words
        "فريقنا الطبي يقدم رعاية شاملة ومتابعة دقيقة للمرضى"  # 10 words
    ]
    
    # Short sentences (exactly 3 words)
    short_sentences = [
        "مرحباً وأهلاً بكم",  # 3 words
        "كيف يمكنني مساعدتكم",  # 3 words
        "احجزوا موعدكم الآن",  # 3 words
        "شكراً لثقتكم بنا",  # 3 words
        "نتائج ممتازة ومضمونة",  # 3 words
        "خدمة طبية متميزة",  # 3 words
        "أطباء ذوو خبرة",  # 3 words
        "تقنيات حديثة ومتطورة"  # 3 words
    ]
    
    return {
        "long": long_sentences,
        "medium": medium_sentences, 
        "short": short_sentences
    }

def test_arabic_speaker_profile(
    speaker_name: str, 
    speaker_profile_path: str, 
    test_sentences: Dict[str, List[str]],
    test_dir: Path
) -> Dict[str, any]:
    """Test an Arabic speaker profile with different sentence types."""
    
    print(f"\n🎤 Testing Arabic Speaker: {speaker_name}")
    print("=" * 60)
    
    try:
        # Import OuteTTS
        import outetts
        
        # Initialize OuteTTS interface
        print("📥 Initializing OuteTTS interface...")
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
            )
        )
        
        # Load the custom speaker profile
        print(f"👤 Loading speaker profile: {speaker_profile_path}")
        if not os.path.exists(speaker_profile_path):
            print(f"❌ Speaker profile not found: {speaker_profile_path}")
            return {"error": "Speaker profile not found"}
            
        speaker = interface.load_speaker(speaker_profile_path)
        print(f"✅ Speaker loaded successfully")
        
        results = {
            "speaker_name": speaker_name,
            "speaker_profile": speaker_profile_path,
            "test_results": [],
            "performance_summary": {}
        }
        
        # Test each sentence type
        for sentence_type, sentences in test_sentences.items():
            print(f"\n📝 Testing {sentence_type} sentences...")
            
            type_results = []
            total_time = 0
            total_chars = 0
            success_count = 0
            
            for i, sentence in enumerate(sentences, 1):
                print(f"   {sentence_type.capitalize()} Test {i}: '{sentence}'")
                
                try:
                    # Measure performance
                    start_time = time.time()
                    
                    # Generate speech with optimized Arabic parameters
                    output = interface.generate(
                        config=outetts.GenerationConfig(
                            text=sentence,
                            generation_type=outetts.GenerationType.CHUNKED,  # Use CHUNKED instead of SENTENCE
                            speaker=speaker,
                            sampler_config=outetts.SamplerConfig(
                                temperature=0.0,          # Much lower for Arabic accuracy
                                top_p=0.75,              # Better control for Arabic morphology
                                top_k=25,                # Lower for more consistent Arabic
                                repetition_penalty=1.02, # Minimal to avoid breaking Arabic words
                                repetition_range=32,     # Shorter for Arabic word structure
                                min_p=0.02              # Lower threshold for Arabic phonemes
                            ),
                            max_length=8192            # Use model's actual max_seq_length
                            # Note: Removed language parameter as it's not supported
                        )
                    )
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # Create output path
                    gender = "female" if "female" in speaker_name.lower() else "male"
                    output_filename = f"{gender}_{sentence_type}_{i:02d}.wav"
                    output_path = test_dir / f"{sentence_type}_sentences" / output_filename
                    
                    # Save audio file
                    output.save(str(output_path))
                    
                    # Verify file was created
                    if output_path.exists():
                        file_size = output_path.stat().st_size
                        chars_per_second = len(sentence) / generation_time if generation_time > 0 else 0
                        
                        print(f"   ✅ Success: {output_path.name} ({file_size:,} bytes, {generation_time:.2f}s)")
                        
                        # Store test result
                        test_result = {
                            "test_number": i,
                            "sentence_type": sentence_type,
                            "text": sentence,
                            "text_length": len(sentence),
                            "generation_time": generation_time,
                            "chars_per_second": chars_per_second,
                            "file_path": str(output_path),
                            "file_size": file_size,
                            "success": True
                        }
                        
                        type_results.append(test_result)
                        results["test_results"].append(test_result)
                        
                        # Update totals
                        total_time += generation_time
                        total_chars += len(sentence)
                        success_count += 1
                        
                    else:
                        print(f"   ❌ Failed: Output file not created")
                        
                except Exception as e:
                    print(f"   ❌ Generation failed: {e}")
                    
                    # Store failed result
                    failed_result = {
                        "test_number": i,
                        "sentence_type": sentence_type,
                        "text": sentence,
                        "text_length": len(sentence),
                        "success": False,
                        "error": str(e)
                    }
                    type_results.append(failed_result)
                    results["test_results"].append(failed_result)
            
            # Calculate performance summary for this sentence type
            if success_count > 0:
                avg_time = total_time / success_count
                avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0
                success_rate = (success_count / len(sentences)) * 100
                
                type_summary = {
                    "total_tests": len(sentences),
                    "successful_tests": success_count,
                    "success_rate": success_rate,
                    "average_generation_time": avg_time,
                    "average_chars_per_second": avg_chars_per_sec,
                    "total_characters": total_chars
                }
                
                results["performance_summary"][sentence_type] = type_summary
                
                print(f"   📊 {sentence_type.capitalize()} Summary: {success_count}/{len(sentences)} successful ({success_rate:.1f}%)")
                print(f"   ⏱️ Average time: {avg_time:.2f}s, Speed: {avg_chars_per_sec:.1f} chars/sec")
        
        return results
        
    except ImportError:
        error_msg = "OuteTTS library not available. Install with: pip install outetts"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Test failed: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

def save_test_results(results: Dict, output_path: Path):
    """Save test results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Test results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """Main function to run Arabic premium speaker profile tests."""
    
    print("🎭 Premium Arabic Speaker Profile Testing Suite")
    print("="*80)
    print("Testing premium Arabic female speaker with:")
    print("   📏 Long sentences (complex medical/beauty terminology)")
    print("   📝 Medium sentences (exactly 10 words)")
    print("   📋 Short sentences (exactly 3 words)")
    print("="*80)
    
    # Create test directories
    test_dir = create_test_directories()
    print(f"📁 Test output directory: {test_dir}")
    
    # Get test sentences
    test_sentences = get_arabic_test_sentences()
    print(f"📝 Test sentences prepared:")
    print(f"   Long sentences: {len(test_sentences['long'])}")
    print(f"   Medium sentences: {len(test_sentences['medium'])} (10 words each)")
    print(f"   Short sentences: {len(test_sentences['short'])} (3 words each)")
    
    # Define speaker profile to test (premium female only)
    speaker_profiles = [
        {
            "name": "arabic_female_premium_19s",
            "profile_path": "/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/arabic_female_premium_19s.json",
            "gender": "female"
        }
    ]
    
    all_results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_description": "Premium Arabic Female Speaker Profile Comprehensive Testing",
        "speaker_results": []
    }
    
    # Test each speaker profile
    for speaker_info in speaker_profiles:
        speaker_name = speaker_info["name"]
        profile_path = speaker_info["profile_path"]
        gender = speaker_info["gender"]
        
        print(f"\n{'='*80}")
        print(f"🎤 TESTING PREMIUM ARABIC SPEAKER: {speaker_name}")
        print(f"{'='*80}")
        
        # Check if profile exists
        if not os.path.exists(profile_path):
            print(f"⚠️ Speaker profile not found: {profile_path}")
            print(f"💡 Run create_arabic_speaker_profiles.py first to create the profiles")
            continue
        
        # Test the speaker
        results = test_arabic_speaker_profile(
            speaker_name=speaker_name,
            speaker_profile_path=profile_path,
            test_sentences=test_sentences,
            test_dir=test_dir
        )
        
        # Add to overall results
        results["gender"] = gender
        all_results["speaker_results"].append(results)
        
        # Save individual results
        individual_results_path = test_dir / f"{speaker_name}_test_results.json"
        save_test_results(results, individual_results_path)
    
    # Save combined results
    combined_results_path = test_dir / "premium_speaker_tests_complete.json"
    save_test_results(all_results, combined_results_path)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='*80}")
    
    for speaker_result in all_results["speaker_results"]:
        if "error" not in speaker_result:
            speaker_name = speaker_result["speaker_name"]
            gender = speaker_result["gender"]
            total_tests = len(speaker_result["test_results"])
            successful_tests = sum(1 for result in speaker_result["test_results"] if result.get("success", False))
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            print(f"🎤 {gender.capitalize()} Speaker ({speaker_name}):")
            print(f"   Tests: {successful_tests}/{total_tests} successful ({success_rate:.1f}%)")
            
            # Show performance by sentence type
            if "performance_summary" in speaker_result:
                for sentence_type, summary in speaker_result["performance_summary"].items():
                    print(f"   {sentence_type.capitalize()}: {summary['success_rate']:.1f}% success, {summary['average_chars_per_second']:.1f} chars/sec")
    
    print(f"\n🎵 Audio files saved in: {test_dir}")
    print(f"📄 Test results saved in: {combined_results_path}")
    print(f"\n✅ Premium Arabic speaker profile testing completed!")

if __name__ == "__main__":
    main()
