#!/usr/bin/env python3
"""
Coqui TTS Professional Test Suite for BeautyAI Framework.
Comprehensive testing of Arabic TTS quality with Whisper transcription validation.
"""

import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService

def get_test_cases() -> List[Dict[str, Any]]:
    """Get comprehensive test cases for different clinic scenarios."""
    return [
        # Greeting & Welcome (Short)
        {
            "category": "greeting",
            "length": "short",
            "text": "أهلاً وسهلاً بكم في عيادة الجمال",
            "description": "Basic greeting message"
        },
        {
            "category": "greeting", 
            "length": "medium",
            "text": "مرحباً بكم في عيادة الجمال المتطورة، نحن هنا لخدمتكم",
            "description": "Extended welcome message"
        },
        
        # Clinic Services Description (Medium)
        {
            "category": "clinic_services",
            "length": "medium", 
            "text": "نقدم أحدث علاجات البشرة والوجه باستخدام التقنيات المتقدمة",
            "description": "General services overview"
        },
        {
            "category": "clinic_services",
            "length": "long",
            "text": "عيادتنا متخصصة في علاجات الجمال المتطورة بما في ذلك تجديد البشرة وعلاج التجاعيد والليزر الطبي المعتمد عالمياً لضمان أفضل النتائج للمرضى",
            "description": "Comprehensive clinic description"
        },
        
        # Botox Treatment Information (Medium/Long)
        {
            "category": "botox_treatment",
            "length": "medium",
            "text": "البوتوكس علاج آمن وفعال لتقليل التجاعيد والخطوط الدقيقة",
            "description": "Basic botox information"
        },
        {
            "category": "botox_treatment", 
            "length": "long",
            "text": "علاج البوتوكس هو إجراء تجميلي غير جراحي يستخدم لتقليل ظهور التجاعيد والخطوط التعبيرية في الوجه، حيث يتم حقن مادة البوتولينوم في العضلات المستهدفة لمنع انقباضها وبالتالي تقليل التجاعيد",
            "description": "Detailed botox procedure explanation"
        },
        
        # Appointment Scheduling (Short/Medium)
        {
            "category": "appointment",
            "length": "short",
            "text": "يمكنكم حجز موعد عبر الهاتف أو الموقع الإلكتروني",
            "description": "Simple appointment booking info"
        },
        {
            "category": "appointment",
            "length": "medium", 
            "text": "لحجز موعدكم، يرجى الاتصال بنا على الرقم المعطى أو زيارة موقعنا الإلكتروني، ونحن متاحون من الأحد إلى الخميس",
            "description": "Detailed appointment scheduling"
        },
        
        # Consultation Information (Medium/Long)
        {
            "category": "consultation",
            "length": "medium",
            "text": "نقدم استشارة مجانية مع طبيب متخصص لتحديد العلاج المناسب لكم",
            "description": "Free consultation offer"
        },
        {
            "category": "consultation",
            "length": "long", 
            "text": "خلال الاستشارة الأولية، سيقوم طبيبنا المتخصص بفحص بشرتكم وتقييم احتياجاتكم الخاصة، ومن ثم سيقترح عليكم خطة علاج مخصصة تناسب نوع بشرتكم وأهدافكم التجميلية مع شرح مفصل للإجراءات والنتائج المتوقعة",
            "description": "Comprehensive consultation process"
        },
        
        # Technical/Medical Terms (Complex)
        {
            "category": "technical",
            "length": "long",
            "text": "نستخدم تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد من إدارة الغذاء والدواء الأمريكية لضمان النتائج المثلى والسلامة القصوى للمرضى",
            "description": "Technical medical terminology"
        }
    ]
def test_coqui_tts_professional():
    """Professional Coqui TTS test suite with comprehensive Arabic scenarios."""
    print("🎙️ Coqui TTS Professional Test Suite")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("/home/lumi/beautyai/voice_tests/coqui_tts_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Get test cases
    test_cases = get_test_cases()
    
    try:
        # Initialize services
        print("\n📥 Initializing Coqui TTS service...")
        tts_service = TextToSpeechService()
        if not tts_service.load_tts_model("coqui-tts-arabic", "coqui"):
            print("❌ Failed to load Coqui TTS model")
            return None
        
        print("📥 Initializing Whisper transcription service...")
        transcription_service = AudioTranscriptionService()
        if not transcription_service.load_whisper_model("whisper-large-v3-turbo-arabic"):
            print("❌ Failed to load Whisper model")
            return None
        
        # Test results storage
        results = []
        
        # Process each test case
        for i, test_case in enumerate(test_cases, 1):
            category = test_case["category"]
            length = test_case["length"]
            text = test_case["text"]
            description = test_case["description"]
            
            print(f"\n🧪 Test {i}/{len(test_cases)}: {category.title()} ({length})")
            print(f"📝 Description: {description}")
            print(f"📜 Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
            
            # Generate meaningful filename
            filename = f"{category}_{length}_{i:02d}.wav"
            output_path = output_dir / filename
            
            # Generate Arabic speech with Coqui TTS
            print(f"🎤 Generating speech...")
            result_path = tts_service.text_to_speech(
                text=text,
                language="ar",
                speaker_voice="female",
                output_path=str(output_path)
            )
            
            if result_path and os.path.exists(result_path):
                print(f"✅ Audio generated: {filename}")
                
                # Transcribe with Whisper
                print(f"🧠 Transcribing...")
                transcription = transcription_service.transcribe_audio_file(result_path)
                
                if transcription:
                    # Calculate accuracy metrics
                    original_words = set(text.split())
                    transcribed_words = set(transcription.split())
                    common_words = original_words.intersection(transcribed_words)
                    word_accuracy = len(common_words) / len(original_words) if original_words else 0
                    
                    # Character accuracy
                    common_chars = set(text) & set(transcription)
                    char_accuracy = len(common_chars) / max(len(set(text)), 1)
                    
                    # Length comparison
                    length_ratio = len(transcription) / len(text) if text else 0
                    
                    # Determine quality
                    if word_accuracy >= 0.7 and char_accuracy >= 0.8:
                        quality = "Excellent"
                        status = "✅"
                    elif word_accuracy >= 0.5 and char_accuracy >= 0.7:
                        quality = "Good"
                        status = "✅"
                    elif word_accuracy >= 0.3 and char_accuracy >= 0.6:
                        quality = "Acceptable"
                        status = "⚠️"
                    else:
                        quality = "Poor"
                        status = "❌"
                    
                    print(f"{status} Quality: {quality} (Word: {word_accuracy:.1%}, Char: {char_accuracy:.1%})")
                    
                    # Store results
                    results.append({
                        "test_id": i,
                        "category": category,
                        "length": length,
                        "description": description,
                        "filename": filename,
                        "original_text": text,
                        "transcription": transcription,
                        "word_accuracy": word_accuracy,
                        "char_accuracy": char_accuracy,
                        "length_ratio": length_ratio,
                        "quality": quality,
                        "status": status
                    })
                    
                else:
                    print("❌ Transcription failed")
                    results.append({
                        "test_id": i,
                        "category": category,
                        "length": length,
                        "description": description,
                        "filename": filename,
                        "original_text": text,
                        "transcription": None,
                        "quality": "Failed",
                        "status": "❌"
                    })
            else:
                print("❌ Speech generation failed")
                results.append({
                    "test_id": i,
                    "category": category,
                    "length": length,
                    "description": description,
                    "filename": "N/A",
                    "original_text": text,
                    "transcription": None,
                    "quality": "Failed",
                    "status": "❌"
                })
        
        return results
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results_report(results: List[Dict[str, Any]]) -> None:
    """Print comprehensive results in markdown format."""
    if not results:
        print("❌ No results to display")
        return
    
    # Calculate overall statistics
    total_tests = len(results)
    successful_tests = [r for r in results if r.get("word_accuracy", 0) is not None and r.get("word_accuracy", 0) > 0.3]
    
    if successful_tests:
        avg_word_acc = sum(r.get("word_accuracy", 0) for r in successful_tests) / len(successful_tests)
        avg_char_acc = sum(r.get("char_accuracy", 0) for r in successful_tests) / len(successful_tests)
    else:
        avg_word_acc = avg_char_acc = 0
    
    # Count by category and quality
    categories = {}
    quality_counts = {"Excellent": 0, "Good": 0, "Acceptable": 0, "Poor": 0, "Failed": 0}
    
    for result in results:
        cat = result.get("category", "unknown")
        quality = result.get("quality", "Failed")
        
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        
        if quality in ["Excellent", "Good", "Acceptable"]:
            categories[cat]["passed"] += 1
        
        quality_counts[quality] += 1
    
    # Print markdown report
    print("\n" + "="*80)
    print("📊 COQUI TTS PROFESSIONAL TEST RESULTS")
    print("="*80)
    print()
    
    print("```markdown")
    print("# Coqui TTS Arabic Testing Report")
    print()
    print("## 📈 Overall Performance")
    print(f"- **Total Tests:** {total_tests}")
    print(f"- **Successful Tests:** {len(successful_tests)}/{total_tests} ({len(successful_tests)/total_tests*100:.1f}%)")
    print(f"- **Average Word Accuracy:** {avg_word_acc:.1%}")
    print(f"- **Average Character Accuracy:** {avg_char_acc:.1%}")
    print()
    
    print("## 🎯 Quality Distribution")
    for quality, count in quality_counts.items():
        percentage = count/total_tests*100 if total_tests > 0 else 0
        emoji = {"Excellent": "🟢", "Good": "🔵", "Acceptable": "🟡", "Poor": "🟠", "Failed": "🔴"}.get(quality, "⚪")
        print(f"- {emoji} **{quality}:** {count} tests ({percentage:.1f}%)")
    print()
    
    print("## 📋 Category Performance")
    for category, stats in categories.items():
        success_rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
        emoji = "✅" if success_rate >= 70 else "⚠️" if success_rate >= 50 else "❌"
        print(f"- {emoji} **{category.replace('_', ' ').title()}:** {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    print()
    
    print("## 📝 Detailed Test Results")
    print()
    print("| Test | Category | Length | Quality | Word Acc | Char Acc | File |")
    print("|------|----------|--------|---------|----------|----------|------|")
    
    for result in results:
        test_id = result.get("test_id", "N/A")
        category = result.get("category", "unknown").replace("_", " ").title()
        length = result.get("length", "N/A").title()
        quality = result.get("quality", "Failed")
        word_acc = result.get("word_accuracy", 0)
        char_acc = result.get("char_accuracy", 0)
        filename = result.get("filename", "N/A")
        
        word_acc_str = f"{word_acc:.1%}" if word_acc is not None else "N/A"
        char_acc_str = f"{char_acc:.1%}" if char_acc is not None else "N/A"
        
        print(f"| {test_id:02d} | {category} | {length} | {quality} | {word_acc_str} | {char_acc_str} | {filename} |")
    
    print()
    print("## 🔍 Sample Transcriptions")
    print()
    
    # Show a few sample transcriptions
    for i, result in enumerate(results[:3]):
        if result.get("transcription"):
            print(f"### Test {result.get('test_id', i+1)}: {result.get('category', 'unknown').replace('_', ' ').title()}")
            print(f"**Original:** {result.get('original_text', 'N/A')}")
            print(f"**Transcribed:** {result.get('transcription', 'N/A')}")
            print(f"**Quality:** {result.get('quality', 'N/A')} (Word: {result.get('word_accuracy', 0):.1%}, Char: {result.get('char_accuracy', 0):.1%})")
            print()
    
    print("## 🎯 Recommendations")
    if avg_word_acc >= 0.7:
        print("- ✅ **Excellent Performance:** Coqui TTS is production-ready for Arabic")
        print("- 🚀 **Next Steps:** Deploy to production environment")
    elif avg_word_acc >= 0.5:
        print("- ✅ **Good Performance:** Coqui TTS shows strong Arabic capabilities") 
        print("- 🔧 **Optimization:** Consider fine-tuning for specific categories with lower scores")
    elif avg_word_acc >= 0.3:
        print("- ⚠️ **Acceptable Performance:** Some improvement needed")
        print("- 🔧 **Actions:** Review model settings and consider alternative models")
    else:
        print("- ❌ **Poor Performance:** Significant improvements required")
        print("- 🛠️ **Actions:** Debug model configuration and test alternative approaches")
    
    print("```")
    print()

def main():
    """Main function to run the professional Coqui TTS test suite."""
    try:
        print("🚀 Starting Coqui TTS Professional Test Suite")
        
        results = test_coqui_tts_professional()
        
        if results:
            print_results_report(results)
            print("\n✅ Coqui TTS testing completed successfully!")
            print("🎯 Check the detailed report above for comprehensive analysis")
            return True
        else:
            print("\n❌ Coqui TTS testing failed")
            print("🔧 Check model configuration and try again")
            return False
            
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
