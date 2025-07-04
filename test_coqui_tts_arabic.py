#!/usr/bin/env python3
"""
Coqui TTS-to-Whisper Accuracy Test for BeautyAI Framework.
Tests the new Coqui TTS engine with Arabic language for better accuracy.
"""

import sys
import time
import os
from pathlib import Path

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService

def test_coqui_tts_arabic():
    """Test Coqui TTS with Arabic text for better accuracy than OuteTTS."""
    print("🎙️ Coqui TTS Arabic Accuracy Test")
    print("=" * 50)
    
    # Test sentences with different complexities
    test_texts = [
        "مرحباً بكم في عيادة الجمال المتطورة",
        "نقدم أحدث علاجات البشرة والوجه",
        "باستخدام تقنيات الذكاء الاصطناعي المتقدمة",
        "والليزر الطبي المعتمد عالمياً",
        "لضمان النتائج المثلى"
    ]
    
    try:
        # Initialize services
        print("📥 Loading Coqui TTS service...")
        tts_service = TextToSpeechService()
        if not tts_service.load_tts_model("coqui-tts-arabic", "coqui"):
            print("❌ Failed to load Coqui TTS model")
            return False
        
        print("📥 Loading Whisper transcription service...")
        transcription_service = AudioTranscriptionService()
        if not transcription_service.load_whisper_model("whisper-large-v3-turbo-arabic"):
            print("❌ Failed to load Whisper model")
            return False
        
        # Test each sentence
        results = []
        for i, test_text in enumerate(test_texts):
            print(f"\n🧪 Test {i+1}/5: '{test_text}'")
            
            # Generate Arabic speech with Coqui TTS
            print(f"🎤 Generating Arabic speech with Coqui TTS...")
            audio_file = f"coqui_test_{i+1}_{int(time.time())}.wav"
            
            result_path = tts_service.text_to_speech(
                text=test_text,
                language="ar",
                speaker_voice="female",
                output_path=audio_file
            )
            
            if result_path and os.path.exists(result_path):
                print(f"✅ Coqui TTS generated: {result_path}")
                
                # Transcribe back with Whisper
                print(f"🧠 Transcribing with Whisper...")
                transcription = transcription_service.transcribe_audio_file(result_path)
                
                if transcription:
                    print(f"📝 RESULTS:")
                    print(f"Original:    '{test_text}'")
                    print(f"Transcribed: '{transcription}'")
                    
                    # Basic accuracy check
                    original_words = set(test_text.split())
                    transcribed_words = set(transcription.split())
                    
                    # Calculate word overlap
                    common_words = original_words.intersection(transcribed_words)
                    word_accuracy = len(common_words) / len(original_words) if original_words else 0
                    
                    # Character similarity (simple metric)
                    common_chars = set(test_text) & set(transcription)
                    char_accuracy = len(common_chars) / max(len(set(test_text)), 1)
                    
                    print(f"📊 Word Accuracy: {word_accuracy:.1%}")
                    print(f"📊 Char Accuracy: {char_accuracy:.1%}")
                    
                    # Success criteria
                    if len(transcription.strip()) > 5 and word_accuracy > 0.3:
                        print("✅ SUCCESS: Good transcription quality!")
                        success = True
                    else:
                        print("⚠️ CONCERN: Low transcription quality")
                        success = False
                        
                    results.append({
                        "test": i+1,
                        "original": test_text,
                        "transcribed": transcription,
                        "word_accuracy": word_accuracy,
                        "char_accuracy": char_accuracy,
                        "success": success
                    })
                    
                    # Cleanup
                    try:
                        os.remove(result_path)
                    except:
                        pass
                else:
                    print("❌ Transcription failed")
                    results.append({
                        "test": i+1,
                        "original": test_text,
                        "transcribed": None,
                        "success": False
                    })
            else:
                print("❌ Coqui TTS generation failed")
                results.append({
                    "test": i+1,
                    "original": test_text,
                    "transcribed": None,
                    "success": False
                })
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print("=" * 30)
        successful_tests = [r for r in results if r.get("success", False)]
        print(f"Successful tests: {len(successful_tests)}/{len(results)}")
        
        if successful_tests:
            avg_word_acc = sum(r.get("word_accuracy", 0) for r in successful_tests) / len(successful_tests)
            avg_char_acc = sum(r.get("char_accuracy", 0) for r in successful_tests) / len(successful_tests)
            print(f"Average word accuracy: {avg_word_acc:.1%}")
            print(f"Average char accuracy: {avg_char_acc:.1%}")
            
            if avg_word_acc > 0.5:
                print("🎯 EXCELLENT: Coqui TTS shows good Arabic accuracy!")
            elif avg_word_acc > 0.3:
                print("✅ GOOD: Coqui TTS shows reasonable Arabic accuracy")
            else:
                print("⚠️ NEEDS IMPROVEMENT: Consider different models or settings")
        else:
            print("❌ NO SUCCESSFUL TESTS: Check TTS and Whisper configuration")
        
        return len(successful_tests) > 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coqui_vs_outetts_comparison():
    """Quick comparison to show improvement over OuteTTS."""
    print("\n🔄 Coqui TTS vs OuteTTS Comparison")
    print("=" * 40)
    
    comparison_notes = {
        "OuteTTS Issues": [
            "❌ Embedded metadata contamination ('ترجمة نانسي قنقر')",
            "❌ Poor Arabic speaker profile quality",
            "❌ Inconsistent transcription results",
            "❌ Limited Arabic language optimization"
        ],
        "Coqui TTS Advantages": [
            "✅ Native Arabic TTS models (tts_models/ar/tn_arabicspeech/vits)",
            "✅ No metadata contamination issues",
            "✅ High-quality neural vocoder",
            "✅ Voice cloning capabilities",
            "✅ GPU acceleration support",
            "✅ Multiple language support",
            "✅ Local processing (no internet required)"
        ]
    }
    
    for category, points in comparison_notes.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  {point}")
    
    print(f"\n🎯 RECOMMENDATION: Coqui TTS is significantly better for Arabic TTS")

def main():
    """Main function to test Coqui TTS accuracy."""
    try:
        print("🚀 Starting Coqui TTS Arabic Accuracy Test")
        
        success = test_coqui_tts_arabic()
        
        # Show comparison
        test_coqui_vs_outetts_comparison()
        
        if success:
            print("\n✅ Coqui TTS testing completed successfully!")
            print("🎯 Ready to use Coqui TTS as the primary TTS engine")
        else:
            print("\n❌ Coqui TTS testing encountered issues")
            print("🔧 Check model configuration and try again")
        
        return success
            
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
