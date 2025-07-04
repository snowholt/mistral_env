#!/usr/bin/env python3
"""
Quick TTS-to-Whisper Accuracy Test Script for BeautyAI Platform.

Simple version that tests one sentence with one TTS configuration and 
measures transcription accuracy using Whisper.

Test sentence: "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."
"""

import sys
import os
import time
import json
import difflib
from pathlib import Path

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test sentence - the one specified by the user
TEST_SENTENCE = "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."

def quick_accuracy_test():
    """Run a quick TTS-to-Whisper accuracy test."""
    
    print("🎯 Quick TTS-to-Whisper Accuracy Test")
    print("="*70)
    print(f"📝 Testing sentence: {TEST_SENTENCE[:60]}...")
    print("="*70)
    
    # Create test directory
    test_dir = Path("/home/lumi/beautyai/voice_tests/quick_accuracy_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique audio file path
    timestamp = int(time.time())
    audio_path = test_dir / f"quick_test_{timestamp}.wav"
    
    try:
        # Step 1: Generate TTS audio
        print("\\n📢 STEP 1: Generating TTS Audio")
        print("-" * 40)
        
        from beautyai_inference.services.text_to_speech_service import TextToSpeechService
        
        # Initialize TTS service
        print("📥 Initializing TTS service...")
        tts_service = TextToSpeechService()
        
        # Load OuteTTS model
        print("📥 Loading OuteTTS model...")
        start_time = time.time()
        success = tts_service.load_tts_model("oute-tts-1b")
        load_time = time.time() - start_time
        
        if not success:
            print("❌ Failed to load OuteTTS model")
            return False
        
        print(f"✅ OuteTTS model loaded in {load_time:.2f}s")
        
        # Generate speech
        print(f"🔊 Generating speech with female Arabic voice...")
        start_gen_time = time.time()
        result = tts_service.text_to_speech(
            text=TEST_SENTENCE,
            language="ar",
            speaker_voice="female",
            output_path=str(audio_path)
        )
        generation_time = time.time() - start_gen_time
        
        if not result or not os.path.exists(result):
            print("❌ TTS generation failed")
            return False
        
        file_size = os.path.getsize(result)
        chars_per_sec = len(TEST_SENTENCE) / generation_time if generation_time > 0 else 0
        
        print(f"✅ TTS generation successful:")
        print(f"   Output: {result}")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Speed: {chars_per_sec:.1f} chars/sec")
        
        # Step 2: Transcribe audio
        print("\\n🎙️ STEP 2: Transcribing Audio")
        print("-" * 40)
        
        from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService
        
        # Initialize transcription service
        print("📥 Initializing transcription service...")
        transcription_service = AudioTranscriptionService()
        
        # Load Whisper model
        print("📥 Loading Whisper Arabic model...")
        start_whisper_time = time.time()
        whisper_success = transcription_service.load_whisper_model("whisper-large-v3-turbo-arabic")
        whisper_load_time = time.time() - start_whisper_time
        
        if not whisper_success:
            print("❌ Failed to load Whisper model")
            return False
        
        print(f"✅ Whisper model loaded in {whisper_load_time:.2f}s")
        
        # Transcribe the audio
        print(f"🎙️ Transcribing audio to text...")
        start_transcription_time = time.time()
        transcribed_text = transcription_service.transcribe_audio_file(
            audio_file_path=str(audio_path),
            language="ar"
        )
        transcription_time = time.time() - start_transcription_time
        
        if not transcribed_text:
            print("❌ Transcription failed")
            return False
        
        print(f"✅ Transcription successful:")
        print(f"   Transcription time: {transcription_time:.2f}s")
        print(f"   Transcribed length: {len(transcribed_text)} characters")
        print(f"   Transcribed words: {len(transcribed_text.split())} words")
        
        # Step 3: Calculate accuracy
        print("\\n📊 STEP 3: Accuracy Analysis")
        print("-" * 40)
        
        # Character-level similarity
        char_similarity = difflib.SequenceMatcher(None, TEST_SENTENCE, transcribed_text).ratio() * 100
        
        # Word-level analysis
        original_words = TEST_SENTENCE.split()
        transcribed_words = transcribed_text.split()
        word_similarity = difflib.SequenceMatcher(None, original_words, transcribed_words).ratio() * 100
        
        # Find common words
        original_set = set(original_words)
        transcribed_set = set(transcribed_words)
        common_words = original_set.intersection(transcribed_set)
        
        # Calculate precision, recall, F1
        precision = len(common_words) / len(transcribed_set) if len(transcribed_set) > 0 else 0
        recall = len(common_words) / len(original_set) if len(original_set) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Word Error Rate (WER)
        def calculate_wer(ref_words, hyp_words):
            d = [[0 for j in range(len(hyp_words) + 1)] for i in range(len(ref_words) + 1)]
            
            for i in range(len(ref_words) + 1):
                d[i][0] = i
            for j in range(len(hyp_words) + 1):
                d[0][j] = j
                
            for i in range(1, len(ref_words) + 1):
                for j in range(1, len(hyp_words) + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        substitution = d[i-1][j-1] + 1
                        insertion = d[i][j-1] + 1
                        deletion = d[i-1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)
            
            return d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
        
        wer = calculate_wer(original_words, transcribed_words) * 100
        
        # Display results
        print("\\n📝 TEXT COMPARISON:")
        print("="*70)
        print(f"Original:    {TEST_SENTENCE}")
        print("="*70)
        print(f"Transcribed: {transcribed_text}")
        print("="*70)
        
        print("\\n📊 ACCURACY METRICS:")
        print(f"✅ Character Similarity: {char_similarity:.1f}%")
        print(f"✅ Word Similarity:      {word_similarity:.1f}%")
        print(f"✅ Word Error Rate:      {wer:.1f}%")
        print(f"✅ Precision:            {precision*100:.1f}%")
        print(f"✅ Recall:               {recall*100:.1f}%")
        print(f"✅ F1 Score:             {f1_score*100:.1f}%")
        
        print("\\n📈 STATISTICS:")
        print(f"   Original words:       {len(original_words)}")
        print(f"   Transcribed words:    {len(transcribed_words)}")
        print(f"   Common words:         {len(common_words)}")
        print(f"   TTS generation time:  {generation_time:.2f}s")
        print(f"   Transcription time:   {transcription_time:.2f}s")
        print(f"   Total processing:     {generation_time + transcription_time:.2f}s")
        
        # Save results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_text": TEST_SENTENCE,
            "transcribed_text": transcribed_text,
            "audio_file": str(audio_path),
            "metrics": {
                "character_similarity": char_similarity,
                "word_similarity": word_similarity,
                "word_error_rate": wer,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1_score": f1_score * 100
            },
            "performance": {
                "tts_load_time": load_time,
                "tts_generation_time": generation_time,
                "whisper_load_time": whisper_load_time,
                "transcription_time": transcription_time,
                "total_time": generation_time + transcription_time,
                "chars_per_second_tts": chars_per_sec,
                "file_size": file_size
            },
            "word_stats": {
                "original_words": len(original_words),
                "transcribed_words": len(transcribed_words),
                "common_words": len(common_words)
            }
        }
        
        results_path = test_dir / f"quick_accuracy_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\\n💾 Results saved to: {results_path}")
        
        # Quality assessment
        print("\\n🎯 QUALITY ASSESSMENT:")
        if char_similarity >= 90:
            print("🌟 EXCELLENT: Character similarity ≥ 90%")
        elif char_similarity >= 80:
            print("✅ GOOD: Character similarity ≥ 80%")
        elif char_similarity >= 70:
            print("⚠️ FAIR: Character similarity ≥ 70%")
        else:
            print("❌ POOR: Character similarity < 70%")
        
        if wer <= 10:
            print("🌟 EXCELLENT: Word Error Rate ≤ 10%")
        elif wer <= 20:
            print("✅ GOOD: Word Error Rate ≤ 20%")
        elif wer <= 30:
            print("⚠️ FAIR: Word Error Rate ≤ 30%")
        else:
            print("❌ POOR: Word Error Rate > 30%")
        
        return True
        
    except ImportError as e:
        print(f"❌ Required library not available: {e}")
        print("💡 Make sure to install required dependencies:")
        print("   pip install outetts")
        print("   pip install transformers torch torchaudio")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    success = quick_accuracy_test()
    
    if success:
        print("\\n🎉 Quick accuracy test completed successfully!")
    else:
        print("\\n💥 Quick accuracy test failed!")
    
    return success

if __name__ == "__main__":
    main()
