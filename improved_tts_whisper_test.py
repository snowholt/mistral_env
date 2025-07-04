#!/usr/bin/env python3
"""
Enhanced TTS-to-Whisper Accuracy Test
Optimized for better Arabic transcription accuracy
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from beautyai_inference.services.tts_service import TTSService
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService

def calculate_enhanced_metrics(original: str, transcribed: str) -> dict:
    """Calculate enhanced accuracy metrics with Arabic-specific considerations"""
    
    # Normalize Arabic text (remove diacritics, normalize spaces)
    def normalize_arabic(text: str) -> str:
        import re
        # Remove Arabic diacritics
        diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        text = diacritics.sub('', text)
        # Normalize punctuation and spaces
        text = re.sub(r'[،؛؟!.\-]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    original_norm = normalize_arabic(original)
    transcribed_norm = normalize_arabic(transcribed)
    
    # Character-level similarity
    char_similarity = SequenceMatcher(None, original_norm, transcribed_norm).ratio() * 100
    
    # Word-level analysis
    original_words = original_norm.split()
    transcribed_words = transcribed_norm.split()
    
    # Find exact word matches
    common_words = set(original_words) & set(transcribed_words)
    
    # Calculate metrics
    precision = len(common_words) / len(transcribed_words) if transcribed_words else 0
    recall = len(common_words) / len(original_words) if original_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Word Error Rate (WER)
    wer = (len(original_words) - len(common_words)) / len(original_words) if original_words else 0
    
    # Word similarity (considering order)
    word_similarity = SequenceMatcher(None, original_words, transcribed_words).ratio() * 100
    
    # Length coverage (how much of original was transcribed)
    length_coverage = len(transcribed_norm) / len(original_norm) if original_norm else 0
    
    return {
        "character_similarity": round(char_similarity, 1),
        "word_similarity": round(word_similarity, 1),
        "word_error_rate": round(wer * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1_score * 100, 1),
        "length_coverage": round(length_coverage * 100, 1),
        "completeness_score": round((length_coverage + recall) * 50, 1)  # Combined metric
    }

def enhanced_tts_whisper_test():
    """Enhanced TTS-to-Whisper accuracy test with optimized parameters"""
    
    print("🎯 Enhanced TTS-to-Whisper Accuracy Test")
    print("=" * 70)
    
    # Test sentence (same as before)
    test_sentence = "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."
    
    print(f"📝 Testing sentence: {test_sentence[:50]}...")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("voice_tests/enhanced_accuracy_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_text": test_sentence,
        "tests": []
    }
    
    try:
        # STEP 1: Enhanced TTS Generation
        print("\\n📢 STEP 1: Enhanced TTS Audio Generation")
        print("-" * 40)
        
        print("📥 Initializing enhanced TTS service...")
        tts_start = time.time()
        tts_service = TTSService()
        tts_service.load_model("oute-tts-1b")
        tts_load_time = time.time() - tts_start
        print(f"✅ OuteTTS model loaded in {tts_load_time:.2f}s")
        
        # Enhanced TTS parameters for better quality
        tts_configs = [
            {
                "name": "high_quality",
                "params": {
                    "temperature": 0.1,      # Lower temperature for more consistent output
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "speed": 0.9,           # Slightly slower for clearer pronunciation
                }
            },
            {
                "name": "medium_quality", 
                "params": {
                    "temperature": 0.3,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                    "speed": 1.0,
                }
            },
            {
                "name": "fast_quality",
                "params": {
                    "temperature": 0.5,
                    "repetition_penalty": 0.9,
                    "length_penalty": 0.9,
                    "speed": 1.1,
                }
            }
        ]
        
        for config in tts_configs:
            print(f"\\n🔊 Testing {config['name']} configuration...")
            
            gen_start = time.time()
            unique_id = str(random.randint(1000000000, 9999999999))
            audio_file = output_dir / f"enhanced_test_{config['name']}_{unique_id}.wav"
            
            # Generate speech with enhanced parameters
            success = tts_service.text_to_speech(
                text=test_sentence,
                output_file=str(audio_file),
                language="ar",
                **config['params']
            )
            
            if not success:
                print(f"❌ TTS generation failed for {config['name']}")
                continue
                
            gen_time = time.time() - gen_start
            file_size = audio_file.stat().st_size if audio_file.exists() else 0
            
            print(f"✅ TTS generation successful:")
            print(f"   Output: {audio_file}")
            print(f"   File size: {file_size:,} bytes")
            print(f"   Generation time: {gen_time:.2f}s")
            print(f"   Speed: {len(test_sentence)/gen_time:.1f} chars/sec")
            
            # STEP 2: Enhanced Whisper Transcription
            print(f"\\n🎙️ STEP 2: Enhanced Transcription ({config['name']})")
            print("-" * 40)
            
            print("📥 Initializing enhanced transcription service...")
            whisper_start = time.time()
            transcription_service = AudioTranscriptionService()
            transcription_service.load_model("whisper-large-v3-turbo-arabic")
            whisper_load_time = time.time() - whisper_start
            print(f"✅ Whisper model loaded in {whisper_load_time:.2f}s")
            
            print("🎙️ Transcribing with enhanced parameters...")
            trans_start = time.time()
            
            # Enhanced transcription parameters
            transcription_result = transcription_service.transcribe_audio(
                audio_file=str(audio_file),
                language="ar",
                task="transcribe",
                # Enhanced Whisper parameters
                max_new_tokens=300,         # Increased for longer transcription
                num_beams=5,               # Better beam search
                do_sample=False,           # Deterministic decoding
                temperature=0.0,           # Most conservative
                return_timestamps=False,
                chunk_length_s=30,         # Process full audio
                batch_size=1               # Single batch for quality
            )
            
            trans_time = time.time() - trans_start
            
            if transcription_result and transcription_result.get("text"):
                transcribed_text = transcription_result["text"].strip()
                print(f"✅ Transcription successful:")
                print(f"   Transcription time: {trans_time:.2f}s")
                print(f"   Transcribed length: {len(transcribed_text)} characters")
                print(f"   Transcribed words: {len(transcribed_text.split())} words")
            else:
                print("❌ Transcription failed")
                continue
            
            # STEP 3: Enhanced Accuracy Analysis
            print(f"\\n📊 STEP 3: Enhanced Accuracy Analysis ({config['name']})")
            print("-" * 40)
            
            metrics = calculate_enhanced_metrics(test_sentence, transcribed_text)
            
            print("\\n📝 TEXT COMPARISON:")
            print("=" * 70)
            print(f"Original:    {test_sentence}")
            print("=" * 70)
            print(f"Transcribed: {transcribed_text}")
            print("=" * 70)
            
            print("\\n📊 ENHANCED ACCURACY METRICS:")
            print(f"✅ Character Similarity: {metrics['character_similarity']}%")
            print(f"✅ Word Similarity:      {metrics['word_similarity']}%")
            print(f"✅ Word Error Rate:      {metrics['word_error_rate']}%")
            print(f"✅ Precision:            {metrics['precision']}%")
            print(f"✅ Recall:               {metrics['recall']}%")
            print(f"✅ F1 Score:             {metrics['f1_score']}%")
            print(f"✅ Length Coverage:      {metrics['length_coverage']}%")
            print(f"✅ Completeness Score:   {metrics['completeness_score']}%")
            
            # Store results
            test_result = {
                "config_name": config['name'],
                "config_params": config['params'],
                "transcribed_text": transcribed_text,
                "audio_file": str(audio_file),
                "metrics": metrics,
                "performance": {
                    "tts_load_time": tts_load_time,
                    "tts_generation_time": gen_time,
                    "whisper_load_time": whisper_load_time,
                    "transcription_time": trans_time,
                    "total_time": gen_time + trans_time,
                    "chars_per_second_tts": len(test_sentence) / gen_time,
                    "file_size": file_size
                },
                "word_stats": {
                    "original_words": len(test_sentence.split()),
                    "transcribed_words": len(transcribed_text.split()),
                    "original_chars": len(test_sentence),
                    "transcribed_chars": len(transcribed_text)
                }
            }
            
            results["tests"].append(test_result)
            
            # Quality assessment
            print("\\n🎯 QUALITY ASSESSMENT:")
            if metrics['completeness_score'] >= 80:
                print("🌟 EXCELLENT: Completeness score ≥ 80%")
            elif metrics['completeness_score'] >= 60:
                print("✅ GOOD: Completeness score ≥ 60%")
            elif metrics['completeness_score'] >= 40:
                print("⚠️ FAIR: Completeness score ≥ 40%")
            else:
                print("❌ POOR: Completeness score < 40%")
            
            print(f"\\n{'='*50}")
        
        # STEP 4: Compare Results and Save
        print("\\n📈 STEP 4: Results Comparison")
        print("-" * 40)
        
        if results["tests"]:
            best_test = max(results["tests"], key=lambda x: x["metrics"]["completeness_score"])
            
            print("\\n🏆 BEST PERFORMING CONFIGURATION:")
            print(f"   Config: {best_test['config_name']}")
            print(f"   Completeness Score: {best_test['metrics']['completeness_score']}%")
            print(f"   Character Similarity: {best_test['metrics']['character_similarity']}%")
            print(f"   Length Coverage: {best_test['metrics']['length_coverage']}%")
            print(f"   F1 Score: {best_test['metrics']['f1_score']}%")
            
            # Save comprehensive results
            results_file = output_dir / f"enhanced_accuracy_results_{unique_id}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\\n💾 Results saved to: {results_file}")
        
        print("\\n🎉 Enhanced accuracy test completed successfully!")
        
    except Exception as e:
        print(f"\\n💥 Enhanced accuracy test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    enhanced_tts_whisper_test()
