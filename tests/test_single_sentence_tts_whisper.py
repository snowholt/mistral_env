#!/usr/bin/env python3
"""
Single Sentence TTS-to-Whisper Accuracy Test for BeautyAI Platform.

A simplified test script that:
1. Takes one Arabic sentence (specified by user)
2. Converts it to speech using Arabic speaker profile 
3. Transcribes the generated audio back to text using Whisper
4. Measures accuracy between original and transcribed text

Test sentence: "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."
"""

import sys
import os
import time
import json
import difflib
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test sentence - the exact one specified by the user
TEST_SENTENCE = "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."

def create_test_directories():
    """Create necessary test directories."""
    test_dir = Path("/home/lumi/beautyai/voice_tests/single_sentence_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

def calculate_accuracy_metrics(original: str, transcribed: str) -> Dict[str, any]:
    """Calculate accuracy metrics between original and transcribed text."""
    
    print(f"\n📊 Calculating accuracy metrics...")
    print(f"📝 Original:    '{original}'")
    print(f"🎙️ Transcribed: '{transcribed}'")
    
    # Basic word-level comparison
    original_words = original.split()
    transcribed_words = transcribed.split()
    
    # Character-level similarity
    char_similarity = difflib.SequenceMatcher(None, original, transcribed).ratio()
    
    # Word-level similarity
    word_similarity = difflib.SequenceMatcher(None, original_words, transcribed_words).ratio()
    
    # Calculate detailed differences
    differ = difflib.unified_diff(
        original.split(), 
        transcribed.split(),
        fromfile='original',
        tofile='transcribed',
        lineterm=''
    )
    differences = list(differ)
    
    metrics = {
        "original_text": original,
        "transcribed_text": transcribed,
        "original_length": len(original),
        "transcribed_length": len(transcribed),
        "original_word_count": len(original_words),
        "transcribed_word_count": len(transcribed_words),
        "character_accuracy": char_similarity * 100,
        "word_accuracy": word_similarity * 100,
        "length_difference": abs(len(original) - len(transcribed)),
        "word_count_difference": abs(len(original_words) - len(transcribed_words)),
        "differences": differences,
        "exact_match": original.strip() == transcribed.strip()
    }
    
    # Print summary
    print(f"✅ Character accuracy: {metrics['character_accuracy']:.2f}%")
    print(f"✅ Word accuracy: {metrics['word_accuracy']:.2f}%")
    print(f"📏 Length difference: {metrics['length_difference']} characters")
    print(f"📊 Word count difference: {metrics['word_count_difference']} words")
    print(f"🎯 Exact match: {'Yes' if metrics['exact_match'] else 'No'}")
    
    return metrics

def generate_tts_audio(text: str, output_path: str, speaker_name: str = "arabic_female_premium_19s") -> Tuple[bool, Dict[str, any]]:
    """Generate TTS audio using OuteTTS with Arabic speaker profile."""
    
    print(f"\n🔊 Generating TTS audio...")
    print(f"📝 Text: {text}")
    print(f"🎤 Speaker: {speaker_name}")
    print(f"💾 Output: {output_path}")
    
    generation_metrics = {
        "speaker": speaker_name,
        "text_length": len(text),
        "word_count": len(text.split()),
        "output_path": output_path
    }
    
    try:
        # Import OuteTTS
        try:
            import outetts
        except ImportError:
            error_msg = "OuteTTS library not available. Install with: pip install outetts"
            print(f"❌ {error_msg}")
            generation_metrics.update({"success": False, "error": error_msg})
            return False, generation_metrics
        
        print("📥 Initializing OuteTTS interface...")
        start_time = time.time()
        
        # Initialize OuteTTS with optimized Arabic configuration
        interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
            )
        )
        
        init_time = time.time() - start_time
        generation_metrics["initialization_time"] = init_time
        print(f"✅ OuteTTS initialized ({init_time:.2f}s)")
        
        # Load Arabic speaker profile
        speaker = None
        speaker_profile_path = f"/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/{speaker_name}.json"
        
        if os.path.exists(speaker_profile_path):
            print(f"👤 Loading speaker profile: {speaker_profile_path}")
            speaker = interface.load_speaker(speaker_profile_path)
            print(f"✅ Speaker profile loaded successfully")
        else:
            print(f"⚠️ Speaker profile not found: {speaker_profile_path}")
            print(f"🔄 Using default OuteTTS Arabic voice")
        
        # Generate speech with optimized Arabic parameters
        print("🎵 Generating speech...")
        generation_start = time.time()
        
        config = outetts.GenerationConfig(
            text=text,
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(
                temperature=0.0,          # Very low for Arabic consistency
                top_p=0.75,              # Better control for Arabic morphology
                top_k=25,                # Lower for more consistent Arabic
                repetition_penalty=1.02, # Minimal to avoid breaking Arabic words
                repetition_range=32,     # Shorter for Arabic word structure
                min_p=0.02              # Lower threshold for Arabic phonemes
            ),
            max_length=8192
        )
        
        output = interface.generate(config)
        output.save(output_path)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            chars_per_second = len(text) / generation_time if generation_time > 0 else 0
            
            generation_metrics.update({
                "success": True,
                "generation_time": generation_time,
                "total_time": total_time,
                "file_size": file_size,
                "chars_per_second": chars_per_second
            })
            
            print(f"✅ Audio generated successfully!")
            print(f"⏱️ Generation time: {generation_time:.2f}s")
            print(f"📊 Speed: {chars_per_second:.1f} chars/sec")
            print(f"📁 File size: {file_size:,} bytes")
            
            return True, generation_metrics
        else:
            error_msg = "Audio file was not created"
            print(f"❌ {error_msg}")
            generation_metrics.update({"success": False, "error": error_msg})
            return False, generation_metrics
            
    except Exception as e:
        error_msg = f"TTS generation failed: {e}"
        print(f"❌ {error_msg}")
        generation_metrics.update({"success": False, "error": error_msg})
        return False, generation_metrics

def transcribe_audio_to_text(audio_path: str) -> Tuple[Optional[str], Dict[str, any]]:
    """Transcribe audio file to text using Whisper."""
    
    print(f"\n🎙️ Transcribing audio using Whisper...")
    print(f"📁 Audio file: {audio_path}")
    
    transcription_metrics = {
        "audio_path": audio_path,
        "whisper_model": "whisper-large-v3-turbo-arabic"
    }
    
    try:
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file size and basic info
        file_size = os.path.getsize(audio_path)
        transcription_metrics["file_size"] = file_size
        print(f"📊 Audio file size: {file_size:,} bytes")
        
        # Use BeautyAI Audio Transcription Service
        from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService
        
        print("📥 Initializing Audio Transcription service...")
        start_time = time.time()
        
        transcription_service = AudioTranscriptionService()
        
        # Load Whisper model
        model_name = "whisper-large-v3-turbo-arabic"
        print(f"🔄 Loading Whisper model: {model_name}")
        
        model_load_start = time.time()
        success = transcription_service.load_whisper_model(model_name)
        model_load_time = time.time() - model_load_start
        
        if not success:
            raise Exception(f"Failed to load Whisper model: {model_name}")
        
        transcription_metrics["model_load_time"] = model_load_time
        print(f"✅ Whisper model loaded ({model_load_time:.2f}s)")
        
        # Transcribe audio with debug information
        print("🎯 Transcribing audio...")
        transcription_start = time.time()
        
        # Add debug info about audio file
        import torchaudio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            print(f"🔍 Audio debug info:")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Channels: {waveform.shape[0]}")
            print(f"   Samples: {waveform.shape[1]:,}")
        except Exception as e:
            print(f"⚠️ Could not load audio for debug: {e}")
        
        # Try transcription with explicit language setting
        transcription = transcription_service.transcribe_audio_file(
            audio_file_path=audio_path,
            language="ar"
        )
        
        # If transcription is too short, try again with automatic language detection
        if transcription and len(transcription.strip()) < 10:
            print(f"⚠️ Short transcription detected: '{transcription}'")
            print(f"🔄 Retrying with automatic language detection...")
            
            transcription_retry = transcription_service.transcribe_audio_file(
                audio_file_path=audio_path,
                language=None  # Auto-detect language
            )
            
            if transcription_retry and len(transcription_retry.strip()) > len(transcription.strip()):
                print(f"✅ Better transcription found: '{transcription_retry[:50]}...'")
                transcription = transcription_retry
            else:
                print(f"❌ Retry didn't improve transcription")
        
        transcription_time = time.time() - transcription_start
        total_time = time.time() - start_time
        
        if transcription:
            transcription_metrics.update({
                "success": True,
                "transcription": transcription,
                "transcription_time": transcription_time,
                "total_time": total_time,
                "transcription_length": len(transcription),
                "transcription_word_count": len(transcription.split())
            })
            
            print(f"✅ Transcription completed successfully!")
            print(f"📝 Result: '{transcription}'")
            print(f"⏱️ Transcription time: {transcription_time:.2f}s")
            print(f"📊 Total time: {total_time:.2f}s")
            
            return transcription, transcription_metrics
        else:
            error_msg = "Transcription returned empty result"
            print(f"❌ {error_msg}")
            transcription_metrics.update({"success": False, "error": error_msg})
            return None, transcription_metrics
            
    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        print(f"❌ {error_msg}")
        transcription_metrics.update({"success": False, "error": error_msg})
        return None, transcription_metrics

def save_test_results(results: Dict, output_path: Path):
    """Save test results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Test results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """Main function to run the single sentence TTS-to-Whisper accuracy test."""
    
    print("🎭 Single Sentence TTS-to-Whisper Accuracy Test")
    print("=" * 80)
    print("Testing Arabic sentence:")
    print(f"📝 '{TEST_SENTENCE}'")
    print("=" * 80)
    
    # Create test directories
    test_dir = create_test_directories()
    print(f"📁 Test output directory: {test_dir}")
    
    # Create output paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_filename = f"arabic_test_sentence_{timestamp}.wav"
    audio_path = test_dir / audio_filename
    results_filename = f"accuracy_test_results_{timestamp}.json"
    results_path = test_dir / results_filename
    
    # Initialize results dictionary
    test_results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_description": "Single Sentence TTS-to-Whisper Accuracy Test",
        "test_sentence": TEST_SENTENCE,
        "audio_file": str(audio_path),
        "results_file": str(results_path),
        "tts_results": {},
        "transcription_results": {},
        "accuracy_metrics": {},
        "overall_success": False
    }
    
    try:
        # Step 1: Generate TTS audio
        print(f"\n{'=' * 60}")
        print("STEP 1: TEXT-TO-SPEECH GENERATION")
        print(f"{'=' * 60}")
        
        tts_success, tts_metrics = generate_tts_audio(
            text=TEST_SENTENCE,
            output_path=str(audio_path),
            speaker_name="arabic_female_premium_19s"
        )
        
        test_results["tts_results"] = tts_metrics
        
        if not tts_success:
            print(f"❌ TTS generation failed. Test cannot continue.")
            test_results["error"] = "TTS generation failed"
            save_test_results(test_results, results_path)
            return
        
        # Step 2: Transcribe audio back to text
        print(f"\n{'=' * 60}")
        print("STEP 2: SPEECH-TO-TEXT TRANSCRIPTION")
        print(f"{'=' * 60}")
        
        transcription, transcription_metrics = transcribe_audio_to_text(str(audio_path))
        
        test_results["transcription_results"] = transcription_metrics
        
        if not transcription:
            print(f"❌ Audio transcription failed. Test cannot continue.")
            test_results["error"] = "Audio transcription failed"
            save_test_results(test_results, results_path)
            return
        
        # Step 3: Calculate accuracy metrics
        print(f"\n{'=' * 60}")
        print("STEP 3: ACCURACY ANALYSIS")
        print(f"{'=' * 60}")
        
        accuracy_metrics = calculate_accuracy_metrics(TEST_SENTENCE, transcription)
        test_results["accuracy_metrics"] = accuracy_metrics
        test_results["overall_success"] = True
        
        # Step 4: Print final summary
        print(f"\n{'=' * 80}")
        print("📊 FINAL TEST SUMMARY")
        print(f"{'=' * 80}")
        
        print(f"🎯 Test Status: {'✅ SUCCESS' if test_results['overall_success'] else '❌ FAILED'}")
        print(f"📝 Original Text: '{TEST_SENTENCE}'")
        print(f"🎙️ Transcribed Text: '{transcription}'")
        print(f"📊 Character Accuracy: {accuracy_metrics['character_accuracy']:.2f}%")
        print(f"📊 Word Accuracy: {accuracy_metrics['word_accuracy']:.2f}%")
        print(f"🎯 Exact Match: {'Yes' if accuracy_metrics['exact_match'] else 'No'}")
        
        # Performance summary
        tts_time = tts_metrics.get('generation_time', 0)
        transcription_time = transcription_metrics.get('transcription_time', 0)
        total_time = tts_time + transcription_time
        
        print(f"\n⏱️ Performance Summary:")
        print(f"   TTS Generation: {tts_time:.2f}s")
        print(f"   STT Transcription: {transcription_time:.2f}s")
        print(f"   Total Processing: {total_time:.2f}s")
        
        # File summary
        audio_size = tts_metrics.get('file_size', 0)
        print(f"\n📁 File Summary:")
        print(f"   Audio File: {audio_path}")
        print(f"   Audio Size: {audio_size:,} bytes")
        print(f"   Results File: {results_path}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        test_results["error"] = str(e)
        test_results["overall_success"] = False
    
    finally:
        # Save results
        save_test_results(test_results, results_path)
        
        print(f"\n✅ Single sentence TTS-to-Whisper accuracy test completed!")
        print(f"📄 Detailed results saved in: {results_path}")

if __name__ == "__main__":
    main()
