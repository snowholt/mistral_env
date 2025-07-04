#!/usr/bin/env python3
"""
TTS-to-Whisper Accuracy Test Script for BeautyAI Platform.

This script tests the accuracy of Arabic TTS generation by:
1. Converting a test sentence to speech using OuteTTS
2. Transcribing the generated audio back to text using Whisper
3. Comparing the original text with the transcribed text for accuracy

Test sentence: "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."
"""

import sys
import os
import time
import json
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the beautyai_inference package to the path
sys.path.insert(0, '/home/lumi/beautyai')

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test sentence - the one specified by the user
TEST_SENTENCE = "مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."

def create_test_directories():
    """Create necessary test directories."""
    test_dir = Path("/home/lumi/beautyai/voice_tests/tts_whisper_accuracy")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (test_dir / "generated_audio").mkdir(exist_ok=True)
    (test_dir / "accuracy_reports").mkdir(exist_ok=True)
    
    return test_dir

def calculate_text_similarity(original: str, transcribed: str) -> Dict[str, any]:
    """Calculate similarity metrics between original and transcribed text."""
    
    # Basic metrics
    original_words = original.split()
    transcribed_words = transcribed.split()
    
    # Character-level comparison
    char_similarity = difflib.SequenceMatcher(None, original, transcribed).ratio()
    
    # Word-level comparison
    word_similarity = difflib.SequenceMatcher(None, original_words, transcribed_words).ratio()
    
    # Calculate word error rate (WER)
    def word_error_rate(ref_words, hyp_words):
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
    
    wer = word_error_rate(original_words, transcribed_words)
    
    # Find common words
    original_set = set(original_words)
    transcribed_set = set(transcribed_words)
    common_words = original_set.intersection(transcribed_set)
    
    # Calculate precision and recall
    precision = len(common_words) / len(transcribed_set) if len(transcribed_set) > 0 else 0
    recall = len(common_words) / len(original_set) if len(original_set) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "character_similarity": char_similarity * 100,  # Convert to percentage
        "word_similarity": word_similarity * 100,
        "word_error_rate": wer * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1_score * 100,
        "original_word_count": len(original_words),
        "transcribed_word_count": len(transcribed_words),
        "common_words_count": len(common_words)
    }

def generate_tts_audio(
    text: str, 
    speaker_name: str, 
    output_path: str,
    tts_engine: str = "beautyai_service"
) -> Tuple[bool, Dict[str, any]]:
    """Generate TTS audio using the specified engine."""
    
    print(f"🔊 Generating TTS audio using {tts_engine}...")
    print(f"📝 Text: {text}")
    print(f"🎤 Speaker: {speaker_name}")
    
    generation_metrics = {
        "engine": tts_engine,
        "speaker": speaker_name,
        "text_length": len(text),
        "word_count": len(text.split()),
        "output_path": output_path
    }
    
    try:
        if tts_engine == "beautyai_service":
            # Use BeautyAI TTS Service
            from beautyai_inference.services.text_to_speech_service import TextToSpeechService
            
            print("📥 Initializing BeautyAI TTS service...")
            tts_service = TextToSpeechService()
            
            # Load the OuteTTS model
            print("📥 Loading OuteTTS model...")
            start_load_time = time.time()
            success = tts_service.load_tts_model("oute-tts-1b")
            load_time = time.time() - start_load_time
            
            if not success:
                raise Exception("Failed to load OuteTTS model")
            
            print(f"✅ OuteTTS model loaded in {load_time:.2f}s")
            generation_metrics["model_load_time"] = load_time
            
            # Generate speech
            start_gen_time = time.time()
            result = tts_service.text_to_speech(
                text=text,
                language="ar",
                speaker_voice=speaker_name,
                output_path=output_path
            )
            generation_time = time.time() - start_gen_time
            
            if result and os.path.exists(result):
                file_size = os.path.getsize(result)
                generation_metrics.update({
                    "generation_time": generation_time,
                    "file_size": file_size,
                    "chars_per_second": len(text) / generation_time if generation_time > 0 else 0,
                    "success": True
                })
                print(f"✅ TTS generation successful: {result}")
                print(f"   File size: {file_size:,} bytes")
                print(f"   Generation time: {generation_time:.2f}s")
                print(f"   Speed: {generation_metrics['chars_per_second']:.1f} chars/sec")
                return True, generation_metrics
            else:
                raise Exception("TTS generation failed - no output file")
                
        elif tts_engine == "outetts_direct":
            # Use OuteTTS directly
            import outetts
            
            print("📥 Initializing OuteTTS interface...")
            start_load_time = time.time()
            interface = outetts.Interface(
                config=outetts.ModelConfig.auto_config(
                    model=outetts.Models.VERSION_1_0_SIZE_1B,
                    backend=outetts.Backend.LLAMACPP,
                    quantization=outetts.LlamaCppQuantization.FP16
                )
            )
            load_time = time.time() - start_load_time
            
            print(f"✅ OuteTTS interface loaded in {load_time:.2f}s")
            generation_metrics["model_load_time"] = load_time
            
            # Load speaker profile if specified
            speaker = None
            if speaker_name != "default":
                speaker_profile_path = f"/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/{speaker_name}.json"
                if os.path.exists(speaker_profile_path):
                    print(f"👤 Loading speaker profile: {speaker_profile_path}")
                    speaker = interface.load_speaker(speaker_profile_path)
                    print(f"✅ Speaker profile loaded")
                else:
                    print(f"⚠️ Speaker profile not found: {speaker_profile_path}")
                    print(f"🔄 Using default OuteTTS Arabic speaker")
            
            # Generate speech
            start_gen_time = time.time()
            
            # Configure generation parameters optimized for Arabic
            config = outetts.GenerationConfig(
                text=text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=speaker,  # Use loaded speaker or None for default
                sampler_config=outetts.SamplerConfig(
                    temperature=0.0,          # Very low for consistency
                    top_p=0.75,              # Better control for Arabic
                    top_k=25,                # Lower for more consistent Arabic
                    repetition_penalty=1.02, # Minimal to avoid breaking Arabic words
                    repetition_range=32,     # Shorter for Arabic word structure
                    min_p=0.02              # Lower threshold for Arabic phonemes
                ),
                max_length=8192
            )
            
            output = interface.generate(config)
            output.save(output_path)
            
            generation_time = time.time() - start_gen_time
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                generation_metrics.update({
                    "generation_time": generation_time,
                    "file_size": file_size,
                    "chars_per_second": len(text) / generation_time if generation_time > 0 else 0,
                    "success": True
                })
                print(f"✅ TTS generation successful: {output_path}")
                print(f"   File size: {file_size:,} bytes")
                print(f"   Generation time: {generation_time:.2f}s")
                print(f"   Speed: {generation_metrics['chars_per_second']:.1f} chars/sec")
                return True, generation_metrics
            else:
                raise Exception("TTS generation failed - no output file")
        
        else:
            raise Exception(f"Unknown TTS engine: {tts_engine}")
            
    except ImportError as e:
        error_msg = f"Required library not available: {e}"
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
    
    print(f"🎙️ Transcribing audio using Whisper...")
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
        transcription_service = AudioTranscriptionService()
        
        # Load the Whisper model
        print("📥 Loading Whisper model for Arabic...")
        start_load_time = time.time()
        success = transcription_service.load_whisper_model("whisper-large-v3-turbo-arabic")
        load_time = time.time() - start_load_time
        
        if not success:
            raise Exception("Failed to load Whisper model")
        
        print(f"✅ Whisper model loaded in {load_time:.2f}s")
        transcription_metrics["model_load_time"] = load_time
        
        # Transcribe the audio
        start_transcription_time = time.time()
        transcribed_text = transcription_service.transcribe_audio_file(
            audio_file_path=audio_path,
            language="ar"  # Arabic language code
        )
        transcription_time = time.time() - start_transcription_time
        
        transcription_metrics["transcription_time"] = transcription_time
        
        if transcribed_text:
            transcription_metrics.update({
                "success": True,
                "transcribed_text": transcribed_text,
                "transcribed_length": len(transcribed_text),
                "transcribed_word_count": len(transcribed_text.split())
            })
            print(f"✅ Transcription successful:")
            print(f"   Transcribed text: {transcribed_text}")
            print(f"   Length: {len(transcribed_text)} characters")
            print(f"   Words: {len(transcribed_text.split())} words")
            print(f"   Transcription time: {transcription_time:.2f}s")
            return transcribed_text, transcription_metrics
        else:
            raise Exception("Transcription failed - no text returned")
            
    except ImportError as e:
        error_msg = f"Required library not available: {e}"
        print(f"❌ {error_msg}")
        transcription_metrics.update({"success": False, "error": error_msg})
        return None, transcription_metrics
    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        print(f"❌ {error_msg}")
        transcription_metrics.update({"success": False, "error": error_msg})
        return None, transcription_metrics

def run_tts_whisper_accuracy_test(
    text: str,
    speaker_name: str = "female",
    tts_engine: str = "beautyai_service",
    test_name: str = "default"
) -> Dict[str, any]:
    """Run complete TTS-to-Whisper accuracy test."""
    
    print(f"\n🧪 Running TTS-to-Whisper Accuracy Test: {test_name}")
    print("=" * 80)
    print(f"📝 Original text: {text}")
    print(f"🎤 Speaker: {speaker_name}")
    print(f"🔧 TTS Engine: {tts_engine}")
    print("=" * 80)
    
    # Create test directories
    test_dir = create_test_directories()
    
    # Generate unique output path
    timestamp = int(time.time())
    audio_filename = f"{test_name}_{speaker_name}_{timestamp}.wav"
    audio_path = test_dir / "generated_audio" / audio_filename
    
    # Initialize test results
    test_results = {
        "test_name": test_name,
        "timestamp": timestamp,
        "original_text": text,
        "speaker_name": speaker_name,
        "tts_engine": tts_engine,
        "audio_path": str(audio_path)
    }
    
    # Step 1: Generate TTS audio
    print(f"\n📢 STEP 1: TTS Generation")
    print("-" * 40)
    
    tts_success, tts_metrics = generate_tts_audio(
        text=text,
        speaker_name=speaker_name,
        output_path=str(audio_path),
        tts_engine=tts_engine
    )
    
    test_results["tts_generation"] = tts_metrics
    
    if not tts_success:
        print(f"❌ TTS generation failed, cannot proceed with transcription")
        test_results["overall_success"] = False
        return test_results
    
    # Step 2: Transcribe audio to text
    print(f"\n🎙️ STEP 2: Audio Transcription")
    print("-" * 40)
    
    transcribed_text, transcription_metrics = transcribe_audio_to_text(str(audio_path))
    test_results["transcription"] = transcription_metrics
    
    if not transcribed_text:
        print(f"❌ Transcription failed, cannot calculate accuracy")
        test_results["overall_success"] = False
        return test_results
    
    # Step 3: Calculate accuracy metrics
    print(f"\n📊 STEP 3: Accuracy Analysis")
    print("-" * 40)
    
    accuracy_metrics = calculate_text_similarity(text, transcribed_text)
    test_results["accuracy"] = accuracy_metrics
    test_results["transcribed_text"] = transcribed_text
    test_results["overall_success"] = True
    
    # Display results
    print(f"📝 Original:    {text}")
    print(f"🎙️ Transcribed: {transcribed_text}")
    print(f"")
    print(f"📊 ACCURACY METRICS:")
    print(f"   Character Similarity: {accuracy_metrics['character_similarity']:.1f}%")
    print(f"   Word Similarity:      {accuracy_metrics['word_similarity']:.1f}%")
    print(f"   Word Error Rate:      {accuracy_metrics['word_error_rate']:.1f}%")
    print(f"   Precision:            {accuracy_metrics['precision']:.1f}%")
    print(f"   Recall:               {accuracy_metrics['recall']:.1f}%")
    print(f"   F1 Score:             {accuracy_metrics['f1_score']:.1f}%")
    print(f"")
    print(f"📈 WORD STATISTICS:")
    print(f"   Original words:       {accuracy_metrics['original_word_count']}")
    print(f"   Transcribed words:    {accuracy_metrics['transcribed_word_count']}")
    print(f"   Common words:         {accuracy_metrics['common_words_count']}")
    
    return test_results

def save_test_results(results: Dict, output_path: Path):
    """Save test results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Test results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """Main function to run TTS-to-Whisper accuracy test."""
    
    print("🎯 TTS-to-Whisper Accuracy Testing Suite")
    print("="*80)
    print("Testing Arabic TTS accuracy by generating speech and transcribing back to text")
    print(f"Test sentence: {TEST_SENTENCE[:60]}...")
    print("="*80)
    
    # Create test directories
    test_dir = create_test_directories()
    print(f"📁 Test output directory: {test_dir}")
    
    # Test configurations to run
    test_configs = [
        {
            "name": "beautyai_service_female",
            "tts_engine": "beautyai_service",
            "speaker": "female",
            "description": "BeautyAI TTS Service with female speaker"
        },
        {
            "name": "beautyai_service_male", 
            "tts_engine": "beautyai_service",
            "speaker": "male",
            "description": "BeautyAI TTS Service with male speaker"
        },
        {
            "name": "outetts_direct_default",
            "tts_engine": "outetts_direct",
            "speaker": "default",
            "description": "Direct OuteTTS with default Arabic speaker"
        },
        {
            "name": "outetts_direct_premium_female",
            "tts_engine": "outetts_direct", 
            "speaker": "arabic_female_premium_19s",
            "description": "Direct OuteTTS with premium female Arabic speaker"
        }
    ]
    
    all_results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_description": "TTS-to-Whisper Accuracy Testing for Arabic Beauty Clinic Sentence",
        "test_sentence": TEST_SENTENCE,
        "test_results": []
    }
    
    # Run each test configuration
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*80}")
        print(f"🧪 TEST {i}/4: {config['description']}")
        print(f"{'='*80}")
        
        try:
            # Run the test
            result = run_tts_whisper_accuracy_test(
                text=TEST_SENTENCE,
                speaker_name=config["speaker"],
                tts_engine=config["tts_engine"],
                test_name=config["name"]
            )
            
            # Add configuration info to results
            result["test_config"] = config
            all_results["test_results"].append(result)
            
            # Save individual test results
            individual_results_path = test_dir / "accuracy_reports" / f"{config['name']}_results.json"
            save_test_results(result, individual_results_path)
            
            # Brief summary
            if result.get("overall_success"):
                accuracy = result["accuracy"]
                print(f"\n✅ Test completed successfully!")
                print(f"   Character similarity: {accuracy['character_similarity']:.1f}%")
                print(f"   Word similarity: {accuracy['word_similarity']:.1f}%")
                print(f"   Word error rate: {accuracy['word_error_rate']:.1f}%")
            else:
                print(f"\n❌ Test failed!")
                
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            error_result = {
                "test_config": config,
                "overall_success": False,
                "error": str(e)
            }
            all_results["test_results"].append(error_result)
    
    # Save combined results
    combined_results_path = test_dir / "accuracy_reports" / "tts_whisper_accuracy_complete.json"
    save_test_results(all_results, combined_results_path)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = 0
    for result in all_results["test_results"]:
        if result.get("overall_success"):
            successful_tests += 1
            config_name = result["test_config"]["name"]
            accuracy = result["accuracy"]
            
            print(f"\n🎤 {config_name}:")
            print(f"   ✅ Character Similarity: {accuracy['character_similarity']:.1f}%")
            print(f"   ✅ Word Similarity: {accuracy['word_similarity']:.1f}%") 
            print(f"   ✅ Word Error Rate: {accuracy['word_error_rate']:.1f}%")
            print(f"   ✅ F1 Score: {accuracy['f1_score']:.1f}%")
        else:
            config_name = result["test_config"]["name"]
            print(f"\n❌ {config_name}: FAILED")
    
    print(f"\n📈 Overall Results: {successful_tests}/{len(test_configs)} tests successful")
    print(f"📁 Audio files saved in: {test_dir / 'generated_audio'}")
    print(f"📄 Accuracy reports saved in: {test_dir / 'accuracy_reports'}")
    print(f"📄 Complete results: {combined_results_path}")
    print(f"\n✅ TTS-to-Whisper accuracy testing completed!")

if __name__ == "__main__":
    main()
