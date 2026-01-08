#!/usr/bin/env python3
"""
XTTS → Whisper Round-Trip Test for Arabic TTS Quality Verification

This test verifies that XTTS is producing correct Arabic pronunciation by:
1. Generating audio from Arabic text using XTTS
2. Transcribing the generated audio with Whisper Large v3 Turbo
3. Comparing the original text with the transcription

If XTTS is speaking correctly, Whisper should transcribe it accurately.

Author: BeautyAI Framework
Date: 2025-12-09
"""

import sys
import os
import time
import wave
import difflib
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

import torch
import numpy as np


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison (remove diacritics, normalize spaces)."""
    import re
    # Remove Arabic diacritics (harakat)
    diacritics = r'[\u064B-\u065F\u0670]'
    text = re.sub(diacritics, '', text)
    # Normalize spaces
    text = ' '.join(text.split())
    # Remove punctuation for comparison
    text = re.sub(r'[،؟!.,?]', '', text)
    return text.strip()


def calculate_similarity(original: str, transcribed: str) -> float:
    """Calculate similarity ratio between two texts."""
    orig_norm = normalize_arabic(original)
    trans_norm = normalize_arabic(transcribed)
    return difflib.SequenceMatcher(None, orig_norm, trans_norm).ratio()


def test_xtts_whisper_roundtrip():
    """Test XTTS audio generation and Whisper transcription round-trip."""
    
    print("=" * 70)
    print("🧪 XTTS → Whisper Round-Trip Test for Arabic TTS Quality")
    print("=" * 70)
    
    # Test texts
    test_texts = [
        "مرحبا! أنا بخير، شكرًا لسؤالك. كيف حالك أنت؟",
        "أهلا وسهلا",
        "كيف حالك؟",
    ]
    
    # Speaker reference files to test
    speaker_refs = {
        "Arabic (q1.wav)": "/home/lumi/beautyai/voice_tests/input_test_questions/q1.wav",
        "English (botox.wav)": "/home/lumi/beautyai/tests/webrtc/botox.wav",
    }
    
    # Output directory
    output_dir = Path("/home/lumi/beautyai/tests/webrtc/xtts_whisper_test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🎯 Testing {len(test_texts)} texts with {len(speaker_refs)} speaker references\n")
    
    # =========================================================================
    # Step 1: Load XTTS Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("📦 Step 1: Loading XTTS Model...")
    print("=" * 70)
    
    try:
        from beautyai_inference.inference_engines.voice.tts.xtts_engine import XTTSEngine
        
        xtts_start = time.time()
        xtts = XTTSEngine()
        if not xtts.load_model():
            print("❌ Failed to load XTTS model!")
            return False
        xtts_load_time = time.time() - xtts_start
        print(f"✅ XTTS loaded in {xtts_load_time:.2f}s")
        print(f"   Device: {xtts.device}")
        print(f"   Sample Rate: {xtts.output_sample_rate}Hz")
        
    except Exception as e:
        print(f"❌ XTTS loading error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Step 2: Load Whisper Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("📦 Step 2: Loading Whisper Large v3 Turbo...")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        whisper_model_id = "openai/whisper-large-v3-turbo"
        
        whisper_start = time.time()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"   Loading from: {whisper_model_id}")
        print(f"   Device: {device}, dtype: {torch_dtype}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(whisper_model_id)
        
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        whisper_load_time = time.time() - whisper_start
        print(f"✅ Whisper loaded in {whisper_load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Whisper loading error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Step 3: Run Round-Trip Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 Step 3: Running Round-Trip Tests...")
    print("=" * 70)
    
    results = []
    
    for speaker_name, speaker_path in speaker_refs.items():
        print(f"\n{'─' * 70}")
        print(f"🎤 Testing with speaker: {speaker_name}")
        print(f"   File: {speaker_path}")
        print(f"{'─' * 70}")
        
        if not Path(speaker_path).exists():
            print(f"   ⚠️ Speaker file not found, skipping...")
            continue
        
        for i, original_text in enumerate(test_texts):
            print(f"\n   📝 Test {i+1}: \"{original_text}\"")
            
            try:
                # Generate TTS audio
                tts_start = time.time()
                audio_path = xtts.text_to_speech(
                    text=original_text,
                    language="ar",
                    speaker_wav=speaker_path,
                    output_path=str(output_dir / f"test_{speaker_name.split()[0].lower()}_{i+1}.wav")
                )
                tts_time = time.time() - tts_start
                print(f"      🔊 TTS generated in {tts_time:.2f}s: {Path(audio_path).name}")
                
                # Get audio duration
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / rate
                print(f"      📊 Audio: {duration:.2f}s @ {rate}Hz")
                
                # Transcribe with Whisper
                stt_start = time.time()
                result = whisper_pipe(
                    audio_path,
                    generate_kwargs={
                        "language": "ar",
                        "task": "transcribe",
                    }
                )
                transcribed_text = result["text"].strip()
                stt_time = time.time() - stt_start
                print(f"      🎧 Whisper transcribed in {stt_time:.2f}s")
                
                # Calculate similarity
                similarity = calculate_similarity(original_text, transcribed_text)
                
                # Print results
                print(f"\n      ┌─ Original:    \"{original_text}\"")
                print(f"      └─ Transcribed: \"{transcribed_text}\"")
                print(f"      📊 Similarity: {similarity:.1%}")
                
                if similarity >= 0.8:
                    status = "✅ PASS"
                elif similarity >= 0.5:
                    status = "⚠️ PARTIAL"
                else:
                    status = "❌ FAIL"
                print(f"      {status}")
                
                results.append({
                    "speaker": speaker_name,
                    "original": original_text,
                    "transcribed": transcribed_text,
                    "similarity": similarity,
                    "tts_time": tts_time,
                    "stt_time": stt_time,
                    "audio_path": audio_path,
                    "status": status,
                })
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "speaker": speaker_name,
                    "original": original_text,
                    "transcribed": f"ERROR: {e}",
                    "similarity": 0.0,
                    "status": "❌ ERROR",
                })
    
    # =========================================================================
    # Step 4: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    # Group by speaker
    for speaker_name in speaker_refs.keys():
        speaker_results = [r for r in results if r.get("speaker") == speaker_name]
        if not speaker_results:
            continue
            
        avg_similarity = sum(r["similarity"] for r in speaker_results) / len(speaker_results)
        pass_count = sum(1 for r in speaker_results if "PASS" in r["status"])
        
        print(f"\n🎤 {speaker_name}:")
        print(f"   Average Similarity: {avg_similarity:.1%}")
        print(f"   Pass Rate: {pass_count}/{len(speaker_results)}")
        
        for r in speaker_results:
            print(f"   • \"{r['original'][:30]}...\" → {r['similarity']:.0%} {r['status']}")
    
    # Overall verdict
    print("\n" + "─" * 70)
    all_similarities = [r["similarity"] for r in results if r["similarity"] > 0]
    if all_similarities:
        overall_avg = sum(all_similarities) / len(all_similarities)
        print(f"📈 Overall Average Similarity: {overall_avg:.1%}")
        
        if overall_avg >= 0.8:
            print("🎉 VERDICT: XTTS Arabic pronunciation is GOOD!")
        elif overall_avg >= 0.5:
            print("⚠️ VERDICT: XTTS Arabic pronunciation needs improvement")
        else:
            print("❌ VERDICT: XTTS Arabic pronunciation is POOR")
    
    print(f"\n📁 Audio files saved to: {output_dir}")
    print("=" * 70)
    
    # Cleanup
    print("\n🧹 Cleaning up models...")
    xtts.unload_model()
    del model, processor, whisper_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Done!")
    
    return True


if __name__ == "__main__":
    success = test_xtts_whisper_roundtrip()
    sys.exit(0 if success else 1)
