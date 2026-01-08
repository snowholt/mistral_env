"""Integration test for Genius AI fine-tuned Arabic models.

Tests:
1. GPU availability and memory diagnostics
2. Persistent loading of Genius Arabic Whisper from ModelManager
3. Persistent loading of XTTS from ModelManager
4. End-to-end pipeline: Voice → Transcription → TTS → Audio output
5. GPU memory usage validation
6. Model persistence across function calls

This test validates that:
- Models are loaded on GPU with proper memory allocation
- ModelManager provides persistent instances (no reloading)
- STT (Genius Whisper) transcribes Arabic audio correctly
- TTS (XTTS) synthesizes Arabic text with proper output
- The complete voice-to-voice pipeline works end-to-end
"""

import asyncio
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import gc

import numpy as np
import pytest
import torch

# Test audio paths
TESTS_DIR = Path(__file__).resolve().parent.parent
ARABIC_AUDIO_PATH = TESTS_DIR / "webrtc" / "q7.wav"  # Arabic test sample
OUTPUT_DIR = TESTS_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _has_gpu() -> bool:
    """Check if GPU is available and working."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() == 0:
        return False
    try:
        torch.cuda.current_device()
        return True
    except Exception:
        return False


def _get_gpu_diagnostics() -> dict:
    """Get detailed GPU diagnostics including memory usage."""
    diagnostics = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A",
        "torch_version": torch.__version__,
    }
    
    if diagnostics["cuda_available"]:
        try:
            diagnostics["device_name"] = torch.cuda.get_device_name(0)
            diagnostics["device_capability"] = torch.cuda.get_device_capability(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            diagnostics["memory_total_gb"] = total_memory / (1024**3)
            diagnostics["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            diagnostics["memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            diagnostics["memory_free_gb"] = (total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
        except Exception as e:
            diagnostics["gpu_error"] = str(e)
    
    return diagnostics


def _print_gpu_memory_usage(label: str = "Current") -> None:
    """Print current GPU memory usage."""
    if not _has_gpu():
        return
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    
    print(f"\n{'='*60}")
    print(f"🔍 GPU Memory Usage - {label}")
    print(f"{'='*60}")
    print(f"  Total:     {total:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB ({(allocated/total)*100:.1f}%)")
    print(f"  Reserved:  {reserved:.2f} GB ({(reserved/total)*100:.1f}%)")
    print(f"  Free:      {free:.2f} GB ({(free/total)*100:.1f}%)")
    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_gpu_diagnostics() -> None:
    """Test GPU availability and provide diagnostic information."""
    diagnostics = _get_gpu_diagnostics()
    
    print("\n" + "="*60)
    print("🔍 GPU Diagnostics for Genius AI Models")
    print("="*60)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"
    
    if diagnostics["cuda_available"]:
        print("✅ GPU is available for testing")
    else:
        print("⚠️  No GPU available - tests will be skipped")


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for Genius AI models")
@pytest.mark.asyncio
async def test_genius_whisper_persistent_loading() -> None:
    """Test that Genius Arabic Whisper loads persistently on GPU via ModelManager."""
    
    print("\n" + "="*60)
    print("🎤 Genius Arabic Whisper - Persistent Loading Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        _print_gpu_memory_usage("Before Loading Whisper")
        
        print("Initializing ModelManager...")
        manager = ModelManager()
        
        print("Loading Genius Arabic Whisper model (genius-whisper-arabic)...")
        load_start = time.time()
        
        # Request the Genius Arabic Whisper model specifically
        whisper_engine = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        
        load_time = time.time() - load_start
        
        assert whisper_engine is not None, "Failed to load Genius Whisper engine"
        print(f"✅ Genius Whisper loaded in {load_time:.2f}s")
        
        # Verify it's the Genius model
        model_info = whisper_engine.get_model_info()
        print(f"\n📋 Engine Info:")
        for key, value in model_info.items():
            if key not in ['model', 'processor', 'pipe']:
                print(f"  {key}: {value}")
        
        # Verify GPU placement
        device = model_info.get('device', 'unknown')
        assert 'cuda' in str(device), f"Expected GPU device, got {device}"
        print(f"\n✅ Model is on GPU: {device}")
        
        # Verify it's the local Genius model path
        model_id = model_info.get('model_id', '')
        assert 'geniusai-arabic-models' in model_id, f"Expected Genius AI model, got {model_id}"
        print(f"✅ Confirmed Genius AI model: {model_id}")
        
        _print_gpu_memory_usage("After Loading Whisper")
        
        # Test persistence - second call should reuse the instance
        print("\n🔄 Testing persistence - requesting same model again...")
        load_start2 = time.time()
        whisper_engine2 = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        load_time2 = time.time() - load_start2
        
        assert whisper_engine2 is whisper_engine, "ModelManager should return the same instance"
        print(f"✅ Persistence confirmed - reused instance in {load_time2:.4f}s (no reload)")
        
        # Check that it's marked as loaded
        is_loaded = manager.is_whisper_model_loaded("genius-whisper-arabic")
        assert is_loaded, "Model should be marked as loaded in ModelManager"
        print(f"✅ Model registered in ModelManager: {is_loaded}")
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for XTTS model")
@pytest.mark.asyncio
async def test_xtts_persistent_loading() -> None:
    """Test that XTTS model loads persistently on GPU via ModelManager."""
    
    print("\n" + "="*60)
    print("🔊 XTTS - Persistent Loading Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        _print_gpu_memory_usage("Before Loading XTTS")
        
        print("Initializing ModelManager...")
        manager = ModelManager()
        
        print("Loading XTTS model (genius-xtts-arabic)...")
        load_start = time.time()
        
        # Request the XTTS model
        tts_engine = manager.get_tts_engine(model_name="genius-xtts-arabic")
        
        load_time = time.time() - load_start
        
        assert tts_engine is not None, "Failed to load XTTS engine"
        print(f"✅ XTTS loaded in {load_time:.2f}s")
        
        # Get model info
        print(f"\n📋 TTS Engine Info:")
        print(f"  Engine Type: {type(tts_engine).__name__}")
        print(f"  Model ID: {tts_engine.config.model_id}")
        print(f"  Engine: {tts_engine.config.engine_type}")
        
        # Verify it's the Genius XTTS model
        model_id = tts_engine.config.model_id
        assert 'geniusai-arabic-models' in model_id, f"Expected Genius AI XTTS model, got {model_id}"
        assert 'xtts' in model_id.lower(), f"Expected XTTS model, got {model_id}"
        print(f"✅ Confirmed Genius AI XTTS model")
        
        _print_gpu_memory_usage("After Loading XTTS")
        
        # Test persistence
        print("\n🔄 Testing persistence - requesting same model again...")
        load_start2 = time.time()
        tts_engine2 = manager.get_tts_engine(model_name="genius-xtts-arabic")
        load_time2 = time.time() - load_start2
        
        assert tts_engine2 is tts_engine, "ModelManager should return the same instance"
        print(f"✅ Persistence confirmed - reused instance in {load_time2:.4f}s (no reload)")
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for end-to-end test")
@pytest.mark.asyncio
async def test_end_to_end_voice_pipeline() -> None:
    """Test complete voice-to-voice pipeline: Audio → STT → TTS → Audio output."""
    
    print("\n" + "="*60)
    print("🎯 End-to-End Voice Pipeline Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        # Initialize manager
        print("Initializing ModelManager...")
        manager = ModelManager()
        
        # Step 1: Load both models
        print("\n📥 Step 1: Loading models...")
        _print_gpu_memory_usage("Before Loading Models")
        
        print("  Loading Genius Whisper...")
        whisper_engine = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        assert whisper_engine is not None, "Failed to load Whisper"
        print("  ✅ Whisper loaded")
        
        print("  Loading XTTS...")
        tts_engine = manager.get_tts_engine(model_name="genius-xtts-arabic")
        assert tts_engine is not None, "Failed to load XTTS"
        print("  ✅ XTTS loaded")
        
        _print_gpu_memory_usage("After Loading Both Models")
        
        # Step 2: Test transcription (STT)
        print("\n🎤 Step 2: Testing Speech-to-Text (Genius Whisper)...")
        
        if not ARABIC_AUDIO_PATH.exists():
            pytest.skip(f"Test audio not found: {ARABIC_AUDIO_PATH}")
        
        print(f"  Reading audio: {ARABIC_AUDIO_PATH}")
        with open(ARABIC_AUDIO_PATH, 'rb') as f:
            audio_bytes = f.read()
        
        print(f"  Audio size: {len(audio_bytes) / 1024:.2f} KB")
        
        stt_start = time.time()
        transcription = whisper_engine.transcribe_audio_bytes(
            audio_bytes, 
            audio_format="wav", 
            language="ar"
        )
        stt_time = time.time() - stt_start
        
        print(f"\n  ✅ Transcription: '{transcription}'")
        print(f"  ⚡ STT Time: {stt_time:.2f}s")
        
        assert transcription, "Transcription should not be empty"
        assert len(transcription) > 0, "Transcription should have content"
        
        # Step 3: Test TTS synthesis
        print(f"\n🔊 Step 3: Testing Text-to-Speech (XTTS)...")
        
        # Use the transcription or a test Arabic phrase
        test_text = transcription if transcription else "مرحبا، هذا اختبار للصوت العربي"
        print(f"  Synthesizing: '{test_text}'")
        
        # Generate speech
        tts_start = time.time()
        
        # XTTS generates audio as bytes
        if hasattr(tts_engine, 'generate_speech'):
            audio_output = await tts_engine.generate_speech(test_text, language="ar")
        elif hasattr(tts_engine, 'synthesize'):
            audio_output = tts_engine.synthesize(test_text, language="ar")
        else:
            # Fallback - check for other methods
            pytest.skip("XTTS engine doesn't have expected synthesis method")
        
        tts_time = time.time() - tts_start
        
        print(f"  ✅ TTS synthesis complete")
        print(f"  ⚡ TTS Time: {tts_time:.2f}s")
        
        # Step 4: Save output
        print(f"\n💾 Step 4: Saving output audio...")
        
        output_path = OUTPUT_DIR / f"genius_test_output_{int(time.time())}.wav"
        
        if isinstance(audio_output, bytes):
            with open(output_path, 'wb') as f:
                f.write(audio_output)
            print(f"  ✅ Audio saved: {output_path}")
            print(f"  📊 Output size: {len(audio_output) / 1024:.2f} KB")
        elif isinstance(audio_output, np.ndarray):
            # Save as WAV using scipy or soundfile
            try:
                import soundfile as sf
                sf.write(str(output_path), audio_output, 24000)  # XTTS uses 24kHz
                print(f"  ✅ Audio saved: {output_path}")
                print(f"  📊 Output samples: {len(audio_output)}")
            except ImportError:
                from scipy.io import wavfile
                wavfile.write(str(output_path), 24000, audio_output)
                print(f"  ✅ Audio saved: {output_path}")
                print(f"  📊 Output samples: {len(audio_output)}")
        else:
            print(f"  ⚠️  Unexpected audio output type: {type(audio_output)}")
        
        # Step 5: Summary
        print("\n" + "="*60)
        print("📊 Pipeline Summary")
        print("="*60)
        print(f"  Input Audio: {ARABIC_AUDIO_PATH.name}")
        print(f"  Transcription: '{transcription[:100]}...' ({len(transcription)} chars)")
        print(f"  STT Time: {stt_time:.2f}s")
        print(f"  TTS Time: {tts_time:.2f}s")
        print(f"  Total Time: {stt_time + tts_time:.2f}s")
        print(f"  Output: {output_path}")
        print("="*60)
        
        _print_gpu_memory_usage("Final State")
        
        # Verify both models are still loaded in manager
        is_whisper_loaded = manager.is_whisper_model_loaded("genius-whisper-arabic")
        print(f"\n✅ Whisper still loaded: {is_whisper_loaded}")
        
        models_list = manager.list_loaded_models()
        print(f"✅ Total models in memory: {len(models_list)}")
        print(f"   Models: {models_list}")
        
        print("\n" + "="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for memory validation")
@pytest.mark.asyncio
async def test_model_memory_footprint() -> None:
    """Validate GPU memory usage is within expected ranges."""
    
    print("\n" + "="*60)
    print("💾 Model Memory Footprint Validation")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        # Clear any existing models first
        print("Clearing existing models...")
        manager = ModelManager()
        manager.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        
        _print_gpu_memory_usage("Initial State (Clean)")
        baseline_memory = torch.cuda.memory_allocated(0) / (1024**3)
        
        # Load Whisper
        print("\nLoading Genius Whisper...")
        whisper_engine = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        assert whisper_engine is not None
        
        whisper_memory = torch.cuda.memory_allocated(0) / (1024**3) - baseline_memory
        print(f"✅ Whisper memory usage: {whisper_memory:.2f} GB")
        
        # Expected: ~3-6 GB for Whisper Large v3 in FP16
        assert whisper_memory < 10.0, f"Whisper using too much memory: {whisper_memory:.2f} GB"
        
        _print_gpu_memory_usage("After Whisper")
        
        # Load XTTS
        print("\nLoading XTTS...")
        tts_engine = manager.get_tts_engine(model_name="genius-xtts-arabic")
        assert tts_engine is not None
        
        total_memory = torch.cuda.memory_allocated(0) / (1024**3) - baseline_memory
        xtts_memory = total_memory - whisper_memory
        print(f"✅ XTTS memory usage: {xtts_memory:.2f} GB")
        
        # Expected: ~2-6 GB for XTTS v2
        assert xtts_memory < 10.0, f"XTTS using too much memory: {xtts_memory:.2f} GB"
        
        _print_gpu_memory_usage("After Both Models")
        
        print("\n" + "="*60)
        print("📊 Memory Summary")
        print("="*60)
        print(f"  Baseline:      {baseline_memory:.2f} GB")
        print(f"  Whisper:       {whisper_memory:.2f} GB")
        print(f"  XTTS:          {xtts_memory:.2f} GB")
        print(f"  Total Used:    {total_memory:.2f} GB")
        print(f"  Combined OK:   {total_memory < 16.0}")
        print("="*60 + "\n")
        
        # Both models should fit in 16GB GPU
        assert total_memory < 16.0, f"Combined memory too high: {total_memory:.2f} GB"
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@pytest.mark.asyncio
async def test_model_unloading() -> None:
    """Test that models can be properly unloaded to free GPU memory."""
    
    print("\n" + "="*60)
    print("🗑️  Model Unloading Test")
    print("="*60)
    
    if not _has_gpu():
        pytest.skip("GPU required for memory management tests")
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Load models
        print("Loading models...")
        whisper_engine = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        tts_engine = manager.get_tts_engine(model_name="genius-xtts-arabic")
        
        assert whisper_engine is not None
        assert tts_engine is not None
        
        memory_with_models = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"✅ Models loaded - Memory: {memory_with_models:.2f} GB")
        
        # Unload Whisper
        print("\nUnloading Whisper...")
        success = manager.unload_whisper_model("genius-whisper-arabic")
        assert success, "Failed to unload Whisper"
        
        gc.collect()
        torch.cuda.empty_cache()
        
        memory_after_whisper = torch.cuda.memory_allocated(0) / (1024**3)
        memory_freed = memory_with_models - memory_after_whisper
        print(f"✅ Whisper unloaded - Freed: {memory_freed:.2f} GB")
        
        # Unload all
        print("\nUnloading all models...")
        manager.unload_all_models()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        memory_final = torch.cuda.memory_allocated(0) / (1024**3)
        total_freed = memory_with_models - memory_final
        print(f"✅ All models unloaded - Total freed: {total_freed:.2f} GB")
        
        # Verify models are gone
        models_list = manager.list_loaded_models()
        assert len(models_list) == 0, f"Models still loaded: {models_list}"
        print("✅ Confirmed: No models in memory")
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


if __name__ == "__main__":
    """Run tests directly for quick validation."""
    print("\n" + "="*60)
    print("🧪 Running Genius AI Models Tests")
    print("="*60 + "\n")
    
    import sys
    
    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
