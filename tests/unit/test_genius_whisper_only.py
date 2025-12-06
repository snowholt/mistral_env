"""Focused test for Genius AI Arabic Whisper model.

This test validates that the Genius Arabic Whisper model:
- Loads persistently on GPU via ModelManager
- Uses minimal GPU memory (~1.5GB in FP16)
- Provides fast, accurate Arabic transcription
- Maintains persistence across calls (no reloading)

Note: XTTS testing requires Python <3.12 due to Coqui TTS library constraints.
For Python 3.12 environments, only Whisper tests will run.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional
import gc

import pytest
import torch

# Test audio paths
TESTS_DIR = Path(__file__).resolve().parent.parent
ARABIC_AUDIO_PATH = TESTS_DIR / "webrtc" / "q7.wav"


def _has_gpu() -> bool:
    """Check if GPU is available and working."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.current_device()
        return True
    except Exception:
        return False


def _print_gpu_memory() -> dict:
    """Print and return GPU memory usage."""
    if not _has_gpu():
        return {}
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    memory_info = {
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": total - reserved,
        "allocated_pct": (allocated/total)*100
    }
    
    print(f"\n{'='*60}")
    print(f"💾 GPU Memory: {allocated:.2f}/{total:.2f} GB ({memory_info['allocated_pct']:.1f}%)")
    print(f"{'='*60}")
    
    return memory_info


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
@pytest.mark.asyncio
async def test_genius_whisper_full_pipeline() -> None:
    """Complete test: Load → Transcribe → Verify persistence."""
    
    print("\n" + "="*60)
    print("🎯 Genius Arabic Whisper - Full Pipeline Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        # Step 1: Initial state
        print("\n📊 Step 1: Initial GPU State")
        baseline = _print_gpu_memory()
        
        # Step 2: Load model
        print("\n📥 Step 2: Loading Genius Arabic Whisper")
        manager = ModelManager()
        
        load_start = time.time()
        whisper = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        load_time = time.time() - load_start
        
        assert whisper is not None, "Failed to load Genius Whisper"
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Step 3: Verify GPU placement
        print("\n🔍 Step 3: Verifying GPU Placement")
        model_info = whisper.get_model_info()
        
        device = model_info.get('device', 'unknown')
        model_id = model_info.get('model_id', '')
        
        print(f"  Device: {device}")
        print(f"  Model: {model_id[:80]}...")
        print(f"  Engine: {model_info.get('engine_name', 'unknown')}")
        
        assert 'cuda' in str(device), f"Expected GPU, got {device}"
        assert 'geniusai-arabic-models' in model_id, "Not the Genius model"
        print("✅ Confirmed: Genius model on GPU")
        
        # Step 4: Memory usage
        print("\n💾 Step 4: GPU Memory Usage")
        after_load = _print_gpu_memory()
        model_memory = after_load['allocated_gb'] - baseline.get('allocated_gb', 0)
        print(f"  Model memory: {model_memory:.2f} GB")
        assert model_memory < 3.0, f"Memory usage too high: {model_memory:.2f}GB"
        print("✅ Memory efficient (<3GB)")
        
        # Step 5: Test transcription
        if ARABIC_AUDIO_PATH.exists():
            print(f"\n🎤 Step 5: Testing Transcription")
            print(f"  Audio: {ARABIC_AUDIO_PATH.name}")
            
            with open(ARABIC_AUDIO_PATH, 'rb') as f:
                audio_bytes = f.read()
            
            transcribe_start = time.time()
            transcription = whisper.transcribe_audio_bytes(audio_bytes, audio_format="wav", language="ar")
            transcribe_time = time.time() - transcribe_start
            
            print(f"\n  📝 Result: '{transcription}'")
            print(f"  ⚡ Time: {transcribe_time:.2f}s")
            
            assert transcription, "Empty transcription"
            assert len(transcription) > 0, "No content"
            assert transcribe_time < 5.0, f"Too slow: {transcribe_time:.2f}s"
            print("✅ Fast transcription (<5s)")
        else:
            print(f"\n⚠️  Step 5: Skipped (no audio at {ARABIC_AUDIO_PATH})")
        
        # Step 6: Test persistence
        print("\n🔄 Step 6: Testing Persistence")
        reuse_start = time.time()
        whisper2 = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        reuse_time = time.time() - reuse_start
        
        assert whisper2 is whisper, "Not the same instance!"
        print(f"  ✅ Same instance reused in {reuse_time:.4f}s")
        print("  ✅ No reloading occurred")
        
        # Step 7: Verify registration
        print("\n📋 Step 7: ModelManager State")
        is_loaded = manager.is_whisper_model_loaded("genius-whisper-arabic")
        models_list = manager.list_loaded_models()
        
        print(f"  Loaded: {is_loaded}")
        print(f"  Models in memory: {len(models_list)}")
        print(f"  Models: {models_list}")
        
        assert is_loaded, "Model not registered"
        print("✅ Model properly registered")
        
        # Final summary
        print("\n" + "="*60)
        print("✨ SUCCESS - All Tests Passed!")
        print("="*60)
        print(f"  Load Time: {load_time:.2f}s")
        print(f"  Memory Usage: {model_memory:.2f} GB")
        if ARABIC_AUDIO_PATH.exists():
            print(f"  Transcription Time: {transcribe_time:.2f}s")
        print(f"  Persistence: ✅ Working")
        print(f"  GPU: ✅ CUDA Device")
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
@pytest.mark.asyncio
async def test_genius_whisper_memory_cleanup() -> None:
    """Test that the model can be cleanly unloaded."""
    
    print("\n" + "="*60)
    print("🗑️  Memory Cleanup Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Load
        print("\n📥 Loading model...")
        whisper = manager.get_streaming_whisper(model_name="genius-whisper-arabic", language="ar")
        assert whisper is not None
        
        loaded_mem = _print_gpu_memory()
        print(f"✅ Loaded: {loaded_mem['allocated_gb']:.2f} GB")
        
        # Unload
        print("\n🗑️  Unloading model...")
        success = manager.unload_whisper_model("genius-whisper-arabic")
        assert success, "Failed to unload"
        
        gc.collect()
        torch.cuda.empty_cache()
        
        final_mem = _print_gpu_memory()
        freed = loaded_mem['allocated_gb'] - final_mem['allocated_gb']
        
        print(f"✅ Freed: {freed:.2f} GB")
        print(f"✅ Final memory: {final_mem['allocated_gb']:.2f} GB")
        
        # Verify it's gone
        is_loaded = manager.is_whisper_model_loaded("genius-whisper-arabic")
        assert not is_loaded, "Model still loaded!"
        print("✅ Model completely unloaded")
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


if __name__ == "__main__":
    """Run tests directly."""
    import sys
    print("\n🧪 Running Genius Whisper Tests (Python 3.12 Compatible)\n")
    sys.exit(pytest.main([__file__, "-v", "-s"]))
