"""GPU smoke test and persistent engine validation for openai/whisper-large-v3-turbo.

Tests:
1. GPU availability and CUDA diagnostics
2. Direct model loading on GPU
3. Persistent Whisper engine from ModelManager
4. Performance benchmarking
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

AUDIO_PATH = Path(__file__).resolve().parents[1] / "webrtc" / "q7.wav"
MODEL_ID = "openai/whisper-large-v3-turbo"


def _has_gpu() -> bool:
    """Check if GPU is available and working."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() == 0:
        return False
    try:
        # Try to actually use the GPU
        torch.cuda.current_device()
        return True
    except Exception:
        return False


def _get_cuda_diagnostics() -> dict:
    """Get detailed CUDA diagnostics for debugging."""
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
            diagnostics["memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception as e:
            diagnostics["gpu_error"] = str(e)
    
    return diagnostics


@pytest.mark.asyncio
async def test_cuda_diagnostics() -> None:
    """Test CUDA availability and provide diagnostic information."""
    diagnostics = _get_cuda_diagnostics()
    
    print("\n" + "="*60)
    print("🔍 CUDA Diagnostics")
    print("="*60)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # This test always passes but provides diagnostic info
    assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for Whisper large-v3-turbo smoke test")
@pytest.mark.asyncio
async def test_whisper_large_v3_turbo_transcribes_sample() -> None:
    """Load the Whisper turbo model on GPU and transcribe the q7 sample."""

    assert AUDIO_PATH.exists(), f"Audio sample missing: {AUDIO_PATH}"
    
    print("\n" + "="*60)
    print("🚀 Direct GPU Model Loading Test")
    print("="*60)

    device = torch.device("cuda:0")
    dtype = torch.float16
    
    print(f"Loading model to {device} with dtype={dtype}...")
    load_start = time.time()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")

    waveform, sample_rate = _load_audio(AUDIO_PATH)
    print(f"📊 Audio: {len(waveform)} samples @ {sample_rate}Hz ({len(waveform)/sample_rate:.2f}s)")

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(device=device, dtype=dtype)

    print("🎤 Transcribing...")
    inference_start = time.time()
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    inference_time = time.time() - inference_start
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print(f"✅ Transcription: '{transcription}'")
    print(f"⚡ Inference time: {inference_time:.2f}s")
    print(f"📈 Real-time factor: {inference_time / (len(waveform)/sample_rate):.2f}x")
    print("="*60 + "\n")

    assert transcription, "Whisper transcription should not be empty"
    assert any(char.isalpha() for char in transcription), "Transcription must contain alphabetic characters"


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for persistent engine test")
@pytest.mark.asyncio
async def test_persistent_whisper_engine() -> None:
    """Test persistent Whisper engine from ModelManager."""
    
    print("\n" + "="*60)
    print("🔧 Persistent Whisper Engine Test")
    print("="*60)
    
    try:
        from beautyai_inference.core.model_manager import ModelManager
        from beautyai_inference.services.voice.transcription import WhisperLargeV3TurboEngine
        
        print("Initializing ModelManager...")
        manager = ModelManager()
        
        print("Getting persistent Whisper engine...")
        engine = manager.get_streaming_whisper()
        
        if engine is None:
            print("⚠️  No persistent engine found, creating new one...")
            engine = WhisperLargeV3TurboEngine()
            success = engine.load_whisper_model()
            assert success, "Failed to load Whisper model"
        else:
            print("✅ Found persistent Whisper engine")
        
        # Check if model is on GPU
        model_info = engine.get_model_info()
        print(f"\nEngine Info:")
        for key, value in model_info.items():
            if key not in ['model', 'processor', 'pipe']:
                print(f"  {key}: {value}")
        
        device = model_info.get('device', 'unknown')
        assert device.startswith('cuda'), f"Expected GPU device, got {device}"
        print(f"\n✅ Engine is on GPU: {device}")
        
        # Test transcription
        if AUDIO_PATH.exists():
            print(f"\n🎤 Testing transcription with {AUDIO_PATH.name}...")
            with open(AUDIO_PATH, 'rb') as f:
                audio_bytes = f.read()
            
            start_time = time.time()
            transcription = engine.transcribe_audio_bytes(audio_bytes, audio_format="wav", language="en")
            transcription_time = time.time() - start_time
            
            print(f"✅ Transcription: '{transcription}'")
            print(f"⚡ Time: {transcription_time:.2f}s")
            
            assert transcription, "Transcription should not be empty"
            assert transcription_time < 5.0, f"GPU transcription too slow: {transcription_time:.2f}s"
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@pytest.mark.asyncio  
async def test_fallback_to_cpu_if_no_gpu() -> None:
    """Test that Whisper falls back to CPU gracefully when GPU unavailable."""
    
    print("\n" + "="*60)
    print("💻 CPU Fallback Test")
    print("="*60)
    
    if _has_gpu():
        pytest.skip("GPU available, skipping CPU fallback test")
    
    try:
        from beautyai_inference.services.voice.transcription import WhisperLargeV3TurboEngine
        
        print("Creating Whisper engine (CPU mode)...")
        engine = WhisperLargeV3TurboEngine()
        
        print("Loading model...")
        success = engine.load_whisper_model()
        assert success, "Failed to load Whisper model on CPU"
        
        model_info = engine.get_model_info()
        device = model_info.get('device', 'unknown')
        print(f"✅ Model loaded on: {device}")
        assert device == 'cpu', f"Expected CPU device, got {device}"
        
        print("="*60 + "\n")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


def _load_audio(path: Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Load and resample waveform for the provided WAV file."""

    try:
        import torchaudio
        import torchaudio.functional as F

        waveform, rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if rate != target_rate:
            waveform = F.resample(waveform, rate, target_rate)
            rate = target_rate

        return waveform.squeeze(0).to(dtype=torch.float32).numpy(), rate

    except ImportError:
        import soundfile as sf

        data, rate = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if rate != target_rate:
            from scipy.signal import resample

            num_samples = int(len(data) * target_rate / rate)
            data = resample(data, num_samples)
            rate = target_rate

        return data.astype(np.float32, copy=False), rate
