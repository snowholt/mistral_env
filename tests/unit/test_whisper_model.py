"""GPU smoke test for openai/whisper-large-v3-turbo."""

import asyncio
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

AUDIO_PATH = Path(__file__).resolve().parents[1] / "webrtc" / "q7.wav"
MODEL_ID = "openai/whisper-large-v3-turbo"


def _has_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for Whisper large-v3-turbo smoke test")
@pytest.mark.asyncio
async def test_whisper_large_v3_turbo_transcribes_sample() -> None:
    """Load the Whisper turbo model on GPU and transcribe the q7 sample."""

    assert AUDIO_PATH.exists(), f"Audio sample missing: {AUDIO_PATH}"

    device = torch.device("cuda:0")
    dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    waveform, sample_rate = _load_audio(AUDIO_PATH)

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(device=device, dtype=dtype)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    assert transcription, "Whisper transcription should not be empty"
    assert any(char.isalpha() for char in transcription), "Transcription must contain alphabetic characters"


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
