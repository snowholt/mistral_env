"""Audio normalization helpers for WebRTC voice pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError("librosa is required for audio normalization") from exc

TARGET_SAMPLE_RATE = 16000


def to_float_mono_16k(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Normalize audio to float32 mono at 16 kHz.

    Args:
        audio: Input audio array. Can be 1-D mono or 2-D multi-channel data.
        sample_rate: Current sampling rate of the input audio.

    Returns:
        A tuple of (normalized_audio, sample_rate) where ``normalized_audio`` is
        float32 mono audio in the ``[-1.0, 1.0]`` range and ``sample_rate`` is
        the (possibly updated) sampling rate.
    """
    if audio.size == 0:
        return audio.astype(np.float32, copy=False), TARGET_SAMPLE_RATE if sample_rate != TARGET_SAMPLE_RATE else sample_rate

    # Ensure float32 dtype in [-1, 1]
    if audio.dtype == np.int16:
        normalized = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        normalized = audio
    else:
        normalized = audio.astype(np.float32)

    # Handle stereo/multi-channel by selecting LEFT channel explicitly
    # This avoids phase cancellation issues from averaging
    if normalized.ndim == 2:
        # Check if shape is (channels, samples) or (samples, channels)
        if normalized.shape[0] <= 2 and normalized.shape[1] > normalized.shape[0]:
            # Shape is (channels, samples) - select first channel (LEFT)
            mono = normalized[0, :]
        else:
            # Shape is (samples, channels) - select first channel (LEFT)
            mono = normalized[:, 0]
    else:
        mono = normalized

    # Resample to the canonical 16 kHz rate when needed.
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = librosa.resample(
            mono,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="polyphase",
        )
        sample_rate = TARGET_SAMPLE_RATE

    return mono.astype(np.float32, copy=False), sample_rate


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert normalized float audio into PCM 16-bit bytes."""
    if audio.size == 0:
        return b""

    float_audio = audio.astype(np.float32, copy=False)
    clipped = np.clip(float_audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def ensure_sample_rate(sample_rate: int | None, fallback: int = TARGET_SAMPLE_RATE) -> int:
    """Return a valid, positive sample rate, falling back to canonical default when missing."""
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return fallback
