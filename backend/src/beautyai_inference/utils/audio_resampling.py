"""
Audio resampling utilities for noise reduction layer comparison.
Handles conversions between 16kHz and 48kHz for RNNoise processing.

Author: Lumina Ashley
Date: November 10, 2025
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def resample_16khz_to_48khz(audio_16khz: np.ndarray) -> np.ndarray:
    """
    Upsample audio from 16kHz to 48kHz for RNNoise processing.
    Uses high-quality polyphase filtering.
    
    Args:
        audio_16khz: Float32 mono audio at 16kHz
    
    Returns:
        Float32 mono audio at 48kHz (3x length)
    """
    # Upsampling ratio: 48000 / 16000 = 3
    upsampling_factor = 3
    
    # Use scipy's resample_poly for high-quality resampling
    # This uses a polyphase FIR filter with Kaiser window
    audio_48khz = signal.resample_poly(
        audio_16khz,
        up=upsampling_factor,
        down=1,
        axis=0,
        window=('kaiser', 5.0)  # Kaiser window with beta=5.0 for good quality
    )
    
    # Ensure float32
    audio_48khz = audio_48khz.astype(np.float32)
    
    logger.debug(f"Upsampled {len(audio_16khz)} samples (16kHz) → {len(audio_48khz)} samples (48kHz)")
    
    return audio_48khz


def resample_48khz_to_16khz(audio_48khz: np.ndarray) -> np.ndarray:
    """
    Downsample audio from 48kHz to 16kHz after RNNoise processing.
    Uses high-quality polyphase filtering with anti-aliasing.
    
    Args:
        audio_48khz: Float32 mono audio at 48kHz
    
    Returns:
        Float32 mono audio at 16kHz (1/3 length)
    """
    # Downsampling ratio: 16000 / 48000 = 1/3
    downsampling_factor = 3
    
    # Use scipy's resample_poly with anti-aliasing filter
    audio_16khz = signal.resample_poly(
        audio_48khz,
        up=1,
        down=downsampling_factor,
        axis=0,
        window=('kaiser', 5.0)  # Kaiser window for anti-aliasing
    )
    
    # Ensure float32
    audio_16khz = audio_16khz.astype(np.float32)
    
    logger.debug(f"Downsampled {len(audio_48khz)} samples (48kHz) → {len(audio_16khz)} samples (16kHz)")
    
    return audio_16khz


def process_with_rnnoise_16khz_pipeline(
    audio_16khz: np.ndarray,
    rnnoise_processor
) -> tuple[np.ndarray, list[float]]:
    """
    Complete pipeline: 16kHz → 48kHz → RNNoise → 16kHz.
    
    This is the bridge function for using RNNoise with 16kHz audio
    (which is what Whisper expects).
    
    Args:
        audio_16khz: Float32 mono audio at 16kHz, range [-1, 1]
        rnnoise_processor: Instance of RNNoiseProcessor
    
    Returns:
        Tuple of (denoised_audio_16khz, vad_probabilities)
        - denoised_audio_16khz: Denoised audio at 16kHz
        - vad_probabilities: List of VAD probabilities from RNNoise
    """
    # Step 1: Upsample to 48kHz (required by RNNoise)
    audio_48khz = resample_16khz_to_48khz(audio_16khz)
    
    # Step 2: Process with RNNoise at 48kHz
    denoised_48khz, vad_probs = rnnoise_processor.process_audio(audio_48khz)
    
    # Step 3: Downsample back to 16kHz (for Whisper)
    denoised_16khz = resample_48khz_to_16khz(denoised_48khz)
    
    logger.debug(f"RNNoise pipeline: {len(audio_16khz)} → {len(audio_48khz)} → {len(denoised_16khz)} samples")
    
    return denoised_16khz, vad_probs


def calculate_resampling_latency(audio_length_seconds: float) -> dict[str, float]:
    """
    Estimate latency added by resampling operations.
    
    Args:
        audio_length_seconds: Length of audio in seconds
    
    Returns:
        Dictionary with latency estimates in milliseconds:
        - 'upsample_16_to_48': Upsampling latency
        - 'downsample_48_to_16': Downsampling latency
        - 'total_resampling': Total resampling overhead
    """
    # Empirical measurements (approximate):
    # scipy.signal.resample_poly is quite fast
    # Upsampling: ~0.5ms per second of audio
    # Downsampling: ~0.3ms per second of audio
    
    upsample_ms = audio_length_seconds * 0.5
    downsample_ms = audio_length_seconds * 0.3
    
    return {
        'upsample_16_to_48': upsample_ms,
        'downsample_48_to_16': downsample_ms,
        'total_resampling': upsample_ms + downsample_ms
    }
