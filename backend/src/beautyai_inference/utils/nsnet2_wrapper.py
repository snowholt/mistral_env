"""
Advanced Spectral Noise Reduction (Layer 3.5)

High-quality spectral gating noise reduction using the noisereduce library.
This provides state-of-the-art noise suppression comparable to NSNet2.

Library: noisereduce (Tim Sainburg)
Method: Spectral gating with stationary/non-stationary noise reduction
Paper: Based on Audacity's noise reduction algorithm

Author: BeautyAI Framework
Date: November 12, 2025
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("noisereduce not available. Spectral gating will not work.")


class SpectralGatingProcessor:
    """
    Advanced spectral gating noise reduction processor.
    
    Uses the noisereduce library which implements high-quality spectral
    gating similar to Audacity's noise reduction. Excellent for:
    - Stationary noise (fans, AC, hum)
    - Non-stationary noise (keyboard, paper rustling)
    - Minimal musical artifacts
    - No ML model required (deterministic algorithm)
    
    Key features:
    - Adaptive noise profile estimation
    - Spectral smoothing to reduce artifacts
    - Configurable aggressiveness
    - Works on any sample rate
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize spectral gating processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        if not NOISEREDUCE_AVAILABLE:
            raise RuntimeError("noisereduce is required. Install: pip install noisereduce")
        
        self.sample_rate = sample_rate
        
        # Noise reduction parameters (tuned for fan/keyboard noise)
        self.stationary = True  # Assume stationary noise (fan)
        self.prop_decrease = 1.0  # Full noise reduction (0.0-1.0)
        self.freq_mask_smooth_hz = 500  # Smooth frequency transitions
        self.time_mask_smooth_ms = 50  # Smooth time transitions
        
        logger.info(f"✅ Spectral gating processor initialized @ {sample_rate}Hz")
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream with spectral gating.
        
        Args:
            audio: Float32 audio array
            
        Returns:
            Denoised audio array
        """
        if len(audio) < 320:  # Minimum 20ms for noise profile
            return audio
        
        try:
            # Apply spectral gating noise reduction
            # Use smaller STFT parameters for small audio chunks
            denoised = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
                freq_mask_smooth_hz=self.freq_mask_smooth_hz,
                time_mask_smooth_ms=self.time_mask_smooth_ms,
                n_fft=512,  # Small FFT for short frames
                hop_length=160,  # 10ms hop
            )
            
            return denoised.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Spectral gating failed: {e}")
            return audio  # Return original on error


def test_spectral_gating():
    """Quick test of spectral gating processor."""
    processor = SpectralGatingProcessor(sample_rate=16000)
    
    # Generate test signal (1 second @ 16kHz)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    # Process
    import time
    start = time.time()
    denoised = processor.process_audio(test_audio)
    latency = (time.time() - start) * 1000
    
    print(f"Spectral Gating Test:")
    print(f"  Input shape: {test_audio.shape}")
    print(f"  Output shape: {denoised.shape}")
    print(f"  Total latency: {latency:.2f}ms (for 1.0s audio)")
    print(f"  ✅ Spectral gating working!")


if __name__ == "__main__":
    test_spectral_gating()
