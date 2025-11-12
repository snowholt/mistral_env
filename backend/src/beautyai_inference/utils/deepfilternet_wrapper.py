"""
DeepFilterNet Wrapper

Lightweight wrapper for DeepFilterNet noise suppression.
Alternative implementation using spectral processing until full model is available.

Author: BeautyAI Framework
Date: November 10, 2025
"""

import logging
from typing import Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class DeepFilterNetProcessor:
    """
    DeepFilterNet-inspired noise suppression processor.
    
    Uses perceptual filtering based on human auditory system characteristics.
    Designed for very low latency (5-10ms) and embedded devices.
    
    Key features:
    - Two-stage filtering (coarse + fine)
    - Perceptual weighting
    - Frame size: 480 samples @ 48kHz (10ms)
    - Works at both 16kHz and 48kHz
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize DeepFilterNet processor.
        
        Args:
            sample_rate: Audio sample rate (16000 or 48000)
        """
        if sample_rate not in [16000, 48000]:
            raise ValueError("DeepFilterNet only supports 16kHz or 48kHz")
        
        self.sample_rate = sample_rate
        self.frame_size = 480 if sample_rate == 48000 else 160  # 10ms frames
        
        # Perceptual filter bank (bark scale approximation)
        self.n_bands = 24
        self._init_filter_bank()
        
        # Noise tracking state
        self.noise_spectrum = None
        self.alpha_noise = 0.98  # Noise update rate (slower than EMA)
        
        logger.info(f"✅ DeepFilterNet processor initialized @ {sample_rate}Hz")
    
    def _init_filter_bank(self):
        """Initialize perceptual filter bank based on Bark scale."""
        # Bark scale critical bands (simplified)
        if self.sample_rate == 16000:
            self.band_edges = np.array([
                0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                4400, 5300, 6400, 7700, 8000
            ])
        else:  # 48kHz
            self.band_edges = np.array([
                0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                4400, 5300, 6400, 7700, 9500, 12000, 15200, 19500, 24000
            ])
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single audio frame through DeepFilterNet-inspired filtering.
        
        Args:
            audio_frame: Float32 audio samples @ configured sample rate
            
        Returns:
            Tuple of (denoised_audio, processing_time_ms)
        """
        import time
        start = time.time()
        
        # Two-stage filtering
        # Stage 1: Coarse spectral suppression
        coarse_filtered = self._coarse_filter(audio_frame)
        
        # Stage 2: Fine perceptual filtering
        fine_filtered = self._fine_filter(coarse_filtered)
        
        processing_time = (time.time() - start) * 1000
        return fine_filtered, processing_time
    
    def _coarse_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Stage 1: Coarse spectral noise suppression.
        Fast broad-spectrum noise reduction.
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Initialize or update noise spectrum estimate
        if self.noise_spectrum is None:
            self.noise_spectrum = magnitude.copy()
        else:
            # Update noise estimate (only during low-energy frames)
            frame_energy = np.mean(magnitude**2)
            if frame_energy < np.mean(self.noise_spectrum**2) * 2:
                self.noise_spectrum = (
                    self.alpha_noise * self.noise_spectrum +
                    (1 - self.alpha_noise) * magnitude
                )
        
        # Wiener-like filtering
        snr = magnitude / (self.noise_spectrum + 1e-10)
        gain = snr**2 / (snr**2 + 1)  # Wiener gain
        
        # Apply gain
        magnitude_clean = magnitude * gain
        
        # Reconstruct
        fft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = np.fft.irfft(fft_clean, n=len(audio))
        
        return audio_clean.astype(np.float32)
    
    def _fine_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Stage 2: Fine perceptual filtering.
        Applies perceptual weighting based on human hearing characteristics.
        """
        # Apply psychoacoustic masking
        # Boost frequencies critical for speech (1-4 kHz)
        
        # Design emphasis filter (boost 1-4 kHz, de-emphasis above/below)
        if len(audio) < 4:
            return audio
        
        # Simple bandpass emphasis for speech frequencies
        nyquist = self.sample_rate / 2
        
        # Boost speech range (1-4 kHz)
        if self.sample_rate >= 8000:
            # Design shelving filter to emphasize speech
            b, a = signal.butter(2, [1000 / nyquist, 4000 / nyquist], btype='band')
            emphasized = signal.filtfilt(b, a, audio)
            
            # Mix with original (50/50)
            audio_clean = 0.7 * audio + 0.3 * emphasized
        else:
            audio_clean = audio
        
        return audio_clean.astype(np.float32)
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream by splitting into frames.
        
        Args:
            audio: Float32 audio array
            
        Returns:
            Denoised audio array
        """
        if len(audio) < self.frame_size:
            denoised, _ = self.process_frame(audio)
            return denoised
        
        # Process frame by frame with overlap-add
        hop_size = self.frame_size // 2  # 50% overlap
        output = np.zeros(len(audio), dtype=np.float32)
        window = signal.hann(self.frame_size)
        
        for i in range(0, len(audio) - self.frame_size + 1, hop_size):
            frame = audio[i:i + self.frame_size]
            
            # Apply window
            windowed = frame * window
            
            # Process
            denoised, _ = self.process_frame(windowed)
            
            # Overlap-add
            output[i:i + self.frame_size] += denoised
        
        # Handle last incomplete frame
        remainder = len(audio) % hop_size
        if remainder > 0:
            last_frame = audio[-self.frame_size:]
            denoised, _ = self.process_frame(last_frame)
            output[-self.frame_size:] = denoised
        
        return output
