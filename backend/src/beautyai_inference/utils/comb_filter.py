"""
Comb Filter for Periodic Noise Removal

Implements a multi-notch (comb) filter to remove periodic impulse noise
by targeting the fundamental frequency and its harmonics.

Author: BeautyAI Framework
Date: November 12, 2025
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class CombFilter:
    """
    Comb filter for removing periodic noise (e.g., 80 Hz electrical interference).
    
    Creates a series of notch filters at the fundamental frequency and all harmonics
    up to the Nyquist frequency. Designed for real-time processing of audio streams.
    """
    
    def __init__(self, sample_rate: int, fundamental_freq: float, quality_factor: float = 30.0, max_harmonics: int = None):
        """
        Initialize comb filter.
        
        Args:
            sample_rate: Audio sample rate in Hz
            fundamental_freq: Fundamental frequency to remove (e.g., 80 Hz)
            quality_factor: Q factor for notch filters (higher = narrower notch, less speech distortion)
            max_harmonics: Maximum number of harmonics to filter (None = auto-calculate to Nyquist)
        """
        self.sample_rate = sample_rate
        self.fundamental_freq = fundamental_freq
        self.quality_factor = quality_factor
        
        # Calculate harmonics up to Nyquist frequency
        nyquist = sample_rate / 2
        if max_harmonics is None:
            max_harmonics = int(nyquist / fundamental_freq)
        
        self.harmonics = [fundamental_freq * (i + 1) for i in range(max_harmonics) 
                         if fundamental_freq * (i + 1) < nyquist]
        
        # Design notch filters for each harmonic
        self.filters = []
        for freq in self.harmonics:
            # Design IIR notch filter (returns b, a coefficients in older scipy)
            b, a = signal.iirnotch(freq, Q=quality_factor, fs=sample_rate)
            # Convert to second-order sections for numerical stability
            sos = signal.tf2sos(b, a)
            self.filters.append(sos)
        
        # Initialize filter states for real-time processing
        self.reset_states()
        
        logger.info(f"✅ Comb filter initialized: {fundamental_freq:.1f} Hz + {len(self.harmonics)-1} harmonics")
        logger.info(f"   Harmonics: {', '.join([f'{h:.0f}Hz' for h in self.harmonics[:10]])}...")
    
    def reset_states(self):
        """Reset filter states (call between different audio streams)."""
        # Initialize filter states for each second-order section
        self.zi = []
        for sos in self.filters:
            # sos is shape (n_sections, 6), need state for each section
            n_sections = sos.shape[0]
            zi = np.zeros((n_sections, 2))
            self.zi.append(zi)
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame with stateful filtering.
        
        Args:
            audio_frame: Float32 audio samples
            
        Returns:
            Filtered audio frame
        """
        filtered = audio_frame.copy().astype(np.float64)
        
        # Apply each notch filter sequentially
        for i, sos in enumerate(self.filters):
            filtered, self.zi[i] = signal.sosfilt(sos, filtered, zi=self.zi[i])
        
        return filtered.astype(np.float32)
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream (stateless, for batch processing).
        
        Args:
            audio: Float32 audio array
            
        Returns:
            Filtered audio array
        """
        filtered = audio.copy()
        
        # Apply each notch filter sequentially
        for sos in self.filters:
            filtered = signal.sosfilt(sos, filtered)
        
        return filtered.astype(np.float32)
    
    def get_frequency_response(self, num_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the frequency response of the combined comb filter.
        
        Returns:
            Tuple of (frequencies, magnitude_dB)
        """
        # Compute frequency response for each filter
        w = np.linspace(0, np.pi, num_points)
        
        # Start with unity response
        combined_response = np.ones(num_points, dtype=complex)
        
        for sos in self.filters:
            w_temp, h = signal.sosfreqz(sos, worN=w, fs=self.sample_rate)
            combined_response *= h
        
        # Convert to dB
        magnitude_db = 20 * np.log10(np.abs(combined_response) + 1e-10)
        frequencies = w * self.sample_rate / (2 * np.pi)
        
        return frequencies, magnitude_db


class AdaptiveCombFilter(CombFilter):
    """
    Adaptive comb filter that can adjust to varying fundamental frequencies.
    Useful when the periodic noise frequency drifts over time.
    """
    
    def __init__(self, sample_rate: int, initial_freq: float, quality_factor: float = 30.0, adaptation_rate: float = 0.01):
        """
        Initialize adaptive comb filter.
        
        Args:
            adaptation_rate: How quickly to adapt to frequency changes (0-1)
        """
        self.adaptation_rate = adaptation_rate
        super().__init__(sample_rate, initial_freq, quality_factor)
    
    def estimate_frequency(self, audio: np.ndarray) -> float:
        """
        Estimate the current fundamental frequency from audio.
        Uses autocorrelation to find the dominant periodicity.
        """
        # Compute autocorrelation
        audio_centered = audio - np.mean(audio)
        autocorr = np.correlate(audio_centered, audio_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Find the first peak after lag 0 (skip first 100 samples to avoid DC)
        min_lag = max(100, int(self.sample_rate / 200))  # Min 5ms period (200 Hz max)
        max_lag = int(self.sample_rate / 20)  # Max 50ms period (20 Hz min)
        
        search_range = autocorr[min_lag:max_lag]
        if len(search_range) > 0:
            peak_lag = np.argmax(search_range) + min_lag
            estimated_freq = self.sample_rate / peak_lag
            return estimated_freq
        
        return self.fundamental_freq  # Return current freq if estimation fails
    
    def adapt_frequency(self, new_freq: float):
        """
        Smoothly adapt the fundamental frequency.
        
        Args:
            new_freq: Newly estimated fundamental frequency
        """
        # Smooth adaptation using exponential moving average
        adapted_freq = self.adaptation_rate * new_freq + (1 - self.adaptation_rate) * self.fundamental_freq
        
        # Only update if change is significant (>1 Hz)
        if abs(adapted_freq - self.fundamental_freq) > 1.0:
            logger.info(f"🔄 Adapting comb filter: {self.fundamental_freq:.1f} Hz → {adapted_freq:.1f} Hz")
            self.__init__(self.sample_rate, adapted_freq, self.quality_factor, self.adaptation_rate)


def test_comb_filter():
    """Test comb filter with synthetic periodic noise."""
    import matplotlib.pyplot as plt
    
    # Generate test signal
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean speech-like signal (sum of sinusoids)
    clean_signal = (
        np.sin(2 * np.pi * 200 * t) +  # F1
        0.5 * np.sin(2 * np.pi * 600 * t) +  # F2
        0.3 * np.sin(2 * np.pi * 1200 * t)  # F3
    )
    
    # Add 80 Hz periodic impulse noise
    impulse_freq = 80
    impulse_samples = int(sample_rate / impulse_freq)
    noise = np.zeros_like(t)
    noise[::impulse_samples] = 0.5  # Click every 12.5ms
    
    # Combined noisy signal
    noisy_signal = clean_signal + noise
    
    # Apply comb filter
    comb = CombFilter(sample_rate, fundamental_freq=80.0, quality_factor=30.0)
    filtered_signal = comb.process_audio(noisy_signal)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time domain
    plot_duration = 0.1  # Show first 100ms
    plot_samples = int(sample_rate * plot_duration)
    axes[0].plot(t[:plot_samples], noisy_signal[:plot_samples], label='Noisy', alpha=0.7)
    axes[0].plot(t[:plot_samples], filtered_signal[:plot_samples], label='Filtered', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain
    from scipy.fft import fft, fftfreq
    freqs = fftfreq(len(noisy_signal), 1/sample_rate)[:len(noisy_signal)//2]
    noisy_fft = np.abs(fft(noisy_signal))[:len(noisy_signal)//2]
    filtered_fft = np.abs(fft(filtered_signal))[:len(noisy_signal)//2]
    
    axes[1].semilogy(freqs, noisy_fft, label='Noisy', alpha=0.7)
    axes[1].semilogy(freqs, filtered_fft, label='Filtered', alpha=0.7)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Domain Comparison')
    axes[1].set_xlim([0, 2000])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Filter frequency response
    freq_response, mag_db = comb.get_frequency_response()
    axes[2].plot(freq_response, mag_db)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_title('Comb Filter Frequency Response')
    axes[2].set_xlim([0, sample_rate//2])
    axes[2].set_ylim([-60, 5])
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB')
    
    plt.tight_layout()
    plt.savefig('/home/lumi/beautyai/reports/debug/analysis/comb_filter_test.png', dpi=150)
    print("✅ Comb filter test complete!")
    print(f"   Removed {len(comb.harmonics)} harmonics of 80 Hz")
    print(f"   Test plot saved to: reports/debug/analysis/comb_filter_test.png")


if __name__ == "__main__":
    test_comb_filter()
