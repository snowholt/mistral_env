"""
Transient Suppressor for Crackle Removal

Applies adaptive median filtering to detect and suppress broadband transient artifacts
(crackles, clicks, pops) at high sample rates BEFORE downsampling. This prevents
smearing of impulses by anti-aliasing filters.

Author: BeautyAI Framework
Date: November 12, 2025
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


class TransientSuppressor:
    """
    Adaptive median filter for transient suppression.
    
    Detects broadband energy spikes (crackles) by comparing frame energy to
    a running median. Detected transients are replaced with median-filtered values.
    Operates at 48kHz to catch artifacts before downsampling smears them.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        kernel_size: int = 3,
        threshold: float = 0.8,
        energy_window: int = 5,
        frame_size: int = 960
    ):
        """
        Initialize transient suppressor.
        
        Args:
            sample_rate: Audio sample rate in Hz (should be 48000)
            kernel_size: Median filter kernel size (3 or 5 samples)
            threshold: Energy spike detection threshold (0.8 = 80% above median)
            energy_window: Window size for running median energy calculation
            frame_size: Audio frame size in samples
        """
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.energy_window = energy_window
        self.frame_size = frame_size
        
        # Energy buffer for running median
        self.energy_buffer = []
        self.max_buffer_size = 100  # Keep last 100 frames for median calculation
        
        logger.info(
            f"✅ Transient Suppressor initialized: kernel={kernel_size}, "
            f"threshold={threshold}, sample_rate={sample_rate} Hz"
        )
    
    def reset(self):
        """Reset internal buffers (call between different audio streams)."""
        self.energy_buffer = []
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame with transient detection and suppression.
        
        Args:
            audio_frame: Float32 audio samples
            
        Returns:
            Audio with transients suppressed
        """
        # Calculate frame RMS energy
        frame_energy = np.sqrt(np.mean(audio_frame ** 2))
        self.energy_buffer.append(frame_energy)
        
        # Keep buffer size manageable
        if len(self.energy_buffer) > self.max_buffer_size:
            self.energy_buffer.pop(0)
        
        # Need minimum buffer for median calculation
        if len(self.energy_buffer) < self.energy_window:
            # Not enough data yet, pass through
            return audio_frame
        
        # Calculate running median energy
        recent_energies = self.energy_buffer[-self.energy_window:]
        median_energy = np.median(recent_energies)
        
        # Detect transient spike
        energy_ratio = frame_energy / (median_energy + 1e-10)
        
        if energy_ratio > (1.0 + self.threshold):
            # Transient detected, apply median filtering
            # Use scipy median_filter for sample-level suppression
            filtered = median_filter(audio_frame, size=self.kernel_size, mode='reflect')
            return filtered.astype(np.float32)
        else:
            # No transient, pass through
            return audio_frame
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream (batch processing).
        
        Args:
            audio: Float32 audio array
            
        Returns:
            Audio with transients suppressed
        """
        # First pass: detect transient locations using energy statistics
        frame_energies = []
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)), mode='constant')
            frame_energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(frame_energy)
        
        # Calculate global median energy
        median_energy = np.median(frame_energies)
        
        # Second pass: suppress transients
        output = audio.copy()
        transient_count = 0
        
        for i, (start, energy) in enumerate(zip(range(0, len(audio), self.frame_size), frame_energies)):
            end = min(start + self.frame_size, len(audio))
            frame = audio[start:end]
            
            # Check for transient
            energy_ratio = energy / (median_energy + 1e-10)
            
            if energy_ratio > (1.0 + self.threshold):
                # Apply median filtering to suppress transient
                filtered = median_filter(frame, size=self.kernel_size, mode='reflect')
                output[start:end] = filtered
                transient_count += 1
        
        logger.info(f"🎯 Transient Suppressor: {transient_count}/{len(frame_energies)} frames filtered")
        
        return output.astype(np.float32)
    
    def get_recent_energy_stats(self) -> Optional[dict]:
        """Get recent energy statistics for monitoring."""
        if len(self.energy_buffer) < self.energy_window:
            return None
        
        recent = self.energy_buffer[-self.energy_window:]
        return {
            'median': float(np.median(recent)),
            'mean': float(np.mean(recent)),
            'max': float(np.max(recent)),
            'std': float(np.std(recent))
        }


class AdvancedTransientSuppressor(TransientSuppressor):
    """
    Enhanced transient suppressor with spectral analysis.
    
    Uses both time-domain energy and spectral flatness to better distinguish
    between speech transients (plosives) and noise artifacts (crackles).
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        kernel_size: int = 3,
        threshold: float = 0.8,
        spectral_threshold: float = 0.7,
        energy_window: int = 5,
        frame_size: int = 960
    ):
        """
        Initialize advanced transient suppressor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            kernel_size: Median filter kernel size
            threshold: Energy spike detection threshold
            spectral_threshold: Spectral flatness threshold (higher = more broadband)
            energy_window: Window size for running median
            frame_size: Audio frame size
        """
        super().__init__(sample_rate, kernel_size, threshold, energy_window, frame_size)
        self.spectral_threshold = spectral_threshold
        
        logger.info(
            f"✅ Advanced Transient Suppressor initialized: "
            f"spectral_threshold={spectral_threshold}"
        )
    
    def calculate_spectral_flatness(self, audio_frame: np.ndarray) -> float:
        """
        Calculate spectral flatness measure.
        
        Spectral flatness close to 1.0 indicates broadband noise (crackles),
        while lower values indicate tonal or speech content.
        
        Args:
            audio_frame: Audio samples
            
        Returns:
            Spectral flatness (0.0 to 1.0)
        """
        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(audio_frame))
        spectrum = spectrum + 1e-10  # Avoid log(0)
        
        # Spectral flatness = geometric_mean / arithmetic_mean
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        
        flatness = geometric_mean / arithmetic_mean
        return float(flatness)
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process frame with energy AND spectral analysis.
        
        Args:
            audio_frame: Float32 audio samples
            
        Returns:
            Audio with transients suppressed
        """
        # Calculate frame RMS energy
        frame_energy = np.sqrt(np.mean(audio_frame ** 2))
        self.energy_buffer.append(frame_energy)
        
        # Keep buffer size manageable
        if len(self.energy_buffer) > self.max_buffer_size:
            self.energy_buffer.pop(0)
        
        # Need minimum buffer
        if len(self.energy_buffer) < self.energy_window:
            return audio_frame
        
        # Calculate running median energy
        recent_energies = self.energy_buffer[-self.energy_window:]
        median_energy = np.median(recent_energies)
        
        # Energy-based detection
        energy_ratio = frame_energy / (median_energy + 1e-10)
        is_energy_spike = energy_ratio > (1.0 + self.threshold)
        
        if is_energy_spike:
            # Check spectral flatness to distinguish crackle from speech plosive
            flatness = self.calculate_spectral_flatness(audio_frame)
            
            if flatness > self.spectral_threshold:
                # High flatness = broadband noise = crackle
                filtered = median_filter(audio_frame, size=self.kernel_size, mode='reflect')
                return filtered.astype(np.float32)
            else:
                # Low flatness = tonal/speech = preserve
                return audio_frame
        else:
            # No spike detected
            return audio_frame


def test_transient_suppressor():
    """Test transient suppressor with synthetic crackles."""
    import matplotlib.pyplot as plt
    
    # Create test signal: speech + random crackles
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Speech-like signal: 200 Hz fundamental + 800 Hz formant
    speech = 0.1 * (np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 800 * t))
    
    # Add 50 random crackle impulses
    crackles = np.zeros_like(t)
    num_crackles = 50
    crackle_positions = np.random.randint(0, len(t), size=num_crackles)
    crackles[crackle_positions] = np.random.randn(num_crackles) * 0.5  # Large amplitude spikes
    
    # Combined signal
    audio = speech + crackles
    
    # Process with transient suppressor
    suppressor = TransientSuppressor(
        sample_rate=sample_rate,
        kernel_size=5,
        threshold=0.8,
        energy_window=5
    )
    
    cleaned_audio = suppressor.process_audio(audio)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time-domain plots (zoom to first 0.1s for clarity)
    zoom_samples = int(0.1 * sample_rate)
    t_zoom = t[:zoom_samples]
    
    axes[0].plot(t_zoom, audio[:zoom_samples])
    axes[0].set_title('Input: Speech + Crackles (Zoomed to 0.1s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_zoom, cleaned_audio[:zoom_samples])
    axes[1].set_title('Output: After Transient Suppression')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Difference signal
    difference = audio - cleaned_audio
    axes[2].plot(t_zoom, difference[:zoom_samples], color='red')
    axes[2].set_title('Removed Transients (Difference)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/transient_suppressor_test.png', dpi=150)
    print(f"✅ Test plot saved to /tmp/transient_suppressor_test.png")
    plt.close()


if __name__ == "__main__":
    test_transient_suppressor()
