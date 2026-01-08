"""
80 Hz Hum Detector with Hysteresis

Detects periodic low-frequency noise (buffer underrun artifacts or electrical hum)
by monitoring energy in 80/160/240 Hz bands relative to neighborhood bins.

Author: BeautyAI Framework
Date: November 13, 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque


class HumDetector:
    """
    Adaptive detector for 80 Hz fundamental and harmonics.
    
    Design:
    - Monitor 80/160/240 Hz bins in FFT
    - Compare to surrounding bins (±20 Hz) for relative threshold
    - Require sustained detection (dwell time) before triggering
    - Hysteresis for stable on/off transitions
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        fft_size: int = 512,
        hop_length: int = 160,
        fundamental_hz: float = 80.0,
        relative_threshold_db: float = 15.0,  # Peak must be 15 dB above neighborhood
        dwell_frames: int = 10,  # Require 10 frames (~200ms @ 20ms frames) before trigger
        hysteresis_db: float = 5.0,  # Once triggered, need 10 dB drop to turn off
    ):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            fft_size: FFT window size
            hop_length: Hop between frames
            fundamental_hz: Target fundamental frequency (80 Hz typical)
            relative_threshold_db: dB above local neighborhood to trigger
            dwell_frames: Consecutive frames required before detection
            hysteresis_db: dB hysteresis for off transition
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.fundamental_hz = fundamental_hz
        self.relative_threshold_db = relative_threshold_db
        self.dwell_frames = dwell_frames
        self.hysteresis_db = hysteresis_db
        
        # State
        self.is_hum_detected = False
        self._detection_buffer = deque(maxlen=dwell_frames)
        self._frame_buffer = []  # Accumulate frames for FFT
        
        # Frequency bin mapping
        self.freq_resolution = sample_rate / fft_size
        self._harmonic_bins = self._compute_harmonic_bins()
        
        # Statistics
        self.detection_count = 0
        self.last_peak_db: Optional[float] = None
        self.last_neighborhood_db: Optional[float] = None
    
    def _compute_harmonic_bins(self) -> list[Tuple[int, int]]:
        """
        Compute FFT bins for fundamental and harmonics with neighborhoods.
        
        Returns:
            List of (center_bin, neighborhood_start, neighborhood_end) tuples
        """
        harmonics = []
        
        for harmonic_num in [1, 2, 3]:  # 80, 160, 240 Hz
            target_freq = self.fundamental_hz * harmonic_num
            
            if target_freq > self.sample_rate / 2:
                break  # Above Nyquist
            
            center_bin = int(round(target_freq / self.freq_resolution))
            
            # Neighborhood: ±20 Hz around target (for relative comparison)
            neighbor_width_hz = 20.0
            neighbor_bins = int(round(neighbor_width_hz / self.freq_resolution))
            
            # Avoid overlap with center peak (±3 bins reserved for peak)
            neighbor_start = center_bin - neighbor_bins - 3
            neighbor_end = center_bin + neighbor_bins + 3
            
            harmonics.append((center_bin, neighbor_start, neighbor_end))
        
        return harmonics
    
    def process_frame(self, audio: np.ndarray) -> bool:
        """
        Process audio frame and update detection state.
        
        Args:
            audio: Audio frame (float32, mono)
        
        Returns:
            True if hum currently detected
        """
        # Accumulate frames
        self._frame_buffer.append(audio)
        
        # Need at least fft_size samples
        if len(self._frame_buffer) * len(audio) < self.fft_size:
            return self.is_hum_detected
        
        # Concatenate and window
        signal = np.concatenate(self._frame_buffer)
        
        # Slide window
        if len(signal) > self.fft_size:
            signal = signal[-self.fft_size:]
            self._frame_buffer = [signal[-len(audio):]]  # Keep last frame
        
        # FFT analysis
        windowed = signal * np.hanning(len(signal))
        spectrum = np.fft.rfft(windowed, n=self.fft_size)
        magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        
        # Check each harmonic
        peak_ratios = []
        
        for center_bin, neighbor_start, neighbor_end in self._harmonic_bins:
            if center_bin >= len(magnitude_db):
                continue
            
            # Peak energy (±1 bin for stability)
            peak_start = max(0, center_bin - 1)
            peak_end = min(len(magnitude_db), center_bin + 2)
            peak_db = np.max(magnitude_db[peak_start:peak_end])
            
            # Neighborhood energy (avoid peak region)
            left_start = max(0, neighbor_start)
            left_end = max(0, center_bin - 3)
            right_start = min(len(magnitude_db), center_bin + 3)
            right_end = min(len(magnitude_db), neighbor_end)
            
            neighborhood_bins = []
            if left_end > left_start:
                neighborhood_bins.extend(magnitude_db[left_start:left_end])
            if right_end > right_start:
                neighborhood_bins.extend(magnitude_db[right_start:right_end])
            
            if not neighborhood_bins:
                continue
            
            neighborhood_db = np.median(neighborhood_bins)
            ratio_db = peak_db - neighborhood_db
            peak_ratios.append(ratio_db)
            
            # Store for debugging
            self.last_peak_db = peak_db
            self.last_neighborhood_db = neighborhood_db
        
        if not peak_ratios:
            self._detection_buffer.append(False)
            return self.is_hum_detected
        
        # Detection logic with hysteresis
        max_ratio_db = max(peak_ratios)
        
        if self.is_hum_detected:
            # Require drop below (threshold - hysteresis) to turn off
            threshold = self.relative_threshold_db - self.hysteresis_db
            frame_detected = max_ratio_db > threshold
        else:
            # Require exceeding threshold to turn on
            frame_detected = max_ratio_db > self.relative_threshold_db
        
        self._detection_buffer.append(frame_detected)
        
        # Update state: require all frames in dwell buffer to be positive
        if all(self._detection_buffer):
            if not self.is_hum_detected:
                self.detection_count += 1
            self.is_hum_detected = True
        elif not any(self._detection_buffer):
            self.is_hum_detected = False
        
        return self.is_hum_detected
    
    def get_stats(self) -> Dict[str, any]:
        """Get current detector statistics"""
        return {
            "is_detected": bool(self.is_hum_detected),
            "detection_count": int(self.detection_count),
            "last_peak_db": float(self.last_peak_db) if self.last_peak_db is not None else None,
            "last_neighborhood_db": float(self.last_neighborhood_db) if self.last_neighborhood_db is not None else None,
            "last_ratio_db": float(self.last_peak_db - self.last_neighborhood_db) if (self.last_peak_db is not None and self.last_neighborhood_db is not None) else None,
            "dwell_buffer": [bool(x) for x in self._detection_buffer],
        }
    
    def reset(self):
        """Reset detector state"""
        self.is_hum_detected = False
        self._detection_buffer.clear()
        self._frame_buffer.clear()
        self.last_peak_db = None
        self.last_neighborhood_db = None
