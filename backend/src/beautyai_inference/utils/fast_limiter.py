"""
Fast Peak Limiter for Transient Suppression

Removes broadband impulses (crackles) with fast attack and moderate release.
Designed for 48kHz pre-downsampling to prevent smearing into speech envelopes.

Author: BeautyAI Framework
Date: November 13, 2025
"""

import numpy as np
from typing import Optional


class FastPeakLimiter:
    """
    Fast peak limiter with envelope follower.
    
    Design:
    - Fast attack (<1ms) to catch transients immediately
    - Moderate release (~20ms) to avoid pumping
    - Threshold relative to recent RMS (adaptive)
    - Zero-latency lookahead not needed (prefer speed)
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        threshold_db: float = -6.0,  # dB relative to recent RMS
        attack_ms: float = 0.5,  # Very fast attack
        release_ms: float = 20.0,  # Moderate release
        rms_window_ms: float = 100.0,  # RMS estimation window
    ):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            threshold_db: Threshold above recent RMS (dB)
            attack_ms: Attack time (milliseconds)
            release_ms: Release time (milliseconds)
            rms_window_ms: Window for RMS estimation (milliseconds)
        """
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.threshold_linear = 10 ** (threshold_db / 20.0)
        
        # Time constants
        self.attack_coeff = self._ms_to_coeff(attack_ms)
        self.release_coeff = self._ms_to_coeff(release_ms)
        
        # State
        self.envelope = 0.0
        self.rms_estimate = 0.01  # Initial RMS estimate
        
        # RMS window
        rms_window_samples = int(rms_window_ms * sample_rate / 1000)
        self.rms_window = np.zeros(rms_window_samples)
        self.rms_index = 0
        
        # Statistics
        self.gain_reduction_count = 0
        self.max_gain_reduction_db = 0.0
    
    def _ms_to_coeff(self, time_ms: float) -> float:
        """Convert time constant (ms) to filter coefficient"""
        if time_ms <= 0:
            return 0.0
        samples = time_ms * self.sample_rate / 1000.0
        return np.exp(-1.0 / samples)
    
    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio frame with peak limiting.
        
        Args:
            audio: Audio frame (float32, mono, range -1.0 to 1.0)
        
        Returns:
            Limited audio frame
        """
        output = np.zeros_like(audio)
        
        # Update RMS estimate
        frame_power = np.mean(audio ** 2)
        self.rms_window[self.rms_index] = frame_power
        self.rms_index = (self.rms_index + 1) % len(self.rms_window)
        self.rms_estimate = np.sqrt(np.mean(self.rms_window))
        
        # Adaptive threshold based on recent RMS
        adaptive_threshold = self.rms_estimate * self.threshold_linear
        
        # Sample-by-sample envelope following
        for i in range(len(audio)):
            sample_abs = abs(audio[i])
            
            # Envelope follower with attack/release
            if sample_abs > self.envelope:
                # Attack
                self.envelope = self.attack_coeff * self.envelope + (1 - self.attack_coeff) * sample_abs
            else:
                # Release
                self.envelope = self.release_coeff * self.envelope + (1 - self.release_coeff) * sample_abs
            
            # Compute gain
            if self.envelope > adaptive_threshold:
                # Apply gain reduction
                gain = adaptive_threshold / (self.envelope + 1e-10)
                self.gain_reduction_count += 1
                
                # Track max reduction
                gain_reduction_db = 20 * np.log10(gain + 1e-10)
                self.max_gain_reduction_db = min(self.max_gain_reduction_db, gain_reduction_db)
            else:
                gain = 1.0
            
            output[i] = audio[i] * gain
        
        return output
    
    def get_stats(self) -> dict:
        """Get limiter statistics"""
        return {
            "rms_estimate": float(self.rms_estimate),
            "current_envelope": float(self.envelope),
            "adaptive_threshold": float(self.rms_estimate * self.threshold_linear),
            "gain_reduction_count": int(self.gain_reduction_count),
            "max_gain_reduction_db": float(self.max_gain_reduction_db),
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.gain_reduction_count = 0
        self.max_gain_reduction_db = 0.0
