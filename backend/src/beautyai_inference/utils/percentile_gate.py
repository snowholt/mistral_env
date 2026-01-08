"""
Percentile-Based Adaptive Noise Gate

Implements a more robust noise gate using rolling percentile-based thresholding
instead of simple EMA multiplication. This adapts to the true noise floor and
includes hysteresis to prevent gate chatter.

Author: BeautyAI Framework
Date: November 12, 2025
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PercentileNoiseGate:
    """
    Percentile-based adaptive noise gate with hysteresis.
    
    Unlike simple EMA gates that can be biased by speech energy, this gate
    calculates the noise floor as the Nth percentile of a rolling energy window.
    Separate open/close thresholds prevent gate chatter.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: int = 200,
        percentile: float = 10.0,
        threshold_multiplier: float = 1.5,
        open_threshold_db: float = -45.0,
        close_threshold_db: float = -50.0,
        attack_ms: int = 5,
        release_ms: int = 50,
        frame_size: int = 320
    ):
        """
        Initialize percentile-based noise gate.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_ms: Rolling window duration for percentile calculation
            percentile: Energy percentile for noise floor (e.g., 10.0 = 10th percentile)
            threshold_multiplier: Gate closes at (percentile_energy * multiplier)
            open_threshold_db: dB level to open gate (higher = less sensitive)
            close_threshold_db: dB level to close gate (lower = holds open longer)
            attack_ms: Gate opening time in milliseconds
            release_ms: Gate closing time in milliseconds
            frame_size: Audio frame size in samples
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.percentile = percentile
        self.threshold_multiplier = threshold_multiplier
        
        # Hysteresis thresholds
        self.open_threshold_db = open_threshold_db
        self.close_threshold_db = close_threshold_db
        
        # Rolling buffer for percentile calculation
        self.window_size = int(window_ms * sample_rate / 1000 / frame_size)
        self.energy_buffer = deque(maxlen=self.window_size)
        
        # Gate state and smoothing coefficients
        self.gate_state = 1.0  # 0.0 = closed, 1.0 = open
        self.attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * sample_rate / 1000 / frame_size))
        self.release_coeff = 1.0 - np.exp(-1.0 / (release_ms * sample_rate / 1000 / frame_size))
        
        logger.info(
            f"✅ Percentile Gate initialized: {percentile}th percentile, "
            f"window={window_ms}ms, hysteresis=[{close_threshold_db}, {open_threshold_db}] dB"
        )
    
    def reset(self):
        """Reset gate state and buffers (call between different audio streams)."""
        self.energy_buffer.clear()
        self.gate_state = 1.0
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame with adaptive gating.
        
        Args:
            audio_frame: Float32 audio samples
            
        Returns:
            Gated audio frame
        """
        # Calculate frame RMS energy
        frame_energy = np.mean(audio_frame ** 2)
        self.energy_buffer.append(frame_energy)
        
        # Need minimum buffer for percentile calculation
        if len(self.energy_buffer) < min(10, self.window_size // 2):
            # Not enough data yet, pass through
            return audio_frame
        
        # Calculate noise floor as Nth percentile of rolling window
        noise_floor = np.percentile(list(self.energy_buffer), self.percentile)
        adaptive_threshold = noise_floor * self.threshold_multiplier
        
        # Convert current frame energy to dB for hysteresis comparison
        frame_db = 10 * np.log10(frame_energy + 1e-10)
        
        # Gate decision with hysteresis
        if self.gate_state < 0.5:  # Currently closed
            if frame_db > self.open_threshold_db:
                target_gate = 1.0  # Open gate
            else:
                target_gate = 0.0  # Stay closed
        else:  # Currently open
            if frame_db < self.close_threshold_db:
                target_gate = 0.0  # Close gate
            else:
                target_gate = 1.0  # Stay open
        
        # Smooth gate transition (attack or release)
        if target_gate > self.gate_state:
            alpha = self.attack_coeff  # Fast attack
        else:
            alpha = self.release_coeff  # Slow release
        
        self.gate_state += alpha * (target_gate - self.gate_state)
        
        # Apply gate envelope
        return audio_frame * self.gate_state
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream (stateless batch processing).
        
        Args:
            audio: Float32 audio array
            
        Returns:
            Gated audio
        """
        output = []
        
        # Process in frames
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            
            # Pad last frame if needed
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)), mode='constant')
            
            processed_frame = self.process_frame(frame)
            output.append(processed_frame)
        
        # Concatenate and trim to original length
        result = np.concatenate(output)[:len(audio)]
        return result.astype(np.float32)
    
    def get_gate_state(self) -> float:
        """Get current gate state (0.0 = closed, 1.0 = open)."""
        return self.gate_state
    
    def get_noise_floor_db(self) -> Optional[float]:
        """Get current estimated noise floor in dB."""
        if len(self.energy_buffer) < 10:
            return None
        
        noise_floor = np.percentile(list(self.energy_buffer), self.percentile)
        return 10 * np.log10(noise_floor + 1e-10)


def test_percentile_gate():
    """Test percentile gate with synthetic signal."""
    import matplotlib.pyplot as plt
    
    # Create test signal: speech burst + silence + noise
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Speech segment (0.5-1.5s): 440 Hz tone at -20 dB
    speech = np.zeros_like(t)
    speech_mask = (t >= 0.5) & (t <= 1.5)
    speech[speech_mask] = 0.1 * np.sin(2 * np.pi * 440 * t[speech_mask])
    
    # Background noise: -50 dB throughout
    noise = 0.003 * np.random.randn(len(t))
    
    # Combined signal
    audio = speech + noise
    
    # Process with percentile gate
    gate = PercentileNoiseGate(
        sample_rate=sample_rate,
        window_ms=200,
        percentile=10.0,
        threshold_multiplier=1.5,
        open_threshold_db=-45.0,
        close_threshold_db=-50.0
    )
    
    gated_audio = gate.process_audio(audio)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(t, audio)
    axes[0].set_title('Input: Speech + Noise')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, gated_audio)
    axes[1].set_title('Output: After Percentile Gate')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Gate state over time
    gate_states = []
    gate.reset()
    for i in range(0, len(audio), 320):
        frame = audio[i:i + 320]
        if len(frame) < 320:
            frame = np.pad(frame, (0, 320 - len(frame)), mode='constant')
        gate.process_frame(frame)
        gate_states.append(gate.get_gate_state())
    
    t_gate = np.linspace(0, duration, len(gate_states))
    axes[2].plot(t_gate, gate_states)
    axes[2].set_title('Gate State (0=Closed, 1=Open)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Gate State')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/percentile_gate_test.png', dpi=150)
    print(f"✅ Test plot saved to /tmp/percentile_gate_test.png")
    plt.close()


if __name__ == "__main__":
    test_percentile_gate()
