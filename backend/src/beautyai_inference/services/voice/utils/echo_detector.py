"""
Echo Detection Utility for Full Duplex Voice Streaming

This module provides sophisticated echo detection capabilities using:
1. Cross-correlation between input/output audio streams
2. Spectral analysis for feedback detection
3. Adaptive threshold based on environment
4. Echo probability score calculation

Used by the echo suppression service to detect and prevent audio feedback
in full duplex voice streaming scenarios.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from dataclasses import dataclass
from scipy import signal
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class EchoMetrics:
    """Metrics for echo detection analysis."""
    correlation_score: float
    spectral_similarity: float
    echo_probability: float
    delay_samples: int
    confidence: float
    timestamp: float


class EchoDetector:
    """
    Advanced echo detection using cross-correlation and spectral analysis.
    
    This detector can identify echo/feedback by analyzing the correlation
    between microphone input and TTS output audio streams.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size_ms: int = 20,
        correlation_threshold: float = 0.3,
        spectral_threshold: float = 0.4,
        max_delay_ms: int = 500,
        adaptive_threshold: bool = True,
        history_size: int = 50
    ):
        """
        Initialize echo detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size_ms: Frame size in milliseconds
            correlation_threshold: Minimum correlation for echo detection
            spectral_threshold: Minimum spectral similarity for echo
            max_delay_ms: Maximum expected echo delay in milliseconds
            adaptive_threshold: Whether to use adaptive thresholding
            history_size: Number of historical frames to keep
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.correlation_threshold = correlation_threshold
        self.spectral_threshold = spectral_threshold
        self.max_delay_samples = int(sample_rate * max_delay_ms / 1000)
        self.adaptive_threshold = adaptive_threshold
        self.history_size = history_size
        
        # Audio buffers for correlation analysis
        self.mic_buffer = deque(maxlen=self.max_delay_samples * 2)
        self.tts_buffer = deque(maxlen=self.max_delay_samples * 2)
        
        # Historical metrics for adaptive thresholding
        self.correlation_history = deque(maxlen=history_size)
        self.spectral_history = deque(maxlen=history_size)
        
        # Adaptive threshold parameters
        self.background_correlation_mean = 0.0
        self.background_correlation_std = 0.1
        self.background_spectral_mean = 0.0
        self.background_spectral_std = 0.1
        
        # State tracking
        self.frames_processed = 0
        self.last_echo_time = 0.0
        
        logger.info(f"Echo detector initialized: sr={sample_rate}, frame={frame_size_ms}ms, "
                   f"thresh={correlation_threshold:.2f}")
    
    def process_audio_frames(
        self,
        mic_frame: np.ndarray,
        tts_frame: Optional[np.ndarray] = None
    ) -> EchoMetrics:
        """
        Process audio frames and detect echo.
        
        Args:
            mic_frame: Microphone audio frame (mono, float32)
            tts_frame: TTS output frame (mono, float32), optional
            
        Returns:
            EchoMetrics with detection results
        """
        current_time = time.time()
        
        # Add to buffers
        if len(mic_frame) > 0:
            self.mic_buffer.extend(mic_frame)
        
        if tts_frame is not None and len(tts_frame) > 0:
            self.tts_buffer.extend(tts_frame)
        
        # Need minimum data for analysis
        if len(self.mic_buffer) < self.frame_size or len(self.tts_buffer) < self.frame_size:
            return EchoMetrics(
                correlation_score=0.0,
                spectral_similarity=0.0,
                echo_probability=0.0,
                delay_samples=0,
                confidence=0.0,
                timestamp=current_time
            )
        
        # Extract recent frames for analysis
        mic_data = np.array(list(self.mic_buffer)[-self.max_delay_samples:])
        tts_data = np.array(list(self.tts_buffer)[-self.max_delay_samples:])
        
        # Perform echo detection analysis
        correlation_score, delay_samples = self._compute_cross_correlation(mic_data, tts_data)
        spectral_similarity = self._compute_spectral_similarity(mic_data, tts_data, delay_samples)
        
        # Update adaptive thresholds
        if self.adaptive_threshold:
            self._update_adaptive_thresholds(correlation_score, spectral_similarity)
        
        # Calculate echo probability
        echo_probability = self._calculate_echo_probability(correlation_score, spectral_similarity)
        confidence = self._calculate_confidence(correlation_score, spectral_similarity)
        
        # Update state
        self.frames_processed += 1
        if echo_probability > 0.5:
            self.last_echo_time = current_time
        
        metrics = EchoMetrics(
            correlation_score=correlation_score,
            spectral_similarity=spectral_similarity,
            echo_probability=echo_probability,
            delay_samples=delay_samples,
            confidence=confidence,
            timestamp=current_time
        )
        
        # Log high-confidence detections
        if echo_probability > 0.7 and confidence > 0.8:
            logger.warning(f"Strong echo detected: prob={echo_probability:.3f}, "
                         f"corr={correlation_score:.3f}, delay={delay_samples}")
        
        return metrics
    
    def _compute_cross_correlation(
        self,
        mic_data: np.ndarray,
        tts_data: np.ndarray
    ) -> Tuple[float, int]:
        """
        Compute cross-correlation between mic and TTS audio.
        
        Returns:
            Tuple of (max_correlation, delay_samples)
        """
        if len(mic_data) == 0 or len(tts_data) == 0:
            return 0.0, 0
        
        # Ensure data is normalized
        mic_norm = mic_data / (np.std(mic_data) + 1e-8)
        tts_norm = tts_data / (np.std(tts_data) + 1e-8)
        
        # Compute cross-correlation
        correlation = signal.correlate(mic_norm, tts_norm, mode='full')
        
        # Find peak correlation and delay
        max_idx = np.argmax(np.abs(correlation))
        max_correlation = correlation[max_idx]
        delay_samples = max_idx - len(tts_norm) + 1
        
        # Normalize correlation to [-1, 1]
        max_correlation = max_correlation / len(tts_norm)
        
        return abs(max_correlation), abs(delay_samples)
    
    def _compute_spectral_similarity(
        self,
        mic_data: np.ndarray,
        tts_data: np.ndarray,
        delay_samples: int
    ) -> float:
        """
        Compute spectral similarity between audio streams.
        
        Uses power spectral density comparison to detect echo.
        """
        if len(mic_data) < 128 or len(tts_data) < 128:
            return 0.0
        
        try:
            # Apply delay compensation
            if delay_samples > 0 and delay_samples < len(tts_data):
                tts_aligned = tts_data[:-delay_samples] if delay_samples > 0 else tts_data
                mic_aligned = mic_data[delay_samples:] if delay_samples > 0 else mic_data
            else:
                tts_aligned = tts_data
                mic_aligned = mic_data
            
            # Ensure same length
            min_len = min(len(mic_aligned), len(tts_aligned))
            if min_len < 64:
                return 0.0
                
            mic_aligned = mic_aligned[:min_len]
            tts_aligned = tts_aligned[:min_len]
            
            # Compute power spectral density
            f_mic, psd_mic = signal.welch(mic_aligned, self.sample_rate, nperseg=min(64, min_len//2))
            f_tts, psd_tts = signal.welch(tts_aligned, self.sample_rate, nperseg=min(64, min_len//2))
            
            # Normalize PSDs
            psd_mic = psd_mic / (np.sum(psd_mic) + 1e-8)
            psd_tts = psd_tts / (np.sum(psd_tts) + 1e-8)
            
            # Compute similarity (using cosine similarity)
            similarity = np.dot(psd_mic, psd_tts) / (
                np.sqrt(np.sum(psd_mic**2)) * np.sqrt(np.sum(psd_tts**2)) + 1e-8
            )
            
            return max(0.0, similarity)
        
        except Exception as e:
            logger.debug(f"Spectral similarity computation failed: {e}")
            return 0.0
    
    def _update_adaptive_thresholds(self, correlation: float, spectral: float):
        """Update adaptive thresholds based on background noise."""
        self.correlation_history.append(correlation)
        self.spectral_history.append(spectral)
        
        if len(self.correlation_history) >= 10:
            # Update background statistics (exclude outliers)
            corr_array = np.array(self.correlation_history)
            spec_array = np.array(self.spectral_history)
            
            # Use 25th percentile as background level
            self.background_correlation_mean = np.percentile(corr_array, 25)
            self.background_correlation_std = np.std(corr_array)
            self.background_spectral_mean = np.percentile(spec_array, 25)
            self.background_spectral_std = np.std(spec_array)
    
    def _calculate_echo_probability(self, correlation: float, spectral: float) -> float:
        """
        Calculate probability of echo based on correlation and spectral similarity.
        
        Returns value between 0.0 (no echo) and 1.0 (definite echo).
        """
        if self.adaptive_threshold:
            # Use adaptive thresholds based on background
            corr_threshold = self.background_correlation_mean + 2 * self.background_correlation_std
            spec_threshold = self.background_spectral_mean + 2 * self.background_spectral_std
        else:
            corr_threshold = self.correlation_threshold
            spec_threshold = self.spectral_threshold
        
        # Weighted combination of correlation and spectral similarity
        corr_score = max(0.0, (correlation - corr_threshold) / (1.0 - corr_threshold))
        spec_score = max(0.0, (spectral - spec_threshold) / (1.0 - spec_threshold))
        
        # Combine scores (correlation weighted higher)
        echo_probability = 0.7 * corr_score + 0.3 * spec_score
        
        return min(1.0, echo_probability)
    
    def _calculate_confidence(self, correlation: float, spectral: float) -> float:
        """Calculate confidence level of echo detection."""
        # High confidence when both metrics agree
        if correlation > 0.5 and spectral > 0.3:
            return min(1.0, (correlation + spectral) / 2.0)
        elif correlation < 0.2 and spectral < 0.2:
            return min(1.0, 1.0 - (correlation + spectral) / 2.0)
        else:
            return 0.5  # Medium confidence when metrics disagree
    
    def get_adaptive_thresholds(self) -> Tuple[float, float]:
        """Get current adaptive thresholds."""
        if self.adaptive_threshold:
            corr_thresh = self.background_correlation_mean + 2 * self.background_correlation_std
            spec_thresh = self.background_spectral_mean + 2 * self.background_spectral_std
        else:
            corr_thresh = self.correlation_threshold
            spec_thresh = self.spectral_threshold
        
        return corr_thresh, spec_thresh
    
    def reset_buffers(self):
        """Reset audio buffers (call when audio stream restarts)."""
        self.mic_buffer.clear()
        self.tts_buffer.clear()
        logger.debug("Echo detector buffers reset")
    
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        corr_thresh, spec_thresh = self.get_adaptive_thresholds()
        
        return {
            "frames_processed": self.frames_processed,
            "correlation_threshold": corr_thresh,
            "spectral_threshold": spec_thresh,
            "background_correlation_mean": self.background_correlation_mean,
            "background_correlation_std": self.background_correlation_std,
            "background_spectral_mean": self.background_spectral_mean,
            "background_spectral_std": self.background_spectral_std,
            "buffer_size_mic": len(self.mic_buffer),
            "buffer_size_tts": len(self.tts_buffer),
            "last_echo_time": self.last_echo_time,
            "adaptive_threshold_enabled": self.adaptive_threshold
        }


class SimpleEchoDetector:
    """
    Simplified echo detector for environments with limited processing power.
    
    Uses basic correlation analysis without spectral features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        correlation_threshold: float = 0.25,
        window_size_ms: int = 100
    ):
        self.sample_rate = sample_rate
        self.correlation_threshold = correlation_threshold
        self.window_size = int(sample_rate * window_size_ms / 1000)
        
        self.mic_window = deque(maxlen=self.window_size)
        self.tts_window = deque(maxlen=self.window_size)
    
    def detect_echo(self, mic_frame: np.ndarray, tts_frame: np.ndarray) -> float:
        """Simple echo detection returning correlation score."""
        if len(mic_frame) == 0 or len(tts_frame) == 0:
            return 0.0
        
        self.mic_window.extend(mic_frame)
        self.tts_window.extend(tts_frame)
        
        if len(self.mic_window) < 32 or len(self.tts_window) < 32:
            return 0.0
        
        # Simple normalized cross-correlation
        mic_data = np.array(self.mic_window)
        tts_data = np.array(self.tts_window)
        
        # Normalize
        mic_norm = mic_data / (np.std(mic_data) + 1e-8)
        tts_norm = tts_data / (np.std(tts_data) + 1e-8)
        
        # Correlation
        correlation = np.corrcoef(mic_norm, tts_norm)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0


# Factory function for easy initialization
def create_echo_detector(
    mode: str = "full",
    sample_rate: int = 16000,
    **kwargs
) -> EchoDetector:
    """
    Factory function to create echo detector instance.
    
    Args:
        mode: "full" for full-featured detector, "simple" for lightweight
        sample_rate: Audio sample rate
        **kwargs: Additional parameters for detector
        
    Returns:
        EchoDetector instance
    """
    if mode == "simple":
        return SimpleEchoDetector(sample_rate=sample_rate, **kwargs)
    else:
        return EchoDetector(sample_rate=sample_rate, **kwargs)