"""
Echo Suppression Service for Full Duplex Voice Communication.

Provides application-layer echo control and barge-in detection to prevent
TTS output from leaking back into the microphone input stream.

Uses the advanced EchoDetector utility for sophisticated correlation analysis.
"""

import asyncio
import logging
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from collections import deque

from .utils.echo_detector import EchoDetector, create_echo_detector, EchoMetrics as DetectorMetrics

logger = logging.getLogger(__name__)


class EchoState(Enum):
    """Echo suppression state machine states."""
    IDLE = "idle"
    TTS_PLAYING = "tts_playing" 
    USER_SPEAKING = "user_speaking"
    BARGE_IN = "barge_in"


@dataclass
class EchoConfig:
    """Configuration for echo suppression."""
    # VAD thresholds
    energy_threshold: float = 0.01  # Voice activity detection threshold
    zero_crossing_rate_threshold: float = 0.1  # Additional VAD criterion
    min_speech_duration_ms: int = 300  # Minimum speech duration to trigger barge-in
    
    # Echo suppression
    tts_duck_db: float = -12.0  # Duck TTS volume during barge-in (dB)
    tts_pause_threshold_ms: int = 500  # Pause TTS if user speaks this long
    resume_delay_ms: int = 500  # Wait before resuming TTS after end-of-speech
    
    # Gating
    gate_mic_during_tts: bool = True  # Gate mic input during TTS playback
    correlation_threshold: float = 0.3  # Echo correlation threshold
    
    # Buffers
    analysis_window_ms: int = 100  # Analysis window for VAD and echo detection
    history_buffer_ms: int = 2000  # History buffer for correlation analysis


@dataclass 
class EchoMetrics:
    """Metrics for echo suppression performance."""
    frames_processed: int = 0
    gated_frames: int = 0
    vad_active_frames: int = 0
    barge_in_events: int = 0
    tts_duck_events: int = 0
    echo_correlation_mean: float = 0.0
    echo_correlation_max: float = 0.0
    recent_correlations: deque = field(default_factory=lambda: deque(maxlen=100))
    session_start_time: float = field(default_factory=time.time)
    
    # Enhanced metrics using EchoDetector
    detector_metrics: Optional[DetectorMetrics] = None
    echo_probability_mean: float = 0.0
    spectral_similarity_mean: float = 0.0
    confidence_mean: float = 0.0
    """Metrics for echo suppression performance."""
    state_changes: int = 0
    barge_in_count: int = 0
    tts_duck_count: int = 0
    tts_pause_count: int = 0
    mic_gate_count: int = 0
    echo_correlation_max: float = 0.0
    last_echo_score: float = 0.0
    avg_user_energy: float = 0.0
    avg_tts_energy: float = 0.0


class EchoSuppressor:
    """
    Application-layer echo suppression for duplex voice streaming.
    
    Features:
    - Voice Activity Detection (VAD) with energy and ZCR analysis
    - Advanced echo detection using correlation and spectral analysis
    - Barge-in detection and TTS interruption
    - Microphone gating during TTS playback
    - Automatic TTS resume after end-of-speech
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.02,
        echo_threshold: float = 0.3,
        barge_in_delay_ms: int = 300,
        resume_delay_ms: int = 500,
        adaptive_mode: bool = True,
        config: Optional[EchoConfig] = None
    ):
        """
        Initialize echo suppressor with enhanced echo detection.
        
        Args:
            sample_rate: Audio sample rate in Hz
            vad_threshold: Voice activity detection threshold
            echo_threshold: Echo correlation threshold
            barge_in_delay_ms: Minimum speech duration to trigger barge-in
            resume_delay_ms: Wait time before resuming TTS
            adaptive_mode: Enable adaptive threshold learning
            config: Optional legacy configuration (deprecated)
        """
        # Use new parameter-based configuration over legacy config
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.echo_threshold = echo_threshold
        self.barge_in_delay_ms = barge_in_delay_ms
        self.resume_delay_ms = resume_delay_ms
        self.adaptive_mode = adaptive_mode
        
        # Legacy config support
        self.config = config or EchoConfig()
        
        # Initialize state
        self.state = EchoState.IDLE
        self.metrics = EchoMetrics()
        
        # Initialize advanced echo detector
        self.echo_detector = create_echo_detector(
            mode="full",
            sample_rate=sample_rate,
            correlation_threshold=echo_threshold,
            adaptive_threshold=adaptive_mode,
            frame_size_ms=20
        )
        
        # Audio buffers for analysis
        self._mic_buffer = deque(maxlen=self._samples_for_ms(2000))  # 2s history
        self._tts_buffer = deque(maxlen=self._samples_for_ms(2000))
        
        # State tracking
        self._tts_start_time: Optional[float] = None
        self._user_speech_start: Optional[float] = None
        self._last_state_change: float = time.time()
        self._resume_task: Optional[asyncio.Task] = None
        
        # VAD state
        self._vad_history = deque(maxlen=10)  # 10 frame VAD history
        self._energy_history = deque(maxlen=50)  # Energy history for adaptation
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_tts_duck: Optional[Callable] = None
        self._on_tts_pause: Optional[Callable] = None
        self._on_tts_resume: Optional[Callable] = None
        self._on_mic_gate: Optional[Callable] = None
        
        logger.info(f"Enhanced echo suppressor initialized: sr={sample_rate}, "
                   f"vad_thresh={vad_threshold}, echo_thresh={echo_threshold}, "
                   f"adaptive={adaptive_mode}")
    
    def _samples_for_ms(self, ms: int) -> int:
        """Convert milliseconds to number of samples at current sample rate."""
        return int(self.sample_rate * ms / 1000)
    
    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_tts_duck: Optional[Callable] = None,
        on_tts_pause: Optional[Callable] = None,
        on_tts_resume: Optional[Callable] = None,
        on_mic_gate: Optional[Callable] = None,
    ):
        """Set callback functions for echo suppression events."""
        self._on_state_change = on_state_change
        self._on_tts_duck = on_tts_duck
        self._on_tts_pause = on_tts_pause
        self._on_tts_resume = on_tts_resume
        self._on_mic_gate = on_mic_gate
    
    async def process_mic_audio(self, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Process microphone audio chunk with enhanced echo suppression.
        
        Args:
            audio_chunk: Raw PCM16 audio from microphone
            
        Returns:
            Dictionary with processing results and actions
        """
        # Convert to numpy for analysis
        try:
            audio_samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            logger.warning("Invalid audio chunk received")
            return {
                "processed_audio": audio_chunk,
                "should_gate": False,
                "vad_active": False,
                "echo_detected": False,
                "state": self.state.value
            }
        
        # Add to analysis buffer
        self._mic_buffer.extend(audio_samples)
        
        # Get TTS audio for echo analysis
        tts_samples = None
        if len(self._tts_buffer) > 0:
            tts_samples = np.array(list(self._tts_buffer)[-len(audio_samples):])
        
        # Perform advanced echo detection
        detector_metrics = self.echo_detector.process_audio_frames(audio_samples, tts_samples)
        
        # Update metrics with detector results
        self.metrics.detector_metrics = detector_metrics
        self.metrics.echo_correlation_mean = 0.9 * self.metrics.echo_correlation_mean + 0.1 * detector_metrics.correlation_score
        self.metrics.echo_correlation_max = max(self.metrics.echo_correlation_max, detector_metrics.correlation_score)
        self.metrics.echo_probability_mean = 0.9 * self.metrics.echo_probability_mean + 0.1 * detector_metrics.echo_probability
        self.metrics.spectral_similarity_mean = 0.9 * self.metrics.spectral_similarity_mean + 0.1 * detector_metrics.spectral_similarity
        self.metrics.confidence_mean = 0.9 * self.metrics.confidence_mean + 0.1 * detector_metrics.confidence
        
        # Advanced VAD with energy and spectral features
        vad_active = await self._detect_user_speech_enhanced(audio_samples)
        
        # Update VAD history
        self._vad_history.append(vad_active)
        
        # Update energy history for adaptive thresholding
        if len(audio_samples) > 0:
            energy = np.mean(np.abs(audio_samples))
            self._energy_history.append(energy)
            self.metrics.avg_user_energy = 0.9 * self.metrics.avg_user_energy + 0.1 * energy
        
        # State machine logic with echo detection
        echo_detected = detector_metrics.echo_probability > 0.5
        await self._update_state_enhanced(vad_active, echo_detected)
        
        # Determine if we should gate mic input
        should_gate = self._should_gate_microphone(vad_active, echo_detected)
        
        # Apply gating if needed
        processed_audio = audio_chunk
        if should_gate:
            self.metrics.gated_frames += 1
            processed_audio = b'\x00' * len(audio_chunk)  # Return silence
            
            if self._on_mic_gate:
                await self._on_mic_gate(True)
        
        self.metrics.frames_processed += 1
        if vad_active:
            self.metrics.vad_active_frames += 1
        
        return {
            "processed_audio": processed_audio,
            "should_gate": should_gate,
            "vad_active": vad_active,
            "echo_detected": echo_detected,
            "echo_probability": detector_metrics.echo_probability,
            "correlation_score": detector_metrics.correlation_score,
            "confidence": detector_metrics.confidence,
            "state": self.state.value,
            "detector_metrics": detector_metrics
        }
    
    async def process_tts_audio(self, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Process TTS audio chunk with ducking and echo tracking.
        
        Args:
            audio_chunk: Raw PCM16 audio from TTS engine
            
        Returns:
            Dictionary with processing results and actions  
        """
        # Convert to numpy for analysis
        try:
            audio_samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            logger.warning("Invalid TTS audio chunk received")
            return {
                "processed_audio": audio_chunk,
                "should_duck": False,
                "should_pause": False,
                "state": self.state.value
            }
        
        # Add to TTS buffer for echo analysis
        self._tts_buffer.extend(audio_samples)
        
        # Update TTS energy metrics
        if len(audio_samples) > 0:
            energy = np.mean(np.abs(audio_samples))
            self.metrics.avg_tts_energy = 0.9 * self.metrics.avg_tts_energy + 0.1 * energy
        
        # Determine processing actions based on state
        should_duck = self.state == EchoState.BARGE_IN
        should_pause = False
        
        # Check if we should pause TTS (long user speech)
        if self.state == EchoState.BARGE_IN and self._user_speech_start:
            speech_duration = time.time() - self._user_speech_start
            if speech_duration > self.barge_in_delay_ms / 1000.0:
                should_pause = True
        
        # Process audio based on actions
        processed_samples = audio_samples.copy()
        
        if should_duck:
            # Apply ducking (reduce volume by 12dB)
            duck_factor = 10 ** (-12.0 / 20.0)  # -12dB in linear scale
            processed_samples *= duck_factor
            self.metrics.tts_duck_events += 1
            
            if self._on_tts_duck:
                await self._on_tts_duck(duck_factor)
        
        if should_pause:
            # Pause TTS completely
            processed_samples.fill(0.0)
            
            if self._on_tts_pause:
                await self._on_tts_pause()
        
        # Convert back to int16
        processed_audio = (processed_samples * 32767.0).astype(np.int16).tobytes()
        
        return {
            "processed_audio": processed_audio,
            "should_duck": should_duck,
            "should_pause": should_pause,
            "duck_factor": duck_factor if should_duck else 1.0,
            "state": self.state.value
        }
    
    async def _detect_user_speech_enhanced(self, audio_samples: np.ndarray) -> bool:
        """Enhanced VAD using energy, ZCR, and spectral features."""
        if len(audio_samples) == 0:
            return False
        
        # Energy-based detection with adaptive threshold
        energy = np.mean(np.abs(audio_samples))
        
        # Adapt threshold based on recent energy history
        if len(self._energy_history) > 10:
            energy_percentile = np.percentile(list(self._energy_history), 75)
            adaptive_threshold = max(self.vad_threshold, energy_percentile * 1.5)
        else:
            adaptive_threshold = self.vad_threshold
        
        energy_speech = energy > adaptive_threshold
        
        # Zero-crossing rate for voiced speech detection
        zcr_speech = False
        if len(audio_samples) > 1:
            zero_crossings = np.sum(np.diff(np.sign(audio_samples)) != 0)
            zcr = zero_crossings / len(audio_samples)
            zcr_speech = 0.05 < zcr < 0.3  # Typical ZCR range for speech
        
        # Spectral flatness (differentiates speech from noise)
        spectral_speech = False
        if len(audio_samples) >= 64:
            # Compute power spectrum
            fft = np.fft.rfft(audio_samples)
            power_spectrum = np.abs(fft) ** 2
            
            # Spectral flatness (geometric mean / arithmetic mean)
            if np.mean(power_spectrum) > 1e-10:
                geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
                arithmetic_mean = np.mean(power_spectrum)
                spectral_flatness = geometric_mean / arithmetic_mean
                spectral_speech = spectral_flatness < 0.5  # Speech is less flat than noise
        
        # Combine all criteria with weights
        vad_score = (
            0.5 * energy_speech +
            0.3 * zcr_speech +
            0.2 * spectral_speech
        )
        
        return vad_score > 0.5
    
    async def _update_state_enhanced(self, vad_active: bool, echo_detected: bool):
        """Enhanced state machine with echo detection."""
        current_time = time.time()
        old_state = self.state
        
        if self.state == EchoState.IDLE:
            if vad_active and not echo_detected:
                self.state = EchoState.USER_SPEAKING
                self._user_speech_start = current_time
                
        elif self.state == EchoState.TTS_PLAYING:
            if vad_active and not echo_detected:
                # User started speaking during TTS - enter barge-in
                self.state = EchoState.BARGE_IN
                self._user_speech_start = current_time
                self.metrics.barge_in_events += 1
                
                if self._on_state_change:
                    await self._on_state_change(old_state, self.state)
                    
        elif self.state == EchoState.USER_SPEAKING:
            if not vad_active:
                # User stopped speaking - back to idle
                self.state = EchoState.IDLE
                self._user_speech_start = None
                
        elif self.state == EchoState.BARGE_IN:
            if not vad_active:
                # User stopped speaking during barge-in
                # Schedule TTS resume after delay
                if self._resume_task:
                    self._resume_task.cancel()
                
                self._resume_task = asyncio.create_task(
                    self._schedule_tts_resume()
                )
                
                # Go back to TTS playing state
                self.state = EchoState.TTS_PLAYING
                self._user_speech_start = None
        
        # Update metrics
        if old_state != self.state:
            self.metrics.state_changes += 1
            self._last_state_change = current_time
            logger.debug(f"State transition: {old_state.value} -> {self.state.value}")
    
    def _should_gate_microphone(self, vad_active: bool, echo_detected: bool) -> bool:
        """Determine if microphone should be gated based on current conditions."""
        # Always gate if echo is detected (unless user is clearly speaking)
        if echo_detected and not vad_active:
            return True
        
        # Gate during TTS playback unless user is speaking
        if self.state == EchoState.TTS_PLAYING and not vad_active:
            return True
        
        # Don't gate during user speech or barge-in
        if self.state in [EchoState.USER_SPEAKING, EchoState.BARGE_IN]:
            return False
        
        return False
    
    async def _schedule_tts_resume(self):
        """Schedule TTS resume after delay."""
        await asyncio.sleep(self.resume_delay_ms / 1000.0)
        
        # Check if we should still resume (state might have changed)
        if self.state == EchoState.TTS_PLAYING:
            if self._on_tts_resume:
                await self._on_tts_resume()
            
            logger.debug("TTS resumed after user speech ended")
            self._resume_task.cancel()
        
            
            logger.debug("TTS resumed after user speech ended")
    
    # State management methods
    def start_tts_playback(self):
        """Start TTS playbook - transition to TTS_PLAYING state."""
        old_state = self.state
        self.state = EchoState.TTS_PLAYING
        self._tts_start_time = time.time()
        
        if self._on_state_change:
            # Use sync callback for state changes
            self._call_sync_callback(self._on_state_change, old_state, self.state)
        
        logger.debug("TTS playbook started - entering TTS_PLAYING state")
    
    def stop_tts_playbook(self):
        """Stop TTS playbook - return to IDLE state."""
        if self.state in (EchoState.TTS_PLAYING, EchoState.BARGE_IN):
            old_state = self.state
            self.state = EchoState.IDLE
            
            if self._on_state_change:
                self._call_sync_callback(self._on_state_change, old_state, self.state)
        
        self._tts_start_time = None
        
        # Cancel any pending resume task
        if self._resume_task:
            self._resume_task.cancel()
            self._resume_task = None
        
        logger.debug("TTS playbook stopped - returning to IDLE state")
    
    def _call_sync_callback(self, callback: Callable, *args):
        """Helper to call sync callbacks safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                # Schedule async callback
                asyncio.create_task(callback(*args))
            else:
                # Call sync callback directly
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    def get_state(self) -> EchoState:
        """Get current echo suppression state."""
        return self.state
    
    def get_metrics(self) -> EchoMetrics:
        """Get current echo suppression metrics."""
        # Update session duration
        session_duration = time.time() - self.metrics.session_start_time
        
        # Update correlation metrics from detector
        if self.metrics.detector_metrics:
            self.metrics.recent_correlations.append(self.metrics.detector_metrics.correlation_score)
        
        return self.metrics
    
    def get_echo_statistics(self) -> Dict[str, Any]:
        """Get detailed echo detection statistics."""
        detector_stats = self.echo_detector.get_statistics()
        
        return {
            "echo_suppression": {
                "state": self.state.value,
                "frames_processed": self.metrics.frames_processed,
                "vad_active_frames": self.metrics.vad_active_frames,
                "gated_frames": self.metrics.gated_frames,
                "barge_in_events": self.metrics.barge_in_events,
                "tts_duck_events": self.metrics.tts_duck_events,
                "session_duration_s": time.time() - self.metrics.session_start_time,
                "avg_user_energy": self.metrics.avg_user_energy,
                "avg_tts_energy": self.metrics.avg_tts_energy
            },
            "echo_detection": detector_stats,
            "current_metrics": {
                "echo_probability": self.metrics.echo_probability_mean,
                "correlation_score": self.metrics.echo_correlation_mean,
                "spectral_similarity": self.metrics.spectral_similarity_mean,
                "confidence": self.metrics.confidence_mean
            }
        }
    
    def get_duplex_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for duplex mode based on echo performance."""
        echo_prob = self.metrics.echo_probability_mean
        correlation = self.metrics.echo_correlation_mean
        
        if echo_prob > 0.7 or correlation > 0.5:
            return {
                "recommended_mode": "half",
                "reason": "high_echo_detected",
                "echo_probability": echo_prob,
                "correlation": correlation,
                "confidence": "high"
            }
        elif echo_prob > 0.4 or correlation > 0.3:
            return {
                "recommended_mode": "full",
                "reason": "moderate_echo_manageable",
                "echo_probability": echo_prob,
                "correlation": correlation,
                "confidence": "medium",
                "suggestion": "enable_hardware_echo_cancellation"
            }
        else:
            return {
                "recommended_mode": "full",
                "reason": "low_echo_optimal",
                "echo_probability": echo_prob,
                "correlation": correlation,
                "confidence": "high"
            }
    
    def reset_state(self):
        """Reset echo suppressor to initial state."""
        self.state = EchoState.IDLE
        self._tts_start_time = None
        self._user_speech_start = None
        self._last_state_change = time.time()
        
        # Cancel any pending tasks
        if self._resume_task:
            self._resume_task.cancel()
            self._resume_task = None
        
        # Reset buffers and detector
        self._mic_buffer.clear()
        self._tts_buffer.clear()
        self._vad_history.clear()
        self._energy_history.clear()
        
        self.echo_detector.reset_buffers()
        
        # Reset metrics but preserve session start time
        session_start = self.metrics.session_start_time
        self.metrics = EchoMetrics()
        self.metrics.session_start_time = session_start
        
        logger.info("Echo suppressor state reset")
