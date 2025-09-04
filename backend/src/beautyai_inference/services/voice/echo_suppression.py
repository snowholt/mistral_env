"""
Echo Suppression Service for Full Duplex Voice Communication.

Provides application-layer echo control and barge-in detection to prevent
TTS output from leaking back into the microphone input stream.
"""

import asyncio
import logging
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from collections import deque

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
    - Voice Activity Detection (VAD)
    - Barge-in detection and TTS interruption
    - Microphone gating during TTS playback
    - Echo correlation analysis
    - Automatic TTS resume after end-of-speech
    """
    
    def __init__(self, config: Optional[EchoConfig] = None):
        self.config = config or EchoConfig()
        self.state = EchoState.IDLE
        self.metrics = EchoMetrics()
        
        # Audio buffers for analysis
        self._mic_buffer = deque(maxlen=self._samples_for_ms(self.config.history_buffer_ms))
        self._tts_buffer = deque(maxlen=self._samples_for_ms(self.config.history_buffer_ms))
        
        # State tracking
        self._tts_start_time: Optional[float] = None
        self._user_speech_start: Optional[float] = None
        self._last_state_change: float = time.time()
        self._resume_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_tts_duck: Optional[Callable] = None
        self._on_tts_pause: Optional[Callable] = None
        self._on_tts_resume: Optional[Callable] = None
        self._on_mic_gate: Optional[Callable] = None
        
        # Sample rate (assumed 16kHz)
        self._sample_rate = 16000
        
        logger.info(f"Echo suppressor initialized with config: {self.config}")
    
    def _samples_for_ms(self, ms: int) -> int:
        """Convert milliseconds to number of samples at current sample rate."""
        return int(self._sample_rate * ms / 1000)
    
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
    
    async def process_mic_audio(self, audio_chunk: bytes) -> bytes:
        """
        Process microphone audio chunk with echo suppression.
        
        Args:
            audio_chunk: Raw PCM16 audio from microphone
            
        Returns:
            Processed audio chunk (may be gated/suppressed)
        """
        # Convert to numpy for analysis
        try:
            audio_samples = np.frombuffer(audio_chunk, dtype=np.int16)
        except ValueError:
            logger.warning("Invalid audio chunk received")
            return audio_chunk
        
        # Add to analysis buffer
        self._mic_buffer.extend(audio_samples)
        
        # Perform VAD analysis
        user_speaking = await self._detect_user_speech(audio_samples)
        
        # Update user energy metrics
        if len(audio_samples) > 0:
            energy = np.mean(np.abs(audio_samples)) / 32768.0  # Normalize to 0-1
            self.metrics.avg_user_energy = 0.9 * self.metrics.avg_user_energy + 0.1 * energy
        
        # State machine logic
        await self._update_state(user_speaking)
        
        # Apply gating based on state
        if self.config.gate_mic_during_tts and self.state == EchoState.TTS_PLAYING:
            if not user_speaking:
                # Gate (silence) microphone input during TTS playback
                self.metrics.mic_gate_count += 1
                if self._on_mic_gate:
                    await self._on_mic_gate(True)
                return b'\x00' * len(audio_chunk)  # Return silence
        
        return audio_chunk
    
    async def process_tts_audio(self, audio_chunk: bytes) -> bytes:
        """
        Process TTS audio chunk for echo analysis.
        
        Args:
            audio_chunk: Raw PCM16 audio from TTS engine
            
        Returns:
            Processed audio chunk (may be ducked during barge-in)
        """
        # Convert to numpy for analysis
        try:
            audio_samples = np.frombuffer(audio_chunk, dtype=np.int16)
        except ValueError:
            logger.warning("Invalid TTS audio chunk received")
            return audio_chunk
        
        # Add to TTS buffer
        self._tts_buffer.extend(audio_samples)
        
        # Update TTS energy metrics
        if len(audio_samples) > 0:
            energy = np.mean(np.abs(audio_samples)) / 32768.0
            self.metrics.avg_tts_energy = 0.9 * self.metrics.avg_tts_energy + 0.1 * energy
        
        # Apply ducking if in barge-in state
        if self.state == EchoState.BARGE_IN:
            # Apply ducking (reduce volume)
            duck_factor = 10 ** (self.config.tts_duck_db / 20.0)  # Convert dB to linear
            ducked_samples = (audio_samples * duck_factor).astype(np.int16)
            return ducked_samples.tobytes()
        
        return audio_chunk
    
    async def _detect_user_speech(self, audio_samples: np.ndarray) -> bool:
        """Detect if user is speaking using VAD."""
        if len(audio_samples) == 0:
            return False
        
        # Energy-based detection
        energy = np.mean(np.abs(audio_samples)) / 32768.0  # Normalize
        energy_speech = energy > self.config.energy_threshold
        
        # Zero-crossing rate (indicates voiced speech)
        if len(audio_samples) > 1:
            zero_crossings = np.sum(np.diff(np.sign(audio_samples)) != 0)
            zcr = zero_crossings / len(audio_samples)
            zcr_speech = zcr > self.config.zero_crossing_rate_threshold
        else:
            zcr_speech = False
        
        # Combine criteria
        user_speaking = energy_speech and zcr_speech
        
        return user_speaking
    
    async def _update_state(self, user_speaking: bool):
        """Update echo suppression state machine."""
        current_time = time.time()
        old_state = self.state
        
        if self.state == EchoState.IDLE:
            if user_speaking:
                self.state = EchoState.USER_SPEAKING
                self._user_speech_start = current_time
                
        elif self.state == EchoState.TTS_PLAYING:
            if user_speaking:
                # User started speaking during TTS - enter barge-in
                self.state = EchoState.BARGE_IN
                self._user_speech_start = current_time
                self.metrics.barge_in_count += 1
                
                # Duck TTS audio
                if self._on_tts_duck:
                    await self._on_tts_duck(self.config.tts_duck_db)
                self.metrics.tts_duck_count += 1
                
        elif self.state == EchoState.USER_SPEAKING:
            if not user_speaking:
                # User stopped speaking
                self.state = EchoState.IDLE
                self._user_speech_start = None
                
        elif self.state == EchoState.BARGE_IN:
            if user_speaking:
                # Check if user has been speaking long enough to pause TTS
                if (self._user_speech_start and 
                    (current_time - self._user_speech_start) * 1000 >= self.config.tts_pause_threshold_ms):
                    
                    if self._on_tts_pause:
                        await self._on_tts_pause()
                    self.metrics.tts_pause_count += 1
                    
            else:
                # User stopped speaking during barge-in
                self.state = EchoState.USER_SPEAKING  # Transition through user speaking
                self._schedule_tts_resume()
        
        # Emit state change event
        if old_state != self.state:
            self._last_state_change = current_time
            self.metrics.state_changes += 1
            
            if self._on_state_change:
                await self._on_state_change(old_state, self.state)
            
            logger.debug(f"Echo suppressor state: {old_state.value} -> {self.state.value}")
    
    def _schedule_tts_resume(self):
        """Schedule TTS resume after delay."""
        if self._resume_task:
            self._resume_task.cancel()
        
        async def _resume_after_delay():
            await asyncio.sleep(self.config.resume_delay_ms / 1000.0)
            if self.state == EchoState.USER_SPEAKING:  # Still not speaking
                self.state = EchoState.TTS_PLAYING
                if self._on_tts_resume:
                    await self._on_tts_resume()
                logger.debug("TTS resumed after user end-of-speech")
        
        self._resume_task = asyncio.create_task(_resume_after_delay())
    
    async def on_tts_start(self):
        """Called when TTS playback starts."""
        self.state = EchoState.TTS_PLAYING
        self._tts_start_time = time.time()
        
        if self._on_state_change:
            await self._on_state_change(EchoState.IDLE, EchoState.TTS_PLAYING)
        
        logger.debug("TTS playback started - entering TTS_PLAYING state")
    
    async def on_tts_complete(self):
        """Called when TTS playback completes."""
        if self.state in (EchoState.TTS_PLAYING, EchoState.BARGE_IN):
            self.state = EchoState.IDLE
            
            if self._on_state_change:
                await self._on_state_change(self.state, EchoState.IDLE)
        
        self._tts_start_time = None
        
        # Cancel any pending resume task
        if self._resume_task:
            self._resume_task.cancel()
            self._resume_task = None
        
        logger.debug("TTS playback completed - returning to IDLE state")
    
    def calculate_echo_correlation(self) -> float:
        """Calculate cross-correlation between mic and TTS buffers to detect echo."""
        if len(self._mic_buffer) < 100 or len(self._tts_buffer) < 100:
            return 0.0
        
        try:
            # Convert to numpy arrays
            mic_samples = np.array(list(self._mic_buffer))
            tts_samples = np.array(list(self._tts_buffer))
            
            # Ensure same length
            min_len = min(len(mic_samples), len(tts_samples))
            mic_samples = mic_samples[-min_len:]
            tts_samples = tts_samples[-min_len:]
            
            # Normalize
            if np.std(mic_samples) > 0 and np.std(tts_samples) > 0:
                mic_norm = (mic_samples - np.mean(mic_samples)) / np.std(mic_samples)
                tts_norm = (tts_samples - np.mean(tts_samples)) / np.std(tts_samples)
                
                # Calculate correlation
                correlation = np.corrcoef(mic_norm, tts_norm)[0, 1]
                
                if not np.isnan(correlation):
                    self.metrics.last_echo_score = abs(correlation)
                    self.metrics.echo_correlation_max = max(
                        self.metrics.echo_correlation_max, 
                        abs(correlation)
                    )
                    return abs(correlation)
            
        except Exception as e:
            logger.warning(f"Error calculating echo correlation: {e}")
        
        return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current echo suppression metrics."""
        echo_score = self.calculate_echo_correlation()
        
        return {
            "state": self.state.value,
            "state_changes": self.metrics.state_changes,
            "barge_in_count": self.metrics.barge_in_count,
            "tts_duck_count": self.metrics.tts_duck_count,
            "tts_pause_count": self.metrics.tts_pause_count,
            "mic_gate_count": self.metrics.mic_gate_count,
            "echo_correlation_current": echo_score,
            "echo_correlation_max": self.metrics.echo_correlation_max,
            "last_echo_score": self.metrics.last_echo_score,
            "avg_user_energy": self.metrics.avg_user_energy,
            "avg_tts_energy": self.metrics.avg_tts_energy,
            "time_since_last_state_change": time.time() - self._last_state_change,
            "tts_active": self.state in (EchoState.TTS_PLAYING, EchoState.BARGE_IN),
        }
    
    def reset(self):
        """Reset echo suppressor state and buffers."""
        self.state = EchoState.IDLE
        self.metrics = EchoMetrics()
        self._mic_buffer.clear()
        self._tts_buffer.clear()
        self._tts_start_time = None
        self._user_speech_start = None
        self._last_state_change = time.time()
        
        if self._resume_task:
            self._resume_task.cancel()
            self._resume_task = None
        
        logger.debug("Echo suppressor reset")
