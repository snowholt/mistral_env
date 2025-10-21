"""
WebRTC Dual VAD Service for BeautyAI Voice Pipeline

This module implements dual-stage Voice Activity Detection for WebRTC audio streams:
1. Fast path: WebRTC VAD for quick initial detection (browser hints)
2. Confirmation: Silero VAD for accurate speech verification

Inspired by KoljaB/RealtimeSTT dual VAD pattern:
- _is_voice_active(): WebRTC VAD for fast initial detection
- _is_silero_speech(): Silero VAD for confirmation
- Language-specific thresholds for Arabic vs English

Integrates with:
- WebRTC audio processor for PCM input
- WebRTC buffer manager for speech accumulation
- SimpleVoiceService for STT/LLM/TTS pipeline

Author: BeautyAI Framework
Date: 2025-10-15
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("webrtcvad not available, dual VAD will use Silero only")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


class VADState(Enum):
    """Voice activity detection states."""
    INACTIVE = "inactive"  # No voice detected
    VOICE_START = "voice_start"  # Initial voice detected (WebRTC)
    VOICE_ACTIVE = "voice_active"  # Confirmed voice activity (Silero)
    VOICE_END_PENDING = "voice_end_pending"  # Silence detected, waiting for confirmation
    VOICE_END = "voice_end"  # Confirmed end of speech


@dataclass
class WebRTCVADConfig:
    """Configuration for WebRTC dual VAD service."""
    
    # WebRTC VAD settings (fast path)
    webrtc_sensitivity: int = 0  # 0-3, higher = less sensitive (0=most aggressive, TESTING)
    webrtc_frame_duration_ms: int = 30  # 10, 20, or 30 ms frames
    
    # Silero VAD settings (confirmation path)
    silero_sensitivity: float = 0.5  # 0.0-1.0, higher = more sensitive
    silero_sample_rate: int = 16000  # Silero requires 16kHz
    
    # Language-specific thresholds (from migration plan)
    language_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "ar": 0.002,  # Arabic: lowered to prioritize detection over noise filtering
        "en": 0.002,  # English: lowered to prioritize detection over noise filtering
        "default": 0.002
    })
    
    # Speech detection timing
    min_speech_duration_ms: int = 300  # Minimum to register as speech (RealtimeSTT pattern)
    post_speech_silence_ms: int = 500  # Silence duration to end speech (RealtimeSTT: 600ms)
    pre_speech_buffer_ms: int = 200  # Pre-roll buffer (RealtimeSTT: 200ms)
    
    # State management
    enable_browser_hints: bool = True  # Use WebRTC VAD as first pass
    require_silero_confirmation: bool = True  # Require Silero to confirm WebRTC
    
    # Performance
    silero_use_onnx: bool = False  # Use ONNX for faster Silero inference
    
    # Monitoring
    log_vad_decisions: bool = False  # Log detailed VAD decisions for debugging


@dataclass
class VADMetrics:
    """Metrics for VAD performance."""
    webrtc_detections: int = 0
    silero_confirmations: int = 0
    false_positives: int = 0  # WebRTC detected but Silero rejected
    speech_segments: int = 0
    total_speech_duration_ms: float = 0.0
    average_silence_gap_ms: float = 0.0


class WebRTCVADService:
    """
    Dual-stage Voice Activity Detection service for WebRTC audio.
    
    Implements the RealtimeSTT dual VAD pattern:
    1. WebRTC VAD: Fast, lightweight initial detection (browser hints)
    2. Silero VAD: Accurate ML-based confirmation
    
    Benefits:
    - Low latency: WebRTC VAD responds immediately
    - High accuracy: Silero VAD filters false positives
    - Language-aware: Different thresholds for Arabic/English
    - Efficient: Only runs Silero when WebRTC detects voice
    
    Usage:
        vad = WebRTCVADService(peer_id="peer123", language="ar")
        await vad.initialize()
        
        # Process audio chunks
        result = await vad.process_audio_chunk(pcm_bytes, metadata)
        if result["voice_state"] == VADState.VOICE_ACTIVE:
            # Start buffering/transcription
            pass
    """
    
    def __init__(
        self,
        peer_id: str,
        language: str = "en",
        config: Optional[WebRTCVADConfig] = None,
        on_voice_start: Optional[Callable[[str], None]] = None,
        on_voice_end: Optional[Callable[[str, float], None]] = None,
        on_vad_state_change: Optional[Callable[[str, VADState], None]] = None
    ):
        """
        Initialize WebRTC dual VAD service.
        
        Args:
            peer_id: Unique identifier for the peer connection
            language: Language code (ar, en) for language-specific thresholds
            config: VAD configuration
            on_voice_start: Callback when voice activity starts
            on_voice_end: Callback when voice activity ends (peer_id, duration)
            on_vad_state_change: Callback on state changes
        """
        self.peer_id = peer_id
        self.language = language.lower()
        self.config = config or WebRTCVADConfig()
        self.logger = logging.getLogger(__name__)
        
        # Callbacks
        self._on_voice_start = on_voice_start
        self._on_voice_end = on_voice_end
        self._on_vad_state_change = on_vad_state_change
        
        # VAD models
        self.webrtc_vad = None
        self.silero_vad_model = None
        self.is_initialized = False
        
        # State tracking
        self.current_state = VADState.INACTIVE
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        
        # Metrics
        self.metrics = VADMetrics()
        
        # Audio buffering for confirmation
        self._audio_buffer = deque(maxlen=100)  # Pre-speech buffer
        self._silero_remainder: np.ndarray = np.array([], dtype=np.float32)
        self._processing_lock = asyncio.Lock()
        
        # Get language-specific threshold
        self.silero_threshold = self.config.language_thresholds.get(
            self.language,
            self.config.language_thresholds["default"]
        )
        
        self.logger.info(
            f"WebRTC dual VAD service created for peer {peer_id} "
            f"(language: {language}, threshold: {self.silero_threshold})"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize VAD models (WebRTC and Silero).
        
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            return True
        
        try:
            # Initialize WebRTC VAD (fast path)
            if WEBRTC_VAD_AVAILABLE and self.config.enable_browser_hints:
                self.webrtc_vad = webrtcvad.Vad(self.config.webrtc_sensitivity)
                self.logger.info(
                    f"WebRTC VAD initialized (sensitivity: {self.config.webrtc_sensitivity})"
                )
            else:
                self.logger.warning("WebRTC VAD not available or disabled, using Silero only")
            
            # Initialize Silero VAD (confirmation path)
            if TORCH_AVAILABLE:
                try:
                    self.silero_vad_model, _ = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        verbose=False,
                        onnx=self.config.silero_use_onnx
                    )
                    self.logger.info(
                        f"Silero VAD initialized (threshold: {self.silero_threshold}, "
                        f"onnx: {self.config.silero_use_onnx})"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to initialize Silero VAD: {e}")
                    return False
            else:
                self.logger.error("PyTorch not available, cannot initialize Silero VAD")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Dual VAD service initialized for peer {self.peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD service: {e}")
            return False
    
    async def process_audio_chunk(
        self,
        audio_data: bytes,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process audio chunk with dual VAD.
        
        Implements the RealtimeSTT dual VAD pattern:
        1. Quick WebRTC VAD check (fast path)
        2. If voice detected, confirm with Silero VAD
        3. Update state machine based on results
        
        Args:
            audio_data: PCM audio bytes (16kHz mono 16-bit)
            metadata: Audio metadata (sample_rate, duration, etc.)
            
        Returns:
            dict: VAD result with state, probabilities, and timing
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "VAD service not initialized"
            }
        
        async with self._processing_lock:
            try:
                start_time = time.time()
                
                # Stage 1: WebRTC VAD (fast path, inspired by _is_voice_active)
                webrtc_detected = False
                if self.webrtc_vad and self.config.enable_browser_hints:
                    webrtc_detected = self._is_voice_active_webrtc(audio_data)
                    if webrtc_detected:
                        self.metrics.webrtc_detections += 1
                
                # Stage 2: Silero VAD (confirmation path, inspired by _is_silero_speech)
                silero_detected = False
                silero_probability = 0.0
                
                if webrtc_detected or not self.config.enable_browser_hints:
                    # Only run Silero if WebRTC detected voice, or if WebRTC disabled
                    silero_detected, silero_probability = self._is_silero_speech(audio_data)
                    
                    if silero_detected and webrtc_detected:
                        self.metrics.silero_confirmations += 1
                    elif not silero_detected and webrtc_detected:
                        self.metrics.false_positives += 1
                
                # Determine final voice detection based on dual VAD strategy
                voice_detected = self._determine_voice_detection(
                    webrtc_detected,
                    silero_detected,
                    silero_probability
                )
                
                # Update state machine
                previous_state = self.current_state
                new_state = await self._update_state(voice_detected, metadata)
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Log VAD decision if enabled
                if self.config.log_vad_decisions:
                    self.logger.debug(
                        f"VAD decision for {self.peer_id}: "
                        f"webrtc={webrtc_detected}, silero={silero_detected} "
                        f"(prob={silero_probability:.3f}), final={voice_detected}, "
                        f"state={new_state.value}"
                    )
                
                return {
                    "success": True,
                    "peer_id": self.peer_id,
                    "voice_detected": voice_detected,
                    "voice_state": new_state,
                    "previous_state": previous_state,
                    "webrtc_detected": webrtc_detected,
                    "silero_detected": silero_detected,
                    "silero_probability": silero_probability,
                    "speech_duration_ms": (
                        (time.time() - self.speech_start_time) * 1000
                        if self.speech_start_time else 0.0
                    ),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                self.logger.error(f"Error processing audio chunk: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def _is_voice_active_webrtc(self, audio_data: bytes) -> bool:
        """
        WebRTC VAD: Fast initial voice activity detection.
        
        Inspired by RealtimeSTT _is_voice_active() and _is_webrtc_speech().
        Processes audio in 10/20/30ms frames as required by WebRTC VAD.
        
        Args:
            audio_data: PCM audio bytes
            
        Returns:
            bool: True if voice detected by WebRTC VAD
        """
        if not self.webrtc_vad:
            print(f"[DEBUG-WEBRTC-VAD] WebRTC VAD not initialized!")
            return False
        
        try:
            # WebRTC VAD requires specific frame sizes at 16kHz
            # 10ms = 320 bytes, 20ms = 640 bytes, 30ms = 960 bytes
            frame_size_bytes = int(
                self.config.webrtc_frame_duration_ms * 16000 * 2 / 1000
            )
            
            print(f"[DEBUG-WEBRTC-VAD] Chunk size={len(audio_data)}, frame_size={frame_size_bytes}, sensitivity={self.config.webrtc_sensitivity}")
            
            # Split audio into frames
            num_frames = len(audio_data) // frame_size_bytes
            
            if num_frames == 0:
                print(f"[DEBUG-WEBRTC-VAD] No complete frames (chunk too small)")
                return False
            
            # Check if any frame contains speech
            speech_frames = 0
            for i in range(num_frames):
                start = i * frame_size_bytes
                end = start + frame_size_bytes
                frame = audio_data[start:end]
                
                if len(frame) == frame_size_bytes:
                    is_speech = self.webrtc_vad.is_speech(frame, self.config.silero_sample_rate)
                    if is_speech:
                        speech_frames += 1
            
            print(f"[DEBUG-WEBRTC-VAD] Frames checked={num_frames}, speech_frames={speech_frames}")
            return speech_frames > 0
            
        except Exception as e:
            self.logger.error(f"WebRTC VAD error: {e}")
            return False
    
    def _is_silero_speech(self, audio_data: bytes) -> tuple[bool, float]:
        """
        Silero VAD: Accurate speech detection confirmation.
        
        Inspired by RealtimeSTT _is_silero_speech().
        Uses ML model for precise speech detection with language-specific thresholds.
        
        Args:
            audio_data: PCM audio bytes
            
        Returns:
            tuple: (is_speech, probability)
        """
        if not self.silero_vad_model:
            return False, 0.0
        
        try:
            # Convert PCM bytes to float32 numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Append to remainder so we always feed exact frame sizes
            if self._silero_remainder.size:
                audio_float = np.concatenate((self._silero_remainder, audio_float))

            frame_size = 512 if self.config.silero_sample_rate == 16000 else 256
            probabilities: list[float] = []

            while audio_float.size >= frame_size:
                frame = audio_float[:frame_size]
                audio_float = audio_float[frame_size:]
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(frame)
                    prob = self.silero_vad_model(
                        audio_tensor,
                        self.config.silero_sample_rate
                    ).item()
                probabilities.append(prob)

            # Store leftover for the next chunk
            if audio_float.size:
                self._silero_remainder = audio_float
            else:
                self._silero_remainder = np.array([], dtype=np.float32)

            if not probabilities:
                return False, 0.0

            probability = max(probabilities)
            is_speech = probability > self.silero_threshold
            return is_speech, probability

        except Exception as e:
            self.logger.error(f"Silero VAD error: {e}")
            return False, 0.0
    
    def _determine_voice_detection(
        self,
        webrtc_detected: bool,
        silero_detected: bool,
        silero_probability: float
    ) -> bool:
        """
        Determine final voice detection based on dual VAD strategy.
        
        Strategy:
        - If browser hints disabled: Use Silero only
        - If confirmation required: WebRTC AND Silero both must detect
        - If confirmation optional: WebRTC OR Silero
        
        Args:
            webrtc_detected: WebRTC VAD result
            silero_detected: Silero VAD result
            silero_probability: Silero confidence score
            
        Returns:
            bool: Final voice detection decision
        """
        if not self.config.enable_browser_hints:
            # Silero only mode
            return silero_detected
        
        if self.config.require_silero_confirmation:
            # Strict mode: Both must agree (RealtimeSTT pattern)
            return webrtc_detected and silero_detected
        else:
            # Permissive mode: Either can trigger
            return webrtc_detected or silero_detected
    
    async def _update_state(
        self,
        voice_detected: bool,
        metadata: Dict[str, Any]
    ) -> VADState:
        """
        Update VAD state machine based on voice detection.
        
        State transitions (inspired by RealtimeSTT _set_state):
        INACTIVE → VOICE_START (voice detected, timing < min_duration)
        VOICE_START → VOICE_ACTIVE (voice sustained for min_duration)
        VOICE_ACTIVE → VOICE_END_PENDING (silence detected)
        VOICE_END_PENDING → VOICE_END (silence sustained for post_speech_duration)
        VOICE_END_PENDING → VOICE_ACTIVE (voice resumed before timeout)
        
        Args:
            voice_detected: Current voice detection result
            metadata: Audio metadata
            
        Returns:
            VADState: New state
        """
        current_time = time.time()
        previous_state = self.current_state
        
        if voice_detected:
            self.last_voice_time = current_time
            
            if self.current_state == VADState.INACTIVE:
                # Start of potential speech
                self.speech_start_time = current_time
                self.silence_start_time = None
                self.current_state = VADState.VOICE_START
                
            elif self.current_state == VADState.VOICE_START:
                # Check if speech duration exceeds minimum
                speech_duration_ms = (current_time - self.speech_start_time) * 1000
                if speech_duration_ms >= self.config.min_speech_duration_ms:
                    self.current_state = VADState.VOICE_ACTIVE
                    if self._on_voice_start:
                        self._on_voice_start(self.peer_id)
                    self.metrics.speech_segments += 1
                
            elif self.current_state == VADState.VOICE_END_PENDING:
                # Voice resumed, cancel silence
                self.silence_start_time = None
                self.current_state = VADState.VOICE_ACTIVE
        
        else:  # No voice detected
            if self.current_state in [VADState.VOICE_START, VADState.VOICE_ACTIVE]:
                # Start measuring silence
                if not self.silence_start_time:
                    self.silence_start_time = current_time
                self.current_state = VADState.VOICE_END_PENDING
                
            elif self.current_state == VADState.VOICE_END_PENDING:
                # Check if silence duration exceeds threshold
                silence_duration_ms = (current_time - self.silence_start_time) * 1000
                if silence_duration_ms >= self.config.post_speech_silence_ms:
                    # Confirmed end of speech
                    speech_duration = current_time - self.speech_start_time if self.speech_start_time else 0
                    self.metrics.total_speech_duration_ms += speech_duration * 1000
                    
                    if self._on_voice_end:
                        self._on_voice_end(self.peer_id, speech_duration)
                    
                    self.current_state = VADState.VOICE_END
                    # Reset to inactive after end
                    self.speech_start_time = None
                    self.silence_start_time = None
                    self.current_state = VADState.INACTIVE
        
        # Notify state change
        if self.current_state != previous_state:
            if self._on_vad_state_change:
                self._on_vad_state_change(self.peer_id, self.current_state)
        
        return self.current_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get VAD performance metrics.
        
        Returns:
            dict: Current metrics
        """
        return {
            "peer_id": self.peer_id,
            "language": self.language,
            "webrtc_detections": self.metrics.webrtc_detections,
            "silero_confirmations": self.metrics.silero_confirmations,
            "false_positives": self.metrics.false_positives,
            "speech_segments": self.metrics.speech_segments,
            "total_speech_duration_ms": self.metrics.total_speech_duration_ms,
            "current_state": self.current_state.value,
            "silero_threshold": self.silero_threshold
        }
    
    def reset(self):
        """Reset VAD state for new utterance."""
        self.current_state = VADState.INACTIVE
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_voice_time = None
        self._audio_buffer.clear()
        self._silero_remainder = np.array([], dtype=np.float32)
        
        self.logger.debug(f"VAD state reset for peer {self.peer_id}")
    
    async def cleanup(self):
        """Cleanup VAD resources."""
        self.reset()
        self.is_initialized = False
        self.logger.info(f"VAD service cleaned up for peer {self.peer_id}")


# Factory function
def create_webrtc_vad_service(
    peer_id: str,
    language: str = "en",
    config: Optional[WebRTCVADConfig] = None,
    **callbacks
) -> WebRTCVADService:
    """
    Factory function to create a WebRTC dual VAD service.
    
    Args:
        peer_id: Unique identifier for the peer
        language: Language code (ar, en)
        config: VAD configuration
        **callbacks: Optional callbacks (on_voice_start, on_voice_end, etc.)
        
    Returns:
        WebRTCVADService instance
    """
    return WebRTCVADService(
        peer_id=peer_id,
        language=language,
        config=config,
        on_voice_start=callbacks.get('on_voice_start'),
        on_voice_end=callbacks.get('on_voice_end'),
        on_vad_state_change=callbacks.get('on_vad_state_change')
    )
