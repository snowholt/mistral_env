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
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

from ..utils.audio import to_float_mono_16k, float_to_pcm16, ensure_sample_rate

if TYPE_CHECKING:
    from ....core.webrtc_buffer_manager import WebRTCBufferManager

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
    webrtc_sensitivity: int = 2  # 0-3, higher = less sensitive (2=less aggressive, FIXED)
    webrtc_frame_duration_ms: int = 30  # 10, 20, or 30 ms frames
    
    # Silero VAD settings (confirmation path)
    silero_sensitivity: float = 0.3  # 0.0-1.0, higher = more sensitive (OPTIMIZED)
    silero_sample_rate: int = 16000  # Silero requires 16kHz
    
    # Language-specific thresholds (from migration plan)
    language_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "ar": 0.001,  # Arabic: very low threshold for maximum capture (OPTIMIZED)
        "en": 0.002,  # English: lowered to prioritize detection over noise filtering
        "default": 0.002
    })
    
    # Speech detection timing
    min_speech_duration_ms: int = 30   # Minimum to register as speech (OPTIMIZED for immediate capture)
    post_speech_silence_ms: int = 1000 # Silence duration to end speech (OPTIMIZED for natural pauses)
    pre_speech_buffer_ms: int = 200  # Pre-roll buffer (RealtimeSTT: 200ms)
    
    # Warmup filter (prevents premature VOICE_START during codec initialization)
    warmup_filter_duration_ms: int = 250  # Ignore initial audio period (lowered to 250ms to ensure completion with short audio files)
    min_sustained_speech_frames: int = 3  # Require N consecutive speech frames before VOICE_START
    
    # State management
    enable_browser_hints: bool = False  # Disable WebRTC VAD, use Silero only for maximum accuracy
    require_silero_confirmation: bool = True  # Require Silero confirmation for quality
    
    # Performance
    silero_use_onnx: bool = False  # Use ONNX for faster Silero inference
    
    # Monitoring
    log_vad_decisions: bool = True  # Log detailed VAD decisions for debugging (ENABLED)


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
        
        # Warmup filter state
        self.connection_start_time: Optional[float] = None
        self.warmup_complete = False
        self.sustained_speech_counter = 0  # Counter for sustained speech frames
        
        # Metrics
        self.metrics = VADMetrics()
        
        # Audio buffering for confirmation
        self._audio_buffer = deque(maxlen=100)  # Pre-speech buffer
        self._silero_remainder: np.ndarray = np.array([], dtype=np.float32)
        self._processing_lock = asyncio.Lock()

        # Debug capture configuration
        self.debug_enabled = os.getenv("BEAUTYAI_VAD_DEBUG", "1") not in {"0", "false", "False"}
        self._debug_webrtc_chunks: list[bytes] = []
        self._debug_silero_chunks: list[bytes] = []
        self._debug_segment_index: int = 0

        if self.debug_enabled:
            try:
                backend_root = Path(__file__).resolve().parents[5]
            except IndexError:
                backend_root = Path.cwd()
            self._debug_dump_dir = backend_root / "logs" / "webrtc" / "vad_debug"
            self._debug_dump_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._debug_dump_dir = None

        # Optional buffer manager linkage for end-of-stream handling
        self.buffer_manager: Optional['WebRTCBufferManager'] = None

        # Debug tracking for wider post-warmup logging window
        self._warmup_completion_chunk: Optional[int] = None
        
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
    
    def attach_buffer_manager(self, buffer_manager: Optional['WebRTCBufferManager']) -> None:
        """Attach buffer manager so VAD can finalize audio on stream end."""
        self.buffer_manager = buffer_manager
        if buffer_manager:
            self.logger.debug(f"[VAD] Buffer manager attached for {self.peer_id}")
        else:
            self.logger.debug(f"[VAD] Buffer manager detached for {self.peer_id}")

    async def process_audio_chunk(
        self,
        audio_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
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
        # Log chunk reception every 10 chunks for debugging
        self.metrics.chunks_processed = getattr(self.metrics, 'chunks_processed', 0) + 1
        if self.metrics.chunks_processed % 10 == 0:
            self.logger.info(f"[VAD\u2190PROCESSOR] Received chunk #{self.metrics.chunks_processed}: {len(audio_data)} bytes")
            print(f"[VAD\u2190PROCESSOR] Received chunk #{self.metrics.chunks_processed}: {len(audio_data)} bytes")
        
        if not self.is_initialized:
            return {
                "success": False,
                "error": "VAD service not initialized"
            }
        
        async with self._processing_lock:
            try:
                start_time = time.time()

                metadata = metadata or {}
                sample_rate_hint = metadata.get("sample_rate") or self.config.silero_sample_rate
                original_len = len(audio_data)

                # Audio comes in as 16kHz mono PCM from audio processor
                # No resampling needed - just convert to numpy array
                try:
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                except ValueError:
                    self.logger.warning("[VAD] Unable to interpret audio chunk as int16 PCM; skipping")
                    return {
                        "success": False,
                        "error": "invalid audio format"
                    }

                # DEBUG: Track audio sizes
                if self.metrics.chunks_processed % 10 == 0:
                    print(f"[VAD-IN] Chunk #{self.metrics.chunks_processed}: received {original_len} bytes")

                # Convert to float32 [-1.0, 1.0] for Silero VAD
                normalized_audio = audio_int16.astype(np.float32) / 32768.0
                sample_rate_used = ensure_sample_rate(sample_rate_hint, self.config.silero_sample_rate)

                metadata["sample_rate"] = sample_rate_used
                metadata["duration_sec"] = len(audio_data) / (sample_rate_used * 2)

                self.logger.debug(
                    f"[VAD] Received chunk: {original_len} bytes at {sample_rate_used}Hz"
                )
                
                # Track connection start time (first audio chunk)
                if self.connection_start_time is None:
                    self.connection_start_time = start_time
                    print(f"[WARMUP-INIT] Connection started for {self.peer_id}, warmup={self.config.warmup_filter_duration_ms}ms")
                    self.logger.info(
                        f"[WARMUP] Connection started for {self.peer_id}, "
                        f"warmup filter active for {self.config.warmup_filter_duration_ms}ms"
                    )
                
                # Check if warmup period has passed
                elapsed_ms = (start_time - self.connection_start_time) * 1000
                warmup_active = elapsed_ms < self.config.warmup_filter_duration_ms
                
                # Log warmup progress
                if warmup_active:
                    print(f"[WARMUP-FILTER] Active: {elapsed_ms:.0f}/{self.config.warmup_filter_duration_ms}ms for {self.peer_id}")
                    self.logger.debug(
                        f"[WARMUP] Filter active for {self.peer_id}: "
                        f"{elapsed_ms:.0f}ms / {self.config.warmup_filter_duration_ms}ms"
                    )
                elif not self.warmup_complete:
                    # Warmup just completed (log once)
                    self.warmup_complete = True
                    self._warmup_completion_chunk = self.metrics.chunks_processed
                    self.logger.info(
                        f"[WARMUP] Filter complete for {self.peer_id} after {elapsed_ms:.0f}ms, "
                        f"STT trigger enabled"
                    )
                    print(f"[WARMUP] ✅ Complete for {self.peer_id}, STT trigger enabled")
                
                # Stage 1: WebRTC VAD (fast path, inspired by _is_voice_active)
                # NOW RUNS DURING WARMUP - we process all audio but delay STT trigger
                webrtc_detected = False
                if self.webrtc_vad and self.config.enable_browser_hints:
                    webrtc_detected = self._is_voice_active_webrtc(audio_data)
                    
                    # Log WebRTC VAD for first 30 chunks
                    if self.metrics.chunks_processed <= 30:
                        self.logger.info(f"[WEBRTC-VAD] Chunk #{self.metrics.chunks_processed}: "
                                       f"detected={webrtc_detected}, warmup={warmup_active}")
                        print(f"[WEBRTC-VAD] detected={webrtc_detected}")
                    
                    if webrtc_detected:
                        self.metrics.webrtc_detections += 1
                
                # Stage 2: Silero VAD (confirmation path, inspired by _is_silero_speech)
                silero_detected = False
                silero_probability = 0.0
                
                post_warmup_window = (
                    self._warmup_completion_chunk is not None and
                    self.metrics.chunks_processed - self._warmup_completion_chunk <= 30
                )

                if normalized_audio.size and (webrtc_detected or not self.config.enable_browser_hints):
                    # Only run Silero if WebRTC detected voice, or if WebRTC disabled
                    silero_detected, silero_probability = self._is_silero_speech(normalized_audio)
                    
                    # Log Silero output for first 30 chunks or when speech detected
                    if (
                        self.metrics.chunks_processed <= 30
                        or silero_detected
                        or post_warmup_window
                    ):
                        self.logger.info(f"[SILERO-VAD] Chunk #{self.metrics.chunks_processed}: "
                                       f"prob={silero_probability:.4f}, threshold={self.silero_threshold:.4f}, "
                                       f"detected={silero_detected}, warmup={warmup_active}")
                        print(f"[SILERO-VAD] prob={silero_probability:.4f}, detected={silero_detected}")
                    
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
                
                if self.debug_enabled:
                    self._collect_debug_chunks(audio_data, webrtc_detected, silero_detected)

                # Sustained speech detection: Require min_sustained_speech_frames consecutive detections
                # before transitioning from INACTIVE to VOICE_START
                # In Silero-only mode, only check silero_detected; in dual mode, check both
                sustained_check = (silero_detected and voice_detected) if not self.config.enable_browser_hints else (voice_detected and webrtc_detected and silero_detected)
                
                if sustained_check:
                    self.sustained_speech_counter += 1
                    if self.metrics.chunks_processed <= 15:
                        self.logger.debug(
                            f"[SUSTAINED] Speech frame {self.sustained_speech_counter}/"
                            f"{self.config.min_sustained_speech_frames} for {self.peer_id}, warmup={warmup_active}"
                        )
                        print(f"[SUSTAINED] {self.sustained_speech_counter}/{self.config.min_sustained_speech_frames}, warmup={warmup_active}")
                else:
                    # Reset counter on non-voice or partial detection
                    if self.sustained_speech_counter > 0:
                        self.logger.debug(
                            f"[SUSTAINED] Counter reset for {self.peer_id} "
                            f"(was {self.sustained_speech_counter})"
                        )
                    self.sustained_speech_counter = 0
                
                # WARMUP FILTER: Delay STT trigger until warmup completes
                # VAD runs normally, but we suppress VOICE_START events during warmup
                if warmup_active and self.current_state == VADState.INACTIVE:
                    # During warmup, suppress VOICE_START (but allow VAD to track speech)
                    original_voice_detected = voice_detected
                    voice_detected = False
                    if original_voice_detected and self.sustained_speech_counter >= self.config.min_sustained_speech_frames:
                        self.logger.info(
                            f"[WARMUP] Delaying VOICE_START for {self.peer_id} "
                            f"(warmup {elapsed_ms:.0f}/{self.config.warmup_filter_duration_ms}ms, "
                            f"sustained={self.sustained_speech_counter} frames)"
                        )
                        print(f"[WARMUP] 🔇 Speech detected but delaying STT trigger (warmup not complete)")
                
                # Override voice_detected if sustained speech requirement not met (only during INACTIVE state)
                elif (self.current_state == VADState.INACTIVE and 
                    self.sustained_speech_counter < self.config.min_sustained_speech_frames):
                    # Not enough sustained speech yet, force to inactive
                    original_voice_detected = voice_detected
                    voice_detected = False
                    if original_voice_detected:
                        self.logger.debug(
                            f"[SUSTAINED] Suppressing VOICE_START for {self.peer_id} "
                            f"(need {self.config.min_sustained_speech_frames - self.sustained_speech_counter} more frames)"
                        )
                
                # Update state machine
                previous_state = self.current_state
                new_state = await self._update_state(voice_detected, metadata)
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Log VAD decision if enabled (using INFO level for visibility in production logs)
                if self.config.log_vad_decisions:
                    self.logger.info(
                        f"[VAD-DECISION] {self.peer_id}: "
                        f"webrtc={webrtc_detected}, silero={silero_detected} "
                        f"(prob={silero_probability:.3f}), sustained={self.sustained_speech_counter}/{self.config.min_sustained_speech_frames}, "
                        f"warmup={warmup_active}, final={voice_detected}, state={new_state.value}"
                    )
                
                if self.debug_enabled:
                    self._handle_debug_state_transition(previous_state, self.current_state, metadata)

                return {
                    "success": True,
                    "peer_id": self.peer_id,
                    "voice_detected": voice_detected,
                    "voice_state": new_state,
                    "previous_state": previous_state,
                    "webrtc_detected": webrtc_detected,
                    "silero_detected": silero_detected,
                    "silero_probability": silero_probability,
                    "sustained_speech_frames": self.sustained_speech_counter,
                    "speech_duration_ms": (
                        (time.time() - self.speech_start_time) * 1000
                        if self.speech_start_time else 0.0
                    ),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": time.time(),
                    "warmup_active": warmup_active,
                    "warmup_elapsed_ms": elapsed_ms
                }
                
            except Exception as e:
                self.logger.error(f"Error processing audio chunk: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    async def handle_end_of_stream(self, peer_id: str) -> None:
        """Finalize buffered audio when upstream stream ends."""
        self.logger.info(f"[VAD] END_OF_STREAM received for {peer_id}")

        buffer_manager = self.buffer_manager
        if buffer_manager is None:
            self.logger.warning(f"[VAD] No buffer manager attached for {peer_id}; resetting state")
            self.reset()
            return

        active_state = self.current_state in {
            VADState.VOICE_START,
            VADState.VOICE_ACTIVE,
            VADState.VOICE_END_PENDING
        }

        buffered_bytes = buffer_manager.get_buffer_size_bytes()
        if not active_state and buffered_bytes == 0:
            self.logger.info(f"[VAD] No buffered audio to finalize for {peer_id}; resetting state")
            self.reset()
            return

        forced_metadata = {
            "peer_id": peer_id,
            "end_of_stream": True,
            "vad_state": self.current_state.value if isinstance(self.current_state, VADState) else str(self.current_state)
        }

        try:
            segment = await buffer_manager.force_finalize_segment(forced_metadata)
            if segment:
                metadata = segment.get("metadata", {})
                duration = metadata.get("duration_sec")
                bytes_len = metadata.get("total_bytes", buffer_manager.get_buffer_size_bytes())
                duration_part = f", duration={duration:.2f}s" if isinstance(duration, (int, float)) else ""
                self.logger.info(
                    f"[VAD] Finalized buffered segment for {peer_id} on END_OF_STREAM"
                    f" (bytes={bytes_len}{duration_part})"
                )
            else:
                self.logger.warning(
                    f"[VAD] Buffer finalize returned no segment for {peer_id}"
                    f" (active_state={active_state}, buffered_bytes={buffered_bytes})"
                )
        except Exception as exc:
            self.logger.error(f"[VAD] Failed to finalize buffered audio for {peer_id}: {exc}", exc_info=True)
        finally:
            if self.debug_enabled:
                self._persist_debug_chunks(self.config.silero_sample_rate)
            self.reset()
    
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
    
    def _is_silero_speech(self, audio_float: np.ndarray) -> tuple[bool, float]:
        """
        Silero VAD: Accurate speech detection confirmation.
        
        Inspired by RealtimeSTT _is_silero_speech().
        Uses ML model for precise speech detection with language-specific thresholds.
        
        Args:
            audio_float: Normalized float audio samples
            
        Returns:
            tuple: (is_speech, probability)
        """
        if not self.silero_vad_model:
            return False, 0.0
        
        try:
            if audio_float.size == 0:
                return False, 0.0

            if audio_float.dtype != np.float32:
                audio_float = audio_float.astype(np.float32)

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
                self._silero_remainder = audio_float.astype(np.float32, copy=False)
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
                    
                    # Transition to VOICE_END (will be reset to INACTIVE on next cycle)
                    self.current_state = VADState.VOICE_END
                    # Reset timing but keep VOICE_END state for this cycle
                    self.speech_start_time = None
                    self.silence_start_time = None
            
            elif self.current_state == VADState.VOICE_END:
                # One cycle after VOICE_END, transition to INACTIVE
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

    async def process_audio(
        self,
        audio_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Backward-compatible alias for process_audio_chunk."""
        return await self.process_audio_chunk(audio_data, metadata)
    
    def reset(self):
        """Reset VAD state for new utterance."""
        self.current_state = VADState.INACTIVE
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_voice_time = None
        self._audio_buffer.clear()
        self._silero_remainder = np.array([], dtype=np.float32)
        
        # Reset warmup filter state
        self.connection_start_time = None
        self.warmup_complete = False
        self.sustained_speech_counter = 0

        if self.debug_enabled:
            self._debug_webrtc_chunks.clear()
            self._debug_silero_chunks.clear()
        
        self.logger.debug(f"VAD state reset for peer {self.peer_id}")
    
    async def cleanup(self):
        """Cleanup VAD resources."""
        self.reset()
        self.is_initialized = False
        self.logger.info(f"VAD service cleaned up for peer {self.peer_id}")

    def _collect_debug_chunks(self, audio_chunk: bytes, webrtc_detected: bool, silero_detected: bool) -> None:
        """Accumulate debug audio slices for WebRTC and Silero detections."""
        try:
            if webrtc_detected:
                self._debug_webrtc_chunks.append(audio_chunk)
            if silero_detected:
                self._debug_silero_chunks.append(audio_chunk)
        except Exception as exc:  # pragma: no cover - debug path
            self.logger.debug(f"[VAD-DEBUG] Failed to collect debug chunks: {exc}")

    def _handle_debug_state_transition(
        self,
        previous_state: VADState,
        new_state: VADState,
        metadata: Dict[str, Any]
    ) -> None:
        """Persist debug audio when a speech segment completes."""
        if not self.debug_enabled:
            return

        terminal_states = {
            VADState.VOICE_ACTIVE,
            VADState.VOICE_END_PENDING,
            VADState.VOICE_START,
            VADState.VOICE_END
        }

        if previous_state in terminal_states and new_state == VADState.INACTIVE:
            sample_rate = ensure_sample_rate(
                metadata.get("sample_rate") if metadata else None,
                self.config.silero_sample_rate
            )
            self._persist_debug_chunks(sample_rate)

    def _persist_debug_chunks(self, sample_rate: int) -> None:
        """Write collected debug chunks to WAV files and reset buffers."""
        if not self.debug_enabled or not self._debug_dump_dir:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_name = f"{timestamp}_{self.peer_id}_{self._debug_segment_index:02d}"

        sample_rate = ensure_sample_rate(sample_rate, self.config.silero_sample_rate)

        for label, chunks in (("webrtc", self._debug_webrtc_chunks), ("silero", self._debug_silero_chunks)):
            if not chunks:
                continue

            raw_audio = b"".join(chunks)
            output_path = self._debug_dump_dir / f"{base_name}_{label}.wav"

            try:
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(raw_audio)
                self.logger.info(
                    f"[VAD-DEBUG] Saved {label.upper()} debug audio to {output_path}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"[VAD-DEBUG] Failed to persist {label} debug audio for {self.peer_id}: {exc}"
                )

        self._debug_webrtc_chunks.clear()
        self._debug_silero_chunks.clear()
        self._debug_segment_index += 1


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
