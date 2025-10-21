"""
WebRTC Buffer Manager for BeautyAI Voice Pipeline

This module implements audio buffering for WebRTC voice streams:
- Pre-roll buffering: Capture audio before speech starts (300ms)
- Post-roll buffering: Continue capture after speech ends (300ms)
- Frame-based accumulation with RTP timing awareness
- Integration with VAD for speech boundary detection

Inspired by KoljaB/RealtimeSTT buffering strategy:
- audio_buffer: Pre-speech rolling buffer (RealtimeSTT pattern)
- frames: Accumulated speech frames during active recording
- feed_audio: Continuous audio ingestion with VAD-driven segmentation

Integrates with:
- WebRTC audio processor for PCM input
- WebRTC VAD service for speech boundaries
- SimpleVoiceService for STT processing

Author: BeautyAI Framework
Date: 2025-10-15
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuration for WebRTC buffer manager."""
    
    # Pre-roll/post-roll settings (RealtimeSTT inspired)
    pre_roll_duration_ms: int = 300  # 300ms before speech start (RealtimeSTT: 200ms)
    post_roll_duration_ms: int = 300  # 300ms after speech end
    
    # Buffer sizing
    max_buffer_duration_ms: int = 30000  # Maximum total buffer (30 seconds)
    max_buffer_size_bytes: int = 16000 * 2 * 30  # 30s at 16kHz mono 16-bit
    
    # Frame management
    frame_duration_ms: int = 30  # Standard frame size for VAD
    sample_rate: int = 16000  # Target sample rate
    
    # Overflow handling
    enable_overflow_protection: bool = True  # Prevent buffer overflow
    overflow_strategy: str = "drop_oldest"  # "drop_oldest" or "drop_newest"
    
    # Performance
    enable_metrics: bool = True  # Track buffer performance


@dataclass
class BufferMetrics:
    """Metrics for buffer manager performance."""
    chunks_received: int = 0
    chunks_buffered: int = 0
    chunks_dropped: int = 0
    pre_roll_chunks: int = 0
    speech_chunks: int = 0
    post_roll_chunks: int = 0
    total_bytes_buffered: int = 0
    buffer_overflows: int = 0
    segments_completed: int = 0


class WebRTCBufferManager:
    """
    Manages audio buffering for WebRTC voice streams with VAD integration.
    
    Implements RealtimeSTT-inspired buffering strategy:
    1. Pre-roll buffer: Continuously rolls, capturing audio before speech
    2. Active buffer: Accumulates audio during speech (VAD-driven)
    3. Post-roll buffer: Continues after silence to avoid cutting words
    
    Workflow:
    1. Audio chunks continuously fed via feed_audio()
    2. Pre-roll buffer maintains recent audio (300ms rolling window)
    3. When VAD detects voice_start, pre-roll is copied to active buffer
    4. During speech, chunks added to active buffer
    5. When VAD detects voice_end, post-roll continues for 300ms
    6. Complete segment ready for STT processing
    
    Usage:
        buffer_mgr = WebRTCBufferManager(peer_id="peer123")
        
        # Feed audio continuously
        await buffer_mgr.feed_audio(pcm_bytes, vad_state, metadata)
        
        # When speech segment complete
        complete_audio = buffer_mgr.get_complete_segment()
    """
    
    def __init__(
        self,
        peer_id: str,
        config: Optional[BufferConfig] = None,
        on_segment_ready: Optional[Callable[[str, bytes, Dict[str, Any]], None]] = None,
        on_buffer_overflow: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize WebRTC buffer manager.
        
        Args:
            peer_id: Unique identifier for the peer connection
            config: Buffer configuration
            on_segment_ready: Callback when complete speech segment ready
            on_buffer_overflow: Callback when buffer overflows
        """
        self.peer_id = peer_id
        self.config = config or BufferConfig()
        self.logger = logging.getLogger(__name__)
        
        # Callbacks
        self._on_segment_ready = on_segment_ready
        self._on_buffer_overflow = on_buffer_overflow
        
        # Calculate buffer sizes in frames
        self.pre_roll_frames = self._ms_to_frames(self.config.pre_roll_duration_ms)
        self.post_roll_frames = self._ms_to_frames(self.config.post_roll_duration_ms)
        self.max_buffer_frames = self._ms_to_frames(self.config.max_buffer_duration_ms)
        
        # Buffers (RealtimeSTT pattern)
        self._pre_roll_buffer: deque = deque(maxlen=self.pre_roll_frames)  # audio_buffer
        self._active_buffer: List[bytes] = []  # frames during speech
        self._post_roll_counter = 0  # Track post-roll frames
        
        # State tracking
        self.is_recording = False
        self.is_in_post_roll = False
        self.speech_start_time: Optional[float] = None
        
        # Metrics
        self.metrics = BufferMetrics()
        
        # Processing lock
        self._buffer_lock = asyncio.Lock()
        
        self.logger.info(
            f"WebRTC buffer manager initialized for peer {peer_id} "
            f"(pre-roll: {self.config.pre_roll_duration_ms}ms, "
            f"post-roll: {self.config.post_roll_duration_ms}ms)"
        )
    
    def _ms_to_frames(self, duration_ms: int) -> int:
        """Convert milliseconds to number of frames."""
        bytes_per_frame = int(
            self.config.sample_rate * 2 * self.config.frame_duration_ms / 1000
        )
        bytes_needed = int(self.config.sample_rate * 2 * duration_ms / 1000)
        return bytes_needed // bytes_per_frame + 1
    
    async def feed_audio(
        self,
        audio_chunk: bytes,
        vad_state: str,  # VADState enum value
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Feed audio chunk to buffer manager (RealtimeSTT feed_audio pattern).
        
        Implements adaptive buffering based on VAD state:
        - INACTIVE: Feed to pre-roll buffer only
        - VOICE_START: Copy pre-roll to active, start recording
        - VOICE_ACTIVE: Add to active buffer
        - VOICE_END_PENDING: Continue adding (might resume)
        - VOICE_END: Add post-roll frames, finalize segment
        
        Args:
            audio_chunk: PCM audio bytes
            vad_state: Current VAD state
            metadata: Audio metadata
            
        Returns:
            dict: Buffer status and segment readiness
        """
        async with self._buffer_lock:
            try:
                self.metrics.chunks_received += 1
                
                # Convert VAD state string to enum-like comparison
                from ..services.voice.vad.webrtc_vad_service import VADState
                
                # Always feed to pre-roll buffer (continuous rolling window)
                self._pre_roll_buffer.append(audio_chunk)
                
                # Handle state-specific buffering
                if vad_state == VADState.INACTIVE.value:
                    # Just maintain pre-roll buffer
                    return {
                        "status": "buffering_pre_roll",
                        "segment_ready": False
                    }
                
                elif vad_state == VADState.VOICE_START.value:
                    if not self.is_recording:
                        # Start of speech detected, copy pre-roll to active buffer
                        self._active_buffer = list(self._pre_roll_buffer)
                        self.is_recording = True
                        self.speech_start_time = time.time()
                        self.metrics.pre_roll_chunks = len(self._active_buffer)
                        
                        self.logger.debug(
                            f"Speech started for {self.peer_id}, "
                            f"copied {len(self._active_buffer)} pre-roll frames"
                        )
                    
                    # Add current chunk
                    self._active_buffer.append(audio_chunk)
                    self.metrics.speech_chunks += 1
                    
                    return {
                        "status": "recording_speech",
                        "segment_ready": False,
                        "buffer_size": len(self._active_buffer)
                    }
                
                elif vad_state == VADState.VOICE_ACTIVE.value:
                    if self.is_recording:
                        # Continue recording speech
                        self._check_buffer_overflow()
                        self._active_buffer.append(audio_chunk)
                        self.metrics.speech_chunks += 1
                        
                        return {
                            "status": "recording_speech",
                            "segment_ready": False,
                            "buffer_size": len(self._active_buffer)
                        }
                
                elif vad_state == VADState.VOICE_END_PENDING.value:
                    if self.is_recording:
                        # Silence detected, but might resume
                        self._active_buffer.append(audio_chunk)
                        
                        return {
                            "status": "silence_pending",
                            "segment_ready": False,
                            "buffer_size": len(self._active_buffer)
                        }
                
                elif vad_state == VADState.VOICE_END.value:
                    if self.is_recording:
                        # Speech confirmed ended, start post-roll
                        if not self.is_in_post_roll:
                            self.is_in_post_roll = True
                            self._post_roll_counter = 0
                            self.logger.debug(
                                f"Speech ended for {self.peer_id}, starting post-roll"
                            )
                        
                        # Add post-roll frames
                        if self._post_roll_counter < self.post_roll_frames:
                            self._active_buffer.append(audio_chunk)
                            self._post_roll_counter += 1
                            self.metrics.post_roll_chunks += 1
                            
                            return {
                                "status": "post_roll",
                                "segment_ready": False,
                                "post_roll_remaining": self.post_roll_frames - self._post_roll_counter
                            }
                        else:
                            # Post-roll complete, segment ready
                            segment_data = await self._finalize_segment(metadata)
                            
                            return {
                                "status": "segment_complete",
                                "segment_ready": True,
                                **segment_data
                            }
                
                return {
                    "status": "unknown_state",
                    "segment_ready": False
                }
                
            except Exception as e:
                self.logger.error(f"Error feeding audio to buffer: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "segment_ready": False
                }
    
    async def _finalize_segment(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize complete speech segment and prepare for STT.
        
        Returns:
            dict: Segment data and metadata
        """
        try:
            # Concatenate all buffered chunks
            complete_audio = b''.join(self._active_buffer)
            
            # Calculate segment duration
            segment_duration = (
                (time.time() - self.speech_start_time)
                if self.speech_start_time else 0.0
            )
            
            # Update metrics
            self.metrics.segments_completed += 1
            self.metrics.chunks_buffered = len(self._active_buffer)
            self.metrics.total_bytes_buffered += len(complete_audio)
            
            # Create segment metadata
            segment_metadata = {
                "peer_id": self.peer_id,
                "duration_sec": segment_duration,
                "num_frames": len(self._active_buffer),
                "pre_roll_frames": self.metrics.pre_roll_chunks,
                "speech_frames": self.metrics.speech_chunks,
                "post_roll_frames": self.metrics.post_roll_chunks,
                "total_bytes": len(complete_audio),
                "sample_rate": self.config.sample_rate,
                "timestamp": time.time(),
                **metadata
            }
            
            self.logger.info(
                f"Segment finalized for {self.peer_id}: "
                f"{len(complete_audio)} bytes, {segment_duration:.2f}s, "
                f"{len(self._active_buffer)} frames"
            )
            
            # Trigger callback if set
            if self._on_segment_ready:
                self._on_segment_ready(self.peer_id, complete_audio, segment_metadata)
            
            # Reset for next segment
            self._reset_for_next_segment()
            
            return {
                "audio_data": complete_audio,
                "metadata": segment_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error finalizing segment: {e}")
            self._reset_for_next_segment()
            raise
    
    def _check_buffer_overflow(self):
        """Check and handle buffer overflow conditions."""
        if not self.config.enable_overflow_protection:
            return
        
        current_size = len(self._active_buffer)
        
        if current_size >= self.max_buffer_frames:
            self.metrics.buffer_overflows += 1
            
            self.logger.warning(
                f"Buffer overflow for {self.peer_id}: "
                f"{current_size}/{self.max_buffer_frames} frames"
            )
            
            if self._on_buffer_overflow:
                self._on_buffer_overflow(self.peer_id)
            
            # Apply overflow strategy
            if self.config.overflow_strategy == "drop_oldest":
                # Remove oldest frames to make room
                excess = current_size - self.max_buffer_frames
                self._active_buffer = self._active_buffer[excess:]
                self.metrics.chunks_dropped += excess
            elif self.config.overflow_strategy == "drop_newest":
                # Don't add new frames
                self.metrics.chunks_dropped += 1
    
    def _reset_for_next_segment(self):
        """Reset buffer state for next speech segment."""
        self._active_buffer.clear()
        self.is_recording = False
        self.is_in_post_roll = False
        self._post_roll_counter = 0
        self.speech_start_time = None
        
        # Reset frame counters
        self.metrics.pre_roll_chunks = 0
        self.metrics.speech_chunks = 0
        self.metrics.post_roll_chunks = 0
    
    def get_complete_segment(self) -> Optional[bytes]:
        """
        Get the complete audio segment if ready.
        
        Returns:
            bytes: Complete audio segment or None if not ready
        """
        if not self.is_recording and self._active_buffer:
            return b''.join(self._active_buffer)
        return None
    
    def get_current_duration(self) -> float:
        """
        Get current segment duration in seconds.
        
        Returns:
            float: Duration in seconds
        """
        if not self.speech_start_time:
            return 0.0
        return time.time() - self.speech_start_time
    
    def get_buffer_size_bytes(self) -> int:
        """
        Get current active buffer size in bytes.
        
        Returns:
            int: Buffer size in bytes
        """
        return sum(len(chunk) for chunk in self._active_buffer)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get buffer performance metrics.
        
        Returns:
            dict: Current metrics
        """
        return {
            "peer_id": self.peer_id,
            "chunks_received": self.metrics.chunks_received,
            "chunks_buffered": self.metrics.chunks_buffered,
            "chunks_dropped": self.metrics.chunks_dropped,
            "pre_roll_chunks": self.metrics.pre_roll_chunks,
            "speech_chunks": self.metrics.speech_chunks,
            "post_roll_chunks": self.metrics.post_roll_chunks,
            "total_bytes_buffered": self.metrics.total_bytes_buffered,
            "buffer_overflows": self.metrics.buffer_overflows,
            "segments_completed": self.metrics.segments_completed,
            "current_buffer_size": len(self._active_buffer),
            "is_recording": self.is_recording,
            "is_in_post_roll": self.is_in_post_roll
        }
    
    def reset(self):
        """Reset buffer manager completely."""
        self._pre_roll_buffer.clear()
        self._active_buffer.clear()
        self._reset_for_next_segment()
        self.metrics = BufferMetrics()
        
        self.logger.debug(f"Buffer manager reset for peer {self.peer_id}")
    
    async def cleanup(self):
        """Cleanup buffer manager resources."""
        self.reset()
        self.logger.info(f"Buffer manager cleaned up for peer {self.peer_id}")


# Factory function
def create_buffer_manager(
    peer_id: str,
    config: Optional[BufferConfig] = None,
    **callbacks
) -> WebRTCBufferManager:
    """
    Factory function to create a WebRTC buffer manager.
    
    Args:
        peer_id: Unique identifier for the peer
        config: Buffer configuration
        **callbacks: Optional callbacks (on_segment_ready, on_buffer_overflow)
        
    Returns:
        WebRTCBufferManager instance
    """
    return WebRTCBufferManager(
        peer_id=peer_id,
        config=config,
        on_segment_ready=callbacks.get('on_segment_ready'),
        on_buffer_overflow=callbacks.get('on_buffer_overflow')
    )
