"""
WebRTC Audio Processor for BeautyAI Voice Pipeline

This module handles audio track processing for WebRTC voice connections:
- Converts MediaStreamTrack AudioFrame → PCM 16kHz mono
- Enforces server-side 10-second utterance limit
- Provides frame-level audio processing for real-time voice input

Integrates with:
- aiortc RTCPeerConnection for WebRTC media tracks
- WebRTC VAD service for voice activity detection
- WebRTC buffer manager for audio accumulation
- SimpleVoiceService for STT/LLM/TTS pipeline

Author: BeautyAI Framework
Date: 2025-10-15
"""

import asyncio
import inspect
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from collections import deque
import numpy as np

try:
    from aiortc import MediaStreamTrack, RTCPeerConnection
    from aiortc.mediastreams import AudioFrame
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    # Provide dummy types for graceful degradation
    MediaStreamTrack = object
    AudioFrame = object

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingConfig:
    """Configuration for WebRTC audio processing."""
    
    # Audio format parameters
    target_sample_rate: int = 16000  # Target sample rate for STT (Whisper expects 16kHz)
    target_channels: int = 1  # Mono audio for STT
    target_format: str = "s16"  # 16-bit signed integer PCM
    
    # Utterance limits (server-side enforcement)
    max_utterance_duration_sec: int = 10  # Maximum recording duration
    max_utterance_bytes: int = 16000 * 2 * 10  # 10 seconds at 16kHz mono 16-bit
    
    # Frame processing
    frame_chunk_size_ms: int = 30  # Process audio in 30ms chunks (Silero VAD standard)
    frame_buffer_size: int = 100  # Maximum frames to buffer before processing
    
    # Quality settings
    auto_gain_control: bool = True  # Enable AGC on audio track
    noise_suppression: bool = True  # Enable noise suppression
    echo_cancellation: bool = True  # Enable echo cancellation
    
    # Monitoring
    enable_level_monitoring: bool = True  # Track audio levels for debugging
    log_processing_stats: bool = False  # Log detailed processing statistics


@dataclass
class AudioStreamMetrics:
    """Metrics for audio stream processing."""
    frames_received: int = 0
    frames_processed: int = 0
    bytes_processed: int = 0
    processing_time_ms: float = 0.0
    average_level: float = 0.0
    peak_level: float = 0.0
    utterance_duration_sec: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    dropped_frames: int = 0


class WebRTCAudioProcessor:
    """
    Processes audio from WebRTC MediaStreamTrack for voice pipeline.
    
    Responsibilities:
    - Convert incoming AudioFrame objects to PCM format
    - Resample audio to 16kHz mono for STT models
    - Enforce server-side 10-second utterance limits
    - Provide audio level monitoring and metrics
    - Feed processed audio to WebRTC buffer manager
    
    This processor sits between the RTCPeerConnection audio track and
    the WebRTC VAD/buffer pipeline, ensuring audio is in the correct
    format for downstream processing.
    """
    
    def __init__(
        self,
        peer_id: str,
        config: Optional[AudioProcessingConfig] = None,
        on_audio_chunk: Optional[Callable[[bytes, Dict[str, Any]], None]] = None,
        on_utterance_limit_exceeded: Optional[Callable[[str], None]] = None,
        on_processing_error: Optional[Callable[[str, Exception], None]] = None
    ):
        """
        Initialize WebRTC audio processor.
        
        Args:
            peer_id: Unique identifier for the peer connection
            config: Audio processing configuration
            on_audio_chunk: Callback when audio chunk is ready (chunk_bytes, metadata)
            on_utterance_limit_exceeded: Callback when utterance exceeds 10s limit
            on_processing_error: Callback for processing errors
        """
        self.peer_id = peer_id
        self.config = config or AudioProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self._processing_task = None
        
        # Callbacks
        self._on_audio_chunk = on_audio_chunk
        self._on_utterance_limit_exceeded = on_utterance_limit_exceeded
        self._on_processing_error = on_processing_error
        
        # Processing state
        self.is_processing = False
        self.start_time: Optional[float] = None
        self.current_utterance_bytes = 0
        self.current_utterance_duration = 0.0
        
        # Audio metrics
        self.metrics = AudioStreamMetrics()
        self._level_history = deque(maxlen=100)  # Track recent audio levels
        
        # Frame buffer for chunking
        self._frame_buffer: deque = deque(maxlen=self.config.frame_buffer_size)
        self._processing_lock = asyncio.Lock()
        
        self.logger.info(
            f"WebRTC audio processor initialized for peer {peer_id} "
            f"(target: {self.config.target_sample_rate}Hz mono, "
            f"max: {self.config.max_utterance_duration_sec}s)"
        )
    
    async def start_processing(self, audio_track: MediaStreamTrack) -> bool:
        """
        Start processing audio from a WebRTC audio track.
        
        Args:
            audio_track: The MediaStreamTrack to process
            
        Returns:
            bool: True if processing started successfully
        """
        if not AIORTC_AVAILABLE:
            self.logger.error("aiortc not available, cannot process audio track")
            return False
        
        if self.is_processing:
            self.logger.warning(f"Audio processor already processing for peer {self.peer_id}")
            return False
        
        try:
            self.is_processing = True
            self.start_time = time.time()
            self.current_utterance_bytes = 0
            self.current_utterance_duration = 0.0
            
            self.logger.debug(f"[PROCESSOR] Starting audio processing for peer {self.peer_id}, track={audio_track}")
            self.logger.debug(f"[PROCESSOR] Callback registered: {self._on_audio_chunk is not None}")
            
            # Start processing loop and store task reference
            self._processing_task = asyncio.create_task(self._process_audio_track(audio_track))
            self.logger.debug(f"[PROCESSOR] Created processing task for {self.peer_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start audio processing: {e}")
            self.is_processing = False
            if self._on_processing_error:
                self._on_processing_error(self.peer_id, e)
            return False
    
    async def stop_processing(self):
        """Stop processing audio and cleanup resources."""
        if not self.is_processing:
            return
        
        self.logger.info(f"Stopping audio processing for peer {self.peer_id}")
        self.is_processing = False
        
        # Process any remaining buffered frames
        if self._frame_buffer:
            await self._flush_frame_buffer()
        
        # Log final metrics
        if self.config.log_processing_stats:
            self._log_metrics()
    
    async def _process_audio_track(self, audio_track: MediaStreamTrack):
        """
        Main processing loop for audio track.
        
        Continuously reads AudioFrame objects from the track,
        converts them to PCM, and feeds them to the buffer manager.
        """
        self.logger.debug(f"[PROCESSOR] Entered _process_audio_track loop for {self.peer_id}")
        frame_count = 0
        try:
            while self.is_processing:
                try:
                    # Read frame from track with timeout
                    frame = await asyncio.wait_for(
                        audio_track.recv(),
                        timeout=1.0
                    )
                    
                    frame_count += 1
                    if frame_count == 1:
                        self.logger.debug(f"[PROCESSOR] Received first audio frame for {self.peer_id}")
                    elif frame_count % 100 == 0:
                        self.logger.debug(f"[PROCESSOR] Processed {frame_count} frames for {self.peer_id}")
                    
                    # Process the frame
                    await self._process_audio_frame(frame)
                    
                except asyncio.TimeoutError:
                    # No frame received in timeout period, continue
                    if frame_count == 0:
                        self.logger.debug(f"[PROCESSOR] No frames received yet for {self.peer_id} (timeout)")
                    continue
                    
                except Exception as e:
                    self.logger.error(f"[PROCESSOR] Error receiving audio frame: {e}", exc_info=True)
                    if self._on_processing_error:
                        self._on_processing_error(self.peer_id, e)
                    break
                
        except Exception as e:
            self.logger.error(f"[PROCESSOR] Fatal error in audio track processing: {e}", exc_info=True)
            if self._on_processing_error:
                self._on_processing_error(self.peer_id, e)
        finally:
            self.logger.debug(f"[PROCESSOR] Exiting _process_audio_track loop for {self.peer_id}, total frames: {frame_count}")
            self.is_processing = False
    
    async def _process_audio_frame(self, frame: AudioFrame):
        """
        Process a single AudioFrame from WebRTC.
        
        Args:
            frame: The AudioFrame object from aiortc
        """
        if not frame:
            return
        
        async with self._processing_lock:
            try:
                start_time = time.time()
                
                # Update metrics with safeguards for missing metadata
                self.metrics.frames_received += 1

                frame_rate = getattr(frame, "sample_rate", None) or 0
                if frame_rate <= 0:
                    frame_rate = self.config.target_sample_rate
                    self.logger.warning(
                        f"[PROCESSOR] Invalid frame sample rate for {self.peer_id}, "
                        f"falling back to target {frame_rate}Hz"
                    )
                self.metrics.sample_rate = frame_rate

                frame_layout = getattr(frame, "layout", None)
                if frame_layout and getattr(frame_layout, "channels", None):
                    self.metrics.channels = len(frame_layout.channels)
                else:
                    self.metrics.channels = self.config.target_channels
                
                # Convert AudioFrame to numpy array
                audio_array = self._frame_to_numpy(frame)

                if audio_array.size == 0:
                    self.logger.warning(
                        f"[PROCESSOR] Empty audio frame received for {self.peer_id}, skipping"
                    )
                    return
                
                # Resample to target sample rate if needed
                if frame_rate != self.config.target_sample_rate:
                    audio_array = self._resample_audio(
                        audio_array,
                        frame_rate,
                        self.config.target_sample_rate
                    )
                
                # Convert to mono if needed
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=0)
                
                # Convert to 16-bit PCM bytes
                pcm_bytes = self._numpy_to_pcm(audio_array)
                
                # Check utterance duration limit
                target_rate = max(self.config.target_sample_rate, 1)
                chunk_duration = len(pcm_bytes) / (target_rate * 2)
                self.current_utterance_duration += chunk_duration
                self.current_utterance_bytes += len(pcm_bytes)
                
                if self.current_utterance_duration > self.config.max_utterance_duration_sec:
                    self.logger.warning(
                        f"Utterance limit exceeded for peer {self.peer_id}: "
                        f"{self.current_utterance_duration:.2f}s"
                    )
                    if self._on_utterance_limit_exceeded:
                        callback_result = self._on_utterance_limit_exceeded(self.peer_id)
                        if inspect.isawaitable(callback_result):
                            await callback_result
                    await self.stop_processing()
                    return
                
                # Calculate audio level if monitoring enabled
                if self.config.enable_level_monitoring:
                    level = self._calculate_audio_level(audio_array)
                    self._level_history.append(level)
                    self.metrics.average_level = np.mean(list(self._level_history))
                    self.metrics.peak_level = max(self.metrics.peak_level, level)
                
                # Update metrics
                self.metrics.frames_processed += 1
                self.metrics.bytes_processed += len(pcm_bytes)
                self.metrics.utterance_duration_sec = self.current_utterance_duration
                processing_time = (time.time() - start_time) * 1000
                self.metrics.processing_time_ms += processing_time
                
                # Create metadata for chunk
                metadata = {
                    "peer_id": self.peer_id,
                    "sample_rate": self.config.target_sample_rate,
                    "channels": self.config.target_channels,
                    "duration_sec": chunk_duration,
                    "utterance_duration_sec": self.current_utterance_duration,
                    "timestamp": time.time(),
                    "audio_level": self.metrics.average_level if self.config.enable_level_monitoring else 0.0
                }
                
                # Send chunk to callback (buffer manager) - must await async callback
                self.logger.debug(f"[PROCESSOR] About to send chunk: {len(pcm_bytes)} bytes, callback={self._on_audio_chunk is not None}")
                if self._on_audio_chunk:
                    # Check if callback is async (coroutine function)
                    is_async = inspect.iscoroutinefunction(self._on_audio_chunk)
                    self.logger.debug(f"[PROCESSOR] Callback is async: {is_async}")
                    if is_async:
                        await self._on_audio_chunk(pcm_bytes, metadata)
                        self.logger.debug(f"[PROCESSOR] Async callback completed")
                    else:
                        # For backward compatibility with sync callbacks
                        self._on_audio_chunk(pcm_bytes, metadata)
                        self.logger.debug(f"[PROCESSOR] Sync callback completed")
                else:
                    self.logger.warning(f"[PROCESSOR] No audio chunk callback registered!")
                
            except Exception as e:
                self.logger.error(f"Error processing audio frame: {e}")
                self.metrics.dropped_frames += 1
                if self._on_processing_error:
                    self._on_processing_error(self.peer_id, e)
    
    def _frame_to_numpy(self, frame: AudioFrame) -> np.ndarray:
        """
        Convert AudioFrame to numpy array.
        
        Args:
            frame: The AudioFrame object
            
        Returns:
            numpy array of audio samples
        """
        # AudioFrame.to_ndarray() returns audio as float32 in range [-1.0, 1.0]
        audio_array = frame.to_ndarray()
        return audio_array
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate using scipy.
        
        Args:
            audio: Input audio array
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if source_rate <= 0:
            self.logger.warning(
                f"[PROCESSOR] Invalid source sample rate {source_rate}, skipping resample"
            )
            return audio

        if target_rate <= 0:
            self.logger.warning(
                f"[PROCESSOR] Invalid target sample rate {target_rate}, skipping resample"
            )
            return audio

        try:
            from scipy import signal
            
            if source_rate == target_rate:
                return audio
            
            # Calculate number of samples in output
            num_samples = int(len(audio) * target_rate / source_rate)

            if num_samples <= 0:
                self.logger.warning(
                    f"[PROCESSOR] Computed resample length {num_samples}, skipping resample"
                )
                return audio
            
            # Resample
            resampled = signal.resample(audio, num_samples)
            
            return resampled
            
        except ImportError:
            self.logger.warning("scipy not available, using simple resampling")
            # Fallback to simple linear interpolation
            if len(audio) == 0:
                return audio
            return np.interp(
                np.linspace(0, len(audio), int(len(audio) * target_rate / source_rate)),
                np.arange(len(audio)),
                audio
            )
    
    def _numpy_to_pcm(self, audio: np.ndarray) -> bytes:
        """
        Convert numpy audio array to 16-bit PCM bytes.
        
        Args:
            audio: Audio array (float32 in range [-1.0, 1.0])
            
        Returns:
            bytes: PCM audio data
        """
        # Clip to [-1.0, 1.0] range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to 16-bit signed integer
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_int16.tobytes()
    
    def _calculate_audio_level(self, audio: np.ndarray) -> float:
        """
        Calculate RMS audio level for monitoring.
        
        Args:
            audio: Audio array
            
        Returns:
            float: RMS level (0.0 to 1.0)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        return float(rms)
    
    async def _flush_frame_buffer(self):
        """Process any remaining frames in the buffer."""
        if not self._frame_buffer:
            return
        
        self.logger.debug(f"Flushing {len(self._frame_buffer)} remaining frames")
        
        while self._frame_buffer:
            frame = self._frame_buffer.popleft()
            await self._process_audio_frame(frame)
    
    def _log_metrics(self):
        """Log processing metrics."""
        avg_processing_time = (
            self.metrics.processing_time_ms / self.metrics.frames_processed
            if self.metrics.frames_processed > 0 else 0
        )
        
        self.logger.info(
            f"Audio processing metrics for peer {self.peer_id}:\n"
            f"  Frames received: {self.metrics.frames_received}\n"
            f"  Frames processed: {self.metrics.frames_processed}\n"
            f"  Frames dropped: {self.metrics.dropped_frames}\n"
            f"  Bytes processed: {self.metrics.bytes_processed}\n"
            f"  Utterance duration: {self.metrics.utterance_duration_sec:.2f}s\n"
            f"  Average processing time: {avg_processing_time:.2f}ms/frame\n"
            f"  Average audio level: {self.metrics.average_level:.4f}\n"
            f"  Peak audio level: {self.metrics.peak_level:.4f}"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current processing metrics.
        
        Returns:
            dict: Current metrics
        """
        return {
            "peer_id": self.peer_id,
            "is_processing": self.is_processing,
            "frames_received": self.metrics.frames_received,
            "frames_processed": self.metrics.frames_processed,
            "frames_dropped": self.metrics.dropped_frames,
            "bytes_processed": self.metrics.bytes_processed,
            "utterance_duration_sec": self.metrics.utterance_duration_sec,
            "average_level": self.metrics.average_level,
            "peak_level": self.metrics.peak_level,
            "sample_rate": self.metrics.sample_rate,
            "channels": self.metrics.channels
        }
    
    def reset_metrics(self):
        """Reset processing metrics for new utterance."""
        self.metrics = AudioStreamMetrics()
        self.current_utterance_bytes = 0
        self.current_utterance_duration = 0.0
        self._level_history.clear()
        self.start_time = time.time()
        
        self.logger.debug(f"Metrics reset for peer {self.peer_id}")


# Factory function for creating audio processors
def create_audio_processor(
    peer_id: str,
    config: Optional[AudioProcessingConfig] = None,
    **callbacks
) -> WebRTCAudioProcessor:
    """
    Factory function to create a WebRTC audio processor.
    
    Args:
        peer_id: Unique identifier for the peer
        config: Audio processing configuration
        **callbacks: Optional callbacks (on_audio_chunk, on_utterance_limit_exceeded, etc.)
        
    Returns:
        WebRTCAudioProcessor instance
    """
    return WebRTCAudioProcessor(
        peer_id=peer_id,
        config=config,
        on_audio_chunk=callbacks.get('on_audio_chunk'),
        on_utterance_limit_exceeded=callbacks.get('on_utterance_limit_exceeded'),
        on_processing_error=callbacks.get('on_processing_error')
    )
