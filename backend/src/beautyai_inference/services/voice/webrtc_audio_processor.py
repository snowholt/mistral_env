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
from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from .vad.webrtc_vad_service import WebRTCVADService

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
        self.utterance_start_time: Optional[float] = None  # Real-time clock for utterance
        self.current_utterance_bytes = 0
        self.current_utterance_duration = 0.0
        self._stream_terminated = False
        
        # Audio metrics
        self.metrics = AudioStreamMetrics()
        self._level_history = deque(maxlen=100)  # Track recent audio levels
        
        # Frame buffer for chunking
        self._frame_buffer: deque = deque(maxlen=self.config.frame_buffer_size)
        self._processing_lock = asyncio.Lock()
        
        # Frame accumulator for tiny frames (before resampling)
        self._frame_accumulator = []
        self._accumulator_samples = 0
        self._min_samples_before_resample = 48  # Just 1ms at 48kHz - very conservative

        # Linked services (assigned by adapter)
        self.vad_service: Optional['WebRTCVADService'] = None
        
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
            self.utterance_start_time = time.time()  # Track real-time utterance duration
            self.current_utterance_bytes = 0
            self.current_utterance_duration = 0.0
            self._stream_terminated = False
            
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
        print(f"[PROCESSOR] Entered _process_audio_track loop for {self.peer_id}")
        frame_count = 0
        timeout_count = 0
        try:
            while self.is_processing:
                try:
                    # Read frame from track with timeout
                    print(f"[PROCESSOR] About to call audio_track.recv() for {self.peer_id}, frame_count={frame_count}")
                    frame = await asyncio.wait_for(
                        audio_track.recv(),
                        timeout=1.0
                    )
                    
                    frame_count += 1
                    timeout_count = 0  # Reset timeout counter on successful recv
                    if frame_count == 1:
                        self.logger.debug(f"[PROCESSOR] Received first audio frame for {self.peer_id}")
                        print(f"[PROCESSOR] Received first audio frame for {self.peer_id}")
                    elif frame_count % 10 == 0:
                        self.logger.debug(f"[PROCESSOR] Processed {frame_count} frames for {self.peer_id}")
                        print(f"[PROCESSOR] Processed {frame_count} frames for {self.peer_id}")
                    
                    # Process the frame
                    await self._process_audio_frame(frame)
                    
                except asyncio.TimeoutError:
                    # No frame received in timeout period, continue
                    timeout_count += 1
                    if frame_count == 0:
                        self.logger.debug(f"[PROCESSOR] No frames received yet for {self.peer_id} (timeout #{timeout_count})")
                        print(f"[PROCESSOR] No frames received yet for {self.peer_id} (timeout #{timeout_count})")
                    else:
                        self.logger.warning(f"[PROCESSOR] Timeout waiting for frame #{frame_count+1} for {self.peer_id} (timeout #{timeout_count})")
                        print(f"[PROCESSOR] Timeout waiting for frame #{frame_count+1} for {self.peer_id} (timeout #{timeout_count})")
                    
                    # If we have many consecutive timeouts after receiving frames, something is wrong
                    if frame_count > 0 and timeout_count >= 3:
                        self.logger.error(f"[PROCESSOR] Audio stream stopped unexpectedly for {self.peer_id} after {frame_count} frames")
                        print(f"[PROCESSOR] Audio stream stopped unexpectedly for {self.peer_id} after {frame_count} frames")
                        self.is_processing = False
                        await self._finalize_stream("consecutive_timeouts")
                        break
                    continue
                    
                except Exception as e:
                    self.logger.error(f"[PROCESSOR] Error receiving audio frame: {e}", exc_info=True)
                    print(f"[PROCESSOR] Error receiving audio frame: {e}")
                    if self._on_processing_error:
                        self._on_processing_error(self.peer_id, e)
                    await self._finalize_stream("recv_exception")
                    break
                
        except Exception as e:
            self.logger.error(f"[PROCESSOR] Fatal error in audio track processing: {e}", exc_info=True)
            print(f"[PROCESSOR] Fatal error in audio track processing: {e}")
            if self._on_processing_error:
                self._on_processing_error(self.peer_id, e)
            await self._finalize_stream("fatal_exception")
        finally:
            self.logger.debug(f"[PROCESSOR] Exiting _process_audio_track loop for {self.peer_id}, total frames: {frame_count}")
            print(f"[PROCESSOR] Exiting _process_audio_track loop for {self.peer_id}, total frames: {frame_count}")
            self.is_processing = False
            self.logger.info(f"[Audio] Stream stopped for {self.peer_id} after {frame_count} frames")
            await self._finalize_stream("loop_exit")
    
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
                
                # Debug: Log frame shape before resampling (first few frames only)
                if self.metrics.frames_received <= 3:
                    print(f"[PROCESSOR] Frame {self.metrics.frames_received}: "
                          f"shape={audio_array.shape}, size={audio_array.size}, "
                          f"frame_rate={frame_rate}Hz")
                
                # Resample to target sample rate if needed
                if frame_rate != self.config.target_sample_rate:
                    audio_array = self._resample_audio(
                        audio_array,
                        frame_rate,
                        self.config.target_sample_rate
                    )
                
                # Debug: Log array shape after resampling
                if self.metrics.frames_received <= 3:
                    print(f"[PROCESSOR] After resample: shape={audio_array.shape}, size={audio_array.size}")
                
                # DEBUG: Check after resample
                if audio_array.size > 10:
                    non_zero = np.count_nonzero(np.abs(audio_array) > 0.001)
                    if non_zero == 0:
                        self.logger.warning(f"[PROCESSOR] Audio is zeros after resample!")
                
                # Convert to 16-bit PCM bytes
                pcm_bytes = self._numpy_to_pcm(audio_array)
                
                # DEBUG: Check PCM bytes content
                if len(pcm_bytes) >= 100:
                    import struct
                    samples = struct.unpack('<50h', pcm_bytes[:100])
                    non_zero = sum(1 for s in samples if abs(s) > 10)
                    if non_zero == 0:
                        self.logger.warning(f"[PROCESSOR] PCM chunk is all zeros! len={len(pcm_bytes)}")
                    elif non_zero < 5:
                        self.logger.warning(f"[PROCESSOR] PCM chunk mostly zeros: {non_zero}/50 non-zero")
                
                # Check utterance duration limit (use real-time clock as safeguard)
                target_rate = max(self.config.target_sample_rate, 1)
                chunk_duration = len(pcm_bytes) / (target_rate * 2)
                self.current_utterance_duration += chunk_duration
                self.current_utterance_bytes += len(pcm_bytes)
                
                # Calculate real elapsed time as safeguard against byte-count errors
                real_elapsed_time = time.time() - self.utterance_start_time if self.utterance_start_time else 0.0
                
                # Debug logging every 20 frames
                if self.metrics.frames_processed % 20 == 0:
                    print(f"[PROCESSOR] Frame {self.metrics.frames_processed}: "
                          f"pcm_bytes={len(pcm_bytes)}, chunk_duration={chunk_duration:.3f}s, "
                          f"calculated_duration={self.current_utterance_duration:.3f}s, "
                          f"real_time={real_elapsed_time:.3f}s, "
                          f"total_bytes={self.current_utterance_bytes}")
                
                # Use real elapsed time (more reliable) or calculated duration (fallback)
                effective_duration = max(real_elapsed_time, self.current_utterance_duration)
                
                if effective_duration > self.config.max_utterance_duration_sec:
                    self.logger.warning(
                        f"Utterance limit exceeded for peer {self.peer_id}: "
                        f"real_time={real_elapsed_time:.2f}s, calculated={self.current_utterance_duration:.2f}s "
                        f"(after {self.metrics.frames_processed} frames)"
                    )
                    print(f"[PROCESSOR] LIMIT EXCEEDED: real_time={real_elapsed_time:.2f}s, "
                          f"calculated={self.current_utterance_duration:.2f}s after {self.metrics.frames_processed} frames")
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
                self.logger.debug(f"[PROCESSOR] Sending chunk to VAD: {len(pcm_bytes)} bytes")
                if self.metrics.frames_processed % 10 == 0:
                    self.logger.info(f"[PROCESSOR→VAD] Sending chunk #{self.metrics.frames_processed}: {len(pcm_bytes)} bytes, {chunk_duration*1000:.1f}ms")
                    print(f"[PROCESSOR→VAD] Sending chunk #{self.metrics.frames_processed}: {len(pcm_bytes)} bytes to callback")
                
                if self._on_audio_chunk:
                    # Check if callback is async (coroutine function)
                    is_async = inspect.iscoroutinefunction(self._on_audio_chunk)
                    if is_async:
                        await self._on_audio_chunk(pcm_bytes, metadata)
                    else:
                        # For backward compatibility with sync callbacks
                        self._on_audio_chunk(pcm_bytes, metadata)
                    
                    if self.metrics.frames_processed % 10 == 0:
                        self.logger.info(f"[PROCESSOR→VAD] Callback completed for frame #{self.metrics.frames_processed}")
                else:
                    self.logger.error(f"[PROCESSOR] No audio chunk callback registered!")
                    print(f"[PROCESSOR] ERROR: No audio chunk callback registered!")
                
            except Exception as e:
                self.logger.error(f"Error processing audio frame: {e}")
                self.metrics.dropped_frames += 1
                if self._on_processing_error:
                    self._on_processing_error(self.peer_id, e)
    
    async def _finalize_stream(self, reason: str) -> None:
        """Invoke VAD end-of-stream finalization exactly once."""
        if self._stream_terminated:
            return
        self._stream_terminated = True
        self.logger.info(f"[PROCESSOR] Finalizing stream for {self.peer_id} (reason: {reason})")
        vad_service = getattr(self, "vad_service", None)
        if not vad_service:
            self.logger.debug(f"[PROCESSOR] No VAD service attached for {self.peer_id}; skipping end-of-stream handling")
            return
        try:
            result = vad_service.handle_end_of_stream(self.peer_id)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self.logger.error(f"[PROCESSOR] Failed to finalize VAD for {self.peer_id}: {exc}")

    def _frame_to_numpy(self, frame: AudioFrame) -> np.ndarray:
        """
        Convert AudioFrame to numpy array and flatten to 1D.
        
        Args:
            frame: The AudioFrame object
            
        Returns:
            1D numpy array of audio samples (mono, flattened)
        """
        # AudioFrame.to_ndarray() returns audio as float32 in range [-1.0, 1.0]
        # Shape can be (samples,) for mono or (channels, samples) for multi-channel
        audio_array = frame.to_ndarray()
        
        # Convert to mono immediately if multi-channel
        if audio_array.ndim > 1:
            # Take mean across channel axis to convert to mono
            audio_array = np.mean(audio_array, axis=0)
        
        # Diagnostic logging (first few frames only)
        if self.metrics.frames_received <= 3:
            rms = np.sqrt(np.mean(audio_array**2))
            max_val = np.max(np.abs(audio_array))
            self.logger.info(f"[AUDIO-DIAG] Frame {self.metrics.frames_received}: "
                           f"dtype={audio_array.dtype}, shape={audio_array.shape}, "
                           f"RMS={rms:.6f}, max={max_val:.6f}")
            print(f"[AUDIO-DIAG] Frame {self.metrics.frames_received}: "
                  f"RMS={rms:.6f}, max={max_val:.6f}")
        
        # DEBUG: Check if audio has actual content
        if audio_array.size > 0:
            non_zero = np.count_nonzero(np.abs(audio_array) > 0.001)
            if non_zero == 0 and audio_array.size > 10:
                self.logger.warning(f"[PROCESSOR] Frame is all zeros! size={audio_array.size}")
        
        return audio_array
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate using scipy resample_poly.
        
        Args:
            audio: Input audio array (float32, mono, [-1.0, 1.0])
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio array with preserved amplitude
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
            from scipy.signal import resample_poly
            
            if source_rate == target_rate:
                return audio
            
            # Calculate GCD for resample_poly (more efficient than arbitrary ratio)
            from math import gcd
            ratio_gcd = gcd(source_rate, target_rate)
            up = target_rate // ratio_gcd
            down = source_rate // ratio_gcd
            
            # Use resample_poly for better energy preservation
            resampled = resample_poly(audio, up, down)
            
            # Diagnostic logging (first few frames)
            if self.metrics.frames_received <= 3:
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(resampled**2))
                self.logger.info(f"[RESAMPLE-DIAG] {source_rate}Hz→{target_rate}Hz: "
                               f"RMS before={rms_before:.6f}, after={rms_after:.6f}")
                print(f"[RESAMPLE-DIAG] RMS before={rms_before:.6f}, after={rms_after:.6f}")
            
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
        Convert numpy audio array to 16-bit PCM bytes with normalization.
        
        Args:
            audio: Audio array (float32 in range [-1.0, 1.0])
            
        Returns:
            bytes: PCM audio data (16-bit signed integer)
        """
        # Ensure float32 dtype
        audio = audio.astype(np.float32)
        
        # Check amplitude and apply gain boost if too quiet (for Silero VAD)
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.0:
            if max_amplitude < 0.05:
                # Audio is very quiet, boost it for VAD detection
                gain = 0.3 / max_amplitude  # Target 0.3 peak amplitude
                gain = min(gain, 20.0)  # Limit max gain to 20x
                audio = audio * gain
                if self.metrics.frames_received <= 3:
                    self.logger.info(f"[NORMALIZE] Boosted quiet audio by {gain:.2f}x (was {max_amplitude:.6f})")
                    print(f"[NORMALIZE] Boosted quiet audio by {gain:.2f}x")
        
        # Clip to [-1.0, 1.0] range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to 16-bit signed integer
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Diagnostic logging (first few frames)
        if self.metrics.frames_received <= 3:
            rms_final = np.sqrt(np.mean(audio**2))
            max_final = np.max(np.abs(audio))
            self.logger.info(f"[PCM-DIAG] Final normalized: RMS={rms_final:.6f}, max={max_final:.6f}")
            print(f"[PCM-DIAG] Final: RMS={rms_final:.6f}, max={max_final:.6f}")
        
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
        self.utterance_start_time = time.time()  # Reset real-time tracker
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
