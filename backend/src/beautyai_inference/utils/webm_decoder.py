"""
WebM/Opus Audio Decoder Utility for BeautyAI Voice Processing.

This utility provides unified WebM and Opus audio decoding capabilities for
both real-time streaming and batch processing use cases. It extracts the
hardcoded FFmpeg processing logic from endpoints into a reusable service.

Key Features:
- Real-time streaming mode with async PCM chunk generation
- Batch mode for complete WebM file/chunk processing  
- Audio format detection and validation
- Configurable FFmpeg parameters for quality/performance tuning
- Proper resource management and cleanup
- Error handling and fallback strategies

Author: BeautyAI Framework
Date: 2025-08-19
"""

import asyncio
import logging
import shlex
import subprocess
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class WebMDecodingMode(Enum):
    """WebM decoding operation modes."""
    REALTIME_STREAM = "realtime_stream"  # Real-time FFmpeg subprocess streaming
    BATCH_CHUNKS = "batch_chunks"        # Batch process accumulated chunks
    BATCH_FILE = "batch_file"            # Process complete WebM file


class WebMDecodingError(Exception):
    """Raised when WebM decoding operations fail."""
    pass


class WebMDecoder:
    """
    Unified WebM/Opus decoder for BeautyAI voice processing.
    
    This class provides multiple decoding strategies optimized for different
    use cases:
    
    1. Real-time streaming: FFmpeg subprocess with continuous PCM output
    2. Batch chunk processing: Accumulate WebM chunks and decode as complete file
    3. Direct file processing: Convert WebM files to PCM/numpy arrays
    
    Examples:
        # Real-time streaming mode
        decoder = WebMDecoder()
        async for pcm_chunk in decoder.stream_realtime_pcm(webm_chunks):
            await audio_buffer.add_pcm(pcm_chunk)
        
        # Batch processing mode
        pcm_data = await decoder.decode_chunks_to_pcm(chunk_list)
        
        # File processing mode  
        audio_array = await decoder.decode_file_to_numpy(webm_path)
    """
    
    def __init__(self, 
                 target_sample_rate: int = 16000,
                 target_channels: int = 1,
                 chunk_size: int = 640,
                 ffmpeg_timeout: int = 30):
        """
        Initialize WebM decoder with audio processing parameters.
        
        Args:
            target_sample_rate: Target sample rate for output PCM (default: 16kHz for Whisper)
            target_channels: Target number of audio channels (default: 1 for mono)
            chunk_size: PCM chunk size in bytes (default: 640 = 20ms at 16kHz mono int16)
            ffmpeg_timeout: FFmpeg operation timeout in seconds
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.chunk_size = chunk_size
        self.ffmpeg_timeout = ffmpeg_timeout
        
        # Real-time streaming state
        self._ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self._streaming_active = False
        
        # Temporary file management
        self.temp_dir = Path(tempfile.gettempdir()) / "beautyai_webm_decoder"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"WebMDecoder initialized: {target_sample_rate}Hz, {target_channels}ch, chunk_size={chunk_size}")
    
    def detect_audio_format(self, audio_data: bytes) -> str:
        """
        Detect audio format from binary data headers.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Detected format string ('webm', 'wav', 'mp3', 'ogg', or 'unknown')
        """
        if not audio_data or len(audio_data) < 4:
            return "unknown"
        
        # WebM/Matroska header
        if audio_data[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        
        # WAV header  
        if audio_data[:4] == b"RIFF" and len(audio_data) >= 12 and audio_data[8:12] == b"WAVE":
            return "wav"
        
        # MP3 frame header (MPEG audio)
        if len(audio_data) >= 2 and audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0:
            return "mp3"
        
        # Ogg header
        if audio_data[:4] == b"OggS":
            return "ogg"
            
        # Default assumption for MediaRecorder streams
        logger.debug(f"Unknown audio format, first 16 bytes: {audio_data[:16].hex()}")
        return "webm"  # Most common for WebSocket streaming
    
    def is_webm_compatible(self, audio_format: str) -> bool:
        """
        Check if the audio format can be processed by this decoder.
        
        Args:
            audio_format: Format string from detect_audio_format()
            
        Returns:
            True if format is supported
        """
        supported_formats = {"webm", "ogg", "mp3", "wav"}
        return audio_format.lower() in supported_formats
    
    async def stream_realtime_pcm(self, 
                                  webm_chunks: AsyncGenerator[bytes, None],
                                  enable_logging: bool = True) -> AsyncGenerator[bytes, None]:
        """
        Real-time WebM decoding with streaming PCM output.
        
        This method spawns an FFmpeg subprocess and feeds WebM chunks to it,
        yielding PCM chunks as they become available. Optimized for low-latency
        streaming applications.
        
        Args:
            webm_chunks: Async generator yielding WebM chunk bytes
            enable_logging: Whether to log debug information
            
        Yields:
            PCM chunk bytes (target_sample_rate, mono, int16 little-endian)
            
        Raises:
            WebMDecodingError: If FFmpeg process fails or times out
        """
        try:
            # Spawn FFmpeg subprocess for real-time decoding
            await self._start_ffmpeg_process()
            
            # Start async tasks for feeding input and reading output
            feed_task = asyncio.create_task(self._feed_webm_chunks(webm_chunks, enable_logging))
            
            # Yield PCM chunks as they become available from _read_pcm_chunks
            async for pcm_chunk in self._read_pcm_chunks(enable_logging):
                yield pcm_chunk
                
        except Exception as e:
            logger.error(f"Real-time WebM decoding failed: {e}")
            raise WebMDecodingError(f"Streaming decode error: {e}")
        finally:
            await self._cleanup_ffmpeg_process()
    
    async def decode_chunks_to_pcm(self, 
                                   webm_chunks: List[bytes],
                                   preserve_header: bool = True) -> bytes:
        """
        Batch decode accumulated WebM chunks to complete PCM data.
        
        This method handles the MediaRecorder chunk accumulation pattern where
        only the first chunk contains WebM headers and subsequent chunks are
        raw data fragments.
        
        Args:
            webm_chunks: List of WebM chunk bytes (first chunk should contain header)
            preserve_header: Whether to preserve the first chunk header for reassembly
            
        Returns:
            Complete PCM data as bytes
            
        Raises:
            WebMDecodingError: If chunk processing or FFmpeg conversion fails
        """
        if not webm_chunks:
            raise WebMDecodingError("No WebM chunks provided")
        
        try:
            # Reassemble complete WebM data from chunks
            if preserve_header and len(webm_chunks) > 1:
                # MediaRecorder pattern: header + data fragments
                complete_webm = webm_chunks[0] + b''.join(webm_chunks[1:])
                logger.debug(f"Reassembled WebM from {len(webm_chunks)} chunks: {len(complete_webm)} bytes")
            else:
                # Simple concatenation
                complete_webm = b''.join(webm_chunks)
                logger.debug(f"Concatenated {len(webm_chunks)} chunks: {len(complete_webm)} bytes")
            
            # Create temporary WebM file
            webm_file = self.temp_dir / f"batch_{uuid.uuid4().hex}.webm"
            pcm_file = self.temp_dir / f"batch_{uuid.uuid4().hex}.pcm"
            
            try:
                # Write WebM data to file
                with open(webm_file, 'wb') as f:
                    f.write(complete_webm)
                
                # Convert to PCM using FFmpeg
                await self._convert_file_to_pcm(webm_file, pcm_file)
                
                # Read PCM result
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                logger.info(f"Batch decoded {len(webm_chunks)} chunks → {len(pcm_data)} PCM bytes")
                return pcm_data
                
            finally:
                # Cleanup temporary files
                webm_file.unlink(missing_ok=True)
                pcm_file.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Batch chunk decoding failed: {e}")
            raise WebMDecodingError(f"Batch decode error: {e}")
    
    async def decode_file_to_numpy(self, 
                                   webm_path: Union[str, Path],
                                   normalize: bool = True) -> np.ndarray:
        """
        Decode WebM file directly to numpy array for Whisper inference.
        
        Args:
            webm_path: Path to WebM file
            normalize: Whether to normalize audio to [-1, 1] range
            
        Returns:
            Audio as numpy array (float32, target_sample_rate, mono)
            
        Raises:
            WebMDecodingError: If file processing fails
        """
        webm_path = Path(webm_path)
        if not webm_path.exists():
            raise WebMDecodingError(f"WebM file not found: {webm_path}")
        
        try:
            # Create temporary PCM file
            pcm_file = self.temp_dir / f"numpy_{uuid.uuid4().hex}.pcm"
            
            try:
                # Convert to PCM
                await self._convert_file_to_pcm(webm_path, pcm_file)
                
                # Load PCM as numpy array
                with open(pcm_file, 'rb') as f:
                    pcm_bytes = f.read()
                
                # Convert int16 PCM to numpy array
                audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                
                if normalize:
                    # Normalize to [-1, 1] range for Whisper
                    audio_array = audio_array / 32768.0
                
                logger.info(f"Decoded WebM file to numpy: {webm_path.name} → {audio_array.shape} array")
                return audio_array
                
            finally:
                pcm_file.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"File to numpy conversion failed: {e}")
            raise WebMDecodingError(f"File decode error: {e}")
    
    async def _start_ffmpeg_process(self):
        """Start FFmpeg subprocess for real-time streaming."""
        if self._ffmpeg_process is not None:
            await self._cleanup_ffmpeg_process()
        
        # FFmpeg command for real-time WebM→PCM conversion
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",  # Input from stdin
            "-f", "s16le",   # Output format: signed 16-bit little-endian
            "-ac", str(self.target_channels),  # Audio channels
            "-ar", str(self.target_sample_rate),  # Sample rate
            "pipe:1"  # Output to stdout
        ]
        
        self._ffmpeg_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self._streaming_active = True
        logger.debug("FFmpeg real-time process started")
    
    async def _feed_webm_chunks(self, webm_chunks: AsyncGenerator[bytes, None], enable_logging: bool):
        """Feed WebM chunks to FFmpeg stdin."""
        try:
            if not self._ffmpeg_process or not self._ffmpeg_process.stdin:
                raise WebMDecodingError("FFmpeg process not available for input")
            
            chunk_count = 0
            async for chunk in webm_chunks:
                if not self._streaming_active:
                    break
                
                self._ffmpeg_process.stdin.write(chunk)
                await self._ffmpeg_process.stdin.drain()
                
                chunk_count += 1
                if enable_logging and chunk_count <= 5:
                    logger.debug(f"Fed WebM chunk {chunk_count}: {len(chunk)} bytes")
            
            # Close stdin to signal end of input
            self._ffmpeg_process.stdin.close()
            await self._ffmpeg_process.stdin.wait_closed()
            
        except Exception as e:
            logger.error(f"Error feeding WebM chunks: {e}")
            self._streaming_active = False
    
    async def _read_pcm_chunks(self, enable_logging: bool) -> AsyncGenerator[bytes, None]:
        """Read PCM chunks from FFmpeg stdout."""
        try:
            if not self._ffmpeg_process or not self._ffmpeg_process.stdout:
                raise WebMDecodingError("FFmpeg process not available for output")
            
            chunk_count = 0
            buffer = bytearray()
            
            while self._streaming_active:
                # Read data from FFmpeg
                try:
                    data = await asyncio.wait_for(
                        self._ffmpeg_process.stdout.read(4096),
                        timeout=self.ffmpeg_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg output timeout")
                    break
                
                if not data:
                    break  # EOF
                
                buffer.extend(data)
                
                # Yield complete chunks
                while len(buffer) >= self.chunk_size:
                    chunk = bytes(buffer[:self.chunk_size])
                    del buffer[:self.chunk_size]
                    
                    chunk_count += 1
                    if enable_logging and chunk_count <= 5:
                        logger.debug(f"Yielding PCM chunk {chunk_count}: {len(chunk)} bytes")
                    
                    yield chunk
            
            # Yield remaining data if any
            if buffer:
                yield bytes(buffer)
                
        except Exception as e:
            logger.error(f"Error reading PCM chunks: {e}")
            self._streaming_active = False
    
    async def _convert_file_to_pcm(self, input_path: Path, output_path: Path):
        """Convert audio file to PCM using FFmpeg."""
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-f", "s16le",   # Signed 16-bit little-endian
            "-ac", str(self.target_channels),
            "-ar", str(self.target_sample_rate),
            str(output_path)
        ]
        
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=self.ffmpeg_timeout
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown FFmpeg error"
                raise WebMDecodingError(f"FFmpeg conversion failed: {error_msg}")
            
            if not output_path.exists():
                raise WebMDecodingError(f"FFmpeg did not create output file: {output_path}")
            
        except asyncio.TimeoutError:
            raise WebMDecodingError(f"FFmpeg conversion timed out after {self.ffmpeg_timeout}s")
    
    async def _cleanup_ffmpeg_process(self):
        """Clean up FFmpeg subprocess and resources."""
        self._streaming_active = False
        
        if self._ffmpeg_process:
            try:
                if self._ffmpeg_process.returncode is None:
                    self._ffmpeg_process.terminate()
                    try:
                        await asyncio.wait_for(self._ffmpeg_process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self._ffmpeg_process.kill()
                        await self._ffmpeg_process.wait()
            except Exception as e:
                logger.warning(f"Error cleaning up FFmpeg process: {e}")
            finally:
                self._ffmpeg_process = None
        
        logger.debug("FFmpeg process cleanup completed")
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*"):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
            
            logger.debug("WebMDecoder cleanup completed")
        except Exception as e:
            logger.error(f"Error during WebMDecoder cleanup: {e}")

    def reset_for_new_utterance(self):
        """
        Reset decoder state for new utterance to prevent audio bleeding.
        
        This method ensures that each new audio stream starts with a clean
        decoder state, preventing accumulation of audio from previous utterances.
        """
        try:
            # Reset streaming state
            self._streaming_active = False
            
            # Clean up any active FFmpeg process
            if self._ffmpeg_process:
                logger.debug("Resetting WebM decoder: terminating existing FFmpeg process")
                try:
                    if self._ffmpeg_process.returncode is None:
                        self._ffmpeg_process.terminate()
                except Exception as e:
                    logger.warning(f"Error terminating FFmpeg during reset: {e}")
                finally:
                    self._ffmpeg_process = None
            
            # Clear temporary files from previous processing
            try:
                if self.temp_dir.exists():
                    for temp_file in self.temp_dir.glob("webm_temp_*"):
                        temp_file.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning temp files during reset: {e}")
                
            logger.debug("WebMDecoder reset for new utterance completed")
            
        except Exception as e:
            logger.error(f"Error during WebMDecoder reset: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get decoder statistics and configuration."""
        return {
            "target_sample_rate": self.target_sample_rate,
            "target_channels": self.target_channels,
            "chunk_size": self.chunk_size,
            "ffmpeg_timeout": self.ffmpeg_timeout,
            "temp_dir": str(self.temp_dir),
            "streaming_active": self._streaming_active,
            "temp_files_count": len(list(self.temp_dir.glob("*"))) if self.temp_dir.exists() else 0
        }


# Convenience factory functions

def create_realtime_decoder() -> WebMDecoder:
    """Create WebM decoder optimized for real-time streaming."""
    return WebMDecoder(
        target_sample_rate=16000,
        target_channels=1,
        chunk_size=640,  # 20ms at 16kHz mono
        ffmpeg_timeout=30
    )


def create_batch_decoder() -> WebMDecoder:
    """Create WebM decoder optimized for batch processing."""
    return WebMDecoder(
        target_sample_rate=16000,
        target_channels=1,
        chunk_size=1600,  # 50ms at 16kHz mono (larger chunks for batch)
        ffmpeg_timeout=60   # Longer timeout for batch processing
    )