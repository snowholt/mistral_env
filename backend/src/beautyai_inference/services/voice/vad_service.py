"""
Real-time Voice Activity Detection Service using Silero VAD.

This service provides real-time voice activity detection for streaming audio,
implementing server-side VAD-driven turn-taking for smooth voice conversations
like Gemini Live or GPT voice.

Features:
- Real-time VAD using Silero VAD model
- Audio chunk buffering and concatenation
- Configurable silence thresholds
- WebM/PCM audio format support
- Low-latency processing (20-30ms chunks)

Author: BeautyAI Framework
Date: 2025-01-08
"""

import asyncio
import logging
import tempfile
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import numpy as np
import torch
import torchaudio
from dataclasses import dataclass
from collections import deque
import threading
import io

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for VAD service."""
    chunk_size_ms: int = 30  # Audio chunk size in milliseconds
    silence_threshold_ms: int = 500  # Silence duration to trigger end-of-turn
    sampling_rate: int = 16000  # Sampling rate for VAD model
    speech_threshold: float = 0.5  # Speech probability threshold
    buffer_max_duration_ms: int = 30000  # Maximum buffer duration (30 seconds)
    vad_window_size_frames: int = 512  # VAD window size in frames


@dataclass
class AudioChunk:
    """Audio chunk with metadata."""
    data: np.ndarray
    timestamp: float
    is_speech: bool
    probability: float


class RealTimeVADService:
    """
    Real-time Voice Activity Detection Service.
    
    Processes streaming audio in real-time using Silero VAD to detect
    voice activity and implement server-side turn-taking logic.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize the VAD service.
        
        Args:
            config: VAD configuration object
        """
        self.config = config or VADConfig()
        self.logger = logging.getLogger(__name__)
        
        # VAD model and utilities
        self.vad_model = None
        self.vad_utils = None
        self.model_loaded = False
        
        # Audio processing state
        self.sampling_rate = self.config.sampling_rate
        self.chunk_size_samples = int(self.config.chunk_size_ms * self.sampling_rate / 1000)
        self.silence_threshold_samples = int(self.config.silence_threshold_ms * self.sampling_rate / 1000)
        
        # Buffering
        self.audio_buffer = deque(maxlen=1000)  # Ring buffer for audio chunks
        self.speech_chunks = []  # Current speech segment chunks
        self.silence_counter = 0
        self.is_speaking = False
        self.last_speech_time = 0
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None
        
        # Threading
        self._processing_lock = threading.Lock()
        self._stop_processing = False
        
        self.logger.info("RealTimeVADService initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize and load the VAD model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Loading Silero VAD model...")
            
            # Set number of threads for optimal performance
            torch.set_num_threads(1)
            
            # Load Silero VAD model via torch.hub
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                verbose=False
            )
            
            # Extract utilities
            (self.get_speech_timestamps, 
             self.save_audio, 
             self.read_audio, 
             self.VADIterator, 
             self.collect_chunks) = self.vad_utils
            
            self.model_loaded = True
            self.logger.info("✅ Silero VAD model loaded successfully")
            
            # Test the model with a dummy input
            await self._test_vad_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize VAD model: {e}")
            return False
    
    async def _test_vad_model(self):
        """Test the VAD model with dummy audio."""
        try:
            # Create dummy audio (1 second of silence)
            dummy_audio = torch.zeros(self.sampling_rate, dtype=torch.float32)
            
            # Test VAD prediction
            speech_timestamps = self.get_speech_timestamps(
                dummy_audio, 
                self.vad_model, 
                sampling_rate=self.sampling_rate
            )
            
            self.logger.info(f"VAD model test successful - detected {len(speech_timestamps)} speech segments in dummy audio")
            
        except Exception as e:
            self.logger.error(f"VAD model test failed: {e}")
            raise
    
    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_turn_complete: Optional[Callable] = None
    ):
        """
        Set callback functions for VAD events.
        
        Args:
            on_speech_start: Called when speech is detected
            on_speech_end: Called when speech ends
            on_turn_complete: Called when turn is complete (audio ready for processing)
        """
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_turn_complete = on_turn_complete
        
        self.logger.info("VAD callbacks registered")
    
    async def process_audio_chunk(self, audio_data: bytes, audio_format: str = "webm") -> Dict[str, Any]:
        """
        Process incoming audio chunk in real-time.
        
        Args:
            audio_data: Raw audio data
            audio_format: Format of audio data (webm, wav, etc.)
            
        Returns:
            Processing result with VAD information
        """
        if not self.model_loaded:
            return {"error": "VAD model not loaded"}
        
        try:
            start_time = time.time()
            
            # Convert audio data to tensor
            audio_tensor = await self._convert_audio_to_tensor(audio_data, audio_format)
            if audio_tensor is None:
                return {"error": "Failed to convert audio data"}
            
            # Process audio chunks
            results = []
            chunk_start = 0
            
            while chunk_start < len(audio_tensor):
                chunk_end = min(chunk_start + self.chunk_size_samples, len(audio_tensor))
                chunk = audio_tensor[chunk_start:chunk_end]
                
                # Pad chunk if too small
                if len(chunk) < self.chunk_size_samples:
                    padding = torch.zeros(self.chunk_size_samples - len(chunk))
                    chunk = torch.cat([chunk, padding])
                
                # Run VAD on chunk
                chunk_result = await self._process_single_chunk(chunk, time.time())
                results.append(chunk_result)
                
                chunk_start = chunk_end
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time_ms": int(processing_time * 1000),
                "chunks_processed": len(results),
                "current_state": {
                    "is_speaking": self.is_speaking,
                    "silence_duration_ms": self.silence_counter * self.config.chunk_size_ms,
                    "buffered_chunks": len(self.speech_chunks)
                },
                "chunks": results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    async def _convert_audio_to_tensor(self, audio_data: bytes, audio_format: str) -> Optional[torch.Tensor]:
        """
        Convert audio data to tensor format suitable for VAD.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (webm, wav, etc.)
            
        Returns:
            Audio tensor or None if conversion failed
        """
        try:
            # Create temporary file for audio conversion
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Load audio using torchaudio
                waveform, sample_rate = torchaudio.load(temp_path)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if necessary
                if sample_rate != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                    waveform = resampler(waveform)
                
                # Convert to 1D tensor
                audio_tensor = waveform.squeeze(0)
                
                self.logger.debug(f"Converted audio: {len(audio_data)} bytes -> {len(audio_tensor)} samples")
                return audio_tensor
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Failed to convert audio data: {e}")
            return None
    
    async def _process_single_chunk(self, chunk: torch.Tensor, timestamp: float) -> Dict[str, Any]:
        """
        Process a single audio chunk with VAD.
        
        Args:
            chunk: Audio chunk tensor
            timestamp: Timestamp of the chunk
            
        Returns:
            Chunk processing result
        """
        try:
            # Run VAD prediction on chunk
            with torch.no_grad():
                # Use the VAD model to get speech probability
                speech_prob = self.vad_model(chunk.unsqueeze(0), self.sampling_rate).item()
            
            is_speech = speech_prob > self.config.speech_threshold
            
            # Create audio chunk object
            audio_chunk = AudioChunk(
                data=chunk.numpy(),
                timestamp=timestamp,
                is_speech=is_speech,
                probability=speech_prob
            )
            
            # Add to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Update state and handle speech detection
            state_change = await self._update_speech_state(audio_chunk)
            
            return {
                "timestamp": timestamp,
                "is_speech": is_speech,
                "probability": speech_prob,
                "state_change": state_change,
                "buffer_size": len(self.audio_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing single chunk: {e}")
            return {
                "timestamp": timestamp,
                "is_speech": False,
                "probability": 0.0,
                "error": str(e)
            }
    
    async def _update_speech_state(self, chunk: AudioChunk) -> Optional[str]:
        """
        Update speech state based on current chunk and trigger callbacks.
        
        Args:
            chunk: Current audio chunk
            
        Returns:
            State change description if any
        """
        state_change = None
        
        if chunk.is_speech:
            # Speech detected
            self.speech_chunks.append(chunk)
            self.silence_counter = 0
            self.last_speech_time = chunk.timestamp
            
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                state_change = "speech_start"
                
                if self.on_speech_start:
                    try:
                        await self._safe_callback(self.on_speech_start)
                    except Exception as e:
                        self.logger.error(f"Error in speech_start callback: {e}")
        else:
            # No speech detected
            self.silence_counter += 1
            
            if self.is_speaking:
                # Check if silence threshold exceeded
                silence_duration_ms = self.silence_counter * self.config.chunk_size_ms
                
                if silence_duration_ms >= self.config.silence_threshold_ms:
                    # End of turn detected
                    self.is_speaking = False
                    state_change = "turn_complete"
                    
                    # Process accumulated speech
                    await self._handle_turn_complete()
                    
                    if self.on_speech_end:
                        try:
                            await self._safe_callback(self.on_speech_end)
                        except Exception as e:
                            self.logger.error(f"Error in speech_end callback: {e}")
        
        return state_change
    
    async def _handle_turn_complete(self):
        """Handle when a complete turn is detected."""
        if not self.speech_chunks:
            return
        
        try:
            # Concatenate all speech chunks
            audio_data = await self._concatenate_speech_chunks()
            
            # Save to temporary file
            audio_file_path = await self._save_concatenated_audio(audio_data)
            
            # Trigger turn complete callback
            if self.on_turn_complete:
                await self._safe_callback(self.on_turn_complete, audio_file_path, audio_data)
            
            # Clear speech buffer
            self.speech_chunks.clear()
            self.silence_counter = 0
            
            self.logger.info(f"Turn complete - saved {len(audio_data)} samples to {audio_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling turn complete: {e}")
    
    async def _concatenate_speech_chunks(self) -> np.ndarray:
        """
        Concatenate all buffered speech chunks into a single audio array.
        
        Returns:
            Concatenated audio data
        """
        if not self.speech_chunks:
            return np.array([])
        
        # Extract audio data from chunks
        audio_arrays = [chunk.data for chunk in self.speech_chunks]
        
        # Concatenate all chunks
        concatenated_audio = np.concatenate(audio_arrays)
        
        return concatenated_audio
    
    async def _save_concatenated_audio(self, audio_data: np.ndarray) -> str:
        """
        Save concatenated audio to a temporary file.
        
        Args:
            audio_data: Audio data to save
            
        Returns:
            Path to saved audio file
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Convert numpy array to tensor and save
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dimension
        torchaudio.save(temp_path, audio_tensor, self.sampling_rate)
        
        return temp_path
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Callback execution failed: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current VAD state information.
        
        Returns:
            Current state dictionary
        """
        return {
            "is_speaking": self.is_speaking,
            "silence_duration_ms": self.silence_counter * self.config.chunk_size_ms,
            "buffered_chunks": len(self.speech_chunks),
            "buffer_size": len(self.audio_buffer),
            "last_speech_time": self.last_speech_time,
            "model_loaded": self.model_loaded,
            "config": {
                "chunk_size_ms": self.config.chunk_size_ms,
                "silence_threshold_ms": self.config.silence_threshold_ms,
                "sampling_rate": self.config.sampling_rate,
                "speech_threshold": self.config.speech_threshold
            }
        }
    
    async def cleanup(self):
        """Clean up resources and stop processing."""
        self._stop_processing = True
        self.speech_chunks.clear()
        self.audio_buffer.clear()
        
        self.logger.info("VAD service cleaned up")


# Singleton instance for global access
_vad_service_instance: Optional[RealTimeVADService] = None


def get_vad_service(config: Optional[VADConfig] = None) -> RealTimeVADService:
    """
    Get the global VAD service instance.
    
    Args:
        config: VAD configuration (only used for first initialization)
        
    Returns:
        VAD service instance
    """
    global _vad_service_instance
    
    if _vad_service_instance is None:
        _vad_service_instance = RealTimeVADService(config)
    
    return _vad_service_instance


async def initialize_vad_service(config: Optional[VADConfig] = None) -> bool:
    """
    Initialize the global VAD service.
    
    Args:
        config: VAD configuration
        
    Returns:
        True if initialization successful
    """
    service = get_vad_service(config)
    return await service.initialize()
