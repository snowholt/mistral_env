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
    
    # ENHANCED: Adaptive threshold settings
    adaptive_threshold: bool = True  # Enable adaptive threshold
    min_speech_duration_ms: int = 300  # Minimum speech duration to register
    max_silence_duration_ms: int = 1500  # Maximum silence before force end
    pre_speech_buffer_ms: int = 200  # Buffer to capture speech beginning
    
    # ENHANCED: Language-specific settings
    language_specific_thresholds: Dict[str, float] = None  # Per-language thresholds
    arabic_speech_threshold: float = 0.45  # Lower threshold for Arabic
    english_speech_threshold: float = 0.5  # Standard threshold for English
    
    # ENHANCED: Environmental adaptation
    noise_adaptation: bool = True  # Enable noise level adaptation
    energy_threshold_multiplier: float = 2.0  # Energy threshold multiplier
    ambient_noise_window_ms: int = 2000  # Window for ambient noise calculation
    
    def __post_init__(self):
        """Initialize language-specific thresholds if not provided."""
        if self.language_specific_thresholds is None:
            self.language_specific_thresholds = {
                'ar': self.arabic_speech_threshold,
                'arabic': self.arabic_speech_threshold,
                'en': self.english_speech_threshold,
                'english': self.english_speech_threshold,
                'auto': self.speech_threshold
            }


@dataclass
class AudioChunk:
    """Audio chunk with metadata."""
    data: np.ndarray
    timestamp: float
    is_speech: bool
    probability: float
    energy_level: float = 0.0  # ENHANCED: Audio energy level
    language_hint: Optional[str] = None  # ENHANCED: Language hint for threshold adaptation


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
        self.turn_being_processed = False  # Prevent multiple turn completions
        self.last_turn_timestamp = 0  # Track last turn completion time
        
        # ENHANCED: Adaptive threshold state
        self.current_language = 'auto'
        self.adaptive_threshold = self.config.speech_threshold
        self.ambient_noise_level = 0.0
        self.ambient_noise_samples = deque(maxlen=int(self.config.ambient_noise_window_ms / self.config.chunk_size_ms))
        self.pre_speech_buffer = deque(maxlen=int(self.config.pre_speech_buffer_ms / self.config.chunk_size_ms))
        
        # ENHANCED: Speech timing validation
        self.speech_start_time = 0
        self.continuous_speech_duration = 0
        self.continuous_silence_duration = 0
        
        # ENHANCED: Energy-based detection
        self.energy_threshold = 0.01  # Will be adapted based on ambient noise
        self.energy_window = deque(maxlen=50)  # Rolling window for energy calculation
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None
        
        # Threading
        self._processing_lock = threading.Lock()
        self._stop_processing = False
        
        self.logger.info("Enhanced RealTimeVADService initialized with adaptive features")
    
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
    
    def set_language_context(self, language: str):
        """
        Set language context for adaptive threshold adjustment.
        
        Args:
            language: Language code ('ar', 'en', 'auto')
        """
        old_language = self.current_language
        self.current_language = language.lower()
        
        # Update adaptive threshold based on language
        if self.config.adaptive_threshold:
            self.adaptive_threshold = self.config.language_specific_thresholds.get(
                self.current_language, 
                self.config.speech_threshold
            )
            
            self.logger.info(f"Language context updated: {old_language} -> {self.current_language}, threshold: {self.adaptive_threshold}")
    
    def _calculate_audio_energy(self, audio_data: np.ndarray) -> float:
        """
        Calculate the energy level of audio data.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Energy level (RMS)
        """
        if len(audio_data) == 0:
            return 0.0
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        return float(rms_energy)
    
    def _update_ambient_noise_level(self, energy: float, is_speech: bool):
        """
        Update the ambient noise level estimation.
        
        Args:
            energy: Current audio energy
            is_speech: Whether current chunk contains speech
        """
        # Only use non-speech segments for ambient noise calculation
        if not is_speech:
            self.ambient_noise_samples.append(energy)
            
            if len(self.ambient_noise_samples) > 10:  # Need sufficient samples
                # Use median to avoid outliers
                self.ambient_noise_level = np.median(list(self.ambient_noise_samples))
                
                # Adapt energy threshold based on ambient noise
                if self.config.noise_adaptation:
                    self.energy_threshold = max(
                        0.005,  # Minimum threshold
                        self.ambient_noise_level * self.config.energy_threshold_multiplier
                    )
    
    def _should_register_as_speech(self, probability: float, energy: float, duration_ms: float) -> bool:
        """
        Enhanced speech detection logic combining VAD probability, energy, and duration.
        
        Args:
            probability: VAD model probability
            energy: Audio energy level
            duration_ms: Duration of continuous speech
            
        Returns:
            True if should be registered as speech
        """
        # Basic VAD threshold check with language adaptation
        vad_check = probability > self.adaptive_threshold
        
        # Energy-based check (helps with low-volume speech)
        energy_check = energy > self.energy_threshold
        
        # Duration check (prevents very short spurious detections)
        duration_check = duration_ms >= self.config.min_speech_duration_ms or self.is_speaking
        
        # Combine all checks
        if self.config.adaptive_threshold:
            # More sophisticated logic when adaptive mode is enabled
            if vad_check and energy_check:
                return duration_check
            elif vad_check and duration_ms > self.config.min_speech_duration_ms:
                # Strong VAD signal can override weak energy
                return True
            elif energy_check and probability > (self.adaptive_threshold * 0.7):
                # Strong energy with reasonable VAD can work
                return duration_check
            else:
                return False
        else:
            # Simple mode - just use VAD threshold
            return vad_check and duration_check
    
    def _detect_speech_end(self, silence_duration_ms: float) -> bool:
        """
        Enhanced speech end detection with multiple criteria.
        
        Args:
            silence_duration_ms: Current silence duration
            
        Returns:
            True if speech end should be triggered
        """
        # Basic silence threshold
        basic_threshold_met = silence_duration_ms >= self.config.silence_threshold_ms
        
        # Force end if silence is too long
        force_end = silence_duration_ms >= self.config.max_silence_duration_ms
        
        # Consider speech duration - longer speech gets more patience
        if self.continuous_speech_duration > 0:
            # Give more time for longer utterances
            patience_factor = min(2.0, 1.0 + (self.continuous_speech_duration / 5000))  # Up to 2x patience for 5s+ speech
            adapted_threshold = self.config.silence_threshold_ms * patience_factor
            adapted_threshold_met = silence_duration_ms >= adapted_threshold
        else:
            adapted_threshold_met = basic_threshold_met
        
        return force_end or adapted_threshold_met
    
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
        Process a single audio chunk with enhanced VAD.
        
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
            
            # ENHANCED: Calculate audio energy
            chunk_numpy = chunk.numpy()
            energy_level = self._calculate_audio_energy(chunk_numpy)
            
            # Add energy to rolling window
            self.energy_window.append(energy_level)
            
            # ENHANCED: Use sophisticated speech detection
            current_time = time.time()
            
            # Calculate continuous speech/silence duration
            if self.is_speaking:
                self.continuous_speech_duration = (current_time - self.speech_start_time) * 1000
                self.continuous_silence_duration = 0
            else:
                self.continuous_silence_duration = self.silence_counter * self.config.chunk_size_ms
            
            # Enhanced speech detection
            is_speech = self._should_register_as_speech(
                speech_prob, 
                energy_level, 
                self.continuous_speech_duration
            )
            
            # Update ambient noise level
            self._update_ambient_noise_level(energy_level, is_speech)
            
            # Create enhanced audio chunk object
            audio_chunk = AudioChunk(
                data=chunk_numpy,
                timestamp=timestamp,
                is_speech=is_speech,
                probability=speech_prob,
                energy_level=energy_level,
                language_hint=self.current_language
            )
            
            # Add to pre-speech buffer for capturing speech beginning
            if self.config.pre_speech_buffer_ms > 0:
                self.pre_speech_buffer.append(audio_chunk)
            
            # Add to main buffer
            self.audio_buffer.append(audio_chunk)
            
            # Update state and handle speech detection
            state_change = await self._update_speech_state_enhanced(audio_chunk)
            
            return {
                "timestamp": timestamp,
                "is_speech": is_speech,
                "probability": speech_prob,
                "energy_level": energy_level,
                "adaptive_threshold": self.adaptive_threshold,
                "ambient_noise": self.ambient_noise_level,
                "continuous_speech_ms": self.continuous_speech_duration,
                "continuous_silence_ms": self.continuous_silence_duration,
                "state_change": state_change,
                "buffer_size": len(self.audio_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing single chunk: {e}")
            return {
                "timestamp": timestamp,
                "is_speech": False,
                "probability": 0.0,
                "energy_level": 0.0,
                "error": str(e)
            }
    
    async def _update_speech_state_enhanced(self, chunk: AudioChunk) -> Optional[str]:
        """
        Enhanced speech state update with better detection logic.
        
        Args:
            chunk: Current audio chunk
            
        Returns:
            State change description if any
        """
        state_change = None
        
        # Don't process new chunks if a turn is being processed
        if self.turn_being_processed:
            return None
        
        current_time = time.time()
        
        if chunk.is_speech:
            # Speech detected
            
            # Add pre-speech buffer if this is start of speech
            if not self.is_speaking and self.config.pre_speech_buffer_ms > 0:
                # Add buffered chunks from before speech started
                for buffered_chunk in self.pre_speech_buffer:
                    if buffered_chunk not in self.speech_chunks:
                        self.speech_chunks.append(buffered_chunk)
            
            self.speech_chunks.append(chunk)
            self.silence_counter = 0
            self.last_speech_time = chunk.timestamp
            
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = current_time
                self.continuous_speech_duration = 0
                state_change = "speech_start"
                
                self.logger.debug(f"Speech started - threshold: {self.adaptive_threshold}, prob: {chunk.probability}, energy: {chunk.energy_level}")
                
                if self.on_speech_start:
                    try:
                        await self._safe_callback(self.on_speech_start)
                    except Exception as e:
                        self.logger.error(f"Error in speech_start callback: {e}")
        else:
            # No speech detected
            self.silence_counter += 1
            
            if self.is_speaking:
                # Check if silence threshold exceeded using enhanced logic
                silence_duration_ms = self.silence_counter * self.config.chunk_size_ms
                
                if self._detect_speech_end(silence_duration_ms):
                    # End of turn detected
                    
                    # Check minimum speech duration
                    speech_duration = (current_time - self.speech_start_time) * 1000
                    if speech_duration < self.config.min_speech_duration_ms:
                        self.logger.debug(f"Speech too short ({speech_duration:.0f}ms < {self.config.min_speech_duration_ms}ms), ignoring")
                        # Reset speech state but don't trigger turn complete
                        self.is_speaking = False
                        self.speech_chunks.clear()
                        self.silence_counter = 0
                        return "speech_cancelled"
                    
                    # Prevent rapid duplicate turn completions
                    if current_time - self.last_turn_timestamp < 1.0:  # 1 second minimum between turns
                        self.logger.warning(f"Turn completion too soon - ignoring (last: {self.last_turn_timestamp}, current: {current_time})")
                        return None
                    
                    self.is_speaking = False
                    self.turn_being_processed = True  # Mark turn as being processed
                    self.last_turn_timestamp = current_time
                    state_change = "turn_complete"
                    
                    self.logger.info(f"Turn complete detected - speech: {speech_duration:.0f}ms, silence: {silence_duration_ms}ms")
                    
                    # Process accumulated speech
                    await self._handle_turn_complete()
                    
                    if self.on_speech_end:
                        try:
                            await self._safe_callback(self.on_speech_end)
                        except Exception as e:
                            self.logger.error(f"Error in speech_end callback: {e}")
        
        return state_change
        """
        Update speech state based on current chunk and trigger callbacks.
        
        Args:
            chunk: Current audio chunk
            
        Returns:
            State change description if any
        """
        state_change = None
        
        # 🛡️ CRITICAL FIX: Don't process new chunks if a turn is being processed
        if self.turn_being_processed:
            return None
        
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
                    current_time = time.time()
                    
                    # 🛡️ CRITICAL FIX: Prevent rapid duplicate turn completions
                    if current_time - self.last_turn_timestamp < 1.0:  # 1 second minimum between turns
                        self.logger.warning(f"🚫 Turn completion too soon - ignoring (last: {self.last_turn_timestamp}, current: {current_time})")
                        return None
                    
                    self.is_speaking = False
                    self.turn_being_processed = True  # Mark turn as being processed
                    self.last_turn_timestamp = current_time
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
            # Reset turn processing state if no chunks
            self.turn_being_processed = False
            return
        
        try:
            # Concatenate all speech chunks
            audio_data = await self._concatenate_speech_chunks()
            
            # Save to temporary file
            audio_file_path = await self._save_concatenated_audio(audio_data)
            
            # Trigger turn complete callback
            if self.on_turn_complete:
                await self._safe_callback(self.on_turn_complete, audio_file_path, audio_data)
            
            # Clear speech buffer for next turn
            self.speech_chunks.clear()
            self.silence_counter = 0
            
            self.logger.info(f"Turn complete - saved {len(audio_data)} samples to {audio_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling turn complete: {e}")
        finally:
            # Always reset turn processing state after completion
            # Note: This will be reset by the WebSocket handler after processing
            pass
    
    def reset_turn_processing(self):
        """Reset turn processing state to allow new turns."""
        self.turn_being_processed = False
        self.logger.debug("Turn processing state reset")
    
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
        Get current enhanced VAD state information.
        
        Returns:
            Current state dictionary with enhanced information
        """
        return {
            "is_speaking": self.is_speaking,
            "silence_duration_ms": self.silence_counter * self.config.chunk_size_ms,
            "buffered_chunks": len(self.speech_chunks),
            "buffer_size": len(self.audio_buffer),
            "last_speech_time": self.last_speech_time,
            "model_loaded": self.model_loaded,
            "turn_being_processed": self.turn_being_processed,
            "last_turn_timestamp": self.last_turn_timestamp,
            
            # ENHANCED: Adaptive features
            "current_language": self.current_language,
            "adaptive_threshold": self.adaptive_threshold,
            "ambient_noise_level": self.ambient_noise_level,
            "energy_threshold": self.energy_threshold,
            "continuous_speech_duration_ms": self.continuous_speech_duration,
            "continuous_silence_duration_ms": self.continuous_silence_duration,
            "pre_speech_buffer_size": len(self.pre_speech_buffer),
            
            # Performance metrics
            "average_energy": np.mean(list(self.energy_window)) if self.energy_window else 0.0,
            "energy_window_size": len(self.energy_window),
            "ambient_noise_samples": len(self.ambient_noise_samples),
            
            "config": {
                "chunk_size_ms": self.config.chunk_size_ms,
                "silence_threshold_ms": self.config.silence_threshold_ms,
                "sampling_rate": self.config.sampling_rate,
                "speech_threshold": self.config.speech_threshold,
                "adaptive_threshold_enabled": self.config.adaptive_threshold,
                "min_speech_duration_ms": self.config.min_speech_duration_ms,
                "max_silence_duration_ms": self.config.max_silence_duration_ms,
                "pre_speech_buffer_ms": self.config.pre_speech_buffer_ms,
                "noise_adaptation_enabled": self.config.noise_adaptation,
                "language_specific_thresholds": self.config.language_specific_thresholds
            }
        }
    
    async def cleanup(self):
        """Clean up resources and stop processing."""
        self._stop_processing = True
        self.speech_chunks.clear()
        self.audio_buffer.clear()
        self.turn_being_processed = False
        self.silence_counter = 0
        self.is_speaking = False
        
        # ENHANCED: Clean up additional state
        self.pre_speech_buffer.clear()
        self.energy_window.clear()
        self.ambient_noise_samples.clear()
        self.continuous_speech_duration = 0
        self.continuous_silence_duration = 0
        self.speech_start_time = 0
        
        self.logger.info("Enhanced VAD service cleaned up")


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
