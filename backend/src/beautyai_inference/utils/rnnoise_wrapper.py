"""
RNNoise Python wrapper using ctypes.
Provides noise suppression for 48kHz mono PCM audio.

Author: Lumina Ashley
Date: November 10, 2025
"""

import ctypes
import os
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RNNoiseError(Exception):
    """Exception raised for RNNoise-related errors."""
    pass


class RNNoiseProcessor:
    """
    Wrapper for xiph/rnnoise library.
    
    RNNoise expects:
    - Sample rate: 48kHz (fixed)
    - Frame size: 480 samples (10ms @ 48kHz)
    - Format: Mono float32 in range [-1, 1]
    
    Usage:
        processor = RNNoiseProcessor()
        denoised_audio = processor.process_audio(audio_48khz)
        processor.cleanup()
    """
    
    FRAME_SIZE = 480  # 10ms at 48kHz
    SAMPLE_RATE = 48000
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize RNNoise processor.
        
        Args:
            lib_path: Path to librnnoise.so. If None, uses default installation path.
        """
        if lib_path is None:
            # Default path relative to this file
            backend_dir = Path(__file__).parent.parent.parent.parent
            lib_path = backend_dir / "rnnoise" / "install" / "lib" / "librnnoise.so"
        
        if not os.path.exists(lib_path):
            raise RNNoiseError(f"RNNoise library not found at {lib_path}")
        
        # Load shared library
        try:
            self.lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise RNNoiseError(f"Failed to load RNNoise library: {e}")
        
        # Define function signatures
        self._setup_function_signatures()
        
        # Create denoiser state
        self.state = self.lib.rnnoise_create(None)
        if not self.state:
            raise RNNoiseError("Failed to create RNNoise state")
        
        logger.info(f"RNNoise initialized with library at {lib_path}")
    
    def _setup_function_signatures(self):
        """Define ctypes function signatures for RNNoise API."""
        # DenoiseState* rnnoise_create(RNNModel *model)
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        
        # void rnnoise_destroy(DenoiseState *st)
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_destroy.restype = None
        
        # float rnnoise_process_frame(DenoiseState *st, float *out, const float *in)
        self.lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Process a single 480-sample frame.
        
        Args:
            frame: Float32 array of 480 samples in range [-1, 1]
        
        Returns:
            Tuple of (denoised_frame, vad_probability)
            - denoised_frame: Float32 array of 480 samples
            - vad_probability: Voice activity probability [0, 1]
        """
        if len(frame) != self.FRAME_SIZE:
            raise ValueError(f"Frame must be {self.FRAME_SIZE} samples, got {len(frame)}")
        
        # Ensure float32
        frame = frame.astype(np.float32)
        
        # Create output buffer
        output = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        
        # Convert to ctypes pointers
        in_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Process frame (returns VAD probability)
        vad_prob = self.lib.rnnoise_process_frame(self.state, out_ptr, in_ptr)
        
        return output, float(vad_prob)
    
    def process_audio(self, audio: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """
        Process entire audio buffer.
        
        Args:
            audio: Float32 array at 48kHz, mono, range [-1, 1]
        
        Returns:
            Tuple of (denoised_audio, vad_probabilities)
            - denoised_audio: Denoised float32 array (same length as input)
            - vad_probabilities: List of VAD probability per frame
        """
        # Ensure correct format
        audio = audio.astype(np.float32)
        
        # Calculate number of complete frames
        num_frames = len(audio) // self.FRAME_SIZE
        
        if num_frames == 0:
            logger.warning(f"Audio too short for RNNoise: {len(audio)} samples < {self.FRAME_SIZE}")
            return audio, []
        
        # Process frame by frame
        denoised_frames = []
        vad_probs = []
        
        for i in range(num_frames):
            start = i * self.FRAME_SIZE
            end = start + self.FRAME_SIZE
            frame = audio[start:end]
            
            denoised_frame, vad_prob = self.process_frame(frame)
            denoised_frames.append(denoised_frame)
            vad_probs.append(vad_prob)
        
        # Concatenate all frames
        denoised_audio = np.concatenate(denoised_frames)
        
        # Handle remaining samples (pad with zeros if needed)
        remaining = len(audio) - len(denoised_audio)
        if remaining > 0:
            # Last incomplete frame - pad input, process, trim output
            last_frame = np.zeros(self.FRAME_SIZE, dtype=np.float32)
            last_frame[:remaining] = audio[-remaining:]
            
            denoised_last, vad_prob = self.process_frame(last_frame)
            denoised_audio = np.concatenate([denoised_audio, denoised_last[:remaining]])
            vad_probs.append(vad_prob)
        
        logger.debug(f"Processed {num_frames} frames, avg VAD: {np.mean(vad_probs):.3f}")
        
        return denoised_audio, vad_probs
    
    def cleanup(self):
        """Release RNNoise resources."""
        if hasattr(self, 'state') and self.state:
            self.lib.rnnoise_destroy(self.state)
            self.state = None
            logger.info("RNNoise resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        self.cleanup()


# Convenience function
def denoise_audio_rnnoise(audio_48khz: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """
    Convenience function to denoise audio with RNNoise.
    
    Args:
        audio_48khz: Float32 mono audio at 48kHz, range [-1, 1]
    
    Returns:
        Tuple of (denoised_audio, vad_probabilities)
    """
    with RNNoiseProcessor() as processor:
        return processor.process_audio(audio_48khz)
