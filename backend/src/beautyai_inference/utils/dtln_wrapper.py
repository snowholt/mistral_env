"""
DTLN (Dual-signal Transformation LSTM Network) Wrapper

Lightweight wrapper for DTLN noise suppression using ONNX Runtime.
Model: https://github.com/breizhn/DTLN

Author: BeautyAI Framework
Date: November 10, 2025
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. DTLN will not work.")


class DTLNProcessor:
    """
    DTLN noise suppression processor using ONNX models.
    
    DTLN uses dual-signal transformation with stacked LSTMs for real-time
    noise suppression. It processes 512-sample frames (32ms @ 16kHz).
    
    Key features:
    - Low latency: ~8-12ms
    - Low complexity: <1M parameters
    - Frame size: 512 samples @ 16kHz
    - Robust performance on stationary noise
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize DTLN processor.
        
        Args:
            model_path: Path to DTLN ONNX model. If None, downloads default model.
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime is required for DTLN. Install: pip install onnxruntime")
        
        self.frame_size = 512  # Fixed frame size for DTLN @ 16kHz
        self.sample_rate = 16000
        
        # For now, we'll use a simple spectral subtraction as fallback
        # In production, you'd download/load the ONNX model
        self.model_available = False
        self.states_1 = None
        self.states_2 = None
        
        logger.info("✅ DTLN processor initialized (using fallback spectral method)")
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single audio frame through DTLN.
        
        Args:
            audio_frame: Float32 audio samples, shape (512,) @ 16kHz
            
        Returns:
            Tuple of (denoised_audio, processing_time_ms)
        """
        import time
        start = time.time()
        
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"DTLN requires exactly {self.frame_size} samples, got {len(audio_frame)}")
        
        # Fallback: Simple spectral subtraction for now
        # TODO: Replace with actual ONNX model inference
        denoised = self._spectral_subtraction(audio_frame)
        
        processing_time = (time.time() - start) * 1000
        return denoised, processing_time
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback spectral subtraction method.
        This is a placeholder until ONNX model is loaded.
        """
        # Simple noise estimation and subtraction
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Estimate noise floor (bottom 20% of spectrum)
        noise_floor = np.percentile(magnitude, 20)
        
        # Subtract noise with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        magnitude_clean = np.maximum(magnitude - alpha * noise_floor, 0.1 * magnitude)
        
        # Reconstruct signal
        fft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = np.fft.irfft(fft_clean, n=len(audio))
        
        return audio_clean.astype(np.float32)
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process full audio stream by splitting into frames.
        
        Args:
            audio: Float32 audio array @ 16kHz
            
        Returns:
            Denoised audio array
        """
        if len(audio) < self.frame_size:
            # Pad if too short
            padded = np.zeros(self.frame_size, dtype=np.float32)
            padded[:len(audio)] = audio
            denoised, _ = self.process_frame(padded)
            return denoised[:len(audio)]
        
        # Process frame by frame
        output = []
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            
            # Pad last frame if needed
            if len(frame) < self.frame_size:
                padded = np.zeros(self.frame_size, dtype=np.float32)
                padded[:len(frame)] = frame
                frame = padded
            
            denoised, _ = self.process_frame(frame)
            output.append(denoised)
        
        return np.concatenate(output)[:len(audio)]


def download_dtln_model() -> Path:
    """
    Download DTLN ONNX model from official repository.
    
    Returns:
        Path to downloaded model file
    """
    import urllib.request
    
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "dtln.onnx"
    
    if model_path.exists():
        logger.info(f"DTLN model already exists: {model_path}")
        return model_path
    
    # DTLN model URL (you'd need to host this or download from GitHub releases)
    # For now, this is a placeholder
    logger.warning("DTLN ONNX model download not implemented yet. Using fallback.")
    
    return model_path
