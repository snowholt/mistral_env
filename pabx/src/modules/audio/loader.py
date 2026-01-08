"""
Audio file loader
Loads WAV, MP3, and other audio files for playback
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import wave
import struct

# Lazy import soundfile only when needed to avoid ALSA probe delays
SOUNDFILE_AVAILABLE = None

def _check_soundfile():
    """Check if soundfile is available (lazy check)"""
    global SOUNDFILE_AVAILABLE
    if SOUNDFILE_AVAILABLE is None:
        try:
            import soundfile as sf
            SOUNDFILE_AVAILABLE = True
        except ImportError:
            SOUNDFILE_AVAILABLE = False
    return SOUNDFILE_AVAILABLE


class AudioLoader:
    """
    Audio file loader with format conversion
    Supports WAV, MP3, FLAC, OGG (with soundfile)
    Falls back to basic WAV support without soundfile
    """
    
    @staticmethod
    def load(
        filepath: str,
        target_sample_rate: Optional[int] = None,
        target_channels: int = 1
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            filepath: Path to audio file
            target_sample_rate: Resample to this rate (None = keep original)
            target_channels: Convert to mono (1) or stereo (2)
            
        Returns:
            Tuple of (audio_data as int16 array, sample_rate)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        if _check_soundfile():
            return AudioLoader._load_with_soundfile(
                filepath, target_sample_rate, target_channels
            )
        else:
            return AudioLoader._load_wav_basic(
                filepath, target_sample_rate, target_channels
            )
    
    @staticmethod
    def _load_with_soundfile(
        filepath: Path,
        target_sample_rate: Optional[int],
        target_channels: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile (supports many formats)"""
        import soundfile as sf
        
        # Load audio
        data, sample_rate = sf.read(str(filepath), dtype='float32')
        
        # Convert to mono if needed
        if len(data.shape) > 1 and data.shape[1] > 1:
            if target_channels == 1:
                data = np.mean(data, axis=1)
            elif target_channels == 2 and data.shape[1] > 2:
                data = data[:, :2]
        
        # Resample if needed
        if target_sample_rate and target_sample_rate != sample_rate:
            data = AudioLoader._resample(data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        
        # Convert to int16
        data = (data * 32767).astype(np.int16)
        
        return data, sample_rate
    
    @staticmethod
    def _load_wav_basic(
        filepath: Path,
        target_sample_rate: Optional[int],
        target_channels: int
    ) -> Tuple[np.ndarray, int]:
        """Basic WAV loader without external dependencies"""
        
        with wave.open(str(filepath), 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            
            # Read all frames
            raw_data = wav.readframes(n_frames)
        
        # Parse based on sample width
        if sample_width == 1:
            # 8-bit unsigned
            data = np.frombuffer(raw_data, dtype=np.uint8)
            data = (data.astype(np.int16) - 128) * 256
        elif sample_width == 2:
            # 16-bit signed
            data = np.frombuffer(raw_data, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Reshape if stereo
        if channels > 1:
            data = data.reshape(-1, channels)
            
            if target_channels == 1:
                # Convert to mono
                data = np.mean(data, axis=1).astype(np.int16)
            elif target_channels == 2 and channels > 2:
                data = data[:, :2]
        
        # Resample if needed
        if target_sample_rate and target_sample_rate != sample_rate:
            data = AudioLoader._resample(data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        
        return data, sample_rate
    
    @staticmethod
    def _resample(
        data: np.ndarray,
        orig_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Simple resampling using linear interpolation
        For better quality, use scipy.signal.resample
        """
        if orig_rate == target_rate:
            return data
        
        # Calculate new length
        duration = len(data) / orig_rate
        target_length = int(duration * target_rate)
        
        # Create interpolation indices
        orig_indices = np.arange(len(data))
        target_indices = np.linspace(0, len(data) - 1, target_length)
        
        # Interpolate
        resampled = np.interp(target_indices, orig_indices, data)
        
        return resampled.astype(np.int16)
    
    @staticmethod
    def save_wav(
        filepath: str,
        audio_data: np.ndarray,
        sample_rate: int,
        channels: int = 1
    ):
        """
        Save audio data to WAV file
        
        Args:
            filepath: Output file path
            audio_data: Audio samples (int16)
            sample_rate: Sample rate in Hz
            channels: Number of channels
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if _check_soundfile():
            # Use soundfile if available
            import soundfile as sf
            audio_float = audio_data.astype(np.float32) / 32767.0
            sf.write(str(filepath), audio_float, sample_rate)
        else:
            # Use wave module
            with wave.open(str(filepath), 'wb') as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())
    
    @staticmethod
    def get_duration(filepath: str) -> float:
        """
        Get audio file duration in seconds
        
        Args:
            filepath: Audio file path
            
        Returns:
            Duration in seconds
        """
        if _check_soundfile():
            import soundfile as sf
            info = sf.info(filepath)
            return info.duration
        else:
            with wave.open(filepath, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
