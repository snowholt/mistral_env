"""
Audio processing utilities
Format conversion, effects, and quality analysis
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from .codecs import get_codec, AudioCodec
from .loader import AudioLoader


class AudioProcessor:
    """
    Audio processing and conversion utilities
    """
    
    @staticmethod
    def convert_sample_rate(
        audio_data: np.ndarray,
        from_rate: int,
        to_rate: int
    ) -> np.ndarray:
        """
        Convert audio sample rate
        
        Args:
            audio_data: Input audio samples
            from_rate: Original sample rate
            to_rate: Target sample rate
            
        Returns:
            Resampled audio
        """
        if from_rate == to_rate:
            return audio_data
        
        # Simple linear interpolation
        # For better quality, use scipy.signal.resample
        duration = len(audio_data) / from_rate
        target_length = int(duration * to_rate)
        
        indices = np.linspace(0, len(audio_data) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        return resampled.astype(np.int16)
    
    @staticmethod
    def normalize(
        audio_data: np.ndarray,
        target_level: float = 0.9
    ) -> np.ndarray:
        """
        Normalize audio to target level
        
        Args:
            audio_data: Input audio samples
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio
        """
        # Find current peak
        peak = np.max(np.abs(audio_data))
        
        if peak == 0:
            return audio_data
        
        # Calculate scaling factor
        scale = (target_level * 32767) / peak
        
        # Apply scaling
        normalized = audio_data * scale
        
        # Clip to prevent overflow
        return np.clip(normalized, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def apply_gain(
        audio_data: np.ndarray,
        gain_db: float
    ) -> np.ndarray:
        """
        Apply gain in decibels
        
        Args:
            audio_data: Input audio samples
            gain_db: Gain in dB (positive to increase, negative to decrease)
            
        Returns:
            Audio with gain applied
        """
        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        gained = audio_data * gain_linear
        
        # Clip to prevent overflow
        return np.clip(gained, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def fade_in(
        audio_data: np.ndarray,
        duration_ms: int = 50
    ) -> np.ndarray:
        """
        Apply fade-in effect
        
        Args:
            audio_data: Input audio samples
            duration_ms: Fade duration in milliseconds
            
        Returns:
            Audio with fade-in
        """
        fade_samples = min(len(audio_data), duration_ms * 8)  # Assume 8kHz
        
        # Create fade curve
        fade_curve = np.linspace(0, 1, fade_samples)
        
        # Apply fade
        result = audio_data.copy().astype(np.float32)
        result[:fade_samples] *= fade_curve
        
        return result.astype(np.int16)
    
    @staticmethod
    def fade_out(
        audio_data: np.ndarray,
        duration_ms: int = 50
    ) -> np.ndarray:
        """
        Apply fade-out effect
        
        Args:
            audio_data: Input audio samples
            duration_ms: Fade duration in milliseconds
            
        Returns:
            Audio with fade-out
        """
        fade_samples = min(len(audio_data), duration_ms * 8)  # Assume 8kHz
        
        # Create fade curve
        fade_curve = np.linspace(1, 0, fade_samples)
        
        # Apply fade
        result = audio_data.copy().astype(np.float32)
        result[-fade_samples:] *= fade_curve
        
        return result.astype(np.int16)
    
    @staticmethod
    def mix_audio(
        audio1: np.ndarray,
        audio2: np.ndarray,
        mix_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Mix two audio signals
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            mix_ratio: Mixing ratio (0.0 = all audio1, 1.0 = all audio2)
            
        Returns:
            Mixed audio
        """
        # Pad shorter audio with zeros
        max_len = max(len(audio1), len(audio2))
        
        if len(audio1) < max_len:
            audio1 = np.pad(audio1, (0, max_len - len(audio1)))
        if len(audio2) < max_len:
            audio2 = np.pad(audio2, (0, max_len - len(audio2)))
        
        # Mix
        mixed = (
            audio1.astype(np.float32) * (1 - mix_ratio) +
            audio2.astype(np.float32) * mix_ratio
        )
        
        # Clip to prevent overflow
        return np.clip(mixed, -32767, 32767).astype(np.int16)
    
    @staticmethod
    def encode_to_codec(
        audio_data: np.ndarray,
        payload_type: int
    ) -> Optional[bytes]:
        """
        Encode PCM audio to codec format
        
        Args:
            audio_data: PCM audio samples (int16)
            payload_type: Target codec payload type
            
        Returns:
            Encoded bytes or None if codec not supported
        """
        codec = get_codec(payload_type)
        if not codec:
            return None
        
        return codec.encode(audio_data)
    
    @staticmethod
    def decode_from_codec(
        encoded_data: bytes,
        payload_type: int
    ) -> Optional[np.ndarray]:
        """
        Decode codec format to PCM audio
        
        Args:
            encoded_data: Encoded audio bytes
            payload_type: Source codec payload type
            
        Returns:
            PCM audio samples (int16) or None if codec not supported
        """
        codec = get_codec(payload_type)
        if not codec:
            return None
        
        return codec.decode(encoded_data)
    
    @staticmethod
    def transcode_file(
        input_file: str,
        output_file: str,
        target_codec: int,
        target_sample_rate: int = 8000
    ):
        """
        Transcode audio file to codec format
        
        Args:
            input_file: Input audio file path
            output_file: Output file path (will be saved as raw codec data)
            target_codec: Target codec payload type
            target_sample_rate: Target sample rate
        """
        # Load audio
        audio_data, sample_rate = AudioLoader.load(
            input_file,
            target_sample_rate=target_sample_rate,
            target_channels=1
        )
        
        # Encode to codec
        codec = get_codec(target_codec)
        if not codec:
            raise ValueError(f"Unsupported codec: {target_codec}")
        
        encoded = codec.encode(audio_data)
        
        # Save encoded data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(encoded)
    
    @staticmethod
    def calculate_metrics(
        audio_data: np.ndarray,
        sample_rate: int = 8000
    ) -> dict:
        """
        Calculate audio quality metrics
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of metrics
        """
        # Peak amplitude
        peak = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak / 32767) if peak > 0 else -np.inf
        
        # RMS level
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        rms_db = 20 * np.log10(rms / 32767) if rms > 0 else -np.inf
        
        # Crest factor
        crest_factor = peak / rms if rms > 0 else 0
        crest_factor_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0
        
        # Duration
        duration = len(audio_data) / sample_rate
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zcr = zero_crossings / len(audio_data)
        
        return {
            'duration_seconds': duration,
            'sample_count': len(audio_data),
            'sample_rate': sample_rate,
            'peak_amplitude': int(peak),
            'peak_db': float(peak_db),
            'rms_amplitude': float(rms),
            'rms_db': float(rms_db),
            'crest_factor': float(crest_factor),
            'crest_factor_db': float(crest_factor_db),
            'zero_crossing_rate': float(zcr),
        }
    
    @staticmethod
    def split_into_frames(
        audio_data: np.ndarray,
        frame_size: int,
        overlap: int = 0
    ) -> list:
        """
        Split audio into frames for packet transmission
        
        Args:
            audio_data: Audio samples
            frame_size: Frame size in samples
            overlap: Overlap between frames
            
        Returns:
            List of audio frames
        """
        frames = []
        step = frame_size - overlap
        
        for i in range(0, len(audio_data) - frame_size + 1, step):
            frame = audio_data[i:i + frame_size]
            frames.append(frame)
        
        # Handle remainder
        if len(audio_data) % step != 0:
            remainder = audio_data[-(len(audio_data) % step):]
            # Pad with zeros
            padded = np.pad(remainder, (0, frame_size - len(remainder)))
            frames.append(padded)
        
        return frames
