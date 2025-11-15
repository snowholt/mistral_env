"""
Audio codec implementations
PCMU (G.711 μ-law), PCMA (G.711 A-law), and G.722
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class AudioCodec(ABC):
    """Base class for audio codecs"""
    
    @abstractmethod
    def encode(self, pcm_data: np.ndarray) -> bytes:
        """Encode PCM samples to codec format"""
        pass
    
    @abstractmethod
    def decode(self, encoded_data: bytes) -> np.ndarray:
        """Decode codec format to PCM samples"""
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get codec sample rate"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get codec name"""
        pass


class PCMUCodec(AudioCodec):
    """
    G.711 μ-law codec (PCMU)
    8kHz sample rate, logarithmic compression
    Used primarily in North America and Japan
    """
    
    # μ-law compression tables
    BIAS = 0x84
    CLIP = 32635
    
    # μ-law expansion table (256 entries)
    MULAW_DECODE_TABLE = None
    
    def __init__(self):
        """Initialize PCMU codec"""
        if PCMUCodec.MULAW_DECODE_TABLE is None:
            PCMUCodec.MULAW_DECODE_TABLE = self._build_decode_table()
    
    @property
    def sample_rate(self) -> int:
        return 8000
    
    @property
    def name(self) -> str:
        return "PCMU"
    
    def encode(self, pcm_data: np.ndarray) -> bytes:
        """
        Encode 16-bit PCM to μ-law
        
        Args:
            pcm_data: NumPy array of int16 PCM samples
            
        Returns:
            Encoded μ-law bytes
        """
        # Ensure int16 type
        pcm_data = pcm_data.astype(np.int16)
        
        # Allocate output
        encoded = np.zeros(len(pcm_data), dtype=np.uint8)
        
        for i, sample in enumerate(pcm_data):
            # Get sign and magnitude
            sign = 0 if sample >= 0 else 0x80
            if sample < 0:
                sample = -sample
            
            # Clip
            if sample > self.CLIP:
                sample = self.CLIP
            
            # Add bias
            sample = sample + self.BIAS
            
            # Find exponent
            exponent = 7
            for exp in range(7, -1, -1):
                if sample > (0xFF << exp):
                    exponent = exp
                    break
            
            # Mantissa (4 bits)
            mantissa = (sample >> (exponent + 3)) & 0x0F
            
            # Combine
            mulaw_byte = sign | (exponent << 4) | mantissa
            
            # Invert (per μ-law spec)
            encoded[i] = ~mulaw_byte & 0xFF
        
        return encoded.tobytes()
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        """
        Decode μ-law to 16-bit PCM
        
        Args:
            encoded_data: μ-law encoded bytes
            
        Returns:
            NumPy array of int16 PCM samples
        """
        # Use lookup table for speed
        encoded_array = np.frombuffer(encoded_data, dtype=np.uint8)
        decoded = np.zeros(len(encoded_array), dtype=np.int16)
        
        for i, byte in enumerate(encoded_array):
            decoded[i] = self.MULAW_DECODE_TABLE[byte]
        
        return decoded
    
    def _build_decode_table(self) -> np.ndarray:
        """Build μ-law decode lookup table"""
        table = np.zeros(256, dtype=np.int16)
        
        for i in range(256):
            # Invert
            mulaw = ~i & 0xFF
            
            # Extract components
            sign = mulaw & 0x80
            exponent = (mulaw >> 4) & 0x07
            mantissa = mulaw & 0x0F
            
            # Reconstruct sample
            sample = ((mantissa << 3) + self.BIAS) << exponent
            sample -= self.BIAS
            
            # Apply sign
            if sign:
                sample = -sample
            
            table[i] = sample
        
        return table


class PCMACodec(AudioCodec):
    """
    G.711 A-law codec (PCMA)
    8kHz sample rate, logarithmic compression
    Used primarily in Europe and rest of world
    """
    
    # A-law compression constants
    A = 87.6
    ALAW_MAX = 0xFFF
    
    # A-law decode table
    ALAW_DECODE_TABLE = None
    
    def __init__(self):
        """Initialize PCMA codec"""
        if PCMACodec.ALAW_DECODE_TABLE is None:
            PCMACodec.ALAW_DECODE_TABLE = self._build_decode_table()
    
    @property
    def sample_rate(self) -> int:
        return 8000
    
    @property
    def name(self) -> str:
        return "PCMA"
    
    def encode(self, pcm_data: np.ndarray) -> bytes:
        """
        Encode 16-bit PCM to A-law
        
        Args:
            pcm_data: NumPy array of int16 PCM samples
            
        Returns:
            Encoded A-law bytes
        """
        pcm_data = pcm_data.astype(np.int16)
        encoded = np.zeros(len(pcm_data), dtype=np.uint8)
        
        for i, sample in enumerate(pcm_data):
            # Get sign
            sign = 0 if sample >= 0 else 0x80
            if sample < 0:
                sample = -sample
            
            # Clip to 12 bits
            if sample > self.ALAW_MAX:
                sample = self.ALAW_MAX
            
            # Find segment
            if sample >= 256:
                exponent = 7
                for exp in range(7, 0, -1):
                    if sample >= (1 << (exp + 3)):
                        exponent = exp
                        break
                mantissa = (sample >> (exponent + 3)) & 0x0F
            else:
                exponent = 0
                mantissa = sample >> 4
            
            # Combine
            alaw_byte = sign | (exponent << 4) | mantissa
            
            # XOR with 0x55 (per A-law spec)
            encoded[i] = alaw_byte ^ 0x55
        
        return encoded.tobytes()
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        """
        Decode A-law to 16-bit PCM
        
        Args:
            encoded_data: A-law encoded bytes
            
        Returns:
            NumPy array of int16 PCM samples
        """
        encoded_array = np.frombuffer(encoded_data, dtype=np.uint8)
        decoded = np.zeros(len(encoded_array), dtype=np.int16)
        
        for i, byte in enumerate(encoded_array):
            decoded[i] = self.ALAW_DECODE_TABLE[byte]
        
        return decoded
    
    def _build_decode_table(self) -> np.ndarray:
        """Build A-law decode lookup table"""
        table = np.zeros(256, dtype=np.int16)
        
        for i in range(256):
            # XOR with 0x55
            alaw = i ^ 0x55
            
            # Extract components
            sign = alaw & 0x80
            exponent = (alaw >> 4) & 0x07
            mantissa = alaw & 0x0F
            
            # Reconstruct sample
            if exponent > 0:
                sample = ((mantissa << 4) | 0x08) << (exponent + 2)
            else:
                sample = (mantissa << 4) | 0x08
            
            # Apply sign
            if sign:
                sample = -sample
            
            table[i] = sample
        
        return table


class G722Codec(AudioCodec):
    """
    G.722 codec (wideband)
    16kHz sample rate, sub-band ADPCM
    
    Note: This is a simplified implementation.
    For production use, consider using external library (e.g., opus, ffmpeg)
    """
    
    def __init__(self):
        """Initialize G.722 codec"""
        self._warned = False
    
    @property
    def sample_rate(self) -> int:
        return 16000
    
    @property
    def name(self) -> str:
        return "G.722"
    
    def encode(self, pcm_data: np.ndarray) -> bytes:
        """
        Simplified G.722 encoding
        
        For production, use external library like ffmpeg
        This is a placeholder that does simple downsampling
        
        Args:
            pcm_data: NumPy array of int16 PCM samples at 16kHz
            
        Returns:
            Encoded bytes (simplified)
        """
        if not self._warned:
            print("Warning: Using simplified G.722 encoding. For production, use ffmpeg.")
            self._warned = True
        
        # Simple approach: downsample to 8-bit and compress 2:1
        # Real G.722 uses sub-band ADPCM
        pcm_data = pcm_data.astype(np.int16)
        
        # Normalize to 8-bit range
        normalized = (pcm_data / 256).astype(np.int8)
        
        # Pack 2 samples into 1 byte (simplified)
        packed = []
        for i in range(0, len(normalized) - 1, 2):
            high = (normalized[i] >> 4) & 0x0F
            low = (normalized[i + 1] >> 4) & 0x0F
            packed.append((high << 4) | low)
        
        return bytes(packed)
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        """
        Simplified G.722 decoding
        
        Args:
            encoded_data: Encoded bytes
            
        Returns:
            NumPy array of int16 PCM samples at 16kHz
        """
        if not self._warned:
            print("Warning: Using simplified G.722 decoding. For production, use ffmpeg.")
            self._warned = True
        
        # Unpack bytes to samples (reverse of encode)
        decoded = []
        for byte in encoded_data:
            high = ((byte >> 4) & 0x0F) << 4
            low = (byte & 0x0F) << 4
            
            # Sign extend 4-bit to 8-bit
            if high & 0x80:
                high |= 0xF0
            if low & 0x80:
                low |= 0xF0
            
            decoded.append(high * 256)
            decoded.append(low * 256)
        
        return np.array(decoded, dtype=np.int16)


# Codec factory
def get_codec(payload_type: int) -> Optional[AudioCodec]:
    """
    Get codec instance by payload type
    
    Args:
        payload_type: RTP payload type number
        
    Returns:
        Codec instance or None if not supported
    """
    codecs = {
        0: PCMUCodec,
        8: PCMACodec,
        9: G722Codec,
    }
    
    codec_class = codecs.get(payload_type)
    if codec_class:
        return codec_class()
    return None
