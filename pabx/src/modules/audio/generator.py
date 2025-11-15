"""
Audio tone and signal generators
Generates test tones, DTMF signals, and other audio patterns
"""

import numpy as np
from typing import List, Tuple


class ToneGenerator:
    """
    Generate audio test tones
    Useful for testing audio pipeline and quality
    """
    
    @staticmethod
    def generate_sine(
        frequency: float,
        duration: float,
        sample_rate: int = 8000,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate sine wave tone
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)
            
        Returns:
            Audio samples as int16 array
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate sine wave
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16
        return (signal * 32767).astype(np.int16)
    
    @staticmethod
    def generate_multitone(
        frequencies: List[float],
        duration: float,
        sample_rate: int = 8000,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate multiple sine tones mixed together
        
        Args:
            frequencies: List of frequencies in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude per tone (0.0 to 1.0)
            
        Returns:
            Audio samples as int16 array
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Sum all frequencies
        signal = np.zeros(n_samples)
        for freq in frequencies:
            signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Normalize to prevent clipping
        signal = signal / len(frequencies)
        
        # Convert to int16
        return (signal * 32767).astype(np.int16)
    
    @staticmethod
    def generate_sweep(
        start_freq: float,
        end_freq: float,
        duration: float,
        sample_rate: int = 8000,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate frequency sweep (chirp)
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)
            
        Returns:
            Audio samples as int16 array
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Linear frequency sweep
        frequency = np.linspace(start_freq, end_freq, n_samples)
        phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
        
        signal = amplitude * np.sin(phase)
        
        return (signal * 32767).astype(np.int16)
    
    @staticmethod
    def generate_noise(
        duration: float,
        sample_rate: int = 8000,
        amplitude: float = 0.1,
        noise_type: str = 'white'
    ) -> np.ndarray:
        """
        Generate noise signal
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)
            noise_type: 'white', 'pink', or 'brown'
            
        Returns:
            Audio samples as int16 array
        """
        n_samples = int(duration * sample_rate)
        
        if noise_type == 'white':
            # White noise (flat spectrum)
            signal = np.random.randn(n_samples)
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            white = np.random.randn(n_samples)
            # Simple approximation using running sum
            signal = np.cumsum(white)
            signal = signal - np.mean(signal)
            signal = signal / np.max(np.abs(signal))
        elif noise_type == 'brown':
            # Brown noise (1/f^2 spectrum)
            white = np.random.randn(n_samples)
            signal = np.cumsum(np.cumsum(white))
            signal = signal - np.mean(signal)
            signal = signal / np.max(np.abs(signal))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        signal = signal * amplitude
        return (signal * 32767).astype(np.int16)
    
    @staticmethod
    def generate_silence(
        duration: float,
        sample_rate: int = 8000
    ) -> np.ndarray:
        """
        Generate silence
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio samples as int16 array (all zeros)
        """
        n_samples = int(duration * sample_rate)
        return np.zeros(n_samples, dtype=np.int16)


class DTMFGenerator:
    """
    Generate DTMF (Dual-Tone Multi-Frequency) tones
    Used for telephone keypads
    """
    
    # DTMF frequency pairs (row, column)
    DTMF_FREQS = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633),
    }
    
    @staticmethod
    def generate_dtmf_digit(
        digit: str,
        duration: float = 0.1,
        sample_rate: int = 8000,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate DTMF tone for a single digit
        
        Args:
            digit: Digit character ('0'-'9', '*', '#', 'A'-'D')
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)
            
        Returns:
            Audio samples as int16 array
        """
        if digit.upper() not in DTMFGenerator.DTMF_FREQS:
            raise ValueError(f"Invalid DTMF digit: {digit}")
        
        freq1, freq2 = DTMFGenerator.DTMF_FREQS[digit.upper()]
        
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate two-tone signal
        signal = amplitude * (
            np.sin(2 * np.pi * freq1 * t) +
            np.sin(2 * np.pi * freq2 * t)
        ) / 2
        
        return (signal * 32767).astype(np.int16)
    
    @staticmethod
    def generate_dtmf_sequence(
        digits: str,
        digit_duration: float = 0.1,
        gap_duration: float = 0.05,
        sample_rate: int = 8000,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate DTMF sequence for multiple digits
        
        Args:
            digits: String of digits to generate
            digit_duration: Duration of each digit in seconds
            gap_duration: Gap between digits in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)
            
        Returns:
            Audio samples as int16 array
        """
        segments = []
        
        for digit in digits:
            if digit == ' ':
                # Space = longer pause
                silence = ToneGenerator.generate_silence(gap_duration * 3, sample_rate)
                segments.append(silence)
            elif digit in DTMFGenerator.DTMF_FREQS:
                # Generate digit tone
                tone = DTMFGenerator.generate_dtmf_digit(
                    digit, digit_duration, sample_rate, amplitude
                )
                segments.append(tone)
                
                # Add gap
                silence = ToneGenerator.generate_silence(gap_duration, sample_rate)
                segments.append(silence)
        
        # Concatenate all segments
        return np.concatenate(segments)
    
    @staticmethod
    def detect_dtmf(
        audio_data: np.ndarray,
        sample_rate: int = 8000,
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Detect DTMF tones in audio (simple detection)
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            threshold: Detection threshold
            
        Returns:
            List of (digit, timestamp) tuples
        """
        # This is a placeholder for DTMF detection
        # Real implementation would use Goertzel algorithm or FFT
        # For production, consider using external library
        
        detected = []
        
        # Window size for analysis
        window_size = int(0.05 * sample_rate)  # 50ms windows
        
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            window = audio_data[i:i + window_size]
            
            # Apply FFT
            fft = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Check for DTMF frequency pairs
            # (Simplified - real implementation needs better peak detection)
            for digit, (f1, f2) in DTMFGenerator.DTMF_FREQS.items():
                # Find magnitude at these frequencies
                idx1 = np.argmin(np.abs(freqs - f1))
                idx2 = np.argmin(np.abs(freqs - f2))
                
                mag1 = magnitude[idx1]
                mag2 = magnitude[idx2]
                
                # Simple threshold detection
                if mag1 > threshold and mag2 > threshold:
                    timestamp = i / sample_rate
                    detected.append((digit, timestamp))
                    break
        
        return detected
