#!/usr/bin/env python3
"""
Generate a test WebM audio file with synthetic speech-like waveform.
This ensures VAD can detect "speech" for testing purposes.
"""
import numpy as np
import av
from av import open as av_open

# Generate 5 seconds of synthetic "speech-like" audio
SAMPLE_RATE = 16000
DURATION = 5  # seconds
FREQUENCY = 440  # Hz (A4 note, speech-like fundamental frequency)

# Generate time array
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)

# Create speech-like modulated tone (amplitude modulation mimics speech envelope)
carrier = np.sin(2 * np.pi * FREQUENCY * t)  # Base tone
modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3Hz amplitude modulation
speech_like = carrier * modulator

# Add some noise for realism
noise = np.random.normal(0, 0.05, len(speech_like))
audio = speech_like + noise

# Normalize to 16-bit PCM range
audio = (audio * 16000).astype(np.int16)

# Write to WebM using av
output_file = "test_speech_synthetic.webm"
with av_open(output_file, 'w', format='webm') as container:
    stream = container.add_stream('opus', rate=SAMPLE_RATE)
    
    # Convert to av frame format
    frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format='s16', layout='mono')
    frame.sample_rate = SAMPLE_RATE
    
    # Encode and write
    for packet in stream.encode(frame):
        container.mux(packet)
    
    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

print(f"✓ Created {output_file} ({DURATION}s, {SAMPLE_RATE}Hz, synthetic speech-like waveform)")
print(f"  Amplitude range: {audio.min()} to {audio.max()}")
print(f"  This file should trigger VAD speech detection for testing.")
