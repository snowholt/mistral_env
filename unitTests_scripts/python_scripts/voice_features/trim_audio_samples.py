#!/usr/bin/env python3
"""
Trim audio samples to create shorter clips for speaker profile creation.
OuteTTS requires audio clips shorter than 20 seconds for optimal results.
"""

import os
import sys
from pathlib import Path
import logging

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError as e:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.error(f"Audio processing libraries not available: {e}")

def get_audio_duration(audio_path):
    """Get the duration of an audio file."""
    try:
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration for {audio_path}: {e}")
        return None

def trim_audio_file(input_path, output_path, start_time=2.0, duration=15.0):
    """
    Trim audio file to specified duration.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file  
        start_time: Start time in seconds (skip initial silence)
        duration: Duration of output clip in seconds
    """
    try:
        # Load audio file
        audio, sr = librosa.load(str(input_path), sr=None)
        
        # Calculate start and end samples
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        
        # Ensure we don't exceed file length
        if end_sample > len(audio):
            end_sample = len(audio)
            logger.warning(f"Trimming to end of file for {input_path}")
        
        # Trim audio
        trimmed_audio = audio[start_sample:end_sample]
        
        # Save trimmed audio
        sf.write(str(output_path), trimmed_audio, sr)
        
        actual_duration = len(trimmed_audio) / sr
        logger.info(f"✅ Trimmed {input_path.name}: {actual_duration:.1f}s → {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error trimming {input_path}: {e}")
        return False

def analyze_audio_file(audio_path):
    """Analyze audio file for quality and characteristics."""
    try:
        audio, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio) / sr
        
        # Basic audio analysis
        rms_energy = np.sqrt(np.mean(audio**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Find the best segment (highest energy)
        segment_length = int(15 * sr)  # 15 seconds
        best_start = 0
        best_energy = 0
        
        if len(audio) > segment_length:
            # Analyze segments to find the one with highest energy
            for start in range(0, len(audio) - segment_length, int(sr)):  # 1-second steps
                segment = audio[start:start + segment_length]
                segment_energy = np.sqrt(np.mean(segment**2))
                if segment_energy > best_energy:
                    best_energy = segment_energy
                    best_start = start
        
        best_start_time = best_start / sr
        
        logger.info(f"📊 Audio analysis for {audio_path.name}:")
        logger.info(f"   Duration: {duration:.1f}s")
        logger.info(f"   Sample rate: {sr} Hz")
        logger.info(f"   RMS Energy: {rms_energy:.4f}")
        logger.info(f"   Best segment starts at: {best_start_time:.1f}s")
        
        return best_start_time
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {audio_path}: {e}")
        return 2.0  # Default start time

def main():
    """Main function to trim audio files."""
    print("🎵 Audio Trimming for Arabic Speaker Profiles")
    print("=" * 60)
    
    if not AUDIO_PROCESSING_AVAILABLE:
        print("❌ Audio processing libraries not available")
        print("💡 Install with: pip install librosa soundfile")
        return False
    
    # Define audio files
    audio_files = [
        ("voice_tests/custom_speakers/audio_1_F.wav", "female"),
        ("voice_tests/custom_speakers/audio_1_M.wav", "male")
    ]
    
    workspace_dir = Path("/home/lumi/beautyai")
    trimmed_dir = workspace_dir / "trimmed_audio"
    trimmed_dir.mkdir(exist_ok=True)
    
    success_count = 0
    
    for filename, gender in audio_files:
        input_path = workspace_dir / filename
        output_path = trimmed_dir / f"arabic_{gender}_15s.wav"
        
        if not input_path.exists():
            logger.error(f"❌ Audio file not found: {input_path}")
            continue
        
        # Get original duration
        original_duration = get_audio_duration(input_path)
        if original_duration:
            logger.info(f"📄 Original {gender} audio: {original_duration:.1f}s")
        
        # Analyze audio to find best segment
        best_start = analyze_audio_file(input_path)
        
        # Trim audio file
        if trim_audio_file(input_path, output_path, start_time=best_start, duration=15.0):
            success_count += 1
            
            # Verify trimmed file
            trimmed_duration = get_audio_duration(output_path)
            if trimmed_duration:
                logger.info(f"✅ Trimmed {gender} audio: {trimmed_duration:.1f}s")
    
    print("\n📋 Summary")
    print("=" * 60)
    print(f"✅ Successfully trimmed {success_count}/2 audio files")
    
    if success_count == 2:
        print("🎯 Ready for speaker profile creation!")
        print(f"📁 Trimmed files saved to: {trimmed_dir}")
        return True
    else:
        print("❌ Audio trimming incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
