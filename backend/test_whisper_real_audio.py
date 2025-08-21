#!/usr/bin/env python3
"""
Test script to verify Whisper engine works with real audio files
"""

import sys
import os
import numpy as np
import soundfile as sf
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(file_path: str) -> np.ndarray:
    """Load audio file and convert to 16kHz mono float32"""
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Ensure float32
        audio_data = audio_data.astype(np.float32)
        
        logger.info(f"Loaded audio: {len(audio_data)} samples, 16kHz")
        return audio_data
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        return None

def main():
    try:
        print("🧪 Testing Whisper engine with real audio...")
        
        # Import after setting up path
        from beautyai_inference.core.model_manager import ModelManager
        
        print("✅ ModelManager imported successfully")
        
        # Get ModelManager instance
        model_manager = ModelManager()
        print("✅ ModelManager instance created")
        
        # Test with multiple audio files
        test_files = [
            "../voice_tests/input_test_questions/q1.wav",
            "../voice_tests/input_test_questions/greeting_ar.wav",
            "../voice_tests/input_test_questions/greeting.wav"
        ]
        
        for audio_file in test_files:
            audio_path = os.path.join(os.path.dirname(__file__), audio_file)
            if not os.path.exists(audio_path):
                print(f"⚠️  Skipping {audio_file} - file not found")
                continue
                
            print(f"\n🔄 Testing with {audio_file}...")
            
            # Load audio
            audio_data = load_audio(audio_path)
            if audio_data is None:
                print(f"❌ Failed to load {audio_file}")
                continue
            
            # Get Whisper engine
            whisper_engine = model_manager.get_streaming_whisper("whisper-large-v3-turbo")
            if not whisper_engine:
                print("❌ Failed to get Whisper engine")
                continue
            
            # Transcribe
            try:
                # Convert audio array to bytes (simulated WAV format)
                import io
                import wave
                
                # Create WAV bytes from numpy array
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    # Convert float32 to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                audio_bytes = buffer.getvalue()
                
                result = whisper_engine.transcribe_audio_bytes(audio_bytes, "wav", "auto")
                print(f"📝 Transcription: '{result}'")
                print("✅ SUCCESS: Transcription completed!")
            except Exception as e:
                print(f"❌ Transcription failed: {e}")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())