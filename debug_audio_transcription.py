#!/usr/bin/env python3
"""
Debug script for audio transcription service.
"""

import sys
import logging
import tempfile
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torchaudio
    print(f"✓ TorchAudio version: {torchaudio.__version__}")
    print(f"✓ TorchAudio backends: {torchaudio.list_audio_backends()}")
except ImportError as e:
    print(f"✗ TorchAudio import failed: {e}")

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    print("✓ Transformers Whisper classes imported successfully")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService
    print("✓ AudioTranscriptionService imported successfully")
    
    # Test creating the service
    service = AudioTranscriptionService()
    print("✓ AudioTranscriptionService instantiated successfully")
    
    # Test model loading
    print("\n🔄 Testing Whisper model loading...")
    success = service.load_whisper_model("whisper-large-v3-turbo-arabic")
    print(f"Model loading result: {success}")
    
    if success:
        print(f"✓ Loaded model: {service.get_loaded_model_name()}")
        print(f"✓ Model status: {service.is_model_loaded()}")
        
        # Test supported formats
        print("\n📁 Supported audio formats:")
        for fmt in service.get_supported_formats():
            print(f"  - {fmt}")
        
        # Test format validation
        print("\n🔍 Testing format validation:")
        test_files = ["test.mp3", "test.webm", "test.wav", "test.flac", "test.xyz"]
        for file in test_files:
            valid = service.validate_audio_format(file)
            print(f"  {file}: {'✓' if valid else '✗'}")
    
    else:
        print("✗ Failed to load Whisper model")
        
except ImportError as e:
    print(f"✗ AudioTranscriptionService import failed: {e}")
except Exception as e:
    print(f"✗ Error testing AudioTranscriptionService: {e}")
    import traceback
    traceback.print_exc()

# Test audio file formats
print("\n🎵 Testing audio format support...")

def test_audio_format(test_data: bytes, format_name: str, extension: str):
    """Test if a specific audio format can be processed."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as temp_file:
            temp_file.write(test_data)
            temp_file_path = temp_file.name
        
        try:
            # Try to load with torchaudio
            waveform, sample_rate = torchaudio.load(temp_file_path)
            print(f"✓ {format_name} ({extension}): Successfully loaded")
            return True
        except Exception as e:
            print(f"✗ {format_name} ({extension}): Failed to load - {e}")
            return False
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"✗ {format_name} ({extension}): Error creating temp file - {e}")
        return False

# Create minimal test audio data (silence)
print("Creating test audio data...")
try:
    # Generate 1 second of silence at 16kHz
    import numpy as np
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    silence = np.zeros(samples, dtype=np.float32)
    
    # Convert to tensor
    waveform_tensor = torch.tensor(silence).unsqueeze(0)
    
    # Test WAV format
    test_wav_path = "/tmp/test_audio.wav"
    torchaudio.save(test_wav_path, waveform_tensor, sample_rate)
    
    with open(test_wav_path, 'rb') as f:
        wav_data = f.read()
    
    print(f"✓ Generated test WAV file ({len(wav_data)} bytes)")
    
    # Test loading the generated WAV
    test_audio_format(wav_data, "WAV", "wav")
    
    # Clean up
    os.unlink(test_wav_path)
    
except Exception as e:
    print(f"✗ Failed to generate test audio: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug completed!")
