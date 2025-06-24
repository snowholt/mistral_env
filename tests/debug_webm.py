#!/usr/bin/env python3
"""
Debug WebM format support in the audio transcription system.
"""

import torch
import torchaudio
import os
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_torchaudio_support():
    """Debug torchaudio format support."""
    print("🔍 TorchAudio Debug Information")
    print("="*50)
    
    print(f"TorchAudio version: {torchaudio.__version__}")
    print(f"Available backends: {torchaudio.list_audio_backends()}")
    
    # Test file paths
    mp3_file = "/home/lumi/beautyai/q1.mp3"
    webm_file = "/home/lumi/beautyai/q1.webm"
    
    # Test MP3 loading
    print(f"\n🎵 Testing MP3 file: {mp3_file}")
    if os.path.exists(mp3_file):
        try:
            waveform, sample_rate = torchaudio.load(mp3_file)
            print(f"✅ MP3 loaded successfully")
            print(f"   Shape: {waveform.shape}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Duration: {waveform.shape[1] / sample_rate:.2f}s")
        except Exception as e:
            print(f"❌ MP3 loading failed: {e}")
    else:
        print(f"❌ MP3 file not found")
    
    # Test WebM loading
    print(f"\n🎬 Testing WebM file: {webm_file}")
    if os.path.exists(webm_file):
        try:
            waveform, sample_rate = torchaudio.load(webm_file)
            print(f"✅ WebM loaded successfully")
            print(f"   Shape: {waveform.shape}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Duration: {waveform.shape[1] / sample_rate:.2f}s")
        except Exception as e:
            print(f"❌ WebM loading failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
    else:
        print(f"❌ WebM file not found")


def test_ffmpeg_conversion():
    """Test if we can convert WebM to WAV using ffmpeg."""
    print(f"\n🔧 Testing FFmpeg WebM conversion")
    print("="*50)
    
    webm_file = "/home/lumi/beautyai/q1.webm"
    
    if not os.path.exists(webm_file):
        print("❌ WebM file not found")
        return
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    try:
        # Try to convert WebM to WAV using ffmpeg
        import subprocess
        
        cmd = [
            'ffmpeg', '-i', webm_file, 
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-y',            # Overwrite
            temp_wav_path
        ]
        
        print(f"🔄 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ FFmpeg conversion successful")
            
            # Test loading the converted file
            if os.path.exists(temp_wav_path):
                try:
                    waveform, sample_rate = torchaudio.load(temp_wav_path)
                    print(f"✅ Converted WAV loaded successfully")
                    print(f"   Shape: {waveform.shape}")
                    print(f"   Sample rate: {sample_rate}")
                    print(f"   Duration: {waveform.shape[1] / sample_rate:.2f}s")
                except Exception as e:
                    print(f"❌ Converted WAV loading failed: {e}")
            else:
                print(f"❌ Converted file not created")
        else:
            print(f"❌ FFmpeg conversion failed")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg conversion timed out")
    except FileNotFoundError:
        print(f"❌ FFmpeg not found - install with: sudo apt-get install ffmpeg")
    except Exception as e:
        print(f"❌ FFmpeg conversion error: {e}")
    finally:
        # Clean up
        try:
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
        except:
            pass


def test_whisper_with_files():
    """Test Whisper processor directly with files."""
    print(f"\n🎙️ Testing Whisper Processor Direct Usage")
    print("="*50)
    
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        print(f"✅ Whisper processor loaded")
        
        # Test with MP3
        mp3_file = "/home/lumi/beautyai/q1.mp3"
        if os.path.exists(mp3_file):
            try:
                waveform, sample_rate = torchaudio.load(mp3_file)
                input_features = processor(
                    waveform.squeeze().numpy(), 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).input_features
                print(f"✅ MP3 processed for Whisper: {input_features.shape}")
            except Exception as e:
                print(f"❌ MP3 Whisper processing failed: {e}")
        
    except Exception as e:
        print(f"❌ Whisper processor loading failed: {e}")


def main():
    print("🐛 WebM Audio Format Debug Tool")
    print("Investigating why WebM files fail to transcribe")
    print("="*80)
    
    debug_torchaudio_support()
    test_ffmpeg_conversion()
    test_whisper_with_files()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. If WebM loading fails with torchaudio, the service needs format conversion")
    print("2. Consider adding FFmpeg-based conversion for unsupported formats")
    print("3. Alternative: Ask users to upload MP3/WAV files instead of WebM")


if __name__ == "__main__":
    main()
