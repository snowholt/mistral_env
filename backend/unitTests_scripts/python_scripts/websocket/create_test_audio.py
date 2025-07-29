#!/usr/bin/env python3
"""
Create test audio files for WebSocket testing.
This script generates simple test audio files using text-to-speech.
"""
import subprocess
import sys
from pathlib import Path

def create_test_audio():
    """Create test audio files using espeak or create silence if not available."""
    
    # Create input test questions directory
    test_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test audio content
    test_cases = [
        {
            "filename": "botox.wav",
            "text": "Hello, I would like to ask about botox treatments. What are the benefits and risks?",
            "language": "en"
        },
        {
            "filename": "botox_ar.wav", 
            "text": "مرحبا، أريد أن أسأل عن علاج البوتوكس. ما هي الفوائد والمخاطر؟",
            "language": "ar"
        }
    ]
    
    created_files = []
    
    for test_case in test_cases:
        output_path = test_dir / test_case["filename"]
        
        try:
            # Try using espeak if available
            if test_case["language"] == "en":
                cmd = [
                    "espeak",
                    "-s", "150",  # Speed
                    "-w", str(output_path),  # Output to WAV file
                    test_case["text"]
                ]
            else:  # Arabic
                cmd = [
                    "espeak", 
                    "-s", "150",
                    "-w", str(output_path),
                    test_case["text"]
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_path.exists():
                print(f"✅ Created: {output_path}")
                created_files.append(str(output_path))
            else:
                print(f"⚠️ espeak failed for {test_case['filename']}, creating silence...")
                create_silence_wav(output_path)
                created_files.append(str(output_path))
                
        except FileNotFoundError:
            print(f"⚠️ espeak not found, creating silence for {test_case['filename']}")
            create_silence_wav(output_path)
            created_files.append(str(output_path))
        except Exception as e:
            print(f"❌ Failed to create {test_case['filename']}: {e}")
    
    return created_files

def create_silence_wav(output_path: Path, duration_seconds: float = 3.0):
    """Create a silent WAV file using ffmpeg or dd."""
    try:
        # Try ffmpeg first
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate=16000",
            "-t", str(duration_seconds),
            "-c:a", "pcm_s16le",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and output_path.exists():
            print(f"✅ Created silence: {output_path}")
            return
            
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # Fallback: create a minimal WAV file manually
    try:
        create_minimal_wav(output_path, duration_seconds)
        print(f"✅ Created minimal WAV: {output_path}")
    except Exception as e:
        print(f"❌ Failed to create minimal WAV: {e}")

def create_minimal_wav(output_path: Path, duration_seconds: float = 3.0):
    """Create a minimal WAV file with silence."""
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    
    # Calculate data size
    num_samples = int(sample_rate * duration_seconds)
    data_size = num_samples * channels * (bits_per_sample // 8)
    
    # WAV header
    header = bytearray()
    
    # RIFF header
    header.extend(b"RIFF")
    header.extend((36 + data_size).to_bytes(4, 'little'))  # File size - 8
    header.extend(b"WAVE")
    
    # fmt chunk
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, 'little'))  # Chunk size
    header.extend((1).to_bytes(2, 'little'))   # Audio format (PCM)
    header.extend(channels.to_bytes(2, 'little'))  # Number of channels
    header.extend(sample_rate.to_bytes(4, 'little'))  # Sample rate
    header.extend((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))  # Byte rate
    header.extend((channels * bits_per_sample // 8).to_bytes(2, 'little'))  # Block align
    header.extend(bits_per_sample.to_bytes(2, 'little'))  # Bits per sample
    
    # data chunk
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, 'little'))
    
    # Write to file
    with open(output_path, "wb") as f:
        f.write(header)
        # Write silence (zeros)
        f.write(b'\x00' * data_size)

if __name__ == "__main__":
    print("🎵 Creating test audio files...")
    created_files = create_test_audio()
    
    print(f"\n✅ Created {len(created_files)} test audio files:")
    for file_path in created_files:
        print(f"  - {file_path}")
    
    print("\n🎤 Test files are ready for WebSocket testing!")
