#!/usr/bin/env python3
"""
Convert captured RTP audio to WAV format
Supports various codecs used by HT813
"""

import os
import sys
import json
import argparse
import wave
import struct
from pathlib import Path


def pcmu_to_pcm(data):
    """Convert PCMU (μ-law) to linear PCM"""
    # μ-law decompression table
    pcm_samples = []
    
    for byte in data:
        # μ-law expansion
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        
        # Compute linear value
        value = ((mantissa << 3) + 132) << exponent
        value -= 132
        
        if sign:
            value = -value
        
        # Convert to 16-bit signed integer
        pcm_samples.append(struct.pack('<h', value))
    
    return b''.join(pcm_samples)


def pcma_to_pcm(data):
    """Convert PCMA (A-law) to linear PCM"""
    # A-law decompression table
    pcm_samples = []
    
    for byte in data:
        byte ^= 0x55  # Invert even bits
        
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        
        # Compute linear value
        if exponent == 0:
            value = (mantissa << 4) + 8
        else:
            value = ((mantissa << 4) + 0x108) << (exponent - 1)
        
        if sign:
            value = -value
        
        # Convert to 16-bit signed integer
        pcm_samples.append(struct.pack('<h', value))
    
    return b''.join(pcm_samples)


def convert_to_wav(session_dir, output_file=None):
    """Convert captured RTP audio to WAV file"""
    
    # Read metadata
    metadata_file = os.path.join(session_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        print(f"❌ Error: metadata.json not found in {session_dir}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Read raw audio data
    raw_file = os.path.join(session_dir, 'audio_raw.bin')
    if not os.path.exists(raw_file):
        print(f"❌ Error: audio_raw.bin not found in {session_dir}")
        return False
    
    with open(raw_file, 'rb') as f:
        raw_data = f.read()
    
    if len(raw_data) == 0:
        print(f"❌ Error: No audio data in {raw_file}")
        return False
    
    # Determine output file
    if output_file is None:
        output_file = os.path.join(session_dir, 'audio.wav')
    
    # Get codec parameters
    payload_type = metadata['payload_type']
    
    # Codec-specific parameters
    sample_rate = 8000  # Default for most telephony codecs
    channels = 1  # Mono
    
    # Convert audio based on codec
    print(f"📊 Converting {metadata['codec']} to WAV...")
    print(f"   Input: {raw_file}")
    print(f"   Size: {len(raw_data)} bytes")
    
    if payload_type == 0:  # PCMU (G.711 μ-law)
        pcm_data = pcmu_to_pcm(raw_data)
        sample_width = 2  # 16-bit
    elif payload_type == 8:  # PCMA (G.711 A-law)
        pcm_data = pcma_to_pcm(raw_data)
        sample_width = 2  # 16-bit
    elif payload_type == 9:  # G.722
        # G.722 is 16kHz wideband
        sample_rate = 16000
        pcm_data = raw_data  # Simplified - actual G.722 decoding is complex
        sample_width = 2
        print("⚠️  Warning: G.722 decoding is simplified, may need ffmpeg for proper conversion")
    else:
        print(f"⚠️  Warning: Unsupported codec (payload type {payload_type})")
        print(f"   Saving as raw PCM, may need manual conversion")
        pcm_data = raw_data
        sample_width = 2
    
    # Write WAV file
    try:
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        print(f"✅ WAV file created successfully!")
        print(f"   Output: {output_file}")
        print(f"   Duration: {len(pcm_data) / (sample_rate * sample_width * channels):.2f}s")
        print(f"   Format: {sample_rate}Hz, {channels} channel(s), {sample_width*8}-bit")
        return True
        
    except Exception as e:
        print(f"❌ Error writing WAV file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert captured RTP audio to WAV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a specific session
  python3 convert_rtp_to_wav.py captures/session_20251114_120000_12345/
  
  # Convert with custom output file
  python3 convert_rtp_to_wav.py captures/session_20251114_120000_12345/ -o output.wav
  
  # Convert all sessions in captures directory
  python3 convert_rtp_to_wav.py captures/ --all
        """
    )
    
    parser.add_argument('session_dir', 
                        help='Path to session directory containing metadata.json and audio_raw.bin')
    parser.add_argument('--output', '-o', default=None,
                        help='Output WAV file path (default: audio.wav in session directory)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Convert all sessions in the directory')
    
    args = parser.parse_args()
    
    if args.all:
        # Find all session directories
        captures_dir = Path(args.session_dir)
        session_dirs = [d for d in captures_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('session_')]
        
        if not session_dirs:
            print(f"❌ No session directories found in {captures_dir}")
            return
        
        print(f"🔄 Found {len(session_dirs)} session(s) to convert\n")
        
        success_count = 0
        for session_dir in sorted(session_dirs):
            print(f"{'='*60}")
            print(f"Processing: {session_dir.name}")
            print(f"{'='*60}")
            if convert_to_wav(str(session_dir)):
                success_count += 1
            print()
        
        print(f"\n✅ Successfully converted {success_count}/{len(session_dirs)} session(s)")
    else:
        # Convert single session
        if not os.path.isdir(args.session_dir):
            print(f"❌ Error: {args.session_dir} is not a directory")
            sys.exit(1)
        
        if convert_to_wav(args.session_dir, args.output):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
