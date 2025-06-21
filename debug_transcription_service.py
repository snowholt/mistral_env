#!/usr/bin/env python3
"""
Debug the exact WebM processing issue in the audio transcription service.
"""

import sys
import logging
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the beautyai_inference to the path
sys.path.insert(0, '/home/lumi/beautyai')

try:
    from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService
    print("✅ AudioTranscriptionService imported successfully")
except ImportError as e:
    print(f"❌ AudioTranscriptionService import failed: {e}")
    exit(1)


def test_direct_transcription():
    """Test transcription directly with the service."""
    print("\n🔍 Testing Direct Audio Transcription Service")
    print("="*60)
    
    # Create service
    service = AudioTranscriptionService()
    
    # Load Whisper model
    print("📥 Loading Whisper model...")
    success = service.load_whisper_model("whisper-large-v3-turbo-arabic")
    if not success:
        print("❌ Failed to load Whisper model")
        return
    
    print("✅ Whisper model loaded successfully")
    
    # Test files
    test_files = [
        ("/home/lumi/beautyai/q1.mp3", "mp3"),
        ("/home/lumi/beautyai/q1.webm", "webm")
    ]
    
    for file_path, format_name in test_files:
        print(f"\n🎵 Testing {format_name.upper()} file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        file_size = os.path.getsize(file_path)
        print(f"📊 File size: {file_size} bytes")
        
        # Test transcribe_audio_file method
        print(f"🎙️ Testing transcribe_audio_file...")
        try:
            transcription = service.transcribe_audio_file(file_path, language="ar")
            if transcription:
                print(f"✅ transcribe_audio_file SUCCESS: '{transcription[:100]}...'")
            else:
                print(f"❌ transcribe_audio_file FAILED: No transcription returned")
        except Exception as e:
            print(f"❌ transcribe_audio_file ERROR: {e}")
        
        # Test transcribe_audio_bytes method
        print(f"🎙️ Testing transcribe_audio_bytes...")
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            
            transcription = service.transcribe_audio_bytes(
                audio_bytes, 
                audio_format=format_name, 
                language="ar"
            )
            
            if transcription:
                print(f"✅ transcribe_audio_bytes SUCCESS: '{transcription[:100]}...'")
            else:
                print(f"❌ transcribe_audio_bytes FAILED: No transcription returned")
        except Exception as e:
            print(f"❌ transcribe_audio_bytes ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_format_validation():
    """Test format validation."""
    print("\n🔍 Testing Format Validation")
    print("="*40)
    
    service = AudioTranscriptionService()
    
    formats_to_test = ["mp3", "webm", "wav", "ogg", "flac", "m4a", "wma"]
    
    print("📋 Supported formats:", service.get_supported_formats())
    
    for fmt in formats_to_test:
        is_valid = service.validate_audio_format(fmt)
        status = "✅" if is_valid else "❌"
        print(f"{status} {fmt.upper()}: {is_valid}")


def main():
    print("🐛 WebM Transcription Debug Tool")
    print("Direct testing of AudioTranscriptionService")
    print("="*80)
    
    test_format_validation()
    test_direct_transcription()
    
    print(f"\n💡 This test bypasses the API endpoint and tests the service directly")
    print(f"   If WebM fails here, the issue is in the transcription service")
    print(f"   If WebM works here, the issue is in the API endpoint handling")


if __name__ == "__main__":
    main()
