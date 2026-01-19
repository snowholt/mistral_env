#!/usr/bin/env python3
"""
Download Saudi TTS Model (AhmedEladl/saudi-tts) from HuggingFace.

This script downloads the Saudi Arabic fine-tuned XTTS v2 model to the
local cache directory for persistent use.

Model: https://huggingface.co/AhmedEladl/saudi-tts

Usage:
    python scripts/download_saudi_tts.py

Author: BeautyAI Framework
Date: 2026-01-16
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

# Model configuration
MODEL_ID = "AhmedEladl/saudi-tts"
CACHE_DIR = Path.home() / ".cache" / "beautyai-models" / "saudi-tts"


def download_model():
    """Download Saudi TTS model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
        
        print("=" * 60)
        print("🇸🇦 Saudi TTS Model Downloader")
        print("=" * 60)
        print(f"Model: {MODEL_ID}")
        print(f"Target: {CACHE_DIR}")
        print()
        
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        print("📥 Downloading model from HuggingFace...")
        print("   This may take a few minutes (~5GB download)")
        print()
        
        # Download the model
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(CACHE_DIR),
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        print()
        print("✅ Model downloaded successfully!")
        print(f"   Location: {local_path}")
        print()
        
        # Verify required files
        required_files = ["config.json", "vocab.json", "model.pth"]
        missing_files = []
        
        for filename in required_files:
            file_path = CACHE_DIR / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ✓ {filename} ({size_mb:.1f} MB)")
            else:
                missing_files.append(filename)
                print(f"   ✗ {filename} (MISSING)")
        
        if missing_files:
            print()
            print(f"⚠️ Warning: Missing files: {missing_files}")
            print("   The model may not work correctly.")
            return False
        
        print()
        print("=" * 60)
        print("🎉 Saudi TTS model is ready to use!")
        print()
        print("Next steps:")
        print("1. Add a speaker reference WAV file:")
        print("   mkdir -p backend/speakers/saudi-female/")
        print("   cp your_speaker.wav backend/speakers/saudi-female/reference.wav")
        print()
        print("2. Restart the BeautyAI API service:")
        print("   sudo systemctl restart beautyai-api.service")
        print("=" * 60)
        
        return True
        
    except ImportError:
        print("❌ Error: huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model():
    """Verify the model is correctly installed."""
    print()
    print("🔍 Verifying model installation...")
    
    if not CACHE_DIR.exists():
        print(f"❌ Model directory not found: {CACHE_DIR}")
        return False
    
    required_files = ["config.json", "vocab.json", "model.pth"]
    
    for filename in required_files:
        file_path = CACHE_DIR / filename
        if not file_path.exists():
            print(f"❌ Missing file: {filename}")
            return False
    
    print("✅ All required files present")
    return True


def check_speaker_reference():
    """Check if speaker reference file exists."""
    speaker_dir = BACKEND_DIR / "speakers" / "saudi-female"
    speaker_file = speaker_dir / "reference.wav"
    
    print()
    print("🎤 Checking speaker reference...")
    
    if speaker_file.exists():
        size_mb = speaker_file.stat().st_size / (1024 * 1024)
        print(f"✅ Speaker reference found: {speaker_file} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"⚠️ Speaker reference not found: {speaker_file}")
        print()
        print("📝 Speaker Audio Requirements:")
        print("   • Format: WAV (PCM, 16-bit)")
        print("   • Sample Rate: 22050 Hz or 24000 Hz")
        print("   • Duration: 6-15 seconds of clear speech")
        print("   • Quality: Studio/clean recording, minimal noise")
        print("   • Language: Arabic (matching target output)")
        print("   • Content: Natural conversational text, varied intonation")
        print()
        print("   Place your speaker reference at:")
        print(f"   {speaker_file}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Saudi TTS Model")
    parser.add_argument("--verify", action="store_true", help="Only verify installation")
    parser.add_argument("--check-speaker", action="store_true", help="Check speaker reference")
    args = parser.parse_args()
    
    if args.verify:
        success = verify_model()
        sys.exit(0 if success else 1)
    
    if args.check_speaker:
        success = check_speaker_reference()
        sys.exit(0 if success else 1)
    
    # Download model
    success = download_model()
    
    # Check speaker reference
    check_speaker_reference()
    
    sys.exit(0 if success else 1)
