#!/usr/bin/env python3
"""
Quick TTS Bytes Test
===================

Test the text_to_speech_bytes method specifically to check if it's working.
This will help identify why the WebSocket isn't getting audio bytes.
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import BeautyAI services
sys.path.append('/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService

def test_tts_bytes():
    """Test TTS bytes generation specifically."""
    logger.info("🚀 Testing TTS Bytes Generation")
    logger.info("=" * 50)
    
    # Initialize TTS service
    tts_service = TextToSpeechService()
    
    # Load model
    logger.info("📥 Loading TTS model...")
    load_result = tts_service.load_tts_model("coqui-tts-arabic")
    if not load_result:
        logger.error("❌ Failed to load TTS model")
        return False
    
    logger.info("✅ TTS model loaded successfully")
    
    # Test texts
    test_cases = [
        ("Arabic", "ar", "مرحباً، كيف حالك؟"),
        ("English", "en", "Hello, how are you?"),
    ]
    
    for lang_name, lang_code, text in test_cases:
        logger.info(f"\n🎯 Testing {lang_name} TTS bytes...")
        logger.info(f"📝 Text: {text}")
        
        try:
            # Test bytes method
            audio_bytes = tts_service.text_to_speech_bytes(
                text=text,
                language=lang_code,
                speaker_voice="female"  # Fixed parameter name
            )
            
            if audio_bytes:
                logger.info(f"✅ TTS bytes generated: {len(audio_bytes)} bytes")
                
                # Save to file for verification
                output_dir = Path("/home/lumi/beautyai/tests/websocket/tts_bytes_test")
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"test_bytes_{lang_code}.wav"
                
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                logger.info(f"💾 Saved to: {output_file}")
                
                # Also test the regular file method for comparison
                file_path = tts_service.text_to_speech(
                    text=text,
                    output_path=str(output_dir / f"test_file_{lang_code}.wav"),
                    language=lang_code,
                    speaker_voice="female"  # Fixed parameter name
                )
                
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"📁 File method: {file_size} bytes")
                    logger.info(f"🔍 Size difference: {abs(len(audio_bytes) - file_size)} bytes")
                else:
                    logger.warning(f"⚠️ File method failed for {lang_name}")
                
            else:
                logger.error(f"❌ TTS bytes method returned None for {lang_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TTS bytes test failed for {lang_name}: {e}")
            return False
    
    logger.info("\n✅ All TTS bytes tests passed!")
    return True

if __name__ == "__main__":
    success = test_tts_bytes()
    if success:
        logger.info("🎉 TTS bytes generation is working correctly!")
        logger.info("📋 The issue might be elsewhere in the voice-to-voice pipeline.")
    else:
        logger.error("❌ TTS bytes generation has issues.")
        logger.info("📋 This explains why WebSocket isn't getting audio data.")
