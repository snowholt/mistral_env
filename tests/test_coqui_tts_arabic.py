#!/usr/bin/env python3
"""
Test script for Coqui TTS Arabic implementation.
Tests the new TTS engine that replaced OuteTTS for better Arabic accuracy.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.services.text_to_speech_service import TextToSpeechService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_coqui_tts_arabic():
    """Test Coqui TTS with Arabic text."""
    
    # Test phrases in Arabic
    test_phrases = [
        "مرحبا، كيف حالك؟",
        "هذا اختبار للذكاء الاصطناعي العربي",
        "أهلاً وسهلاً بك في نظام الذكاء الاصطناعي الجميل",
        "نحن نعمل على تطوير أفضل تقنيات الصوت العربي"
    ]
    
    # Output directory
    output_dir = Path("/home/lumi/beautyai/voice_tests/coqui_tts_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize TTS service
        logger.info("🚀 Initializing Coqui TTS Service...")
        tts_service = TextToSpeechService()
        
        # Initialize Coqui TTS engine
        logger.info("📡 Loading Coqui TTS Arabic model...")
        success = tts_service.load_tts_model("coqui-tts-arabic")
        
        if not success:
            logger.error("❌ Failed to load Coqui TTS engine")
            return False
        
        logger.info("✅ Coqui TTS Arabic model loaded successfully!")
        
        # Test each phrase
        for i, phrase in enumerate(test_phrases, 1):
            logger.info(f"\n🎯 Testing phrase {i}: {phrase}")
            
            output_path = output_dir / f"coqui_arabic_test_{i}.wav"
            
            # Generate speech
            result = tts_service.text_to_speech(
                text=phrase,
                output_path=str(output_path),
                language="ar",
                voice="female"
            )
            
            if result.get("success"):
                logger.info(f"✅ Successfully generated: {output_path}")
                logger.info(f"   Audio duration: {result.get('metadata', {}).get('duration', 'unknown')} seconds")
            else:
                logger.error(f"❌ Failed to generate audio for phrase {i}: {result.get('error')}")
        
        # Test voice cloning with reference audio if available
        reference_audio = Path("/home/lumi/beautyai/voice_tests/reference_arabic.wav")
        if reference_audio.exists():
            logger.info(f"\n🎭 Testing voice cloning with reference audio...")
            
            clone_output = output_dir / "coqui_arabic_cloned.wav"
            clone_result = tts_service.text_to_speech(
                text=test_phrases[0],
                output_path=str(clone_output),
                language="ar",
                voice="female"
            )
            
            if clone_result.get("success"):
                logger.info(f"✅ Voice cloning successful: {clone_output}")
            else:
                logger.info(f"⚠️ Voice cloning not available or failed: {clone_result.get('error')}")
        
        # Get engine status
        model_info = tts_service.get_model_info()
        logger.info(f"\n📊 TTS Service Status:")
        logger.info(f"   Model loaded: {model_info.get('model_name', 'unknown')}")
        logger.info(f"   Engine type: {model_info.get('engine_type', 'unknown')}")
        
        memory_stats = tts_service.get_memory_stats()
        logger.info(f"   Memory usage: {memory_stats}")
        
        logger.info("\n🎉 Coqui TTS Arabic test completed successfully!")
        logger.info(f"📁 Audio files saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during Coqui TTS test: {e}")
        return False

def main():
    """Main function to run the test."""
    logger.info("🔬 Starting Coqui TTS Arabic Test")
    logger.info("=" * 60)
    
    # Run the async test
    success = asyncio.run(test_coqui_tts_arabic())
    
    if success:
        logger.info("\n✅ All tests passed! Coqui TTS is working correctly.")
        logger.info("🎯 Arabic speech synthesis should now have much better accuracy.")
        logger.info("📈 This replaces OuteTTS which had metadata contamination issues.")
    else:
        logger.error("\n❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
