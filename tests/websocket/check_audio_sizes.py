#!/usr/bin/env python3
"""
Quick test to check actual audio sizes being generated.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path for BeautyAI imports
sys.path.append('/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService


def check_audio_sizes():
    """Test actual audio sizes from voice-to-voice service."""
    logger.info("🔍 CHECKING ACTUAL AUDIO SIZES")
    logger.info("=" * 50)
    
    # Initialize voice service
    v2v_service = VoiceToVoiceService(content_filter_strictness="disabled")
    
    # Initialize required models
    logger.info("📥 Initializing models...")
    init_result = v2v_service.initialize_models(
        stt_model="whisper-large-v3-turbo-arabic",
        tts_model="coqui-tts-arabic",
        chat_model="qwen3-unsloth-q4ks"
    )
    
    if not init_result.get("success", False):
        logger.error(f"❌ Failed to initialize models: {init_result.get('error', 'Unknown error')}")
        return
    
    logger.info("✅ Models initialized successfully")
    
    # Test files
    test_files = [
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav",
        "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
    ]
    
    size_results = []
    
    for audio_file in test_files:
        if not Path(audio_file).exists():
            logger.warning(f"⚠️ File not found: {audio_file}")
            continue
            
        file_name = Path(audio_file).name
        logger.info(f"\n🎯 Testing: {file_name}")
        logger.info("-" * 30)
        
        try:
            # Process through voice-to-voice pipeline
            result = v2v_service.voice_to_voice_file(
                input_audio_path=audio_file,
                config={
                    "language": "auto",
                    "speaker_voice": "female",
                    "response_max_length": 100,  # Shorter response
                    "enable_content_filter": False
                }
            )
            
            if result.get("success", False):
                # Check different size sources
                output_audio_path = result.get("output_audio_path")
                audio_output_bytes = result.get("audio_output_bytes")
                
                sizes = {
                    "input_file": Path(audio_file).stat().st_size if Path(audio_file).exists() else 0,
                    "output_file": 0,
                    "output_bytes": 0
                }
                
                # Check output file size
                if output_audio_path and Path(output_audio_path).exists():
                    sizes["output_file"] = Path(output_audio_path).stat().st_size
                    logger.info(f"📁 Output file: {output_audio_path}")
                    logger.info(f"📏 Output file size: {sizes['output_file']:,} bytes ({sizes['output_file']/1024/1024:.2f} MB)")
                
                # Check output bytes size
                if audio_output_bytes:
                    sizes["output_bytes"] = len(audio_output_bytes)
                    logger.info(f"📦 Output bytes size: {sizes['output_bytes']:,} bytes ({sizes['output_bytes']/1024/1024:.2f} MB)")
                
                # Input file size
                logger.info(f"📥 Input file size: {sizes['input_file']:,} bytes ({sizes['input_file']/1024/1024:.2f} MB)")
                
                # WebSocket frame limit check
                max_websocket_frame = 5000000  # 5MB as currently set
                logger.info(f"🔧 WebSocket frame limit: {max_websocket_frame:,} bytes ({max_websocket_frame/1024/1024:.2f} MB)")
                
                for size_type, size_bytes in sizes.items():
                    if size_bytes > 0:
                        if size_bytes > max_websocket_frame:
                            logger.warning(f"⚠️ {size_type}: {size_bytes:,} bytes > {max_websocket_frame:,} bytes (TOO LARGE)")
                        else:
                            logger.info(f"✅ {size_type}: {size_bytes:,} bytes < {max_websocket_frame:,} bytes (OK)")
                
                # Check transcription and response
                logger.info(f"📝 Transcription: {result.get('transcription', '')[:100]}...")
                logger.info(f"💬 Response: {result.get('response_text', '')[:100]}...")
                
                size_results.append({
                    "file": file_name,
                    "success": True,
                    "sizes": sizes,
                    "transcription_length": len(result.get('transcription', '')),
                    "response_length": len(result.get('response_text', ''))
                })
                
            else:
                logger.error(f"❌ Voice processing failed: {result.get('error', 'Unknown error')}")
                size_results.append({
                    "file": file_name,
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            size_results.append({
                "file": file_name,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    logger.info("\n📊 AUDIO SIZE ANALYSIS SUMMARY")
    logger.info("=" * 50)
    
    for result in size_results:
        logger.info(f"\n📁 File: {result['file']}")
        if result['success']:
            sizes = result['sizes']
            logger.info(f"   Input: {sizes['input_file']:,} bytes")
            if sizes['output_file'] > 0:
                logger.info(f"   Output File: {sizes['output_file']:,} bytes")
            if sizes['output_bytes'] > 0:
                logger.info(f"   Output Bytes: {sizes['output_bytes']:,} bytes")
            logger.info(f"   Response Length: {result['response_length']} chars")
        else:
            logger.info(f"   ❌ Error: {result['error']}")
    
    # Find maximum sizes
    max_output_size = 0
    max_output_file = ""
    
    for result in size_results:
        if result['success']:
            sizes = result['sizes']
            max_size = max(sizes['output_file'], sizes['output_bytes'])
            if max_size > max_output_size:
                max_output_size = max_size
                max_output_file = result['file']
    
    logger.info(f"\n🎯 MAXIMUM OUTPUT SIZE: {max_output_size:,} bytes ({max_output_size/1024/1024:.2f} MB)")
    logger.info(f"📁 From file: {max_output_file}")
    logger.info(f"🔧 Current WebSocket limit: {5000000:,} bytes ({5000000/1024/1024:.2f} MB)")
    
    if max_output_size > 5000000:
        logger.warning("⚠️ AUDIO SIZE EXCEEDS WEBSOCKET LIMIT!")
        logger.info("💡 Recommendations:")
        logger.info("   1. Increase WebSocket frame limit further")
        logger.info("   2. Reduce response length")
        logger.info("   3. Use lower audio quality/sample rate")
        logger.info("   4. Implement audio chunking")
    else:
        logger.info("✅ Audio sizes are within WebSocket limit")


def main():
    check_audio_sizes()


if __name__ == "__main__":
    main()
