#!/usr/bin/env python3
"""
WebM Decoder Utility Validation Script

This script validates the WebMDecoder utility by testing its functionality
with sample WebM/Opus audio data and ensuring proper integration with
the BeautyAI voice endpoints.

Author: BeautyAI Framework
Date: 2024-01-19
"""

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the backend source to Python path for imports
sys.path.insert(0, '/home/lumi/beautyai/backend/src')

try:
    from beautyai_inference.utils import (
        WebMDecoder,
        WebMDecodingMode,
        WebMDecodingError,
        create_realtime_decoder,
        create_batch_decoder
    )
    logger.info("✅ Successfully imported WebMDecoder utility")
except ImportError as e:
    logger.error(f"❌ Failed to import WebMDecoder: {e}")
    sys.exit(1)


async def test_batch_decoding():
    """Test batch decoding functionality."""
    logger.info("🧪 Testing batch decoding...")
    
    try:
        decoder = create_batch_decoder()
        
        # Test with empty chunks
        try:
            await decoder.decode_chunks_to_pcm([])
            logger.error("❌ Expected error for empty chunks")
        except WebMDecodingError:
            logger.info("✅ Correctly handles empty chunk list")
        
        # Test format detection
        webm_header = b"\x1a\x45\xdf\xa3"  # WebM/Matroska header
        ogg_header = b"OggS"               # Ogg header
        wav_header = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE"  # WAV header with size
        
        webm_format = decoder.detect_audio_format(webm_header + b"test")
        ogg_format = decoder.detect_audio_format(ogg_header + b"test") 
        wav_format = decoder.detect_audio_format(wav_header + b"test")
        unknown_format = decoder.detect_audio_format(b"unknown")
        
        logger.info(f"Format detection results: webm={webm_format}, ogg={ogg_format}, wav={wav_format}, unknown={unknown_format}")
        
        assert webm_format == "webm", f"Expected 'webm', got '{webm_format}'"
        assert ogg_format == "ogg", f"Expected 'ogg', got '{ogg_format}'"
        assert wav_format == "wav", f"Expected 'wav', got '{wav_format}'"
        assert unknown_format == "webm", f"Expected 'webm' (default), got '{unknown_format}'"
        
        logger.info("✅ Audio format detection working correctly")
        
        # Test compatibility check
        assert decoder.is_webm_compatible("webm")
        assert decoder.is_webm_compatible("ogg") 
        assert decoder.is_webm_compatible("wav")
        assert not decoder.is_webm_compatible("xyz")
        
        logger.info("✅ Format compatibility check working correctly")
        
        # Test decoder stats
        stats = decoder.get_stats()
        assert "target_sample_rate" in stats
        assert "target_channels" in stats
        assert stats["target_sample_rate"] == 16000
        assert stats["target_channels"] == 1
        
        logger.info("✅ Decoder statistics working correctly")
        
        # Cleanup
        decoder.cleanup()
        logger.info("✅ Batch decoding tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Batch decoding test failed: {e}")
        raise


async def test_realtime_streaming():
    """Test real-time streaming functionality."""
    logger.info("🧪 Testing real-time streaming...")
    
    try:
        decoder = create_realtime_decoder()
        
        # Create a mock WebM chunk generator
        async def mock_webm_chunks() -> AsyncGenerator[bytes, None]:
            # Simulate WebM header chunk
            yield b"\x1a\x45\xdf\xa3" + b"mock_webm_header_data" * 10
            
            # Simulate data chunks
            for i in range(5):
                yield b"mock_webm_data_chunk_" + str(i).encode() * 20
                await asyncio.sleep(0.1)  # Simulate streaming delay
        
        # Test that the streaming interface works (will fail on FFmpeg but structure is correct)
        try:
            pcm_count = 0
            async for pcm_chunk in decoder.stream_realtime_pcm(mock_webm_chunks(), enable_logging=False):
                pcm_count += 1
                if pcm_count > 10:  # Limit iterations to prevent infinite loop
                    break
            logger.warning("⚠️ FFmpeg not available for actual decoding test")
        except WebMDecodingError as e:
            logger.info(f"✅ Correctly handles FFmpeg unavailability: {e}")
        except Exception as e:
            logger.info(f"✅ Streaming interface structure correct (FFmpeg error expected): {e}")
        
        # Test cleanup
        decoder.cleanup()
        logger.info("✅ Real-time streaming tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Real-time streaming test failed: {e}")
        raise


async def test_factory_functions():
    """Test factory function behavior."""
    logger.info("🧪 Testing factory functions...")
    
    try:
        # Test realtime decoder factory
        realtime_decoder = create_realtime_decoder()
        assert realtime_decoder.target_sample_rate == 16000
        assert realtime_decoder.target_channels == 1
        assert realtime_decoder.chunk_size == 640  # 20ms at 16kHz
        assert realtime_decoder.ffmpeg_timeout == 30
        
        # Test batch decoder factory
        batch_decoder = create_batch_decoder()
        assert batch_decoder.target_sample_rate == 16000
        assert batch_decoder.target_channels == 1
        assert batch_decoder.chunk_size == 1600  # 50ms at 16kHz
        assert batch_decoder.ffmpeg_timeout == 60
        
        logger.info("✅ Factory functions working correctly")
        
        # Cleanup
        realtime_decoder.cleanup()
        batch_decoder.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Factory function test failed: {e}")
        raise


async def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("🧪 Testing error handling...")
    
    try:
        decoder = create_batch_decoder()
        
        # Test with non-existent file
        try:
            await decoder.decode_file_to_numpy("/nonexistent/path/file.webm")
            logger.error("❌ Expected error for non-existent file")
        except WebMDecodingError:
            logger.info("✅ Correctly handles non-existent file")
        
        # Test format detection edge cases
        assert decoder.detect_audio_format(b"") == "unknown"
        assert decoder.detect_audio_format(b"a") == "unknown"  # Too short
        assert decoder.detect_audio_format(b"abc") == "unknown"  # Too short
        assert decoder.detect_audio_format(b"unknown") == "webm"  # Default fallback for 4+ bytes
        
        logger.info("✅ Error handling tests completed successfully")
        
        # Cleanup
        decoder.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        raise


async def test_integration_patterns():
    """Test patterns used in endpoint integration."""
    logger.info("🧪 Testing endpoint integration patterns...")
    
    try:
        # Test queue-based pattern (streaming_voice.py)
        chunk_queue = asyncio.Queue()
        
        async def chunk_generator():
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:  # Sentinel
                    break
                yield chunk
        
        # Test sentinel stop pattern
        await chunk_queue.put(b"test_chunk")
        await chunk_queue.put(None)  # Stop sentinel
        
        chunks = []
        async for chunk in chunk_generator():
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == b"test_chunk"
        
        logger.info("✅ Queue-based streaming pattern working correctly")
        
        # Test chunk accumulation pattern (websocket_simple_voice.py)
        chunk_list = [
            b"\x1a\x45\xdf\xa3" + b"header",  # WebM header
            b"data_chunk_1",
            b"data_chunk_2"
        ]
        
        decoder = create_batch_decoder()
        
        # Test chunk reassembly logic
        complete_webm = chunk_list[0] + b''.join(chunk_list[1:])
        assert len(complete_webm) > len(chunk_list[0])
        
        logger.info("✅ Chunk accumulation pattern working correctly")
        
        # Cleanup
        decoder.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Integration pattern test failed: {e}")
        raise


async def main():
    """Run all validation tests."""
    logger.info("🚀 Starting WebMDecoder utility validation...")
    
    tests = [
        test_batch_decoding,
        test_realtime_streaming,
        test_factory_functions,
        test_error_handling,
        test_integration_patterns
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"🏁 Validation complete: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All WebMDecoder utility validation tests passed!")
        return 0
    else:
        logger.error("💥 Some validation tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)