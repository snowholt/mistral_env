#!/usr/bin/env python3
"""
Test script for VAD-driven voice conversation implementation.

This script tests the new VAD service and integration with the existing
voice WebSocket endpoint.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend src to Python path
sys.path.append(str(Path(__file__).parent / "backend" / "src"))

from beautyai_inference.services.voice.vad_service import RealTimeVADService, VADConfig, initialize_vad_service


async def test_vad_initialization():
    """Test VAD service initialization."""
    print("🔬 Testing VAD service initialization...")
    
    try:
        # Create VAD configuration
        config = VADConfig(
            chunk_size_ms=30,
            silence_threshold_ms=500,
            sampling_rate=16000,
            speech_threshold=0.5
        )
        
        # Initialize VAD service
        success = await initialize_vad_service(config)
        
        if success:
            print("✅ VAD service initialized successfully")
            return True
        else:
            print("❌ VAD service initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ VAD service initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_audio_file_processing():
    """Test processing existing audio files."""
    print("🎵 Testing audio file processing...")
    
    try:
        from beautyai_inference.services.voice.vad_service import get_vad_service
        
        # Get the VAD service
        vad_service = get_vad_service()
        
        # Look for test audio files
        test_audio_dir = Path(__file__).parent / "voice_tests" / "input_test_questions" / "webm"
        
        if not test_audio_dir.exists():
            print(f"⚠️ Test audio directory not found: {test_audio_dir}")
            return False
        
        # Find first available test file
        test_files = list(test_audio_dir.glob("*.webm"))
        if not test_files:
            print("⚠️ No WebM test files found")
            return False
        
        test_file = test_files[0]
        print(f"📁 Testing with file: {test_file.name}")
        
        # Read audio data
        with open(test_file, "rb") as f:
            audio_data = f.read()
        
        print(f"🎶 Audio file size: {len(audio_data)} bytes")
        
        # Process audio with VAD
        result = await vad_service.process_audio_chunk(audio_data, "webm")
        
        if result.get("success"):
            print("✅ Audio processing successful")
            print(f"📊 Processing time: {result.get('processing_time_ms', 0)}ms")
            print(f"🎤 Chunks processed: {result.get('chunks_processed', 0)}")
            
            state = result.get("current_state", {})
            print(f"🗣️ Speaking: {state.get('is_speaking', False)}")
            print(f"🔇 Silence duration: {state.get('silence_duration_ms', 0)}ms")
            
            return True
        else:
            print(f"❌ Audio processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_service_integration():
    """Test integration with existing voice service."""
    print("🔗 Testing voice service integration...")
    
    try:
        from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
        
        # Initialize voice service
        voice_service = SimpleVoiceService()
        await voice_service.initialize()
        
        print("✅ Voice service initialized successfully")
        
        # Get available voices
        voices = voice_service.get_available_voices()
        print(f"🎙️ Available voices: {len(voices)}")
        
        for voice_key, voice_info in voices.items():
            print(f"  - {voice_key}: {voice_info['display_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice service integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_websocket_import():
    """Test if WebSocket endpoint imports work correctly."""
    print("🌐 Testing WebSocket endpoint imports...")
    
    try:
        from beautyai_inference.api.endpoints.websocket_simple_voice import (
            SimpleVoiceWebSocketManager,
            simple_ws_manager
        )
        
        print("✅ WebSocket endpoint imports successful")
        
        # Test manager initialization
        manager = SimpleVoiceWebSocketManager()
        print("✅ WebSocket manager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket endpoint import error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 BeautyAI VAD Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("VAD Initialization", test_vad_initialization),
        ("WebSocket Imports", test_websocket_import),
        ("Voice Service Integration", test_voice_service_integration),
        ("Audio File Processing", test_audio_file_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print("-" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VAD implementation is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
