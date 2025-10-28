"""
Unit tests for WebRTC VAD Service warmup filter and sustained speech detection.

Tests the server-side fix for Q7 audio test failure where premature VOICE_START
detection during warm-up silence leads to Whisper hallucinations.

Author: BeautyAI Framework
Created: October 27, 2025
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
from typing import Dict, Any

from beautyai_inference.services.voice.vad.webrtc_vad_service import (
    WebRTCVADService,
    WebRTCVADConfig,
    VADState
)


@pytest_asyncio.fixture
async def vad_service():
    """Create and initialize a VAD service for testing."""
    config = WebRTCVADConfig(
        warmup_filter_duration_ms=500,  # 0.5s warmup
        min_sustained_speech_frames=3,   # Require 3 consecutive frames
        webrtc_sensitivity=2,
        silero_sensitivity=0.3,
        language_thresholds={"ar": 0.001, "en": 0.002, "default": 0.002}
    )
    
    vad = WebRTCVADService(
        peer_id="test_peer",
        language="ar",
        config=config
    )
    
    await vad.initialize()
    yield vad
    await vad.cleanup()


def generate_silence_chunk(duration_ms: int = 30, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio chunk."""
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()


def generate_speech_chunk(duration_ms: int = 30, sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate synthetic speech-like PCM audio chunk."""
    num_samples = int(duration_ms * sample_rate / 1000)
    
    # Generate speech-like signal (low frequency tone + noise)
    t = np.linspace(0, duration_ms / 1000, num_samples)
    speech = amplitude * (
        np.sin(2 * np.pi * 200 * t) +  # 200Hz fundamental
        0.3 * np.sin(2 * np.pi * 400 * t) +  # 400Hz harmonic
        0.1 * np.random.randn(num_samples)  # Noise
    )
    
    # Convert to int16
    speech_int16 = (speech * 32767).astype(np.int16)
    return speech_int16.tobytes()


def create_metadata(duration_ms: int = 30, sample_rate: int = 16000) -> Dict[str, Any]:
    """Create metadata for audio chunk."""
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_sec": duration_ms / 1000,
        "timestamp": time.time()
    }


@pytest.mark.asyncio
async def test_warmup_filter_blocks_initial_speech(vad_service):
    """
    Test that warmup filter prevents VOICE_START during initial 500ms,
    even when strong speech is present.
    
    This is the core fix for Q7 test failure.
    """
    # Simulate first 500ms with strong speech (simulating warm-up frames with audio)
    results = []
    
    for i in range(20):  # 20 frames * 30ms = 600ms (covers warmup period)
        # Generate strong speech from the start
        chunk = generate_speech_chunk(duration_ms=30, amplitude=0.7)
        metadata = create_metadata(duration_ms=30)
        
        result = await vad_service.process_audio_chunk(chunk, metadata)
        results.append(result)
        
        # Small delay between chunks (realistic timing)
        await asyncio.sleep(0.001)
    
    # Assertions
    # During first 500ms (first ~17 frames), should return INACTIVE due to warmup filter
    warmup_results = results[:17]
    for idx, r in enumerate(warmup_results):
        assert r["success"] == True, f"Frame {idx}: VAD processing failed"
        assert r.get("warmup_active", False) == True, f"Frame {idx}: Warmup not active"
        assert r["voice_state"] == VADState.INACTIVE, f"Frame {idx}: Expected INACTIVE during warmup, got {r['voice_state']}"
        print(f"✅ Frame {idx}: Warmup filter active, returned INACTIVE (elapsed: {r.get('warmup_elapsed_ms', 0):.0f}ms)")
    
    # After 500ms, warmup should complete
    post_warmup_results = results[17:]
    for idx, r in enumerate(post_warmup_results, start=17):
        assert r.get("warmup_active", False) == False, f"Frame {idx}: Warmup still active after 500ms"
        print(f"✅ Frame {idx}: Warmup complete, state={r['voice_state'].value}, detected={r['voice_detected']}")


@pytest.mark.asyncio
async def test_sustained_speech_detection(vad_service):
    """
    Test that sustained speech detection requires 3 consecutive frames
    before declaring VOICE_START.
    """
    # Skip warmup period first
    for _ in range(20):
        chunk = generate_silence_chunk()
        await vad_service.process_audio_chunk(chunk, create_metadata())
        await asyncio.sleep(0.001)
    
    # Now send strong speech frames
    speech_results = []
    for i in range(5):
        chunk = generate_speech_chunk(amplitude=0.8)
        metadata = create_metadata()
        result = await vad_service.process_audio_chunk(chunk, metadata)
        speech_results.append(result)
        print(f"Speech frame {i}: state={result['voice_state'].value}, "
              f"detected={result['voice_detected']}, "
              f"sustained_frames={result.get('sustained_speech_frames', 0)}")
        await asyncio.sleep(0.001)
    
    # First 2 frames should be INACTIVE (not enough sustained speech)
    assert speech_results[0]["voice_state"] == VADState.INACTIVE, "Frame 0: Should be INACTIVE (need sustained speech)"
    assert speech_results[1]["voice_state"] == VADState.INACTIVE, "Frame 1: Should be INACTIVE (need sustained speech)"
    
    # Frame 3+ should trigger VOICE_START (3 consecutive frames)
    # Note: Depending on timing, might be VOICE_START or VOICE_ACTIVE
    assert speech_results[2]["voice_state"] in [VADState.VOICE_START, VADState.VOICE_ACTIVE], \
        f"Frame 2: Should be VOICE_START or VOICE_ACTIVE after 3 sustained frames, got {speech_results[2]['voice_state']}"
    
    print("✅ Sustained speech detection working correctly")


@pytest.mark.asyncio
async def test_warmup_then_sustained_speech_flow(vad_service):
    """
    Test complete flow: warmup filter → sustained speech → VOICE_START.
    
    This simulates the Q7 test scenario:
    1. First 500ms: Warm-up silence/quiet frames → INACTIVE
    2. After 500ms: Real speech begins → Wait for sustained detection → VOICE_START
    """
    all_results = []
    
    # Phase 1: First 500ms with weak audio (warmup period)
    print("\n=== Phase 1: Warmup period (500ms) ===")
    for i in range(17):  # ~510ms
        chunk = generate_speech_chunk(amplitude=0.1)  # Weak signal (warm-up)
        result = await vad_service.process_audio_chunk(chunk, create_metadata())
        all_results.append(result)
        print(f"Warmup frame {i}: state={result['voice_state'].value}, warmup={result.get('warmup_active', False)}")
        await asyncio.sleep(0.001)
    
    # All warmup frames should be INACTIVE
    assert all(r["voice_state"] == VADState.INACTIVE for r in all_results[:17]), \
        "Warmup period should return INACTIVE"
    
    # Phase 2: After warmup, send strong speech
    print("\n=== Phase 2: Strong speech after warmup ===")
    for i in range(10):  # Send strong speech
        chunk = generate_speech_chunk(amplitude=0.8)
        result = await vad_service.process_audio_chunk(chunk, create_metadata())
        all_results.append(result)
        print(f"Speech frame {i}: state={result['voice_state'].value}, "
              f"detected={result['voice_detected']}, "
              f"sustained={result.get('sustained_speech_frames', 0)}")
        await asyncio.sleep(0.001)
    
    # Check that VOICE_START was eventually declared
    post_warmup_states = [r["voice_state"] for r in all_results[17:]]
    assert VADState.VOICE_START in post_warmup_states or VADState.VOICE_ACTIVE in post_warmup_states, \
        f"Speech should be detected after warmup, got states: {[s.value for s in post_warmup_states]}"
    
    print("\n✅ Complete warmup → sustained speech → VOICE_START flow working correctly")


@pytest.mark.asyncio
async def test_reset_clears_warmup_state(vad_service):
    """Test that reset() properly clears warmup filter state."""
    # Process some chunks to activate warmup
    for _ in range(5):
        chunk = generate_speech_chunk()
        await vad_service.process_audio_chunk(chunk, create_metadata())
    
    # Check warmup state is set
    assert vad_service.connection_start_time is not None, "Connection start time should be set"
    
    # Reset
    vad_service.reset()
    
    # Verify state cleared
    assert vad_service.connection_start_time is None, "Connection start time should be cleared"
    assert vad_service.warmup_complete == False, "Warmup complete flag should be reset"
    assert vad_service.sustained_speech_counter == 0, "Sustained speech counter should be reset"
    assert vad_service.current_state == VADState.INACTIVE, "State should be INACTIVE"
    
    print("✅ Reset properly clears warmup state")


@pytest.mark.asyncio
async def test_config_customization():
    """Test that custom warmup and sustained speech configs work."""
    # Create VAD with custom config
    custom_config = WebRTCVADConfig(
        warmup_filter_duration_ms=1000,  # 1 second warmup
        min_sustained_speech_frames=5,   # Require 5 frames
    )
    
    vad = WebRTCVADService(peer_id="test_custom", language="ar", config=custom_config)
    await vad.initialize()
    
    # Test warmup duration
    results = []
    for i in range(40):  # 40 * 30ms = 1200ms (ensure we exceed 1000ms)
        chunk = generate_speech_chunk()
        result = await vad.process_audio_chunk(chunk, create_metadata())
        results.append(result)
        await asyncio.sleep(0.035)  # 35ms delay to ensure proper timing
    
    # Check which frames had warmup active
    warmup_frames = [i for i, r in enumerate(results) if r.get("warmup_active", False)]
    non_warmup_frames = [i for i, r in enumerate(results) if not r.get("warmup_active", False)]
    
    print(f"Warmup active frames: {len(warmup_frames)} / {len(results)}")
    print(f"Non-warmup frames: {len(non_warmup_frames)}")
    
    # Should have warmup frames and non-warmup frames
    assert len(warmup_frames) > 0, "Should have warmup frames"
    assert len(non_warmup_frames) > 0, "Should have frames after warmup"
    
    # Warmup frames should be at the beginning
    if warmup_frames:
        assert warmup_frames[0] == 0, "Warmup should start from first frame"
        # Warmup frames should be consecutive from start
        for i, frame_idx in enumerate(warmup_frames):
            if i > 0:
                assert frame_idx == warmup_frames[i-1] + 1, "Warmup frames should be consecutive"
    
    print(f"✅ Custom config working: warmup={custom_config.warmup_filter_duration_ms}ms, "
          f"sustained_frames={custom_config.min_sustained_speech_frames}")
    
    await vad.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
