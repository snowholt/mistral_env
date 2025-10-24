#!/usr/bin/env python3
"""
Test VAD improvements with 5-second audio samples
Validates that audio processing captures full utterance duration
"""

import sys
import os
import time
import asyncio
from pathlib import Path

sys.path.insert(0, '/home/lumi/beautyai/backend/src')

from beautyai_inference.services.voice.vad.webrtc_vad_service import (
    WebRTCVADService, WebRTCVADConfig, VADState
)
from beautyai_inference.services.voice.webrtc_audio_processor import (
    WebRTCAudioProcessor, AudioProcessingConfig
)
from beautyai_inference.core.webrtc_buffer_manager import (
    WebRTCBufferManager, BufferConfig
)

# Test files with known durations
test_files = [
    ("voice_tests/input_test_questions/pcm/q7.pcm", 5.09, "Arabic question 7"),
    ("voice_tests/input_test_questions/pcm/q10.pcm", 5.37, "Arabic question 10"),
    ("voice_tests/input_test_questions/pcm/q4.pcm", 4.05, "Arabic question 4"),
]

print("=" * 80)
print("VAD Configuration Improvement Test")
print("=" * 80)
print("\nNew Configuration:")
print("  - WebRTC Sensitivity: 2 (less aggressive)")
print("  - Min Speech Duration: 150ms (was 300ms)")
print("  - Post-Speech Silence: 800ms (was 500ms)")
print("  - Arabic Threshold: 0.05 (was 0.002)")
print("  - VAD Logging: ENABLED")
print("=" * 80)


async def test_audio_file(file_path: str, expected_duration: float, description: str):
    """Test audio processing with new VAD configuration"""
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {description}")
    print(f"File: {file_path}")
    print(f"Expected Duration: {expected_duration:.2f}s")
    
    if not os.path.exists(file_path):
        print(f"   ⚠️  File not found!")
        return None
    
    # Read audio file
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    
    file_size = len(audio_data)
    actual_duration = file_size / (16000 * 2)  # 16kHz mono 16-bit
    print(f"Actual File Duration: {actual_duration:.2f}s ({file_size:,} bytes)")
    
    # Create VAD service with new configuration
    vad_config = WebRTCVADConfig(
        webrtc_sensitivity=2,  # Less aggressive
        min_speech_duration_ms=150,
        post_speech_silence_ms=800,
        log_vad_decisions=True
    )
    
    vad = WebRTCVADService(
        peer_id=f"test_{Path(file_path).stem}",
        language="ar",
        config=vad_config
    )
    
    # Initialize VAD
    await vad.initialize()
    
    # Process audio in 30ms chunks (simulating real-time streaming)
    chunk_size = int(16000 * 2 * 0.03)  # 30ms at 16kHz mono 16-bit = 960 bytes
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
    
    print(f"\nProcessing {len(chunks)} chunks...")
    
    speech_detected_chunks = 0
    voice_active_time = 0.0
    state_transitions = []
    last_state = VADState.INACTIVE
    
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < chunk_size:
            # Pad last chunk if needed
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        
        metadata = {
            "chunk_index": i,
            "timestamp": time.time(),
            "sample_rate": 16000
        }
        
        result = await vad.process_audio_chunk(chunk, metadata)
        
        if result["success"]:
            if result["voice_detected"]:
                speech_detected_chunks += 1
            
            # Count VOICE_START, VOICE_ACTIVE, and VOICE_END_PENDING as active speech
            # (VOICE_END_PENDING is still within the utterance, just silent pause)
            if result["voice_state"] in [VADState.VOICE_START, VADState.VOICE_ACTIVE, VADState.VOICE_END_PENDING]:
                voice_active_time += 0.03  # 30ms per chunk
            
            # Track state transitions
            current_state = result["voice_state"]
            if current_state != last_state:
                state_transitions.append({
                    "chunk": i,
                    "time": i * 0.03,
                    "from": last_state.value if isinstance(last_state, VADState) else last_state,
                    "to": current_state.value if isinstance(current_state, VADState) else current_state
                })
                last_state = current_state
    
    processing_time = time.time() - start_time
    
    # Get final metrics
    metrics = vad.get_metrics()
    
    # Calculate results
    capture_percentage = (voice_active_time / actual_duration) * 100 if actual_duration > 0 else 0
    
    print(f"\n{'=' * 80}")
    print("RESULTS:")
    print(f"{'=' * 80}")
    print(f"File Duration:           {actual_duration:.2f}s")
    print(f"Voice Active Time:       {voice_active_time:.2f}s")
    print(f"Capture Percentage:      {capture_percentage:.1f}%")
    print(f"Speech Detected Chunks:  {speech_detected_chunks}/{len(chunks)}")
    print(f"Processing Time:         {processing_time:.3f}s")
    
    print(f"\nVAD Metrics:")
    print(f"  WebRTC Detections:     {metrics['webrtc_detections']}")
    print(f"  Silero Confirmations:  {metrics['silero_confirmations']}")
    print(f"  False Positives:       {metrics['false_positives']}")
    print(f"  Speech Segments:       {metrics['speech_segments']}")
    print(f"  Current State:         {metrics['current_state']}")
    
    if state_transitions:
        print(f"\nState Transitions ({len(state_transitions)} total):")
        for trans in state_transitions[:10]:  # Show first 10
            print(f"  [{trans['time']:.2f}s] {trans['from']} → {trans['to']}")
        if len(state_transitions) > 10:
            print(f"  ... and {len(state_transitions) - 10} more")
    
    # Determine pass/fail
    success = capture_percentage >= 90.0  # At least 90% captured
    status = "✅ PASS" if success else "❌ FAIL"
    
    print(f"\n{status}: {'Audio capture meets target' if success else 'Audio capture below 90% threshold'}")
    
    return {
        "file": file_path,
        "description": description,
        "expected_duration": expected_duration,
        "actual_duration": actual_duration,
        "voice_active_time": voice_active_time,
        "capture_percentage": capture_percentage,
        "success": success,
        "metrics": metrics,
        "state_transitions": len(state_transitions)
    }


async def run_tests():
    """Run all audio file tests"""
    
    results = []
    
    for file_path, expected_duration, description in test_files:
        try:
            result = await test_audio_file(file_path, expected_duration, description)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR testing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    if not results:
        print("No tests completed successfully")
        return
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    print(f"\n{'File':<40} {'Duration':<12} {'Captured':<12} {'%':<8} {'Status'}")
    print("-" * 80)
    
    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"{Path(r['file']).name:<40} {r['actual_duration']:>6.2f}s     "
              f"{r['voice_active_time']:>6.2f}s     {r['capture_percentage']:>6.1f}%  {status}")
    
    avg_capture = sum(r["capture_percentage"] for r in results) / len(results)
    print(f"\nAverage Capture Rate: {avg_capture:.1f}%")
    
    if avg_capture >= 90.0:
        print("\n✅ SUCCESS: VAD improvements effective! Audio capture rate meets target.")
    else:
        print("\n⚠️  WARNING: Audio capture rate still below target. Further tuning needed.")


if __name__ == "__main__":
    asyncio.run(run_tests())
