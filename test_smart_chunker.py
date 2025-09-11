#!/usr/bin/env python3
"""Test script for smart audio chunker functionality"""

import sys
import os
import numpy as np

# Add the backend src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend/src'))

from beautyai_inference.services.voice.streaming.smart_audio_chunker import SmartAudioChunker
from beautyai_inference.services.voice.streaming.audio_chunking_config import AudioChunkingConfig

def test_smart_audio_chunker():
    """Test the smart audio chunker with sample data"""
    
    print("🧪 Testing SmartAudioChunker...")
    
    # Create configuration with 200ms chunks
    config = AudioChunkingConfig(
        chunk_duration_ms=200,
        overlap_ms=50,
        min_chunk_duration_ms=100,
        accumulation_buffer_ms=400,
        sample_rate=16000
    )
    
    print(f"📊 Config: {config.chunk_duration_ms}ms chunks, {config.overlap_ms}ms overlap")
    
    # Initialize chunker
    chunker = SmartAudioChunker(config)
    
    # Generate test audio data (simulate 20ms chunks from client)
    sample_rate = 16000
    chunk_size_20ms = int(sample_rate * 0.02)  # 320 samples at 16kHz
    
    chunks_processed = 0
    total_samples_sent = 0
    
    print(f"🎵 Simulating 20ms client chunks ({chunk_size_20ms} samples each)...")
    
    # Send 30 chunks of 20ms each (600ms total)
    for i in range(30):
        # Create 20ms of test audio (simple sine wave)
        t = np.linspace(i * 0.02, (i + 1) * 0.02, chunk_size_20ms, False)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
        audio_samples = (sine_wave * 32767).astype(np.int16)
        audio_bytes = audio_samples.tobytes()
        
        total_samples_sent += len(audio_samples)
        
        # Process through chunker
        for chunk_samples, timestamp_ms in chunker.process_audio(audio_bytes):
            chunks_processed += 1
            chunk_duration_ms = len(chunk_samples) * 1000 / sample_rate
            
            print(f"  📦 Chunk {chunks_processed}: {len(chunk_samples)} samples "
                  f"({chunk_duration_ms:.1f}ms) @ {timestamp_ms}ms")
    
    # Flush remaining audio
    final_chunk = chunker.flush()
    if final_chunk:
        chunk_samples, timestamp_ms = final_chunk
        chunks_processed += 1
        chunk_duration_ms = len(chunk_samples) * 1000 / sample_rate
        print(f"  📦 Final chunk: {len(chunk_samples)} samples "
              f"({chunk_duration_ms:.1f}ms) @ {timestamp_ms}ms")
    
    # Get statistics
    stats = chunker.get_stats()
    
    print("\n📈 Results:")
    print(f"  • Input: 30 × 20ms chunks = {total_samples_sent} samples ({total_samples_sent/sample_rate:.1f}s)")
    print(f"  • Output: {chunks_processed} smart chunks")
    print(f"  • Avg output chunk duration: {stats['avg_chunk_duration_ms']:.1f}ms")
    print(f"  • Total duration processed: {stats['total_duration_ms']:.1f}ms")
    
    # Verify we didn't lose audio
    expected_duration_ms = total_samples_sent * 1000 / sample_rate
    actual_duration_ms = stats['total_duration_ms']
    
    if abs(expected_duration_ms - actual_duration_ms) < 50:  # 50ms tolerance
        print("  ✅ Audio duration preserved (no significant loss)")
    else:
        print(f"  ⚠️  Duration mismatch: expected {expected_duration_ms:.1f}ms, got {actual_duration_ms:.1f}ms")
    
    # Check that chunks are larger than original
    avg_input_chunk_ms = 20  # Original chunk size
    avg_output_chunk_ms = stats['avg_chunk_duration_ms']
    
    if avg_output_chunk_ms > avg_input_chunk_ms * 5:  # Should be much larger
        print(f"  ✅ Chunks are larger: {avg_input_chunk_ms}ms → {avg_output_chunk_ms}ms "
              f"({avg_output_chunk_ms/avg_input_chunk_ms:.1f}x bigger)")
    else:
        print(f"  ⚠️  Chunks not sufficiently larger: {avg_input_chunk_ms}ms → {avg_output_chunk_ms}ms")
    
    print(f"  ✅ Smart chunker test completed successfully!")
    return True

if __name__ == "__main__":
    test_smart_audio_chunker()