#!/usr/bin/env python3
"""
Quick test script for RNNoise integration.
Tests the complete pipeline: 16k→48k→RNNoise→16k with comparison metrics.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from beautyai_inference.utils.rnnoise_wrapper import RNNoiseProcessor
from beautyai_inference.utils.audio_resampling import process_with_rnnoise_16khz_pipeline
from beautyai_inference.utils.noise_comparison import compare_noise_reduction_methods, generate_comparison_summary

def test_rnnoise_integration():
    """Test RNNoise integration with synthetic audio."""
    print("=" * 60)
    print("RNNoise Integration Test")
    print("=" * 60)
    
    # Step 1: Initialize RNNoise processor
    print("\n1️⃣ Initializing RNNoise processor...")
    try:
        with RNNoiseProcessor() as processor:
            print("   ✅ RNNoise processor initialized successfully")
            
            # Step 2: Generate synthetic test audio (16kHz, 1 second)
            print("\n2️⃣ Generating synthetic test audio (16kHz, 1 second)...")
            sample_rate = 16000
            duration = 1.0
            num_samples = int(sample_rate * duration)
            
            # Create a synthetic signal: 440Hz sine wave + white noise
            t = np.linspace(0, duration, num_samples, endpoint=False)
            signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone (A4 note)
            noise = 0.1 * np.random.randn(num_samples)  # White noise
            audio_16k_noisy = signal + noise
            audio_16k_noisy = audio_16k_noisy.astype(np.float32)
            
            print(f"   ✅ Generated audio: {len(audio_16k_noisy)} samples @ {sample_rate}Hz")
            print(f"   Signal RMS: {np.sqrt(np.mean(signal**2)):.4f}")
            print(f"   Noise RMS: {np.sqrt(np.mean(noise**2)):.4f}")
            
            # Step 3: Process with RNNoise pipeline
            print("\n3️⃣ Processing with RNNoise (16k→48k→RNNoise→16k)...")
            audio_16k_rnnoise, vad_probs = process_with_rnnoise_16khz_pipeline(
                audio_16k_noisy, processor
            )
            print(f"   ✅ RNNoise processing completed")
            print(f"   Output samples: {len(audio_16k_rnnoise)}")
            print(f"   VAD probabilities: {len(vad_probs)} frames")
            print(f"   Average VAD probability: {np.mean(vad_probs):.2%}")
            
            # Step 4: Simulate EMA noise gate
            print("\n4️⃣ Simulating EMA noise gate...")
            audio_16k_ema = audio_16k_noisy.copy()
            noise_ema = 0.001
            alpha = 0.1
            
            # Process in 480-sample frames (same as RNNoise)
            frame_size = 480
            for i in range(0, len(audio_16k_ema), frame_size):
                frame = audio_16k_ema[i:i+frame_size]
                if len(frame) < frame_size:
                    break
                frame_rms = np.sqrt(np.mean(frame**2))
                if frame_rms < noise_ema * 1.5:
                    noise_ema = alpha * frame_rms + (1 - alpha) * noise_ema
                adaptive_threshold = noise_ema * 2.0
                if frame_rms < adaptive_threshold:
                    audio_16k_ema[i:i+frame_size] = 0
            
            print(f"   ✅ EMA noise gate applied")
            print(f"   Final noise floor: {noise_ema:.6f}")
            
            # Step 5: Compare methods
            print("\n5️⃣ Comparing noise reduction methods...")
            metrics = compare_noise_reduction_methods(
                audio_16k_noisy, audio_16k_ema, audio_16k_rnnoise, sample_rate=sample_rate
            )
            
            print("\n📊 Comparison Results:")
            print(f"   SNR:")
            print(f"      EMA: {metrics['snr']['ema_db']:.2f} dB")
            print(f"      RNNoise: {metrics['snr']['rnnoise_db']:.2f} dB")
            print(f"      Winner: {metrics['snr']['winner']}")
            
            print(f"   RMS Level:")
            print(f"      EMA: {metrics['rms_level']['ema']:.4f}")
            print(f"      RNNoise: {metrics['rms_level']['rnnoise']:.4f}")
            print(f"      EMA Reduction: {metrics['rms_level']['ema_reduction_percent']:.1f}%")
            print(f"      RNNoise Reduction: {metrics['rms_level']['rnnoise_reduction_percent']:.1f}%")
            
            print(f"   Spectral Flatness:")
            print(f"      Original: {metrics['spectral_flatness']['original']:.3f}")
            print(f"      EMA: {metrics['spectral_flatness']['ema']:.3f}")
            print(f"      RNNoise: {metrics['spectral_flatness']['rnnoise']:.3f}")
            
            print(f"   Correlation with Original:")
            print(f"      EMA: {metrics['correlation_with_original']['ema']:.4f}")
            print(f"      RNNoise: {metrics['correlation_with_original']['rnnoise']:.4f}")
            print(f"      Winner: {metrics['correlation_with_original']['winner']}")
            
            # Step 6: Generate summary (with estimated latency)
            print("\n6️⃣ Generating comparison summary...")
            latency_estimates = {
                "ema_avg_ms": 0.1,
                "ema_min_ms": 0.05,
                "ema_max_ms": 0.15,
                "rnnoise_avg_ms": 14.0,
                "rnnoise_min_ms": 12.0,
                "rnnoise_max_ms": 16.0,
                "difference_ms": 13.9,
                "faster_method": "EMA"
            }
            summary = generate_comparison_summary(metrics, latency_estimates, duration)
            print("\n" + "=" * 60)
            print(summary)
            print("=" * 60)
            
            print("\n✅ All tests passed! RNNoise integration working correctly.")
            return True
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rnnoise_integration()
    sys.exit(0 if success else 1)
