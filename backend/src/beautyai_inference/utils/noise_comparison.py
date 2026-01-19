"""
Noise reduction comparison utilities for experimental layer testing.
Compares EMA vs RNNoise performance metrics.

Author: Lumina Ashley
Date: November 10, 2025
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    Args:
        signal: Original signal
        noise: Noise component (difference between original and processed)
    
    Returns:
        SNR in decibels (higher is better)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate Root Mean Square of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def calculate_spectral_flatness(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Calculate spectral flatness (measure of how noise-like vs tone-like).
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate in Hz
    
    Returns:
        Spectral flatness (0 = pure tone, 1 = white noise)
    """
    # Apply FFT
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    
    # Avoid log(0)
    magnitude = np.maximum(magnitude, 1e-10)
    
    # Geometric mean / Arithmetic mean
    geometric_mean = np.exp(np.mean(np.log(magnitude)))
    arithmetic_mean = np.mean(magnitude)
    
    if arithmetic_mean == 0:
        return 0.0
    
    flatness = geometric_mean / arithmetic_mean
    return float(flatness)


def compare_noise_reduction_methods(
    original_audio: np.ndarray,
    ema_output: np.ndarray,
    rnnoise_output: np.ndarray,
    sample_rate: int = 16000
) -> Dict[str, any]:
    """
    Compare EMA and RNNoise noise reduction performance.
    
    Args:
        original_audio: Original 16kHz audio before noise reduction
        ema_output: Audio after EMA noise gate
        rnnoise_output: Audio after RNNoise processing
        sample_rate: Sample rate in Hz
    
    Returns:
        Dictionary with comparison metrics
    """
    # Calculate noise components
    ema_noise = original_audio - ema_output
    rnnoise_noise = original_audio - rnnoise_output
    
    # SNR calculations
    ema_snr = calculate_snr(original_audio, ema_noise)
    rnnoise_snr = calculate_snr(original_audio, rnnoise_noise)
    
    # RMS levels
    original_rms = calculate_rms(original_audio)
    ema_rms = calculate_rms(ema_output)
    rnnoise_rms = calculate_rms(rnnoise_output)
    
    # Spectral flatness (noise-likeness)
    original_flatness = calculate_spectral_flatness(original_audio, sample_rate)
    ema_flatness = calculate_spectral_flatness(ema_output, sample_rate)
    rnnoise_flatness = calculate_spectral_flatness(rnnoise_output, sample_rate)
    
    # Cross-correlation (similarity measure)
    ema_correlation = np.corrcoef(original_audio, ema_output)[0, 1]
    rnnoise_correlation = np.corrcoef(original_audio, rnnoise_output)[0, 1]
    
    comparison = {
        "snr": {
            "ema_db": round(ema_snr, 2),
            "rnnoise_db": round(rnnoise_snr, 2),
            "winner": "EMA" if ema_snr > rnnoise_snr else "RNNoise",
            "difference_db": round(abs(ema_snr - rnnoise_snr), 2)
        },
        "rms_level": {
            "original": round(original_rms, 4),
            "ema": round(ema_rms, 4),
            "rnnoise": round(rnnoise_rms, 4),
            "ema_reduction_percent": round((1 - ema_rms / original_rms) * 100, 2) if original_rms > 0 else 0,
            "rnnoise_reduction_percent": round((1 - rnnoise_rms / original_rms) * 100, 2) if original_rms > 0 else 0
        },
        "spectral_flatness": {
            "original": round(original_flatness, 3),
            "ema": round(ema_flatness, 3),
            "rnnoise": round(rnnoise_flatness, 3),
            "ema_improvement": round((original_flatness - ema_flatness), 3),
            "rnnoise_improvement": round((original_flatness - rnnoise_flatness), 3)
        },
        "correlation_with_original": {
            "ema": round(float(ema_correlation), 4),
            "rnnoise": round(float(rnnoise_correlation), 4),
            "winner": "EMA" if ema_correlation > rnnoise_correlation else "RNNoise"
        }
    }
    
    return comparison


def measure_processing_latency(
    audio_16khz: np.ndarray,
    ema_processor_func: callable,
    rnnoise_processor_func: callable,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Measure and compare processing latency for both methods.
    
    Args:
        audio_16khz: Test audio at 16kHz
        ema_processor_func: Function that applies EMA noise gate
        rnnoise_processor_func: Function that applies RNNoise
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with latency measurements in milliseconds
    """
    ema_times = []
    rnnoise_times = []
    
    # Warm-up run
    try:
        _ = ema_processor_func(audio_16khz.copy())
        _ = rnnoise_processor_func(audio_16khz.copy())
    except Exception as e:
        logger.warning(f"Warm-up failed: {e}")
    
    # Measure EMA
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            _ = ema_processor_func(audio_16khz.copy())
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            ema_times.append(elapsed)
        except Exception as e:
            logger.error(f"EMA processing error: {e}")
    
    # Measure RNNoise
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            _ = rnnoise_processor_func(audio_16khz.copy())
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            rnnoise_times.append(elapsed)
        except Exception as e:
            logger.error(f"RNNoise processing error: {e}")
    
    return {
        "ema_avg_ms": round(np.mean(ema_times), 2) if ema_times else 0,
        "ema_min_ms": round(np.min(ema_times), 2) if ema_times else 0,
        "ema_max_ms": round(np.max(ema_times), 2) if ema_times else 0,
        "rnnoise_avg_ms": round(np.mean(rnnoise_times), 2) if rnnoise_times else 0,
        "rnnoise_min_ms": round(np.min(rnnoise_times), 2) if rnnoise_times else 0,
        "rnnoise_max_ms": round(np.max(rnnoise_times), 2) if rnnoise_times else 0,
        "difference_ms": round(abs(np.mean(rnnoise_times) - np.mean(ema_times)), 2) if ema_times and rnnoise_times else 0,
        "faster_method": "EMA" if (ema_times and rnnoise_times and np.mean(ema_times) < np.mean(rnnoise_times)) else "RNNoise"
    }


def generate_comparison_summary(
    quality_metrics: Dict,
    latency_metrics: Dict,
    audio_duration_s: float
) -> str:
    """
    Generate human-readable summary of comparison results.
    
    Args:
        quality_metrics: Output from compare_noise_reduction_methods()
        latency_metrics: Output from measure_processing_latency()
        audio_duration_s: Duration of audio processed
    
    Returns:
        Formatted summary string
    """
    summary_lines = [
        "\n" + "="*80,
        "🔊 NOISE REDUCTION COMPARISON SUMMARY",
        "="*80,
        f"\n📊 Audio Duration: {audio_duration_s:.2f}s\n",
        
        "🎯 QUALITY METRICS:",
        f"  SNR (Signal-to-Noise Ratio):",
        f"    • EMA:     {quality_metrics['snr']['ema_db']:.2f} dB",
        f"    • RNNoise: {quality_metrics['snr']['rnnoise_db']:.2f} dB",
        f"    • Winner:  {quality_metrics['snr']['winner']} (+{quality_metrics['snr']['difference_db']:.2f} dB)",
        f"",
        f"  RMS Noise Reduction:",
        f"    • EMA:     {quality_metrics['rms_level']['ema_reduction_percent']:.1f}%",
        f"    • RNNoise: {quality_metrics['rms_level']['rnnoise_reduction_percent']:.1f}%",
        f"",
        f"  Signal Preservation (correlation with original):",
        f"    • EMA:     {quality_metrics['correlation_with_original']['ema']:.4f}",
        f"    • RNNoise: {quality_metrics['correlation_with_original']['rnnoise']:.4f}",
        f"    • Winner:  {quality_metrics['correlation_with_original']['winner']}",
        f"",
        
        "⚡ LATENCY METRICS:",
        f"  EMA Processing:",
        f"    • Average: {latency_metrics['ema_avg_ms']:.2f} ms",
        f"    • Range:   {latency_metrics['ema_min_ms']:.2f} - {latency_metrics['ema_max_ms']:.2f} ms",
        f"",
        f"  RNNoise Processing:",
        f"    • Average: {latency_metrics['rnnoise_avg_ms']:.2f} ms",
        f"    • Range:   {latency_metrics['rnnoise_min_ms']:.2f} - {latency_metrics['rnnoise_max_ms']:.2f} ms",
        f"",
        f"  Speed Comparison:",
        f"    • Faster:     {latency_metrics['faster_method']}",
        f"    • Difference: {latency_metrics['difference_ms']:.2f} ms",
        f"",
        
        "="*80 + "\n"
    ]
    
    return "\n".join(summary_lines)
