#!/usr/bin/env python3
"""
Periodic Noise Diagnostic Tool

Analyzes audio to detect periodic impulse noise (clicks, pops) and measure
their exact frequency, helping identify buffer underruns or electrical interference.

Usage:
    python tools/diagnose_periodic_noise.py --file path/to/audio.wav
"""

import argparse
import wave
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq


def load_wav(filepath: Path) -> Tuple[np.ndarray, int]:
    """Load WAV file and return audio data + sample rate."""
    with wave.open(str(filepath), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def detect_impulses(audio: np.ndarray, sample_rate: int, threshold_std: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect impulse events using derivative analysis.
    
    Returns:
        impulse_indices: Sample indices where impulses occur
        impulse_magnitudes: Magnitude of each impulse
    """
    # Compute first derivative (rate of change)
    diff = np.diff(audio)
    
    # Find peaks in the absolute derivative
    abs_diff = np.abs(diff)
    threshold = np.mean(abs_diff) + threshold_std * np.std(abs_diff)
    
    # Find peaks above threshold
    peaks, properties = signal.find_peaks(abs_diff, height=threshold, distance=sample_rate//200)
    
    return peaks, properties['peak_heights']


def analyze_periodicity(impulse_times: np.ndarray, sample_rate: int) -> dict:
    """
    Analyze the periodicity of detected impulses.
    
    Returns:
        Dictionary with frequency, interval, and regularity metrics
    """
    if len(impulse_times) < 2:
        return {"periodic": False, "reason": "Too few impulses detected"}
    
    # Convert sample indices to time (seconds)
    times_sec = impulse_times / sample_rate
    
    # Calculate inter-impulse intervals
    intervals = np.diff(times_sec)
    
    # Statistics
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval if mean_interval > 0 else float('inf')  # Coefficient of variation
    
    # Estimate fundamental frequency
    if mean_interval > 0:
        fundamental_freq = 1.0 / mean_interval
    else:
        fundamental_freq = 0
    
    # Check if periodic (low coefficient of variation indicates regularity)
    is_periodic = cv < 0.15  # Less than 15% variation
    
    return {
        "periodic": is_periodic,
        "fundamental_frequency_hz": fundamental_freq,
        "mean_interval_ms": mean_interval * 1000,
        "std_interval_ms": std_interval * 1000,
        "coefficient_of_variation": cv,
        "total_impulses": len(impulse_times),
        "impulses_per_second": len(impulse_times) / (times_sec[-1] - times_sec[0]) if len(times_sec) > 1 else 0,
        "intervals": intervals.tolist()[:20]  # First 20 intervals for inspection
    }


def compute_autocorrelation(audio: np.ndarray, max_lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation to find periodic patterns.
    
    Returns:
        lags: Lag values in samples
        autocorr: Autocorrelation values
    """
    if max_lag is None:
        max_lag = len(audio) // 2
    
    # Compute autocorrelation using FFT (much faster)
    audio_centered = audio - np.mean(audio)
    fft_result = fft(audio_centered, n=2*len(audio))
    autocorr_full = np.real(np.fft.ifft(fft_result * np.conj(fft_result)))
    
    # Normalize
    autocorr_full = autocorr_full / autocorr_full[0]
    
    # Take only positive lags up to max_lag
    lags = np.arange(max_lag)
    autocorr = autocorr_full[:max_lag]
    
    return lags, autocorr


def find_periodic_peaks(lags: np.ndarray, autocorr: np.ndarray, sample_rate: int, min_freq: float = 20.0) -> List[dict]:
    """
    Find peaks in autocorrelation indicating periodic patterns.
    
    Args:
        min_freq: Minimum frequency to consider (Hz)
    """
    # Ignore lag 0 (always maximum)
    search_lags = lags[1:]
    search_autocorr = autocorr[1:]
    
    # Find peaks
    max_lag = int(sample_rate / min_freq)
    search_mask = search_lags <= max_lag
    
    peaks, properties = signal.find_peaks(
        search_autocorr[search_mask],
        height=0.1,  # At least 10% correlation
        prominence=0.05
    )
    
    # Convert to frequencies
    periodic_freqs = []
    for peak_idx in peaks:
        lag = search_lags[search_mask][peak_idx]
        freq = sample_rate / lag
        corr = search_autocorr[search_mask][peak_idx]
        
        periodic_freqs.append({
            "frequency_hz": freq,
            "period_ms": (lag / sample_rate) * 1000,
            "autocorrelation": corr,
            "lag_samples": lag
        })
    
    # Sort by autocorrelation strength
    periodic_freqs.sort(key=lambda x: x['autocorrelation'], reverse=True)
    
    return periodic_freqs


def plot_diagnostics(audio: np.ndarray, sample_rate: int, impulse_indices: np.ndarray, 
                     lags: np.ndarray, autocorr: np.ndarray, output_path: Path):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Waveform with detected impulses
    time = np.arange(len(audio)) / sample_rate
    axes[0].plot(time, audio, 'b-', alpha=0.5, linewidth=0.5, label='Audio')
    if len(impulse_indices) > 0:
        impulse_times = impulse_indices / sample_rate
        axes[0].scatter(impulse_times, audio[impulse_indices], c='r', s=10, alpha=0.7, label='Detected Impulses')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Audio Waveform with Detected Impulses ({len(impulse_indices)} total)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Inter-impulse intervals histogram
    if len(impulse_indices) > 1:
        intervals_ms = np.diff(impulse_indices / sample_rate) * 1000
        axes[1].hist(intervals_ms, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(intervals_ms), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(intervals_ms):.2f} ms')
        axes[1].set_xlabel('Inter-impulse Interval (ms)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Distribution of Time Between Impulses')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient impulses for interval analysis', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # Plot 3: Autocorrelation
    lag_time_ms = lags / sample_rate * 1000
    axes[2].plot(lag_time_ms[:2000], autocorr[:2000], 'b-', linewidth=1)
    axes[2].set_xlabel('Lag (ms)')
    axes[2].set_ylabel('Autocorrelation')
    axes[2].set_title('Autocorrelation Function (First 2000 lags)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Diagnostic plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose periodic noise in audio files")
    parser.add_argument('--file', type=str, required=True, help='Path to WAV file')
    parser.add_argument('--threshold', type=float, default=3.0, help='Impulse detection threshold (std deviations)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # Load audio
    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return
    
    print(f"📂 Loading: {audio_path}")
    audio, sample_rate = load_wav(audio_path)
    duration = len(audio) / sample_rate
    print(f"   Duration: {duration:.2f}s @ {sample_rate} Hz")
    
    # Detect impulses
    print("\n🔍 Detecting impulse events...")
    impulse_indices, impulse_magnitudes = detect_impulses(audio, sample_rate, threshold_std=args.threshold)
    print(f"   Found {len(impulse_indices)} impulses")
    
    # Analyze periodicity
    if len(impulse_indices) >= 2:
        print("\n📊 Analyzing periodicity...")
        periodicity = analyze_periodicity(impulse_indices, sample_rate)
        
        print(f"\n{'='*80}")
        print("PERIODICITY ANALYSIS")
        print(f"{'='*80}")
        print(f"Periodic Pattern Detected: {'✅ YES' if periodicity['periodic'] else '❌ NO'}")
        print(f"Total Impulses: {periodicity['total_impulses']}")
        print(f"Impulses per Second: {periodicity['impulses_per_second']:.2f}")
        print(f"Fundamental Frequency: {periodicity['fundamental_frequency_hz']:.2f} Hz")
        print(f"Mean Interval: {periodicity['mean_interval_ms']:.2f} ms")
        print(f"Std Interval: {periodicity['std_interval_ms']:.2f} ms")
        print(f"Regularity (CV): {periodicity['coefficient_of_variation']:.4f} (lower = more regular)")
        
        if periodicity['periodic']:
            print(f"\n🎯 DIAGNOSIS: Highly periodic noise at ~{periodicity['fundamental_frequency_hz']:.1f} Hz")
            print(f"   Likely causes:")
            print(f"   1. Buffer underrun/overrun (interval = {periodicity['mean_interval_ms']:.2f}ms)")
            print(f"   2. Electrical interference from power supply")
            print(f"   3. Clock signal interference")
    else:
        print("⚠️  Too few impulses to analyze periodicity")
    
    # Autocorrelation analysis
    print("\n🔬 Computing autocorrelation...")
    max_lag = min(sample_rate * 2, len(audio) // 2)  # Up to 2 seconds
    lags, autocorr = compute_autocorrelation(audio, max_lag=max_lag)
    
    # Find periodic patterns
    periodic_patterns = find_periodic_peaks(lags, autocorr, sample_rate, min_freq=20.0)
    
    if periodic_patterns:
        print(f"\n{'='*80}")
        print("AUTOCORRELATION PERIODIC PATTERNS")
        print(f"{'='*80}")
        for i, pattern in enumerate(periodic_patterns[:5], 1):
            print(f"{i}. Frequency: {pattern['frequency_hz']:.2f} Hz | "
                  f"Period: {pattern['period_ms']:.2f} ms | "
                  f"Correlation: {pattern['autocorrelation']:.3f}")
    
    # Generate plots
    output_dir = Path(args.output) if args.output else audio_path.parent
    output_path = output_dir / f"periodic_noise_diagnosis_{audio_path.stem}.png"
    plot_diagnostics(audio, sample_rate, impulse_indices, lags, autocorr, output_path)
    
    print(f"\n{'='*80}")
    print("✅ Diagnostic complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
