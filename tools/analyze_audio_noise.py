#!/usr/bin/env python3
"""
Audio Noise Analysis Tool

Analyzes captured audio files to identify noise characteristics,
aliasing artifacts, and recommend optimal noise filtering strategies.

Usage:
    python tools/analyze_audio_noise.py [--layer LAYER] [--visualize]
"""

import argparse
import json
import wave
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq


def convert_to_native_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


def load_wav(filepath: Path) -> Tuple[np.ndarray, int]:
    """Load WAV file and return audio data + sample rate."""
    with wave.open(str(filepath), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)
        
        # Convert to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio, sample_rate


def compute_spectrum(audio: np.ndarray, sample_rate: int, max_freq: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frequency spectrum using FFT."""
    n = len(audio)
    spectrum = np.abs(fft(audio))
    freqs = fftfreq(n, 1/sample_rate)
    
    # Take only positive frequencies
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    spectrum = spectrum[positive_mask]
    
    if max_freq:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        spectrum = spectrum[freq_mask]
    
    return freqs, spectrum


def compute_noise_floor(spectrum: np.ndarray, percentile: float = 10) -> float:
    """Estimate noise floor using lower percentile of spectrum."""
    return np.percentile(spectrum, percentile)


def detect_aliasing(freqs: np.ndarray, spectrum: np.ndarray, sample_rate: int) -> Dict:
    """Detect aliasing artifacts near Nyquist frequency."""
    nyquist = sample_rate / 2
    
    # Check energy in the "danger zone" (above 75% of Nyquist)
    danger_start = nyquist * 0.75
    danger_mask = freqs >= danger_start
    danger_energy = np.sum(spectrum[danger_mask]**2)
    total_energy = np.sum(spectrum**2)
    danger_ratio = danger_energy / total_energy if total_energy > 0 else 0
    
    # Find peaks near Nyquist (potential aliasing)
    near_nyquist_mask = (freqs >= nyquist * 0.85) & (freqs <= nyquist * 0.98)
    near_nyquist_spectrum = spectrum[near_nyquist_mask]
    near_nyquist_freqs = freqs[near_nyquist_mask]
    
    if len(near_nyquist_spectrum) > 0:
        # Find peaks
        peaks, properties = signal.find_peaks(near_nyquist_spectrum, height=np.percentile(near_nyquist_spectrum, 90))
        peak_freqs = near_nyquist_freqs[peaks] if len(peaks) > 0 else []
    else:
        peak_freqs = []
    
    return {
        "danger_zone_energy_ratio": danger_ratio,
        "aliasing_peaks": peak_freqs.tolist() if len(peak_freqs) > 0 else [],
        "nyquist_frequency": nyquist,
        "assessment": "HIGH" if danger_ratio > 0.05 else "MODERATE" if danger_ratio > 0.01 else "LOW"
    }


def compute_snr(audio: np.ndarray, noise_percentile: float = 10) -> float:
    """Estimate Signal-to-Noise Ratio."""
    noise_floor = np.percentile(np.abs(audio), noise_percentile)
    signal_level = np.percentile(np.abs(audio), 90)
    
    if noise_floor > 0:
        snr_db = 20 * np.log10(signal_level / noise_floor)
    else:
        snr_db = float('inf')
    
    return snr_db


def analyze_crackle_artifacts(audio: np.ndarray, sample_rate: int) -> Dict:
    """Detect sudden discontinuities that cause crackle sounds."""
    # Compute first-order difference (sudden changes)
    diff = np.abs(np.diff(audio))
    
    # Find sudden spikes (potential crackles)
    threshold = np.percentile(diff, 99.5)  # Top 0.5% of changes
    crackle_indices = np.where(diff > threshold)[0]
    
    # Count crackles per second
    duration = len(audio) / sample_rate
    crackles_per_sec = len(crackle_indices) / duration if duration > 0 else 0
    
    return {
        "crackle_count": len(crackle_indices),
        "crackles_per_second": crackles_per_sec,
        "max_discontinuity": np.max(diff),
        "assessment": "HIGH" if crackles_per_sec > 5 else "MODERATE" if crackles_per_sec > 1 else "LOW"
    }


def compare_layers(base_audio: np.ndarray, processed_audio: np.ndarray) -> Dict:
    """Compare original vs processed audio."""
    # Ensure same length
    min_len = min(len(base_audio), len(processed_audio))
    base_audio = base_audio[:min_len]
    processed_audio = processed_audio[:min_len]
    
    # Correlation
    correlation = np.corrcoef(base_audio, processed_audio)[0, 1]
    
    # RMS comparison
    base_rms = np.sqrt(np.mean(base_audio**2))
    processed_rms = np.sqrt(np.mean(processed_audio**2))
    rms_reduction = ((base_rms - processed_rms) / base_rms * 100) if base_rms > 0 else 0
    
    return {
        "correlation": correlation,
        "base_rms": base_rms,
        "processed_rms": processed_rms,
        "rms_reduction_percent": rms_reduction
    }


def generate_spectrogram(audio: np.ndarray, sample_rate: int, output_path: Path, title: str):
    """Generate and save spectrogram visualization."""
    plt.figure(figsize=(12, 6))
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024)
    
    # Plot
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.ylim([0, sample_rate / 2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  📊 Saved spectrogram: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze audio noise characteristics')
    parser.add_argument('--layer', type=str, choices=['all', '3', '31', '32', '4'], default='all',
                        help='Which layer to analyze (default: all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate spectrograms and plots')
    parser.add_argument('--compare', action='store_true',
                        help='Compare EMA vs RNNoise performance')
    args = parser.parse_args()
    
    # Define paths
    audio_dir = Path(__file__).resolve().parents[1] / "reports/debug/webrtc"
    output_dir = Path(__file__).resolve().parents[1] / "reports/debug/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = {
        '3': audio_dir / "layer3_16khz.wav",          # Baseline (downsampled, no noise reduction)
        '31': audio_dir / "layer31_ema_16khz.wav",    # EMA noise gate
        '32': audio_dir / "layer32_rnnoise_16khz.wav", # RNNoise
        '33': audio_dir / "layer33_dtln_16khz.wav",   # DTLN
        '34': audio_dir / "layer34_deepfilternet_16khz.wav", # DeepFilterNet
        '35': audio_dir / "layer35_nsnet2_16khz.wav", # Spectral Gating
        '36': audio_dir / "layer36_comb_16khz.wav",   # Comb Filter (80 Hz removal)
        '4': audio_dir / "layer4_16khz_vad_filtered.wav" # VAD filtered
    }
    
    if args.layer != 'all':
        layers = {args.layer: layers[args.layer]}
    
    results = {}
    
    print("\n" + "="*80)
    print("🔊 AUDIO NOISE ANALYSIS")
    print("="*80 + "\n")
    
    # Load baseline (Layer 3) for comparison
    baseline_audio, baseline_sr = None, None
    if (audio_dir / "layer3_16khz.wav").exists():
        baseline_audio, baseline_sr = load_wav(audio_dir / "layer3_16khz.wav")
    
    for layer_name, layer_path in layers.items():
        if not layer_path.exists():
            print(f"⚠️  Layer {layer_name} not found: {layer_path}")
            continue
        
        print(f"📂 Analyzing Layer {layer_name}: {layer_path.name}")
        print("-" * 80)
        
        # Load audio
        audio, sample_rate = load_wav(layer_path)
        duration = len(audio) / sample_rate
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Samples: {len(audio)}")
        
        # Compute spectrum
        freqs, spectrum = compute_spectrum(audio, sample_rate, max_freq=sample_rate//2)
        
        # SNR
        snr = compute_snr(audio)
        print(f"\n  📊 Signal-to-Noise Ratio: {snr:.2f} dB")
        
        # Aliasing detection
        aliasing = detect_aliasing(freqs, spectrum, sample_rate)
        print(f"\n  🎛️  Aliasing Assessment: {aliasing['assessment']}")
        print(f"      Danger zone energy: {aliasing['danger_zone_energy_ratio']*100:.2f}%")
        print(f"      Nyquist frequency: {aliasing['nyquist_frequency']} Hz")
        if aliasing['aliasing_peaks']:
            print(f"      Aliasing peaks found: {len(aliasing['aliasing_peaks'])} peaks")
        
        # Crackle detection
        crackles = analyze_crackle_artifacts(audio, sample_rate)
        print(f"\n  ⚡ Crackle Assessment: {crackles['assessment']}")
        print(f"      Crackle events: {crackles['crackle_count']}")
        print(f"      Crackles per second: {crackles['crackles_per_second']:.2f}")
        print(f"      Max discontinuity: {crackles['max_discontinuity']:.6f}")
        
        # Comparison with baseline
        if baseline_audio is not None and layer_name != '3':
            comparison = compare_layers(baseline_audio, audio)
            print(f"\n  🔄 Comparison with Layer 3 (baseline):")
            print(f"      Correlation: {comparison['correlation']:.4f}")
            print(f"      RMS reduction: {comparison['rms_reduction_percent']:.2f}%")
        
        # Store results
        results[layer_name] = {
            "duration_s": duration,
            "sample_rate": sample_rate,
            "snr_db": snr,
            "aliasing": aliasing,
            "crackles": crackles
        }
        
        # Generate visualizations
        if args.visualize:
            spectrogram_path = output_dir / f"spectrogram_layer{layer_name}.png"
            generate_spectrogram(audio, sample_rate, spectrogram_path, f"Layer {layer_name} Spectrogram")
        
        print()
    
    # Comparison summary
    if args.compare and '31' in results and '32' in results:
        print("\n" + "="*80)
        print("🆚 EMA vs RNNoise Comparison")
        print("="*80 + "\n")
        
        ema = results['31']
        rnnoise = results['32']
        
        print("Signal-to-Noise Ratio:")
        print(f"  EMA:     {ema['snr_db']:.2f} dB")
        print(f"  RNNoise: {rnnoise['snr_db']:.2f} dB")
        print(f"  Winner:  {'EMA' if ema['snr_db'] > rnnoise['snr_db'] else 'RNNoise'}")
        
        print("\nCrackle Artifacts:")
        print(f"  EMA:     {ema['crackles']['crackles_per_second']:.2f} crackles/sec ({ema['crackles']['assessment']})")
        print(f"  RNNoise: {rnnoise['crackles']['crackles_per_second']:.2f} crackles/sec ({rnnoise['crackles']['assessment']})")
        print(f"  Winner:  {'EMA' if ema['crackles']['crackles_per_second'] < rnnoise['crackles']['crackles_per_second'] else 'RNNoise'}")
        
        print("\nAliasing Risk:")
        print(f"  EMA:     {ema['aliasing']['assessment']} ({ema['aliasing']['danger_zone_energy_ratio']*100:.2f}%)")
        print(f"  RNNoise: {rnnoise['aliasing']['assessment']} ({rnnoise['aliasing']['danger_zone_energy_ratio']*100:.2f}%)")
    
    # Save JSON report
    report_path = output_dir / "noise_analysis_report.json"
    with open(report_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        results_native = convert_to_native_types(results)
        json.dump(results_native, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete! Report saved to: {report_path}")
    print("="*80 + "\n")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 80)
    
    if '3' in results:
        baseline = results['3']
        if baseline['aliasing']['assessment'] in ['HIGH', 'MODERATE']:
            print("  ⚠️  Aliasing detected in baseline! Recommendations:")
            print("      1. Lower anti-aliasing cutoff frequency (try 7000 Hz or 6500 Hz)")
            print("      2. Increase filter order (try 10th-order Butterworth)")
            print("      3. Use multi-stage downsampling (48k→24k→16k)")
        
        if baseline['crackles']['assessment'] in ['HIGH', 'MODERATE']:
            print("  ⚠️  Crackle artifacts detected! Likely causes:")
            print("      1. EMA noise gate too aggressive (increase threshold_multiplier)")
            print("      2. Resampling artifacts (try different Kaiser beta)")
            print("      3. Fan noise + aliasing interaction")
    
    if '31' in results and results['31']['crackles']['assessment'] != 'LOW':
        print("\n  🎛️  EMA noise gate tuning suggestions:")
        print("      - Increase threshold_multiplier: 2.0 → 2.5 or 3.0 (less aggressive)")
        print("      - Decrease alpha: 0.1 → 0.05 (slower adaptation, smoother)")
        print("      - Add attack/release smoothing to prevent sudden zeros")
    
    if '32' in results and results['32']['snr_db'] < results.get('31', {}).get('snr_db', 0):
        print("\n  🤖 RNNoise is underperforming:")
        print("      - RNNoise may not be suitable for this noise type")
        print("      - Consider disabling RNNoise in production")
        print("      - Stick with optimized EMA for real-time performance")
    
    print("\n  📚 Noise reduction methods available:")
    print("      1. EMA Gate (Layer 3.1): Exponential Moving Average noise gate")
    print("      2. RNNoise (Layer 3.2): Xiph.org lightweight denoiser")
    print("      3. DTLN (Layer 3.3): Dual-signal Transformation LSTM")
    print("      4. DeepFilterNet (Layer 3.4): Facebook/Meta multi-band approach")
    print("      5. Spectral Gating (Layer 3.5): noisereduce stationary noise removal")
    print("      6. Comb Filter (Layer 3.6): Multi-notch IIR for periodic noise (80 Hz)")
    print()


if __name__ == "__main__":
    main()
