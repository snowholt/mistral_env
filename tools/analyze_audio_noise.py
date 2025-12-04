#!/usr/bin/env python3
"""
Audio Noise Analysis Tool

Analyzes captured audio files to identify noise characteristics,
aliasing artifacts, WebRTC/Opus issues, and recommend optimal filtering.

Usage:
    python tools/analyze_audio_noise.py [--layer LAYER] [--visualize] [--compare]
    
Features:
    - Crackle/click detection with severity levels
    - Zero-run detection (packet loss concealment artifacts)
    - Periodic pattern analysis (Opus frame artifacts)
    - SNR estimation
    - Aliasing detection near Nyquist
    - Layer comparison (raw vs processed)
"""

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

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
    """
    Detect sudden discontinuities that cause crackle sounds.
    
    Uses multiple severity thresholds to categorize crackles:
    - Severe (top 0.1%): Definite audible crackles
    - Moderate (0.1-0.5%): Likely audible artifacts
    - Mild (0.5-1%): Possible micro-artifacts
    """
    # Compute first-order difference (sudden changes)
    diff = np.abs(np.diff(audio))
    duration = len(audio) / sample_rate
    
    # Multiple severity thresholds
    threshold_severe = np.percentile(diff, 99.9)   # Top 0.1%
    threshold_moderate = np.percentile(diff, 99.5)  # Top 0.5%
    threshold_mild = np.percentile(diff, 99.0)      # Top 1%
    
    severe_count = np.sum(diff > threshold_severe)
    moderate_count = np.sum((diff > threshold_moderate) & (diff <= threshold_severe))
    mild_count = np.sum((diff > threshold_mild) & (diff <= threshold_moderate))
    
    total_crackles = severe_count + moderate_count
    crackles_per_sec = total_crackles / duration if duration > 0 else 0
    severe_per_sec = severe_count / duration if duration > 0 else 0
    
    # Find crackle indices for pattern analysis
    crackle_indices = np.where(diff > threshold_severe)[0]
    
    # Analyze periodic patterns (Opus frame intervals)
    # 48kHz: 2.5ms=120, 5ms=240, 10ms=480, 20ms=960 samples
    # 16kHz: 2.5ms=40, 5ms=80, 10ms=160, 20ms=320 samples
    periodic_patterns = {}
    if len(crackle_indices) > 10:
        intervals = np.diff(crackle_indices)
        for ms, samples in [(2.5, int(sample_rate * 0.0025)), 
                            (5, int(sample_rate * 0.005)),
                            (10, int(sample_rate * 0.01)), 
                            (20, int(sample_rate * 0.02))]:
            tolerance = max(samples // 10, 2)  # 10% tolerance, min 2
            matches = np.sum((intervals >= samples - tolerance) & (intervals <= samples + tolerance))
            if matches > 3:
                periodic_patterns[f"{ms}ms"] = int(matches)
    
    # Determine assessment
    if severe_per_sec > 20:
        assessment = "CRITICAL"
    elif severe_per_sec > 5:
        assessment = "HIGH"
    elif crackles_per_sec > 5:
        assessment = "MODERATE"
    elif crackles_per_sec > 1:
        assessment = "LOW"
    else:
        assessment = "CLEAN"
    
    return {
        "severe_count": int(severe_count),
        "moderate_count": int(moderate_count),
        "mild_count": int(mild_count),
        "total_crackles": int(total_crackles),
        "crackles_per_second": crackles_per_sec,
        "severe_per_second": severe_per_sec,
        "max_discontinuity": float(np.max(diff)),
        "periodic_patterns": periodic_patterns,
        "assessment": assessment
    }


def analyze_zero_runs(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Detect zero-runs in audio that indicate packet loss concealment (PLC).
    
    When Opus doesn't receive a packet, it either:
    1. Generates concealment frames (sounds like underwater/warble)
    2. Inserts zeros (causes clicks/pops)
    
    This function detects both short zeros (clicks) and long zeros (dropouts).
    """
    zero_threshold = 0.0001  # Consider < 0.0001 as zero
    
    zero_runs = []
    in_zero = False
    zero_start = 0
    
    for i, sample in enumerate(audio):
        if abs(sample) < zero_threshold:
            if not in_zero:
                in_zero = True
                zero_start = i
        else:
            if in_zero:
                run_length = i - zero_start
                zero_runs.append((zero_start, run_length))
                in_zero = False
    
    # Handle trailing zeros
    if in_zero:
        zero_runs.append((zero_start, len(audio) - zero_start))
    
    # Categorize by duration
    # At 48kHz: 1 sample = 0.02ms, 48 samples = 1ms, 480 = 10ms
    short_zeros = [r for r in zero_runs if r[1] < 50]       # < 1ms (clicks)
    medium_zeros = [r for r in zero_runs if 50 <= r[1] < 500]  # 1-10ms (pops)
    long_zeros = [r for r in zero_runs if r[1] >= 500]      # >= 10ms (dropouts)
    
    total_zero_samples = sum(r[1] for r in zero_runs)
    duration = len(audio) / sample_rate
    zero_ratio = total_zero_samples / len(audio) if len(audio) > 0 else 0
    
    # Assessment
    if len(long_zeros) > 5 or zero_ratio > 0.05:
        assessment = "CRITICAL"  # Significant packet loss
    elif len(medium_zeros) > 50 or zero_ratio > 0.01:
        assessment = "HIGH"
    elif len(short_zeros) > 1000:
        assessment = "MODERATE"
    elif len(zero_runs) > 100:
        assessment = "LOW"
    else:
        assessment = "CLEAN"
    
    return {
        "total_zero_runs": len(zero_runs),
        "short_zeros_clicks": len(short_zeros),
        "medium_zeros_pops": len(medium_zeros),
        "long_zeros_dropouts": len(long_zeros),
        "total_zero_samples": int(total_zero_samples),
        "zero_ratio_percent": zero_ratio * 100,
        "assessment": assessment
    }


def analyze_clipping(audio: np.ndarray) -> Dict:
    """Detect audio clipping (samples at max amplitude)."""
    clip_threshold = 0.99
    
    clipped_positive = np.sum(audio >= clip_threshold)
    clipped_negative = np.sum(audio <= -clip_threshold)
    total_clipped = clipped_positive + clipped_negative
    
    clip_ratio = total_clipped / len(audio) if len(audio) > 0 else 0
    
    if clip_ratio > 0.01:
        assessment = "CRITICAL"
    elif clip_ratio > 0.001:
        assessment = "HIGH"
    elif total_clipped > 0:
        assessment = "LOW"
    else:
        assessment = "CLEAN"
    
    return {
        "clipped_samples": int(total_clipped),
        "clipped_positive": int(clipped_positive),
        "clipped_negative": int(clipped_negative),
        "clip_ratio_percent": clip_ratio * 100,
        "assessment": assessment
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


def process_layer(layer_name, layer_config, baseline_audio, results, args, output_dir):
    """Process a single audio layer with comprehensive analysis."""
    layer_path = layer_config['path']
    if not layer_path.exists():
        print(f"⚠️  Layer {layer_name} not found: {layer_path}")
        return
    
    filter_name = layer_config['filter']
    sample_rate_tag = layer_config['sample_rate_tag']
    description = layer_config['description']
    
    print(f"📂 Analyzing Layer {layer_name} [{sample_rate_tag}] ({filter_name}): {layer_path.name}")
    print(f"   {description}")
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
    
    # Enhanced crackle detection
    crackles = analyze_crackle_artifacts(audio, sample_rate)
    print(f"\n  ⚡ Crackle Assessment: {crackles['assessment']}")
    print(f"      Severe crackles: {crackles['severe_count']} ({crackles['severe_per_second']:.1f}/sec)")
    print(f"      Moderate crackles: {crackles['moderate_count']}")
    print(f"      Total crackles/sec: {crackles['crackles_per_second']:.1f}")
    print(f"      Max discontinuity: {crackles['max_discontinuity']:.6f}")
    if crackles['periodic_patterns']:
        print(f"      Periodic patterns (Opus frames):")
        for pattern, count in crackles['periodic_patterns'].items():
            print(f"         {pattern}: {count} occurrences")
    
    # Zero-run analysis (packet loss detection)
    zeros = analyze_zero_runs(audio, sample_rate)
    print(f"\n  📡 Packet Loss (Zero-Run) Assessment: {zeros['assessment']}")
    print(f"      Total zero runs: {zeros['total_zero_runs']}")
    print(f"      Short (<1ms, clicks): {zeros['short_zeros_clicks']}")
    print(f"      Medium (1-10ms, pops): {zeros['medium_zeros_pops']}")
    print(f"      Long (>10ms, dropouts): {zeros['long_zeros_dropouts']}")
    print(f"      Zero ratio: {zeros['zero_ratio_percent']:.2f}%")
    
    # Clipping analysis
    clipping = analyze_clipping(audio)
    if clipping['assessment'] != 'CLEAN':
        print(f"\n  🔴 Clipping Assessment: {clipping['assessment']}")
        print(f"      Clipped samples: {clipping['clipped_samples']}")
        print(f"      Clip ratio: {clipping['clip_ratio_percent']:.4f}%")
    
    # Comparison with baseline
    if baseline_audio is not None and layer_name != '3':
        comparison = compare_layers(baseline_audio, audio)
        print(f"\n  🔄 Comparison with Layer 3 (baseline):")
        print(f"      Correlation: {comparison['correlation']:.4f}")
        print(f"      RMS reduction: {comparison['rms_reduction_percent']:.2f}%")
    
    # Store results with metadata
    results[layer_name] = {
        "layer": f"Layer {layer_name}",
        "filter": filter_name,
        "sample_rate_tag": sample_rate_tag,
        "description": description,
        "file_name": layer_path.name,
        "duration_s": duration,
        "sample_rate": sample_rate,
        "snr_db": snr,
        "aliasing": aliasing,
        "crackles": crackles,
        "zero_runs": zeros,
        "clipping": clipping
    }
    
    # Generate visualizations with proper naming
    if args.visualize:
        # Format: spectrogram_Layer3_Baseline_16kHz.png
        spec_filename = f"spectrogram_Layer{layer_name}_{filter_name}_{sample_rate_tag}.png"
        spectrogram_path = output_dir / spec_filename
        plot_title = f"Layer {layer_name} - {filter_name} ({sample_rate_tag})"
        generate_spectrogram(audio, sample_rate, spectrogram_path, plot_title)
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze audio noise characteristics')
    parser.add_argument('--layer', type=str, default='all',
                        help='Which layer to analyze (default: all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate spectrograms and plots')
    parser.add_argument('--compare', action='store_true',
                        help='Compare layers and show WebRTC diagnostics')
    args = parser.parse_args()
    
    # Define paths
    workspace_root = Path(__file__).resolve().parents[1]
    audio_dir = workspace_root / "reports/debug/webrtc"
    vad_debug_dir = workspace_root / "backend/logs/webrtc/vad_debug"
    output_dir = workspace_root / "reports/debug/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Layer configuration with metadata (path, filter_name, sample_rate)
    layers = {
        # VAD Debug Layer
        'vad': {
            'path': vad_debug_dir / "20251128-171357_session_00_silero.wav",
            'filter': 'SileroVAD',
            'sample_rate_tag': '16kHz',
            'description': 'Silero VAD Debug Output'
        },
        # 48kHz layers (raw WebRTC audio)
        '1': {
            'path': audio_dir / "layer1_48000hz_raw.wav",
            'filter': 'Raw',
            'sample_rate_tag': '48kHz',
            'description': 'Raw 48kHz from WebRTC (Opus decoded)'
        },
        '15': {
            'path': audio_dir / "layer15_transient_48000hz.wav",
            'filter': 'TransientSuppressor',
            'sample_rate_tag': '48kHz',
            'description': 'Transient Suppressor (crackle removal @ 48kHz)'
        },
        '2': {
            'path': audio_dir / "layer2_48000hz_float.wav",
            'filter': 'Normalized',
            'sample_rate_tag': '48kHz',
            'description': 'Normalized 48kHz float'
        },
        # VAD-filtered layers (speech only)
        '4': {
            'path': audio_dir / "layer4_16khz_vad_filtered.wav",
            'filter': 'VAD_Filtered',
            'sample_rate_tag': '16kHz',
            'description': 'VAD-filtered speech (sent to Whisper)'
        },
        '5': {
            'path': audio_dir / "layer5_48khz_vad_filtered.wav",
            'filter': 'VAD_Filtered',
            'sample_rate_tag': '48kHz',
            'description': 'VAD-filtered speech @ 48kHz'
        },
        # 16kHz processed layers
        '3': {
            'path': audio_dir / "layer3_16khz.wav",
            'filter': 'Baseline',
            'sample_rate_tag': '16kHz',
            'description': 'Baseline (downsampled, no noise reduction)'
        },
        '31': {
            'path': audio_dir / "layer31_ema_16khz.wav",
            'filter': 'EMA',
            'sample_rate_tag': '16kHz',
            'description': 'EMA noise gate'
        },
        '31b': {
            'path': audio_dir / "layer31b_percentile_16khz.wav",
            'filter': 'PercentileGate',
            'sample_rate_tag': '16kHz',
            'description': 'Percentile Gate (adaptive noise gate)'
        },
        '32': {
            'path': audio_dir / "layer32_rnnoise_16khz.wav",
            'filter': 'RNNoise',
            'sample_rate_tag': '16kHz',
            'description': 'RNNoise denoiser'
        },
        '33': {
            'path': audio_dir / "layer33_dtln_16khz.wav",
            'filter': 'DTLN',
            'sample_rate_tag': '16kHz',
            'description': 'DTLN denoiser'
        },
        '35': {
            'path': audio_dir / "layer35_nsnet2_16khz.wav",
            'filter': 'SpectralGating',
            'sample_rate_tag': '16kHz',
            'description': 'Spectral Gating'
        },
        '36': {
            'path': audio_dir / "layer36_comb_16khz.wav",
            'filter': 'CombFilter_80Hz',
            'sample_rate_tag': '16kHz',
            'description': 'Comb Filter (80 Hz removal)'
        }
    }
    
    # Filter out layers that don't exist to avoid errors
    existing_layers = {}
    for k, v in layers.items():
        if v['path'].exists():
            existing_layers[k] = v
        else:
            # Only warn if user specifically requested this layer
            if args.layer == k:
                print(f"⚠️  Layer {k} not found: {v['path']}")
    
    layers = existing_layers
    
    if args.layer != 'all':
        if args.layer in layers:
            layers = {args.layer: layers[args.layer]}
        else:
            print(f"❌ Invalid layer: {args.layer}")
            print(f"Available layers: {', '.join(layers.keys())}")
            sys.exit(1)
    
    results = {}
    
    print("\n" + "="*80)
    print("🔊 AUDIO NOISE ANALYSIS")
    print("="*80 + "\n")
    
    # Load baseline (Layer 3) for comparison
    baseline_audio, baseline_sr = None, None
    baseline_layer = layers.get('3')
    if baseline_layer and baseline_layer['path'].exists():
        baseline_audio, baseline_sr = load_wav(baseline_layer['path'])
    
    # Group layers by sample rate for organized output
    layers_48k = {k: v for k, v in layers.items() if v['sample_rate_tag'] == '48kHz'}
    layers_16k = {k: v for k, v in layers.items() if v['sample_rate_tag'] == '16kHz'}
    
    # Process 48kHz layers first
    if layers_48k:
        print("┌" + "─"*78 + "┐")
        print("│" + " "*25 + "48 kHz LAYERS" + " "*40 + "│")
        print("└" + "─"*78 + "┘\n")
        
        for layer_name, layer_config in layers_48k.items():
            process_layer(layer_name, layer_config, baseline_audio, results, args, output_dir)
    
    # Process 16kHz layers
    if layers_16k:
        print("\n" + "┌" + "─"*78 + "┐")
        print("│" + " "*25 + "16 kHz LAYERS" + " "*40 + "│")
        print("└" + "─"*78 + "┘\n")
        
        for layer_name, layer_config in layers_16k.items():
            process_layer(layer_name, layer_config, baseline_audio, results, args, output_dir)
    
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
    
    # Save JSON report with metadata
    report_path = output_dir / "noise_analysis_report.json"
    
    # Create organized report structure
    organized_report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_layers_analyzed": len(results),
            "layers_48kHz": [k for k, v in results.items() if v['sample_rate_tag'] == '48kHz'],
            "layers_16kHz": [k for k, v in results.items() if v['sample_rate_tag'] == '16kHz']
        },
        "layers": results
    }
    
    with open(report_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        report_native = convert_to_native_types(organized_report)
        json.dump(report_native, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete! Report saved to: {report_path}")
    print("="*80 + "\n")
    
    # WebRTC/Opus Diagnostics
    print("🌐 WEBRTC/OPUS DIAGNOSTICS:")
    print("-" * 80)
    
    if '1' in results:
        raw = results['1']
        zeros = raw.get('zero_runs', {})
        crackles = raw.get('crackles', {})
        
        print(f"\n  📡 Network Quality Assessment:")
        print(f"      Zero-runs (packet loss): {zeros.get('assessment', 'N/A')}")
        print(f"      Total zero runs: {zeros.get('total_zero_runs', 0)}")
        print(f"      Long dropouts (>10ms): {zeros.get('long_zeros_dropouts', 0)}")
        
        if zeros.get('assessment') in ['CRITICAL', 'HIGH']:
            print("""
      🔴 SIGNIFICANT PACKET LOSS DETECTED in raw WebRTC audio!
      
      Root Cause: Opus PLC (Packet Loss Concealment) artifacts
      
      The crackling is NOT from server-side processing - it's from:
      1. Network packet loss between client and server
      2. Opus codec generating concealment frames
      3. Client-side jitter buffer underruns
      
      Solutions:
      1. Improve network quality (use wired connection, closer server)
      2. Increase client-side jitter buffer (browser WebRTC settings)
      3. Try disabling browser audio processing:
         - echoCancellation: false
         - noiseSuppression: false
         - autoGainControl: false
      4. Use a different browser (Chrome vs Firefox vs Safari)
      5. Check if client microphone has issues
""")
        
        if crackles.get('periodic_patterns'):
            print(f"\n  🔄 Periodic Crackle Patterns (Opus frame related):")
            for pattern, count in crackles['periodic_patterns'].items():
                print(f"      {pattern}: {count} occurrences")
            print("      → These patterns suggest Opus frame boundary issues")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 80)
    
    if '3' in results:
        baseline = results['3']
        if baseline['aliasing']['assessment'] in ['HIGH', 'MODERATE']:
            print("  ⚠️  Aliasing detected in baseline! Recommendations:")
            print("      1. Lower anti-aliasing cutoff frequency (try 7000 Hz or 6500 Hz)")
            print("      2. Increase filter order (try 10th-order Butterworth)")
            print("      3. Use multi-stage downsampling (48k→24k→16k)")
        
        if baseline['crackles']['assessment'] in ['CRITICAL', 'HIGH']:
            print("  ⚠️  Severe crackle artifacts detected!")
            print("      → Check WebRTC Diagnostics above for root cause")
    
    if '4' in results:
        vad = results['4']
        print(f"\n  🎤 VAD-Filtered Audio (Layer 4) Quality:")
        print(f"      Crackles: {vad['crackles']['assessment']}")
        print(f"      Zero-runs: {vad.get('zero_runs', {}).get('assessment', 'N/A')}")
        if vad['crackles']['assessment'] in ['CRITICAL', 'HIGH']:
            print("      → Crackles are being passed to Whisper (affects transcription quality)")
    
    print("\n  📚 Available noise reduction layers:")
    print("      Layer 3: Baseline (resampled only)")
    print("      Layer 4: VAD-filtered (speech segments only) → sent to Whisper")
    print("      Layer 5: VAD-filtered @ 48kHz (high quality reference)")
    print("      Layer 3.2: RNNoise denoised")
    print()


if __name__ == "__main__":
    main()