"""
Lean Audio Processing Pipeline

Single-pass processing chain optimized for real-time audio:
Limiter → Resample → Single Denoiser → Adaptive Comb → Percentile Gate

Author: BeautyAI Framework
Date: November 13, 2025
"""

import numpy as np
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from .fast_limiter import FastPeakLimiter
from .hum_detector import HumDetector
from .percentile_gate import PercentileNoiseGate
from .comb_filter import CombFilter
from .rnnoise_wrapper import RNNoiseProcessor
from .audio_resampling import process_with_rnnoise_16khz_pipeline


class LeanPipeline:
    """
    Optimized single-pass audio processing pipeline.
    
    Design principles:
    - Single denoiser (RNNoise OR DTLN, not both)
    - Adaptive comb filter (only when hum detected)
    - Pre-allocated buffers for resampling
    - ThreadPoolExecutor for CPU-bound ops
    """
    
    def __init__(
        self,
        sample_rate_in: int = 48000,
        sample_rate_out: int = 16000,
        denoiser_type: str = "rnnoise",  # "rnnoise" or "dtln" or "none"
        enable_limiter: bool = True,
        enable_adaptive_comb: bool = True,
        enable_gate: bool = True,
        max_workers: int = 2,  # Fixed thread pool size
    ):
        """
        Args:
            sample_rate_in: Input sample rate (typically 48kHz from browser)
            sample_rate_out: Output sample rate (typically 16kHz for ASR)
            denoiser_type: Which denoiser to use ("rnnoise", "dtln", "none")
            enable_limiter: Enable fast peak limiter at 48kHz
            enable_adaptive_comb: Enable adaptive comb filter (only when hum detected)
            enable_gate: Enable percentile noise gate
            max_workers: Fixed thread pool size for CPU-bound ops
        """
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out
        self.denoiser_type = denoiser_type
        self.enable_limiter = enable_limiter
        self.enable_adaptive_comb = enable_adaptive_comb
        self.enable_gate = enable_gate
        
        # Fixed thread pool (never spawn unbounded tasks)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="audio_worker")
        
        # Initialize components
        self.limiter = FastPeakLimiter(sample_rate=sample_rate_in) if enable_limiter else None
        self.hum_detector = HumDetector(sample_rate=sample_rate_out) if enable_adaptive_comb else None
        self.comb_filter = CombFilter(sample_rate=sample_rate_out, fundamental_freq=80.0, quality_factor=2.0) if enable_adaptive_comb else None
        self.gate = PercentileNoiseGate(sample_rate=sample_rate_out) if enable_gate else None
        
        # Denoiser
        self.denoiser = None
        if denoiser_type == "rnnoise":
            try:
                self.denoiser = RNNoiseProcessor()
                self.denoiser_type = "rnnoise"
            except Exception as e:
                print(f"[LEAN-PIPELINE] Failed to load RNNoise: {e}, falling back to none")
                self.denoiser_type = "none"
        elif denoiser_type == "dtln":
            try:
                from .dtln_wrapper import DTLNProcessor
                self.denoiser = DTLNProcessor()
                self.denoiser_type = "dtln"
            except Exception as e:
                print(f"[LEAN-PIPELINE] Failed to load DTLN: {e}, falling back to none")
                self.denoiser_type = "none"
        
        # Statistics
        self.frame_count = 0
        self.limiter_activations = 0
        self.hum_detections = 0
        self.comb_active_frames = 0
        self.gate_closed_frames = 0
        
        # Pre-allocate resampling filter (scipy butter filter)
        self.lpf_sos = None
    
    def _resample_to_16k(self, audio_48k_int16: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Resample audio from 48kHz to 16kHz with anti-aliasing.
        
        This is moved from the recv loop to avoid blocking the event loop.
        """
        # Convert to float32
        audio_float = audio_48k_int16.astype(np.float32) / 32767.0
        
        # Apply anti-alias filter (initialize once)
        if self.lpf_sos is None and sample_rate > 16000:
            nyquist_freq = sample_rate / 2
            cutoff_freq = 8000  # Target Nyquist for 16kHz
            normalized_cutoff = cutoff_freq / nyquist_freq
            from scipy.signal import butter, sosfiltfilt
            self.lpf_sos = butter(8, normalized_cutoff, btype='low', output='sos')
        
        if self.lpf_sos is not None:
            from scipy.signal import sosfiltfilt
            audio_float = sosfiltfilt(self.lpf_sos, audio_float)
            audio_float = np.clip(audio_float, -1.0, 1.0)
        
        # Resample to 16kHz
        from scipy.signal import resample_poly
        from math import gcd
        
        if sample_rate == 48000:
            # Two-stage: 48→24→16
            audio_24k = resample_poly(audio_float, 1, 2, window=('kaiser', 8.0))
            audio_24k = np.clip(audio_24k, -1.0, 1.0)
            audio_16k = resample_poly(audio_24k, 2, 3, window=('kaiser', 8.0))
            audio_16k = np.clip(audio_16k, -1.0, 1.0)
        else:
            ratio_gcd = gcd(sample_rate, 16000)
            up = 16000 // ratio_gcd
            down = sample_rate // ratio_gcd
            audio_16k = resample_poly(audio_float, up, down, window=('kaiser', 8.0))
            audio_16k = np.clip(audio_16k, -1.0, 1.0)
        
        return audio_16k
    
    def process_frame_sync(
        self,
        audio_48k_int16: np.ndarray,
        audio_16k_float32: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Synchronous processing (called from worker thread).
        
        Args:
            audio_48k_int16: Raw 48kHz PCM (int16)
            audio_16k_float32: Downsampled 16kHz (float32, optional - will be computed if None)
        
        Returns:
            Dict with processed layers and timing stats
        """
        start_time = time.monotonic()
        timing = {}
        result = {}
        
        self.frame_count += 1
        
        # ===== STAGE 0: Resample to 16kHz (if not provided) =====
        if audio_16k_float32 is None:
            stage_start = time.monotonic()
            audio_16k_float32 = self._resample_to_16k(audio_48k_int16)
            timing["resample_ms"] = (time.monotonic() - stage_start) * 1000
            result["audio_16k_float32"] = audio_16k_float32  # Store for disk writer
        
        # ===== STAGE 1: Limiter @ 48kHz (Fast) =====
        stage_start = time.monotonic()
        
        if self.limiter and self.enable_limiter:
            # Convert int16 → float32
            audio_48k_float = audio_48k_int16.astype(np.float32) / 32767.0
            
            # Apply limiter
            audio_48k_limited = self.limiter.process_frame(audio_48k_float)
            
            # Convert back to int16
            result["layer_15_limited_48k"] = (np.clip(audio_48k_limited, -1.0, 1.0) * 32767).astype(np.int16)
            
            # Stats
            limiter_stats = self.limiter.get_stats()
            if limiter_stats["gain_reduction_count"] > 0:
                self.limiter_activations += 1
        else:
            result["layer_15_limited_48k"] = audio_48k_int16.copy()
        
        timing["limiter_ms"] = (time.monotonic() - stage_start) * 1000
        
        # ===== STAGE 2: Denoiser @ 16kHz (CPU-bound) =====
        stage_start = time.monotonic()
        
        audio_16k_denoised = audio_16k_float32.copy()
        
        if self.denoiser and self.denoiser_type == "rnnoise":
            # RNNoise requires 48k→16k pipeline
            audio_16k_denoised, _ = process_with_rnnoise_16khz_pipeline(
                audio_16k_float32, self.denoiser
            )
        elif self.denoiser and self.denoiser_type == "dtln":
            # DTLN processes 16kHz directly
            audio_16k_denoised = self.denoiser.process_audio(audio_16k_float32)
        
        result["layer_32_denoised_16k"] = audio_16k_denoised
        timing["denoiser_ms"] = (time.monotonic() - stage_start) * 1000
        
        # ===== STAGE 3: Hum Detection + Adaptive Comb @ 16kHz =====
        stage_start = time.monotonic()
        
        audio_16k_comb = audio_16k_denoised
        comb_was_active = False
        
        if self.hum_detector and self.comb_filter and self.enable_adaptive_comb:
            # Detect hum
            is_hum_detected = self.hum_detector.process_frame(audio_16k_denoised)
            
            if is_hum_detected:
                # Apply comb filter
                audio_16k_comb = self.comb_filter.process_audio(audio_16k_denoised)
                self.comb_active_frames += 1
                comb_was_active = True
                
                if self.frame_count % 50 == 0:
                    hum_stats = self.hum_detector.get_stats()
                    print(f"[LEAN-PIPELINE] 🔊 Hum detected: ratio={hum_stats.get('last_ratio_db', 0):.1f} dB, comb filter ACTIVE")
        
        result["layer_36_comb_16k"] = audio_16k_comb
        result["comb_active"] = comb_was_active
        timing["hum_comb_ms"] = (time.monotonic() - stage_start) * 1000
        
        # ===== STAGE 4: Percentile Gate @ 16kHz =====
        stage_start = time.monotonic()
        
        audio_16k_gated = audio_16k_comb
        gate_state = 1.0
        
        if self.gate and self.enable_gate:
            audio_16k_gated = self.gate.process_frame(audio_16k_comb)
            gate_state = self.gate.get_gate_state()
            
            if gate_state < 0.1:
                self.gate_closed_frames += 1
        
        result["layer_31b_gated_16k"] = audio_16k_gated
        result["gate_state"] = gate_state
        timing["gate_ms"] = (time.monotonic() - stage_start) * 1000
        
        # ===== Total Timing =====
        timing["total_ms"] = (time.monotonic() - start_time) * 1000
        result["timing"] = timing
        
        return result
    
    async def process_frame_async(
        self,
        audio_48k_int16: np.ndarray,
        audio_16k_float32: np.ndarray,
        loop=None,
    ) -> Dict[str, Any]:
        """
        Async wrapper for process_frame_sync (offloads to executor).
        
        Args:
            audio_48k_int16: Raw 48kHz PCM
            audio_16k_float32: Downsampled 16kHz
            loop: Event loop (optional)
        
        Returns:
            Processing result dict
        """
        import asyncio
        
        if loop is None:
            loop = asyncio.get_running_loop()
        
        # Run CPU-bound processing in executor
        result = await loop.run_in_executor(
            self.executor,
            self.process_frame_sync,
            audio_48k_int16,
            audio_16k_float32,
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        # Get sub-component stats
        limiter_stats = self.limiter.get_stats() if self.limiter else None
        hum_stats = self.hum_detector.get_stats() if self.hum_detector else None
        
        # Get gate stats with proper type conversion
        gate_stats = None
        if self.gate:
            noise_floor = self.gate.get_noise_floor_db()
            gate_state = self.gate.get_gate_state()
            gate_stats = {
                "noise_floor_db": float(noise_floor) if noise_floor is not None else None,
                "gate_state": float(gate_state) if gate_state is not None else None,
            }
        
        return {
            "frame_count": int(self.frame_count),
            "denoiser_type": str(self.denoiser_type),
            "limiter_activations": int(self.limiter_activations),
            "hum_detections": int(self.hum_detections),
            "comb_active_frames": int(self.comb_active_frames),
            "gate_closed_frames": int(self.gate_closed_frames),
            "limiter_stats": limiter_stats,
            "hum_stats": hum_stats,
            "gate_stats": gate_stats,
        }
    
    def shutdown(self):
        """Shutdown thread pool gracefully"""
        self.executor.shutdown(wait=True, cancel_futures=False)
