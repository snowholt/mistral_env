
---
## Revised Expert Consultation Prompt (Gemini WebUI Submission)

Title: BeautyAI Real-Time Voice Noise & Artifact Investigation (Capture with Transient Suppressor + Percentile Gate + Comb Filter)

Date: 13 Nov 2025  
Owner: Lumina Ashley (BeautyAI Arabic Voice Pipeline)  
Target Directory for Analysis: `reports/debug/analysis`  
Included Assets:  
- `noise_analysis_report.json` (multi-layer metrics)  
- `buffer_monitoring.json` (timing & CPU stats)  
- `comparison_summary.json` (EMA vs RNNoise latency/quality)  
- `debug_capture_session_transcriptions.json` (Layer4 vs Layer5 ASR)  
- Spectrograms (current + new layers):  
	- `spectrogram_Layer1_Raw_48kHz.png`  
	- `spectrogram_Layer15_TransientSuppressor_48kHz.png` *(new)*  
	- `spectrogram_Layer2_Normalized_48kHz.png`  
	- `spectrogram_Layer3_Baseline_16kHz.png`  
	- `spectrogram_Layer31_EMA_16kHz.png`  
	- `spectrogram_Layer31b_PercentileGate_16kHz.png` *(new)*  
	- `spectrogram_Layer32_RNNoise_16kHz.png`  
	- `spectrogram_Layer33_DTLN_16kHz.png`  
	- `spectrogram_Layer35_SpectralGating_16kHz.png`  
	- `spectrogram_Layer36_CombFilter_80Hz_16kHz.png`  

### Context
We are refining a low-latency (≤50 ms budget) Arabic-first WebRTC audio pipeline. The laptop microphone introduces fan-induced broadband impulsive artifacts (“crackles”) plus a periodic low-frequency hum near 80 Hz. The analyzer now distinguishes 48 kHz preprocessing layers from 16 kHz downstream enhancement layers and adds an experimental Layer 3.6 Comb Filter (multi-notch IIR targeting the 80 Hz family). DeepFilterNet (previously Layer 3.4) is absent in this capture.

### Processing Chain (Current Capture)
1. Layer 1 (Raw 48 kHz PCM, browser)  
1.5 Layer 1.5 Transient Suppressor @48 kHz (median filter kernel=5, adaptive energy spike removal)  
2. Layer 2 (Normalized 48 kHz float)  
→ 8th‑order Butterworth low-pass @8 kHz (aliasing & hiss suppression pre-downsample)  
3. Layer 3 (Baseline 16 kHz)  
3.1 Layer 3.1 EMA gate (legacy; alpha=0.1; threshold=2.0×EMA)  
3.1b Layer 3.1b Percentile Gate (NEW: 10th percentile noise floor + hysteresis -50/-45 dB)  
3.2 RNNoise (48↔16 bridging)  
3.3 DTLN (two-stage denoiser)  
3.5 Spectral Gating (noisereduce; n_fft=512, hop=160, stationary=True)  
3.6 Comb Filter 80 Hz (multi-notch Q=2.0 updated from 30.0 to reduce ringing)  
→ (VAD + ASR follow: Layer4 16 kHz VAD speech, Layer5 48 kHz VAD speech; ASR parity tracked.)

### Key Metrics (Previous Capture ≈29.22 s each – BEFORE new Layers 1.5 & 3.1b)
Format: Layer – Filter – SNR(dB) – Aliasing Danger Zone Energy Ratio – Crackles (count & per sec) – Max Discontinuity
- 1 Raw 48 kHz: 59.29 dB | 0.008882% | 7011 (239.94/s) | 0.490  
- 2 Normalized 48 kHz: 62.81 dB | 0.008871% | 7007 (239.80/s) | 0.490  
- 3 Baseline 16 kHz: ∞ dB | 0.4863% | 2336 (79.95/s) | 0.477  
- 31 EMA: ∞ dB | 0.4863% | 2336 (79.95/s) | 0.477  
- 32 RNNoise: 68.37 dB | 0.5115% | 2337 (79.98/s) | 0.458  
- 33 DTLN: 68.23 dB | 0.3763% | 2337 (79.98/s) | 0.445  
- 35 Spectral Gating: 68.42 dB | 0.4864% | 2336 (79.95/s) | 0.477  
- 36 Comb Filter 80 Hz: 66.20 dB | 0.0238% | 2338 (80.01/s) | 0.576  

### New Layers (Pending Full Metric Integration)
- 15 Transient Suppressor 48 kHz: (Expected) crackles << 240/s (target >90% reduction pre-downsample)
- 31b Percentile Gate 16 kHz: (Expected) finite SNR; gated silence energy floor stabilization; reduced residual impulsive tails.
*Note:* Analyzer update staged; forthcoming capture will regenerate SNR / crackle metrics including Layers 15 & 31b.

### Immediate Observations (Updated w/ Runtime JSON Metrics)
1. Buffer Health: `buffer_monitoring.json` shows 329 underruns / 1562 frames (≈21.1%); systemic timing jitter likely primary origin of 80 Hz periodic impulses—hardware / scheduling, not acoustic fan hum alone.
2. CPU Headroom: Mean CPU 1.33% (max 4.8%) → ample budget for added transient suppression & adaptive gating without breaching <50 ms latency constraint.
3. Metric Integrity Issues: `comparison_summary.json` reports EMA SNR = +∞, RNNoise SNR = -∞, correlations = NaN → current SNR/noise floor estimator unstable (division by near-zero floor, uninitialized buffers). Must re-implement SNR using robust floor clamp (e.g., P10 energy, epsilon floor) before comparative conclusions.
4. EMA Ineffectiveness: Layer 3.1 identical to baseline (no crackle suppression) confirms static 2×EMA threshold fails under elevated floor; rationale for Percentile Gate (Layer 3.1b) now validated.
5. Transient Suppression Rationale: Pre-downsample merging (~240/s → ~80/s) previously misread as suppression; Layer 1.5 now positioned to eliminate impulses before they smear into speech-like envelopes at 16 kHz.
6. Percentile Gate Expectations: Layer 3.1b (10th percentile + hysteresis) should deliver finite SNR, reduce false open states, and enable stable silence floor tracking—pending next analyzer run.
7. Comb Filter Status: Prior capture (Q=30) achieved ~20× aliasing danger zone reduction (0.4863% → 0.0238%) but increased max discontinuity (0.576). Q lowered to 2.0; need fresh metrics to confirm ringing reduction (<0.40 target) while preserving F0/formants.
8. Residual Artifact Hypothesis: Remaining ‘crackles’ after denoisers likely transformed impulse envelopes; with Layer 1.5 active we expect RNNoise/DTLN spectral models to better target stationary residuals rather than mistaking broadened spikes for plosives.
9. ASR Parity: Layer4 vs Layer5 transcriptions nearly identical (one minor lexical divergence: “catches” vs “caches”), indicating current resample + enhancement stages preserve intelligibility; latency per segment (≈66–115 ms) acceptable for buffered segments—future partial streaming could reduce perceived delay.
10. Next Validation Pass: Run new capture to populate metrics for Layers 15 & 31b, recompute crackle counts, discontinuity distribution, percentile-based noise floor trace, and evaluate buffer underrun mitigation strategies (driver period size / jitter buffer).

### Spectrogram Review Goals (External)
Assess:  
- Persistence of 80 Hz harmonic bands after Layer36.  
- Any formant distortion / harmonic truncation in RNNoise, DTLN, Spectral Gating.  
- Crackle morphology: vertical broadband streaks vs narrow-band clusters.  
- Alignment of comb notches with 80, 160, 240 Hz without harming Arabic consonant clarity.

### Specific Questions
**A. Crackle Characterization**: Nature (broadband impulses vs merged alias artifacts) & pre-downsample suppression viability.  
**B. EMA Failure**: Fix thresholding (adaptive percentile? hysteresis?) & SNR estimation improvement.  
**C. Comb Filter Trade-offs**: Refine notch Q / dynamic activation vs static multi-notch; alternative harmonic tracking.  
**D. Strategy**: Introduce transient suppression at 48 kHz before downsample? Reorder pipeline? Two-band approach (hum vs impulsives).  
**E. Metrics**: Validate SNR; add spectral flux, zero-crossing evolution, modulation spectra, crackle energy %.  
**F. Alternatives**: Reintroduce DeepFilterNet; pair RNNoise with transient classifier; lightweight neural artifact detector under <50 ms.

### Actionable Output Requested
Provide:  
1. Validation / corrections of current interpretations.  
2. Top 3 prioritized interventions.  
3. Parameter adjustment table (Current | Proposed | Rationale).  
4. Speech intelligibility risk notes (Arabic phoneme preservation).  
5. Additional metrics or plots to integrate.  
6. Optional flow diagram / pseudocode for revised ordering.

### Constraints
- Real-time latency <50 ms.  
- Preserve Arabic emphatic/uvular/pharyngeal formants.  
- Modular toggling of experimental layers (e.g., Comb Filter).  

### Deliverable Format
- Executive summary  
- Technical deep dive (reference file names & layer IDs)  
- Parameter adjustment table  
- Risk assessment notes  
- Optional redesigned pipeline pseudocode

### Final Note
Do NOT transform audio yourself. Analyze contents of `reports/debug/analysis` and propose improvements. We can add metrics/modules if justified. Add any requests for further artifacts (e.g., delta spectrograms, per-bin variance). Thank you sweetie for your expert sparkle.



