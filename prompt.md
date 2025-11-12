### Logs
- Beautyai API Service Journal : `reports/logs/journal_backend_service.log`




---
## Revised Expert Consultation Prompt (Gemini WebUI Submission)

Title: BeautyAI Real-Time Voice Noise & Artifact Investigation (Updated Capture with Comb Filter Layer)

Date: 12 Nov 2025  
Owner: Lumina Ashley (BeautyAI Arabic Voice Pipeline)  
Target Directory for Analysis: `reports/debug/analysis`  
Included Assets:  
- `noise_analysis_report.json` (multi-layer metrics)  
- Spectrograms:  
	- `spectrogram_Layer1_Raw_48kHz.png`  
	- `spectrogram_Layer2_Normalized_48kHz.png`  
	- `spectrogram_Layer3_Baseline_16kHz.png`  
	- `spectrogram_Layer31_EMA_16kHz.png`  
	- `spectrogram_Layer32_RNNoise_16kHz.png`  
	- `spectrogram_Layer33_DTLN_16kHz.png`  
	- `spectrogram_Layer35_SpectralGating_16kHz.png`  
	- `spectrogram_Layer36_CombFilter_80Hz_16kHz.png`  

### Context
We are refining a low-latency (≤50 ms budget) Arabic-first WebRTC audio pipeline. The laptop microphone introduces fan-induced broadband impulsive artifacts (“crackles”) plus a periodic low-frequency hum near 80 Hz. The analyzer now distinguishes 48 kHz preprocessing layers from 16 kHz downstream enhancement layers and adds an experimental Layer 3.6 Comb Filter (multi-notch IIR targeting the 80 Hz family). DeepFilterNet (previously Layer 3.4) is absent in this capture.

### Processing Chain (Current Capture)
1. Layer 1 (Raw 48 kHz PCM, browser)  
2. Layer 2 (Normalized 48 kHz float)  
→ 8th‑order Butterworth low-pass @8 kHz to prevent aliasing during 16 kHz downsample  
3. Layer 3 (Baseline 16 kHz)  
3.1 EMA adaptive gate (alpha=0.1, threshold=2.0×EMA)  
3.2 RNNoise (48↔16 bridging)  
3.3 DTLN (spectral subtraction path)  
3.5 Spectral Gating (noisereduce; n_fft=512, hop=160, stationary=True)  
3.6 Comb Filter 80 Hz (experimental multi-notch IIR)  
→ (Subsequent VAD & ASR steps omitted; focus is pre-VAD artifact removal.)

### Key Metrics (Duration ≈29.22 s each)
Format: Layer – Filter – SNR(dB) – Aliasing Danger Zone Energy Ratio – Crackles (count & per sec) – Max Discontinuity
- 1 Raw 48 kHz: 59.29 dB | 0.008882% | 7011 (239.94/s) | 0.490  
- 2 Normalized 48 kHz: 62.81 dB | 0.008871% | 7007 (239.80/s) | 0.490  
- 3 Baseline 16 kHz: ∞ dB | 0.4863% | 2336 (79.95/s) | 0.477  
- 31 EMA: ∞ dB | 0.4863% | 2336 (79.95/s) | 0.477  
- 32 RNNoise: 68.37 dB | 0.5115% | 2337 (79.98/s) | 0.458  
- 33 DTLN: 68.23 dB | 0.3763% | 2337 (79.98/s) | 0.445  
- 35 Spectral Gating: 68.42 dB | 0.4864% | 2336 (79.95/s) | 0.477  
- 36 Comb Filter 80 Hz: 66.20 dB | 0.0238% | 2338 (80.01/s) | 0.576  

### Immediate Observations
1. Crackle density bifurcates: ~240/s (48 kHz) vs ~80/s (16 kHz) → downsampling merges/transforms impulsives.  
2. EMA layer identical to baseline → threshold never triggers (needs redesign).  
3. RNNoise & DTLN raise SNR but crackles persist → artifacts resemble transient speech-like energy.  
4. Spectral Gating near-baseline effect → mask or profile insufficient.  
5. Comb Filter slashes aliasing energy (~20× reduction) but loses ~2 dB SNR and increases discontinuity (possible notch ringing).  
6. Infinite SNR signals faulty noise floor modeling in baseline/EMA.  
7. Aliasing ratio drop only dramatic in comb-filter layer.  
8. Max discontinuity highest post-comb → review notch Q / cascade design.

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



