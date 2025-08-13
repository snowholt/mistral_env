/* Phase 1.5 Minimal Audio Preprocessor Worklet (Scaffold)
   Provides: high-pass DC filter, rolling RMS, soft noise gate, RMS normalization,
   optional debug metric posting. All processing is causal (no lookahead) and
   frame sized according to render quantum (128 samples @ 48k typical). Downsampling
   to 16 kHz and Int16 packing will be implemented in later streaming client file.
*/

class MinimalPreprocWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.hpPrev = 0.0;                 // High-pass previous sample
    this.alphaHP = 0.995;              // ~25 Hz at 48 kHz (simple DC removal)
    this.rmsWindow = [];               // Store last N frame RMS for baseline
    this.rmsWindowMs = 400;            // 400 ms window
    this.sampleRateTarget = sampleRate; // Browser sampleRate (likely 48k)
    this.framesForWindow = Math.ceil((this.rmsWindowMs / 1000) * (this.sampleRateTarget / 128));
    this.targetLoudness = -23;         // dBFS target
    this.maxGainDb = 6;                // +/-6 dB clamp
    this.minGainDb = -6;
    this.lastGainDb = 0;
    this.gateFactor = 1.45;            // threshold multiplier over baseline
    this.attenuation = 0.3;            // soft gate attenuation
    this.debugInterval = 200;          // ms
    this.lastDebugTs = 0;
    this.enableDebug = false;
    this.enableProcessing = true;

    this.port.onmessage = (e) => {
      const data = e.data || {};
      if (data.type === 'config') {
        if (typeof data.enableDebug === 'boolean') this.enableDebug = data.enableDebug;
        if (typeof data.enableProcessing === 'boolean') this.enableProcessing = data.enableProcessing;
        if (typeof data.gateFactor === 'number') this.gateFactor = data.gateFactor;
        if (typeof data.attenuation === 'number') this.attenuation = data.attenuation;
        if (typeof data.targetLoudness === 'number') this.targetLoudness = data.targetLoudness;
      }
    };
  }

  dbfs(linear) {
    return 20 * Math.log10(linear + 1e-12);
  }

  fromDb(db) {
    return Math.pow(10, db / 20);
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input[0] || !output || !output[0]) return true;

    const channelIn = input[0];
    const channelOut = output[0];

    if (!this.enableProcessing) {
      for (let i = 0; i < channelIn.length; i++) channelOut[i] = channelIn[i];
      return true;
    }

    // 1. High-pass (simple DC removal)
    let sumSquares = 0;
    for (let i = 0; i < channelIn.length; i++) {
      const x = channelIn[i];
      const y = x - this.hpPrev + this.alphaHP * channelOut[i === 0 ? 0 : i - 1] || 0;
      this.hpPrev = x;
      channelOut[i] = y;
      sumSquares += y * y;
    }

    // 2. Frame RMS
    const frameRms = Math.sqrt(sumSquares / channelIn.length);
    this.rmsWindow.push(frameRms);
    if (this.rmsWindow.length > this.framesForWindow) this.rmsWindow.shift();
    const baselineRms = this.rmsWindow.reduce((a, b) => a + b, 0) / this.rmsWindow.length;

    // 3. Soft gate (attenuate below gate threshold)
    const gateThreshold = baselineRms * this.gateFactor;
    if (frameRms < gateThreshold && baselineRms > 0) {
      for (let i = 0; i < channelOut.length; i++) channelOut[i] *= this.attenuation;
    }

    // 4. RMS normalization (simple gain toward target loudness)
    const frameDb = this.dbfs(frameRms);
    const gainNeeded = this.targetLoudness - frameDb;
    const clampedGain = Math.max(this.minGainDb, Math.min(this.maxGainDb, gainNeeded));
    // Smooth gain (attack 150 ms, release 400 ms approx → simple lerp)
    const smoothing = clampedGain > this.lastGainDb ? 0.15 : 0.05;
    this.lastGainDb = this.lastGainDb + (clampedGain - this.lastGainDb) * smoothing;
    const linearGain = this.fromDb(this.lastGainDb);
    for (let i = 0; i < channelOut.length; i++) channelOut[i] *= linearGain;

    // 5. Optional debug metrics (every debugInterval ms)
    if (this.enableDebug) {
      const now = currentTime * 1000; // AudioContext time in ms
      if (now - this.lastDebugTs > this.debugInterval) {
        this.lastDebugTs = now;
        this.port.postMessage({
          type: 'debug_metrics',
            frame_rms: frameRms,
            baseline_rms: baselineRms,
            gate_threshold: gateThreshold,
            gain_db: this.lastGainDb,
            samples: channelOut.length,
        });
      }
    }

    // Peak limiter guard
    for (let i = 0; i < channelOut.length; i++) {
      if (channelOut[i] > 0.95) channelOut[i] = 0.95;
      else if (channelOut[i] < -0.95) channelOut[i] = -0.95;
    }

    return true;
  }
}

registerProcessor('minimal-preproc-worklet', MinimalPreprocWorklet);
