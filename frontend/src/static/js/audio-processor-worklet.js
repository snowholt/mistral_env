/**
 * AudioWorklet Processor for Low-Latency Voice Streaming
 * 
 * This replaces the deprecated ScriptProcessor with modern AudioWorklet
 * for optimal performance and minimal latency in voice streaming.
 * 
 * Features:
 * - Hardware-rate capture with real-time downsampling to 16kHz
 * - Direct Float32 to Int16 conversion
 * - Configurable chunk sizes for optimal streaming
 * - Audio level monitoring for UI feedback
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    // Configuration from main thread
    const { targetSampleRate = 16000, chunkSizeMs = 20 } = options.processorOptions || {};
    
    this.targetSampleRate = targetSampleRate;
    this.sourceSampleRate = sampleRate; // Current context sample rate
    this.downsampleRatio = this.sourceSampleRate / this.targetSampleRate;
    
    // Calculate frames per chunk at target rate
    this.targetFramesPerChunk = Math.round(this.targetSampleRate * (chunkSizeMs / 1000));
    
    // Downsampling state
    this.inputBuffer = [];
    this.outputBuffer = [];
    this.downsampleIndex = 0;
    
    // Audio level monitoring
    this.levelSum = 0;
    this.levelSamples = 0;
    this.levelReportInterval = Math.round(this.sourceSampleRate * 0.1); // 100ms
    
    console.log(`[AudioProcessor] Initialized: ${this.sourceSampleRate}Hz → ${this.targetSampleRate}Hz, chunk=${chunkSizeMs}ms (${this.targetFramesPerChunk} frames)`);
  }
  
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    
    const inputData = input[0]; // Mono channel
    
    // Calculate audio level for monitoring
    for (let i = 0; i < inputData.length; i++) {
      this.levelSum += inputData[i] * inputData[i];
      this.levelSamples++;
    }
    
    // Report audio level periodically
    if (this.levelSamples >= this.levelReportInterval) {
      const rms = Math.sqrt(this.levelSum / this.levelSamples);
      this.port.postMessage({
        type: 'audioLevel',
        level: rms
      });
      this.levelSum = 0;
      this.levelSamples = 0;
    }
    
    // Downsample to target rate
    for (let i = 0; i < inputData.length; i++) {
      // Simple linear interpolation downsampling
      if (this.downsampleIndex >= this.downsampleRatio) {
        this.outputBuffer.push(inputData[i]);
        this.downsampleIndex -= this.downsampleRatio;
      }
      this.downsampleIndex += 1;
    }
    
    // Send chunks when we have enough data
    while (this.outputBuffer.length >= this.targetFramesPerChunk) {
      const chunk = this.outputBuffer.splice(0, this.targetFramesPerChunk);
      
      // Convert Float32 to Int16
      const int16Data = this.floatToInt16(chunk);
      
      // Send to main thread
      this.port.postMessage({
        type: 'audioChunk',
        data: int16Data,
        sampleRate: this.targetSampleRate,
        samples: chunk.length
      });
    }
    
    return true;
  }
  
  /**
   * Convert Float32 samples to Int16 with proper clamping
   */
  floatToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let sample = float32Array[i];
      // Clamp to [-1, 1] range
      if (sample > 1) sample = 1;
      else if (sample < -1) sample = -1;
      // Convert to 16-bit signed integer
      int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    return int16Array;
  }
}

registerProcessor('audio-processor', AudioProcessor);