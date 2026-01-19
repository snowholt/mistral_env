/**
 * TTS Player AudioWorklet for PCM16 Real-time Playback
 * 
 * Handles real-time PCM16 audio playback with jitter buffering,
 * volume control, and echo-safe routing.
 */

class TTSPlayerWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    
    // Playback state
    this.isPlaying = false;
    this.isPaused = false;
    this.volume = 1.0;
    
    // Buffer management
    this.audioBuffer = new Float32Array(0);
    this.targetBufferSize = 1280; // 80ms at 16kHz = 1280 samples
    this.minBufferSize = 960;     // 60ms minimum
    this.maxBufferSize = 2400;    // 150ms maximum
    
    // Chunk processing
    this.expectedSequence = 0;
    this.pendingChunks = new Map();
    this.isStreamEnded = false;
    
    // Metrics
    this.samplesPlayed = 0;
    this.bufferUnderruns = 0;
    this.chunksReceived = 0;
    
    // Setup message handling
    this.port.onmessage = this.handleMessage.bind(this);
    
    console.log('[TTSPlayerWorklet] Initialized');
  }
  
  handleMessage(event) {
    const { type, data, sequence, flags, volume } = event.data;
    
    switch (type) {
      case 'audio_chunk':
        this.processAudioChunk(data, sequence, flags);
        break;
        
      case 'start_playback':
        this.startPlayback();
        break;
        
      case 'pause':
        this.isPaused = true;
        break;
        
      case 'resume':
        this.isPaused = false;
        break;
        
      case 'stop':
        this.stopPlayback();
        break;
        
      case 'set_volume':
        this.volume = volume || 1.0;
        break;
        
      default:
        console.warn('[TTSPlayerWorklet] Unknown message type:', type);
    }
  }
  
  processAudioChunk(chunkData, sequence, flags) {
    this.chunksReceived++;
    
    const isStart = (flags & 0x01) !== 0;
    const isEnd = (flags & 0x02) !== 0;
    
    if (isStart) {
      // Reset for new stream
      this.expectedSequence = sequence;
      this.pendingChunks.clear();
      this.audioBuffer = new Float32Array(0);
      this.isStreamEnded = false;
    }
    
    if (isEnd) {
      this.isStreamEnded = true;
    }
    
    // Handle chunk ordering
    if (sequence < this.expectedSequence) {
      return; // Ignore duplicates/late arrivals
    }
    
    if (sequence > this.expectedSequence) {
      // Buffer out-of-order chunk
      this.pendingChunks.set(sequence, { data: chunkData, flags });
      return;
    }
    
    // Process chunk in order
    this.appendAudioData(chunkData);
    this.expectedSequence++;
    
    // Process any pending chunks that are now in order
    while (this.pendingChunks.has(this.expectedSequence)) {
      const pending = this.pendingChunks.get(this.expectedSequence);
      this.appendAudioData(pending.data);
      this.pendingChunks.delete(this.expectedSequence);
      this.expectedSequence++;
    }
    
    // Start playback if we have enough buffer
    if (!this.isPlaying && this.audioBuffer.length >= this.minBufferSize) {
      this.startPlayback();
    }
  }
  
  appendAudioData(pcmData) {
    if (!pcmData || pcmData.byteLength === 0) return;
    
    // Convert PCM16 to Float32
    const int16Array = new Int16Array(pcmData);
    const float32Array = new Float32Array(int16Array.length);
    
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0; // Convert to -1.0 to 1.0
    }
    
    // Append to buffer
    const newBuffer = new Float32Array(this.audioBuffer.length + float32Array.length);
    newBuffer.set(this.audioBuffer, 0);
    newBuffer.set(float32Array, this.audioBuffer.length);
    this.audioBuffer = newBuffer;
    
    // Prevent buffer from growing too large
    if (this.audioBuffer.length > this.maxBufferSize * 2) {
      const excess = this.audioBuffer.length - this.maxBufferSize;
      this.audioBuffer = this.audioBuffer.slice(excess);
    }
  }
  
  startPlayback() {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.port.postMessage({ type: 'playback_started' });
  }
  
  stopPlayback() {
    this.isPlaying = false;
    this.isPaused = false;
    this.audioBuffer = new Float32Array(0);
    this.pendingChunks.clear();
    this.expectedSequence = 0;
    this.isStreamEnded = false;
    this.samplesPlayed = 0;
  }
  
  process(inputs, outputs) {
    const output = outputs[0];
    const outputChannel = output[0];
    const framesToProcess = outputChannel.length;
    
    if (!this.isPlaying || this.isPaused) {
      // Output silence
      outputChannel.fill(0);
      return true;
    }
    
    // Check if we have enough buffered audio
    if (this.audioBuffer.length < framesToProcess) {
      if (this.isStreamEnded && this.audioBuffer.length === 0) {
        // Stream ended and buffer empty - signal completion
        this.isPlaying = false;
        this.port.postMessage({ type: 'playback_complete' });
        outputChannel.fill(0);
        return true;
      } else if (!this.isStreamEnded) {
        // Buffer underrun - output silence and wait for more data
        this.bufferUnderruns++;
        this.port.postMessage({ type: 'buffer_underrun' });
        outputChannel.fill(0);
        return true;
      }
    }
    
    // Copy available audio to output
    const samplesToCopy = Math.min(framesToProcess, this.audioBuffer.length);
    
    for (let i = 0; i < samplesToCopy; i++) {
      outputChannel[i] = this.audioBuffer[i] * this.volume;
    }
    
    // Fill remaining with silence if needed
    for (let i = samplesToCopy; i < framesToProcess; i++) {
      outputChannel[i] = 0;
    }
    
    // Remove processed samples from buffer
    if (samplesToCopy > 0) {
      this.audioBuffer = this.audioBuffer.slice(samplesToCopy);
      this.samplesPlayed += samplesToCopy;
    }
    
    // Send metrics periodically
    if (this.samplesPlayed % 1600 === 0) { // Every ~100ms at 16kHz
      this.port.postMessage({
        type: 'metrics',
        data: {
          bufferSize: this.audioBuffer.length,
          bufferUnderruns: this.bufferUnderruns,
          samplesPlayed: this.samplesPlayed,
          chunksReceived: this.chunksReceived,
        }
      });
    }
    
    return true;
  }
}

registerProcessor('tts-player-worklet', TTSPlayerWorklet);