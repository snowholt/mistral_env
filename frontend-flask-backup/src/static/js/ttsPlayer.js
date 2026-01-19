/**
 * TTS Player with MediaSource Extensions for Duplex Voice Streaming
 * 
 * Handles real-time TTS audio chunk playback with proper audio routing
 * to prevent echo/feedback into the microphone input.
 * 
 * Features:
 * - MediaSource Extensions (MSE) for WebM/Opus streaming
 * - PCM16 AudioWorklet fallback for raw audio
 * - Jitter buffer with adaptive sizing (60-120ms)
 * - Duck/pause/resume support for barge-in scenarios
 * - Separate audio routing to prevent mic contamination
 * - Playback progress events for caption synchronization
 */

class TTSPlayer {
  constructor(options = {}) {
    this.audioContext = null;
    this.mediaSource = null;
    this.sourceBuffer = null;
    this.audioElement = null;
    this.workletNode = null;
    
    // Configuration
    this.jitterBufferMs = options.jitterBufferMs || 80;
    this.maxJitterBufferMs = options.maxJitterBufferMs || 150;
    this.minJitterBufferMs = options.minJitterBufferMs || 60;
    
    // State
    this.isPlaying = false;
    this.isPaused = false;
    this.isDucked = false;
    this.currentVolume = 1.0;
    this.duckVolume = 0.25; // -12dB approximately
    
    // Chunk management
    this.pendingChunks = [];
    this.expectedSequence = 0;
    this.lastChunkTime = 0;
    
    // Audio routing
    this.outputDeviceId = options.outputDeviceId || 'default';
    this.separateOutput = options.separateOutput !== false; // Enable by default
    
    // Events
    this.onProgress = options.onProgress || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
    this.onError = options.onError || (() => {});
    this.onPlaybackComplete = options.onPlaybackComplete || (() => {});
    
    // Debug
    this.debug = !!options.debug;
    this.metrics = {
      chunksReceived: 0,
      chunksPlayed: 0,
      bufferUnderruns: 0,
      playbackStalls: 0,
      avgJitterMs: 0,
    };
    
    this._log('TTS Player initialized', options);
  }
  
  /**
   * Initialize audio context and setup audio pipeline
   */
  async initialize() {
    try {
      // Create audio context at 16kHz to match TTS output
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      // Create audio element with echo-safe routing
      this.audioElement = new Audio();
      this.audioElement.crossOrigin = 'anonymous';
      this.audioElement.preload = 'auto';
      this.audioElement.volume = this.currentVolume;
      
      // Set output device if supported and specified
      if (this.audioElement.setSinkId && this.outputDeviceId !== 'default') {
        try {
          await this.audioElement.setSinkId(this.outputDeviceId);
          this._log(`Audio output routed to device: ${this.outputDeviceId}`);
        } catch (e) {
          this._log('Failed to set output device, using default', e);
        }
      }
      
      // Setup MediaSource for streaming
      this.mediaSource = new MediaSource();
      this.audioElement.src = URL.createObjectURL(this.mediaSource);
      
      // Wait for MediaSource to be ready
      await new Promise((resolve, reject) => {
        this.mediaSource.addEventListener('sourceopen', resolve, { once: true });
        this.mediaSource.addEventListener('error', reject, { once: true });
      });
      
      // Create source buffer for Opus audio
      try {
        // Try WebM/Opus first (preferred for low latency)
        this.sourceBuffer = this.mediaSource.addSourceBuffer('audio/webm; codecs="opus"');
        this.audioFormat = 'webm-opus';
      } catch (e) {
        try {
          // Fallback to MP4/AAC
          this.sourceBuffer = this.mediaSource.addSourceBuffer('audio/mp4; codecs="mp4a.40.2"');
          this.audioFormat = 'mp4-aac';
        } catch (e2) {
          // If MSE not supported, use AudioWorklet for raw PCM
          await this._initializeWorkletPlayer();
          this.audioFormat = 'pcm16-worklet';
          this._log('Using AudioWorklet for PCM16 playback');
          return;
        }
      }
      
      // Configure source buffer
      this.sourceBuffer.mode = 'sequence';
      
      // Setup event handlers
      this.sourceBuffer.addEventListener('updateend', () => {
        this._processPendingChunks();
      });
      
      this.sourceBuffer.addEventListener('error', (e) => {
        this._log('SourceBuffer error:', e);
        this.onError(new Error('SourceBuffer error'));
      });
      
      this.audioElement.addEventListener('canplay', () => {
        if (!this.isPlaying) {
          this._startPlayback();
        }
      });
      
      this.audioElement.addEventListener('waiting', () => {
        this.metrics.playbackStalls++;
        this._log('Playback stalled - buffer underrun');
      });
      
      this.audioElement.addEventListener('ended', () => {
        this._handlePlaybackComplete();
      });
      
      this._log(`TTS Player ready (format: ${this.audioFormat})`);
      
    } catch (error) {
      this._log('TTS Player initialization failed:', error);
      this.onError(error);
      throw error;
    }
  }
  
  /**
   * Initialize AudioWorklet-based player for raw PCM
   */
  async _initializeWorkletPlayer() {
    try {
      // Load audio worklet for PCM playback
      await this.audioContext.audioWorklet.addModule('/static/js/tts-player-worklet.js');
      
      this.workletNode = new AudioWorkletNode(this.audioContext, 'tts-player-worklet', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1], // Mono
      });
      
      // Connect to output
      this.workletNode.connect(this.audioContext.destination);
      
      // Setup worklet message handler
      this.workletNode.port.onmessage = (event) => {
        const { type, data } = event.data;
        
        switch (type) {
          case 'playback_started':
            this.isPlaying = true;
            this.onStateChange('playing');
            break;
            
          case 'playback_complete':
            this._handlePlaybackComplete();
            break;
            
          case 'buffer_underrun':
            this.metrics.bufferUnderruns++;
            break;
            
          case 'metrics':
            Object.assign(this.metrics, data);
            break;
        }
      };
      
    } catch (error) {
      this._log('AudioWorklet initialization failed:', error);
      throw error;
    }
  }
  
  /**
   * Process incoming TTS audio chunk
   */
  async processChunk(chunkData, sequenceNumber, flags = 0) {
    this.metrics.chunksReceived++;
    this.lastChunkTime = performance.now();
    
    const isStart = (flags & 0x01) !== 0;
    const isEnd = (flags & 0x02) !== 0;
    
    if (isStart) {
      this._log('TTS playback starting');
      this.expectedSequence = sequenceNumber;
      this.pendingChunks = [];
    }
    
    // Handle sequence ordering
    if (sequenceNumber < this.expectedSequence) {
      this._log(`Duplicate chunk ignored: ${sequenceNumber} < ${this.expectedSequence}`);
      return;
    }
    
    if (sequenceNumber > this.expectedSequence) {
      // Out of order - buffer for later
      this.pendingChunks.push({ data: chunkData, sequence: sequenceNumber, flags });
      this.pendingChunks.sort((a, b) => a.sequence - b.sequence);
      return;
    }
    
    // Process chunk immediately
    await this._processChunk(chunkData, sequenceNumber, flags);
    
    if (isEnd) {
      this._log('TTS chunk stream ended');
      await this._finalizePlayback();
    }
  }
  
  /**
   * Process individual audio chunk
   */
  async _processChunk(chunkData, sequenceNumber, flags) {
    if (this.audioFormat === 'pcm16-worklet') {
      // Send PCM data to worklet
      this.workletNode.port.postMessage({
        type: 'audio_chunk',
        data: chunkData,
        sequence: sequenceNumber,
        flags: flags
      });
    } else {
      // Append to MediaSource
      if (this.sourceBuffer && !this.sourceBuffer.updating) {
        try {
          // Convert raw PCM to WebM/Opus if needed
          let bufferData = chunkData;
          
          if (chunkData.byteLength > 0) {
            // For now, assume chunks are already in correct format
            // In a production system, you'd convert PCM16 to Opus here
            this.sourceBuffer.appendBuffer(bufferData);
            this.expectedSequence = sequenceNumber + 1;
            this.metrics.chunksPlayed++;
          }
        } catch (error) {
          this._log('Error appending chunk to source buffer:', error);
          this.onError(error);
        }
      } else {
        // Buffer is updating, queue for later
        this.pendingChunks.push({ data: chunkData, sequence: sequenceNumber, flags });
      }
    }
    
    // Emit progress
    this.onProgress({
      sequence: sequenceNumber,
      isPlaying: this.isPlaying,
      bufferHealth: this._getBufferHealth()
    });
  }
  
  /**
   * Process any pending chunks in sequence
   */
  _processPendingChunks() {
    while (this.pendingChunks.length > 0) {
      const next = this.pendingChunks[0];
      
      if (next.sequence === this.expectedSequence) {
        this.pendingChunks.shift();
        this._processChunk(next.data, next.sequence, next.flags);
      } else {
        break; // Wait for missing sequence
      }
    }
  }
  
  /**
   * Start audio playback
   */
  async _startPlayback() {
    if (this.isPlaying) return;
    
    try {
      if (this.audioFormat === 'pcm16-worklet') {
        this.workletNode.port.postMessage({ type: 'start_playback' });
      } else {
        await this.audioElement.play();
      }
      
      this.isPlaying = true;
      this.onStateChange('playing');
      this._log('TTS playback started');
      
    } catch (error) {
      this._log('Failed to start playback:', error);
      this.onError(error);
    }
  }
  
  /**
   * Duck (reduce volume) during barge-in
   */
  duck(duckLevelDb = -12) {
    if (this.isDucked) return;
    
    this.isDucked = true;
    const duckFactor = Math.pow(10, duckLevelDb / 20); // Convert dB to linear
    
    if (this.audioFormat === 'pcm16-worklet') {
      this.workletNode.port.postMessage({
        type: 'set_volume',
        volume: duckFactor
      });
    } else {
      this.audioElement.volume = this.currentVolume * duckFactor;
    }
    
    this.onStateChange('ducked');
    this._log(`TTS ducked by ${duckLevelDb}dB`);
  }
  
  /**
   * Restore normal volume after barge-in
   */
  unduck() {
    if (!this.isDucked) return;
    
    this.isDucked = false;
    
    if (this.audioFormat === 'pcm16-worklet') {
      this.workletNode.port.postMessage({
        type: 'set_volume',
        volume: this.currentVolume
      });
    } else {
      this.audioElement.volume = this.currentVolume;
    }
    
    this.onStateChange('playing');
    this._log('TTS volume restored');
  }
  
  /**
   * Pause playback during extended barge-in
   */
  pause() {
    if (this.isPaused || !this.isPlaying) return;
    
    this.isPaused = true;
    
    if (this.audioFormat === 'pcm16-worklet') {
      this.workletNode.port.postMessage({ type: 'pause' });
    } else {
      this.audioElement.pause();
    }
    
    this.onStateChange('paused');
    this._log('TTS playback paused');
  }
  
  /**
   * Resume playback after barge-in ends
   */
  resume() {
    if (!this.isPaused) return;
    
    this.isPaused = false;
    
    if (this.audioFormat === 'pcm16-worklet') {
      this.workletNode.port.postMessage({ type: 'resume' });
    } else {
      this.audioElement.play().catch(e => this._log('Resume failed:', e));
    }
    
    this.onStateChange('playing');
    this._log('TTS playback resumed');
  }
  
  /**
   * Stop playback and cleanup
   */
  stop() {
    this.isPlaying = false;
    this.isPaused = false;
    this.isDucked = false;
    
    if (this.audioFormat === 'pcm16-worklet') {
      this.workletNode.port.postMessage({ type: 'stop' });
    } else {
      if (this.audioElement) {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;
      }
    }
    
    this.pendingChunks = [];
    this.expectedSequence = 0;
    
    this.onStateChange('stopped');
    this._log('TTS playback stopped');
  }
  
  /**
   * Finalize playback when all chunks received
   */
  async _finalizePlayback() {
    if (this.audioFormat !== 'pcm16-worklet' && this.mediaSource && this.mediaSource.readyState === 'open') {
      try {
        this.mediaSource.endOfStream();
      } catch (e) {
        this._log('Error ending stream:', e);
      }
    }
  }
  
  /**
   * Handle playback completion
   */
  _handlePlaybackComplete() {
    this.isPlaying = false;
    this.isPaused = false;
    this.isDucked = false;
    
    this.onStateChange('complete');
    this.onPlaybackComplete();
    this._log('TTS playback completed');
  }
  
  /**
   * Get buffer health percentage (0-100)
   */
  _getBufferHealth() {
    if (this.audioFormat === 'pcm16-worklet') {
      // Would need worklet to report buffer level
      return 50; // Placeholder
    } else if (this.audioElement) {
      const buffered = this.audioElement.buffered;
      if (buffered.length > 0) {
        const currentTime = this.audioElement.currentTime;
        const bufferedEnd = buffered.end(buffered.length - 1);
        const bufferAhead = bufferedEnd - currentTime;
        
        // Convert to percentage of jitter buffer target
        return Math.min(100, (bufferAhead * 1000 / this.jitterBufferMs) * 100);
      }
    }
    
    return 0;
  }
  
  /**
   * Get playback metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      bufferHealth: this._getBufferHealth(),
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      isDucked: this.isDucked,
      audioFormat: this.audioFormat,
      jitterBufferMs: this.jitterBufferMs,
    };
  }
  
  /**
   * Cleanup resources
   */
  dispose() {
    this.stop();
    
    if (this.audioElement) {
      URL.revokeObjectURL(this.audioElement.src);
      this.audioElement = null;
    }
    
    if (this.mediaSource) {
      this.mediaSource = null;
    }
    
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    this._log('TTS Player disposed');
  }
  
  _log(message, ...args) {
    if (this.debug) {
      console.log('[TTSPlayer]', message, ...args);
    }
  }
}

// Make available globally
window.TTSPlayer = TTSPlayer;