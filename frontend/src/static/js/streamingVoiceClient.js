/* Phase 7 Streaming Voice Client
  Incremental upgrade:
  - Load minimal audio preprocessor worklet
  - Capture microphone PCM via AudioWorkletNode
  - Downsample 48k -> 16k mono Int16
  - Frame & send binary PCM to /api/v1/ws/streaming-voice (20 ms cadence)
  - Handle server events: ready, partial_transcript, final_transcript, endpoint, tts_* events
  - Provide hooks for UI (mic level, live transcript, assistant reply audio)
  - Base64 WAV playback support for tts_audio
*/

class StreamingVoiceClient {
  constructor(opts = {}) {
    this.enabled = opts.enabled ?? true;
    this.language = opts.language || 'ar';
    this.ws = null;
    this.audioContext = null;
    this.workletNode = null;
    this.processorReady = false;
    this.frameSizeMs = 20; // target packet duration
    this.targetSampleRate = 16000;
    this.samplesPerFrame = (this.targetSampleRate * this.frameSizeMs) / 1000; // e.g. 320 samples @16k
    this._floatDownsampleQueue = new Float32Array(0);
    this._int16SendQueue = [];
    this.debug = !!opts.debug;
    this.onEvent = opts.onEvent || (() => {});
    this._connected = false;
    this._lastFrameSentAt = 0;
    this._livePartial = '';
    this.autoplay = opts.autoplay !== false;
    this._audioSink = opts.audioSink || null; // <audio> element reference optional
    // Phase 8 additions
    this.autoRearm = opts.autoRearm !== false; // auto re-arm after TTS complete
    this._suspended = false;
    this._lastMicLevel = 0;
    this._ingestMode = null; // 'pcm16le' | 'webm-opus' etc
    this._diag = { firstFrameAt: null, readyAt: null };
    
    // Duplex streaming additions
    this.duplexEnabled = opts.duplexEnabled !== false;
    this.duplexWebSocket = null;
    this.ttsPlayer = null;
    this.echoCancellation = opts.echoCancellation !== false;
    this.selectedMicDevice = opts.selectedMicDevice || null;
    this.selectedSpeakerDevice = opts.selectedSpeakerDevice || null;
    this._currentStream = null;
  }

  async initAudio() {
    if (this.audioContext) return;
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    try {
      await this.audioContext.audioWorklet.addModule('/static/js/streaming-audio-worklet.js');
      this.workletNode = new AudioWorkletNode(this.audioContext, 'minimal-preproc-worklet');
      this.workletNode.port.onmessage = (e) => {
        if (e.data?.type === 'debug_metrics' && this.debug) {
          this.onEvent({ type: 'preproc_debug', ...e.data });
        }
      };
      this.processorReady = true;
      if (this.debug) console.log('[StreamingVoice] Worklet loaded');
      
      // Initialize duplex components if enabled
      if (this.duplexEnabled) {
        await this._initializeDuplexMode();
      }
    } catch (err) {
      console.error('Failed to load audio worklet', err);
      throw err;
    }
  }

  /**
   * Initialize duplex mode components (TTS player and duplex WebSocket)
   */
  async _initializeDuplexMode() {
    if (this.debug) console.log('[StreamingVoice] Initializing duplex mode');
    
    // Initialize TTS player
    if (window.TTSPlayer) {
      this.ttsPlayer = new window.TTSPlayer({
        outputDeviceId: this.selectedSpeakerDevice,
        separateOutput: true, // Ensure separate audio routing
        debug: this.debug,
        onProgress: (progress) => {
          this.onEvent({ type: 'tts_progress', ...progress });
        },
        onStateChange: (state) => {
          this.onEvent({ type: 'tts_state_change', state });
        },
        onError: (error) => {
          this.onEvent({ type: 'tts_error', error });
        },
        onPlaybackComplete: () => {
          this.onEvent({ type: 'tts_playback_complete' });
        }
      });
      
      await this.ttsPlayer.initialize();
      if (this.debug) console.log('[StreamingVoice] TTS player initialized');
    } else {
      console.warn('[StreamingVoice] TTSPlayer not available, duplex mode limited');
    }
    
    // Initialize duplex WebSocket
    if (window.DuplexWebSocket) {
      // Will be connected later in connect() method
      if (this.debug) console.log('[StreamingVoice] Duplex WebSocket ready');
    } else {
      console.warn('[StreamingVoice] DuplexWebSocket not available, falling back to legacy mode');
      this.duplexEnabled = false;
    }
  }

  async start() {
    if (!this.enabled) return;
    await this.initAudio();
    
    // Enhanced audio constraints for echo cancellation
    const audioConstraints = {
      channelCount: 1,
      sampleRate: 48000, // High sample rate for better quality
      echoCancellation: this.echoCancellation,
      noiseSuppression: true,
      autoGainControl: true,
      // Advanced constraints to prevent monitor/loopback devices
      deviceId: this.selectedMicDevice ? { exact: this.selectedMicDevice } : undefined,
    };
    
    // Get available devices and validate selection
    if (this.selectedMicDevice) {
      const devices = await this._getAudioDevices();
      const selectedDevice = devices.find(d => d.deviceId === this.selectedMicDevice);
      
      if (!selectedDevice) {
        console.warn('[StreamingVoice] Selected mic device not found, using default');
        delete audioConstraints.deviceId;
        this.selectedMicDevice = null;
      } else if (this._isMonitorDevice(selectedDevice)) {
        console.warn('[StreamingVoice] Selected device appears to be a monitor/loopback, using default');
        delete audioConstraints.deviceId;
        this.selectedMicDevice = null;
      }
    }
    
    try {
      this._currentStream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints
      });
      
      // Validate stream settings
      const track = this._currentStream.getAudioTracks()[0];
      const settings = track.getSettings();
      
      if (this.debug) {
        console.log('[StreamingVoice] Audio track settings:', settings);
        console.log('[StreamingVoice] Echo cancellation active:', settings.echoCancellation);
      }
      
      // Emit device info event
      this.onEvent({ 
        type: 'audio_device_info', 
        settings: settings,
        echoCancellationActive: settings.echoCancellation,
        deviceId: settings.deviceId,
        deviceLabel: track.label
      });
      
      const src = this.audioContext.createMediaStreamSource(this._currentStream);
      src.connect(this.workletNode); // no monitor to avoid echo
      this._setupCapture(); // begin capture chain
      this.onEvent({ type: 'audio_ready' });
      
    } catch (error) {
      console.error('[StreamingVoice] Failed to start audio:', error);
      this.onEvent({ type: 'audio_error', error: error.message });
      throw error;
    }
  }

  /**
   * Get available audio input/output devices
   */
  async _getAudioDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter(device => device.kind === 'audioinput');
    } catch (error) {
      console.error('[StreamingVoice] Failed to enumerate devices:', error);
      return [];
    }
  }

  /**
   * Detect if device is likely a monitor/loopback device
   */
  _isMonitorDevice(device) {
    const suspiciousTerms = [
      'monitor', 'loopback', 'stereo mix', 'what u hear', 'wave out mix',
      'speakers', 'headphones', 'output', 'playback'
    ];
    
    const deviceName = (device.label || '').toLowerCase();
    return suspiciousTerms.some(term => deviceName.includes(term));
  }

  /**
   * Set microphone device (with validation)
   */
  async setMicrophoneDevice(deviceId) {
    const devices = await this._getAudioDevices();
    const device = devices.find(d => d.deviceId === deviceId);
    
    if (!device) {
      throw new Error('Device not found');
    }
    
    if (this._isMonitorDevice(device)) {
      throw new Error('Selected device appears to be a monitor/loopback device');
    }
    
    this.selectedMicDevice = deviceId;
    this.onEvent({ type: 'mic_device_selected', deviceId, device });
    
    // Restart audio if currently active
    if (this._currentStream) {
      await this._restartAudio();
    }
  }

  /**
   * Set speaker device for TTS output
   */
  async setSpeakerDevice(deviceId) {
    this.selectedSpeakerDevice = deviceId;
    this.onEvent({ type: 'speaker_device_selected', deviceId });
    
    // Update TTS player output if active
    if (this.ttsPlayer && this.ttsPlayer.audioElement && this.ttsPlayer.audioElement.setSinkId) {
      try {
        await this.ttsPlayer.audioElement.setSinkId(deviceId);
        console.log('[StreamingVoice] TTS output routed to:', deviceId);
      } catch (error) {
        console.warn('[StreamingVoice] Failed to set TTS output device:', error);
      }
    }
  }

  /**
   * Restart audio with new settings
   */
  async _restartAudio() {
    if (this._currentStream) {
      this._currentStream.getTracks().forEach(track => track.stop());
      this._currentStream = null;
    }
    
    await this.start();
  }

  _setupCapture() {
    const gain = this.audioContext.createGain();
    this.workletNode.connect(gain);
    const bufferSize = 1024; // small enough for low latency
    const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!this._connected || !this.ws || this.ws.readyState !== WebSocket.OPEN || this._suspended) return;
      const input = e.inputBuffer.getChannelData(0);
      // quick RMS for mic level
      let sum = 0; for (let i=0;i<input.length;i++){ const v=input[i]; sum+=v*v; }
      const rms = Math.sqrt(sum / input.length);
      // simple decay smoothing
      this._lastMicLevel = this._lastMicLevel * 0.85 + rms * 0.15;
      if (this.debug && (performance.now() - (this._lastMicEmit||0) > 250)) {
        this._lastMicEmit = performance.now();
        this.onEvent({ type: 'mic_level', level: this._lastMicLevel });
      }
      this._enqueueAndDownsample(input);
      this._maybeSendFrames();
    };
    gain.connect(processor);
    processor.connect(this.audioContext.destination); // ensure processing continues
    this._captureNode = processor;
  }

  _enqueueAndDownsample(float32Chunk) {
    const prev = this._floatDownsampleQueue;
    const merged = new Float32Array(prev.length + float32Chunk.length);
    merged.set(prev, 0); merged.set(float32Chunk, prev.length);
    const ratio = 48000 / this.targetSampleRate; // 3
    const outLen = Math.floor(merged.length / ratio);
    const down = new Float32Array(outLen);
    let inIdx = 0;
    for (let i = 0; i < outLen; i++) {
      let sum = 0;
      for (let r = 0; r < ratio; r++) sum += merged[inIdx++];
      down[i] = sum / ratio;
    }
    const int16 = new Int16Array(down.length);
    for (let i = 0; i < down.length; i++) {
      let s = Math.max(-1, Math.min(1, down[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    this._int16SendQueue.push(int16);
    this._floatDownsampleQueue = new Float32Array(0);
  }

  _maybeSendFrames() {
    if (!this._connected) return;
    
    // Check which connection type we're using
    const canSend = this.duplexWebSocket ? 
      this.duplexWebSocket.isConnected : 
      (this.ws && this.ws.readyState === WebSocket.OPEN);
      
    if (!canSend) return;
    
    let total = 0;
    for (const arr of this._int16SendQueue) total += arr.length;
    if (total < this.samplesPerFrame) return;
    
    const concat = new Int16Array(total);
    let offset = 0;
    for (const arr of this._int16SendQueue) { 
      concat.set(arr, offset); 
      offset += arr.length; 
    }
    this._int16SendQueue = [];
    
    for (let pos = 0; pos + this.samplesPerFrame <= concat.length; pos += this.samplesPerFrame) {
      const frame = concat.subarray(pos, pos + this.samplesPerFrame);
      
      try {
        if (this.duplexWebSocket) {
          // Send via duplex WebSocket with binary protocol
          // Ensure proper byte alignment for Int16 data
          const frameBuffer = new ArrayBuffer(frame.byteLength);
          const frameView = new Uint8Array(frameBuffer);
          const sourceView = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
          frameView.set(sourceView);
          this.duplexWebSocket.sendMicChunk(frameBuffer);
        } else {
          // Legacy mode - send raw binary
          // Create properly aligned buffer
          const frameBuffer = new ArrayBuffer(frame.byteLength);
          const frameView = new Uint8Array(frameBuffer);
          const sourceView = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
          frameView.set(sourceView);
          this.ws.send(frameBuffer);
        }
        
        this._lastFrameSentAt = performance.now();
        if (this.debug) this.onEvent({ type: 'frame_sent', samples: frame.length });
        
      } catch (error) {
        console.warn('[StreamingVoice] Failed to send frame:', error);
      }
    }
    
    const leftoverSamples = concat.length % this.samplesPerFrame;
    if (leftoverSamples) {
      const tail = concat.subarray(concat.length - leftoverSamples);
      this._int16SendQueue.push(tail);
    }
  }

  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    
    // Determine appropriate API host. Prefer explicit override, else map frontend -> api subdomain.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let apiHost = window.BEAUTYAI_API_HOST || window.location.host;
    const hn = window.location.hostname;
    // If we're on a non-api gmai.sa host (e.g., dev.gmai.sa, app.gmai.sa) route to api.gmai.sa
    if (!window.BEAUTYAI_API_HOST && /\.gmai\.sa$/i.test(hn) && !/^api\./i.test(hn)) {
      apiHost = 'api.gmai.sa';
    }
    // Normalize unsupported language values (e.g., 'auto') to 'ar' default
    let lang = (this.language === 'auto' || !this.language) ? 'ar' : this.language;
    // Heuristic: if user started speaking English but UI stuck on 'ar', allow override via window.FORCE_STREAM_LANG
    if (window.FORCE_STREAM_LANG) lang = window.FORCE_STREAM_LANG;
    const url = `${protocol}//${apiHost}/api/v1/ws/streaming-voice?language=${encodeURIComponent(lang)}`;
    this._attempts = (this._attempts || 0) + 1;
    
    if (this.duplexEnabled && window.DuplexWebSocket) {
      // Use duplex WebSocket for bidirectional streaming
      if (this.debug) console.log('[StreamingVoice] connecting duplex mode attempt', this._attempts, url);
      
      this.duplexWebSocket = new window.DuplexWebSocket({
        url: url,
        debug: this.debug,
        onTextMessage: (data) => {
          this._handleServerEvent(data);
        },
        onTTSChunk: (audioData, sequenceNumber, flags) => {
          this._handleTTSChunk(audioData, sequenceNumber, flags);
        },
        onConnectionChange: (connected) => {
          this._connected = connected;
          this.onEvent({ type: connected ? 'ws_open' : 'ws_close', duplex: true });
        },
        onError: (error) => {
          this.onEvent({ type: 'ws_error', error, duplex: true });
        }
      });
      
      await this.duplexWebSocket.connect();
      
    } else {
      // Fallback to legacy WebSocket
      if (this.debug) console.log('[StreamingVoice] connecting legacy mode attempt', this._attempts, url);
      
      this.ws = new WebSocket(url);
      this.ws.onopen = () => { this._connected = true; this.onEvent({ type: 'ws_open' }); };
      this.ws.onclose = (ev) => {
        const wasConnected = this._connected;
        this._connected = false;
        this.onEvent({ type: 'ws_close', code: ev.code, reason: ev.reason, was_connected: wasConnected, attempts: this._attempts });
      };
      this.ws.onerror = (e) => { this.onEvent({ type: 'ws_error', error: e }); };
      this.ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          this._handleServerEvent(data);
        } catch {}
      };
    }
  }

  /**
   * Handle incoming TTS audio chunks for duplex streaming
   */
  _handleTTSChunk(audioData, sequenceNumber, flags) {
    if (!audioData) {
      // End marker received
      this.onEvent({ type: 'tts_stream_end', sequence: sequenceNumber });
      return;
    }
    
    // Forward to TTS player
    if (this.ttsPlayer) {
      this.ttsPlayer.processChunk(audioData, sequenceNumber, flags);
    } else {
      // Fallback: emit event for external handling
      this.onEvent({ 
        type: 'tts_chunk_received', 
        audioData, 
        sequence: sequenceNumber, 
        flags 
      });
    }
  }

  disconnect() {
    if (this.duplexWebSocket) {
      this.duplexWebSocket.disconnect();
      this.duplexWebSocket = null;
    }
    
    if (this.ws) { 
      try { this.ws.close(); } catch(_){} 
      this.ws = null;
    }
    
    // Stop TTS player
    if (this.ttsPlayer) {
      this.ttsPlayer.stop();
    }
    
    // Stop current audio stream
    if (this._currentStream) {
      this._currentStream.getTracks().forEach(track => track.stop());
      this._currentStream = null;
    }
  }

  suspend() {
    this._suspended = true;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try { this.ws.close(); } catch(_){}
    }
    this.onEvent({ type: 'suspended' });
  }

  resume() {
    this._suspended = false;
    this.connect();
    this.onEvent({ type: 'resumed' });
  }

  /**
   * Soft suspend without closing socket (for push-to-talk UX in streaming overlay).
   * If flag=false and socket closed, will reconnect.
   */
  setSuspended(flag) {
    const was = this._suspended;
    this._suspended = !!flag;
    if (!this._suspended && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
      this.connect();
    }
    if (was !== this._suspended) {
      this.onEvent({ type: this._suspended ? 'soft_suspended' : 'soft_resumed' });
    }
  }

  _handleServerEvent(ev) {
    const type = ev.type;
    switch (type) {
      case 'ready':
        this.onEvent({ type: 'ready', data: ev });
  this._diag.readyAt = performance.now();
        break;
      case 'partial_transcript':
        this._livePartial = ev.text || '';
        if (this.debug) console.log('[StreamingVoiceClient] partial', ev.text);
        this.onEvent({ type: 'partial', text: ev.text, stable: ev.stable, stable_tokens: ev.stable_tokens });
        break;
      case 'final_transcript':
        this.onEvent({ type: 'final', text: ev.text, utterance_index: ev.utterance_index });
        this._livePartial = '';
        break;
      case 'endpoint':
        this.onEvent({ type: 'endpoint', event: ev.event, reason: ev.reason });
        break;
      case 'tts_start':
        this.onEvent({ type: 'tts_start', utterance_index: ev.utterance_index });
        break;
      case 'assistant_response':
        this.onEvent({ type: 'assistant_response', utterance_index: ev.utterance_index, text: ev.text, chars: ev.chars });
        break;
      case 'tts_audio':
        this.onEvent({ type: 'tts_audio', chars: ev.chars, text: ev.text });
        if (this.autoplay) this._playBase64Wav(ev.audio);
        break;
      case 'tts_complete':
        this.onEvent({ type: 'tts_complete', processing_ms: ev.processing_ms });
        if (this.autoRearm && !this._suspended) {
          this.onEvent({ type: 'auto_rearm' });
        }
        break;
      case 'tts_streaming_complete':
        this.onEvent({ type: 'tts_streaming_complete', utterance_index: ev.utterance_index, total_chunks: ev.total_chunks });
        if (this.autoRearm && !this._suspended) {
          this.onEvent({ type: 'auto_rearm' });
        }
        break;
      case 'tts_progress':
        this.onEvent({ type: 'tts_progress', utterance_index: ev.utterance_index, chunks_sent: ev.chunks_sent });
        break;
      case 'ingest_mode':
        this._ingestMode = ev.mode;
        this.onEvent({ type: 'ingest_mode', mode: ev.mode, bytes: ev.bytes_received });
        break;
      case 'ingest_summary':
        this.onEvent({ type: 'ingest_summary', summary: ev });
        break;
      case 'heartbeat':
        this.onEvent({ type: 'heartbeat', stats: ev });
        break;
      case 'session_end':
        this.onEvent({ type: 'session_end', stats: ev });
        break;
      case 'error':
        this.onEvent({ type: 'error', message: ev.message, stage: ev.stage });
        break;
      default:
        this.onEvent({ type: 'server_event', data: ev });
    }
  }

  _playBase64Wav(b64) {
    try {
      const binary = atob(b64);
      const len = binary.length;
      const buf = new Uint8Array(len);
      for (let i = 0; i < len; i++) buf[i] = binary.charCodeAt(i);
      const blob = new Blob([buf], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const audioEl = this._audioSink || new Audio();
      audioEl.src = url;
      audioEl.play().catch(()=>{});
      if (!this._audioSink) {
        audioEl.addEventListener('ended', () => URL.revokeObjectURL(url), { once: true });
      }
    } catch (e) {
      if (this.debug) console.warn('Failed WAV playback', e);
    }
  }
}

window.StreamingVoiceClient = StreamingVoiceClient;
