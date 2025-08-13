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
    } catch (err) {
      console.error('Failed to load audio worklet', err);
      throw err;
    }
  }

  async start() {
    if (!this.enabled) return;
    await this.initAudio();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: false,
        autoGainControl: false
      }
    });
    const src = this.audioContext.createMediaStreamSource(stream);
    src.connect(this.workletNode); // no monitor to avoid echo
    this._setupCapture(); // begin capture chain
    this.onEvent({ type: 'audio_ready' });
  }

  _setupCapture() {
    const gain = this.audioContext.createGain();
    this.workletNode.connect(gain);
    const bufferSize = 1024; // small enough for low latency
    const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!this._connected || !this.ws || this.ws.readyState !== WebSocket.OPEN || this._suspended) return;
      const input = e.inputBuffer.getChannelData(0);
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
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    let total = 0;
    for (const arr of this._int16SendQueue) total += arr.length;
    if (total < this.samplesPerFrame) return;
    const concat = new Int16Array(total);
    let offset = 0;
    for (const arr of this._int16SendQueue) { concat.set(arr, offset); offset += arr.length; }
    this._int16SendQueue = [];
    for (let pos = 0; pos + this.samplesPerFrame <= concat.length; pos += this.samplesPerFrame) {
      const frame = concat.subarray(pos, pos + this.samplesPerFrame);
      try { this.ws.send(frame.buffer); } catch {}
      this._lastFrameSentAt = performance.now();
      if (this.debug) this.onEvent({ type: 'frame_sent', samples: frame.length });
    }
    const leftoverSamples = concat.length % this.samplesPerFrame;
    if (leftoverSamples) {
      const tail = concat.subarray(concat.length - leftoverSamples);
      this._int16SendQueue.push(tail);
    }
  }

  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/api/v1/ws/streaming-voice?language=${encodeURIComponent(this.language)}`;
    this.ws = new WebSocket(url);
    this.ws.onopen = () => { this._connected = true; this.onEvent({ type: 'ws_open' }); };
    this.ws.onclose = () => { this._connected = false; this.onEvent({ type: 'ws_close' }); };
    this.ws.onerror = (e) => { this.onEvent({ type: 'ws_error', error: e }); };
    this.ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        this._handleServerEvent(data);
      } catch {}
    };
  }

  disconnect() {
    if (this.ws) { try { this.ws.close(); } catch(_){} }
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

  _handleServerEvent(ev) {
    const type = ev.type;
    switch (type) {
      case 'ready':
        this.onEvent({ type: 'ready', data: ev });
        break;
      case 'partial_transcript':
        this._livePartial = ev.text || '';
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
      case 'tts_audio':
        this.onEvent({ type: 'tts_audio', chars: ev.chars });
        if (this.autoplay) this._playBase64Wav(ev.audio);
        break;
      case 'tts_complete':
        this.onEvent({ type: 'tts_complete', processing_ms: ev.processing_ms });
        if (this.autoRearm && !this._suspended) {
          this.onEvent({ type: 'auto_rearm' });
        }
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
