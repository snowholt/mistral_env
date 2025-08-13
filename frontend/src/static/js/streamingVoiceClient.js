/* Phase 1.5 Streaming Voice Client Skeleton
   Responsibilities:
   - Load minimal audio preprocessor worklet
   - Capture microphone PCM via AudioWorkletNode
   - (Future) Downsample 48k -> 16k and send Int16 frames to /api/v1/ws/streaming-voice
   - Provide debug hooks & simple UI integration stub
*/

class StreamingVoiceClient {
  constructor(opts = {}) {
    this.enabled = opts.enabled ?? true;
    this.language = opts.language || 'ar';
    this.ws = null;
    this.audioContext = null;
    this.workletNode = null;
    this.processorReady = false;
    this.frameSizeMs = 20; // planned frame window (aggregate)
    this.int16Buffer = [];
    this.downsampleBuffer = new Float32Array(0);
    this.targetSampleRate = 16000;
    this.debug = !!opts.debug;
    this.onEvent = opts.onEvent || (() => {});
    this._connected = false;
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
    // Connect to worklet, and optionally to destination only for debugging (monitor). Remove in later phase.
    src.connect(this.workletNode).connect(this.audioContext.destination);
    this.onEvent({ type: 'audio_ready' });
  }

  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // includes port
    const url = `${protocol}//${host}/api/v1/ws/streaming-voice?language=${encodeURIComponent(this.language)}`;
    this.ws = new WebSocket(url);
    this.ws.onopen = () => { this._connected = true; this.onEvent({ type: 'ws_open' }); };
    this.ws.onclose = () => { this._connected = false; this.onEvent({ type: 'ws_close' }); };
    this.ws.onerror = (e) => { this.onEvent({ type: 'ws_error', error: e }); };
    this.ws.onmessage = (msg) => {
      try { this.onEvent({ type: 'server_event', data: JSON.parse(msg.data) }); } catch { /* ignore */ }
    };
  }

  disconnect() {
    if (this.ws) { try { this.ws.close(); } catch(_){} }
  }
}

window.StreamingVoiceClient = StreamingVoiceClient;
