/**
 * BeautyAI Debug WebSocket Tester
 * Specialized client for testing the simple WebSocket voice pipeline
 * Designed for debugging STT → LLM → TTS flow with file upload
 */

class DebugWebSocketTester {
  constructor() {
    this.ws = null;
    this.sessionData = {
      sessionId: null,
      connectionId: null,
      startTime: null,
      stats: {
        framesSent: 0,
        bytesSent: 0,
        partials: 0,
        finals: 0,
        responses: 0,
        errors: 0
      },
      timing: {
        firstPartial: null,
        firstFinal: null,
        firstAssistant: null,
        ttsStart: null,
        ttsComplete: null,
        streamStart: null
      }
    };
    this.debugEvents = [];
    this.audioBuffer = null;
    this.streamOffset = 0;
    this.frameTimer = null;
    this.isStreaming = false;
    
    // DOM elements cache
    this.elements = {};
    this.config = {};
  }

  /**
   * Initialize the debug interface
   */
  initialize() {
    this.cacheElements();
    this.bindEvents();
    this.updateStatus('disconnected');
    this.updateUI();
    this.logDebugEvent('SYSTEM', 'info', 'Debug WebSocket Tester initialized');
  }

  /**
   * Cache DOM elements for performance
   */
  cacheElements() {
    const elementIds = [
      'wsStatus', 'configLanguage', 'configFrameSize', 'configPacing', 'configAutoplay',
      'configEndpointUrl', 'connectBtn', 'disconnectBtn', 'streamBtn', 'abortBtn',
      'audioFileInput', 'fileLabel', 'selectFileBtn',
      'pipelineStatus', 'pipelineProgress', 'stageSTT', 'stageLLM', 'stageTTS',
      'stageSTTStatus', 'stageSTTTiming', 'stageSTTData',
      'stageLLMStatus', 'stageLLMTiming', 'stageLLMData',
      'stageTTSStatus', 'stageTTSTiming', 'stageTTSData',
      'responseTranscript', 'responseAssistant', 'responseAudio', 'responsePlayBtn',
      'debugEventsLog', 'showInfoEvents', 'showWarningEvents', 'showErrorEvents', 'showDebugEvents',
      'exportLogsBtn', 'exportSessionBtn', 'exportDebugBtn', 'exportFormat',
      'sessionCount', 'avgResponseTime', 'successRate'
    ];

    elementIds.forEach(id => {
      this.elements[id] = document.getElementById(id);
    });
  }

  /**
   * Bind event handlers
   */
  bindEvents() {
    // Connection controls
    if (this.elements.connectBtn) {
      this.elements.connectBtn.addEventListener('click', () => this.connect());
    }
    if (this.elements.disconnectBtn) {
      this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
    }

    // File handling
    if (this.elements.selectFileBtn) {
      this.elements.selectFileBtn.addEventListener('click', () => {
        if (this.elements.audioFileInput) {
          this.elements.audioFileInput.click();
        }
      });
    }
    if (this.elements.audioFileInput) {
      this.elements.audioFileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    // Streaming controls
    if (this.elements.streamBtn) {
      this.elements.streamBtn.addEventListener('click', () => this.startStreaming());
    }
    if (this.elements.abortBtn) {
      this.elements.abortBtn.addEventListener('click', () => this.abortStreaming());
    }

    // Response controls
    if (this.elements.responsePlayBtn) {
      this.elements.responsePlayBtn.addEventListener('click', () => this.playResponseAudio());
    }

    // Export controls
    if (this.elements.exportLogsBtn) {
      this.elements.exportLogsBtn.addEventListener('click', () => this.exportLogs());
    }
    if (this.elements.exportSessionBtn) {
      this.elements.exportSessionBtn.addEventListener('click', () => this.exportSession());
    }
    if (this.elements.exportDebugBtn) {
      this.elements.exportDebugBtn.addEventListener('click', () => this.exportDebugData());
    }

    // Log filtering
    ['showInfoEvents', 'showWarningEvents', 'showErrorEvents', 'showDebugEvents'].forEach(id => {
      if (this.elements[id]) {
        this.elements[id].addEventListener('change', () => this.filterDebugEvents());
      }
    });

    // Configuration changes
    if (this.elements.configLanguage) {
      this.elements.configLanguage.addEventListener('change', () => this.updateEndpointUrl());
    }
  }

  /**
   * Update configuration from UI
   */
  updateConfig() {
    this.config = {
      language: this.elements.configLanguage?.value || 'ar',
      frameSize: parseInt(this.elements.configFrameSize?.value || '20'),
      pacing: this.elements.configPacing?.value || 'realtime',
      autoplay: this.elements.configAutoplay?.checked || false,
      endpointUrl: this.elements.configEndpointUrl?.value || this.getDefaultEndpointUrl()
    };
  }

  /**
   * Get default endpoint URL based on current configuration
   */
  getDefaultEndpointUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.BEAUTYAI_API_HOST || 'localhost:8000';
    const language = this.elements.configLanguage?.value || 'ar';
    return `${protocol}//${host}/api/v1/ws/simple-voice-chat?language=${encodeURIComponent(language)}&voice_type=female&debug=1`;
  }

  /**
   * Update endpoint URL when language changes
   */
  updateEndpointUrl() {
    if (this.elements.configEndpointUrl) {
      this.elements.configEndpointUrl.value = this.getDefaultEndpointUrl();
    }
  }

  /**
   * Connect to WebSocket
   */
  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.logDebugEvent('CONNECTION', 'warning', 'Already connected');
      return;
    }

    this.updateConfig();
    this.updateStatus('connecting');
    this.resetSession();

    try {
      this.ws = new WebSocket(this.config.endpointUrl);
      
      this.ws.onopen = () => {
        this.updateStatus('connected');
        this.logDebugEvent('CONNECTION', 'info', 'WebSocket connected', { url: this.config.endpointUrl });
        this.updateUI();
      };

      this.ws.onclose = (event) => {
        this.updateStatus('disconnected');
        this.logDebugEvent('CONNECTION', 'info', 'WebSocket disconnected', { 
          code: event.code, 
          reason: event.reason 
        });
        this.abortStreaming();
        this.updateUI();
      };

      this.ws.onerror = (error) => {
        this.logDebugEvent('CONNECTION', 'error', 'WebSocket error', { error: error.message });
        this.sessionData.stats.errors++;
      };

      this.ws.onmessage = (event) => {
        this.handleServerMessage(event.data);
      };

    } catch (error) {
      this.updateStatus('disconnected');
      this.logDebugEvent('CONNECTION', 'error', 'Failed to connect', { error: error.message });
      this.updateUI();
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    if (this.ws) {
      this.abortStreaming();
      this.ws.close();
      this.ws = null;
    }
    this.updateStatus('disconnected');
    this.updateUI();
  }

  /**
   * Handle incoming server messages
   */
  handleServerMessage(data) {
    try {
      const message = JSON.parse(data);
      const now = performance.now();

      // Log the raw message for debugging
      this.logDebugEvent('SERVER', 'debug', 'Received message', { type: message.type, data: message });

      switch (message.type) {
        case 'ready':
          this.handleReadyMessage(message);
          break;

        case 'partial_transcript':
          this.handlePartialTranscript(message, now);
          break;

        case 'final_transcript':
          this.handleFinalTranscript(message, now);
          break;

        case 'assistant_response':
          this.handleAssistantResponse(message, now);
          break;

        case 'tts_start':
          this.handleTTSStart(message, now);
          break;

        case 'tts_audio':
          this.handleTTSAudio(message);
          break;

        case 'tts_complete':
          this.handleTTSComplete(message, now);
          break;

        case 'error':
          this.handleError(message);
          break;

        case 'debug_event':
          this.handleDebugEvent(message);
          break;

        default:
          this.logDebugEvent('SERVER', 'debug', 'Unknown message type', { type: message.type, message });
      }

    } catch (error) {
      this.logDebugEvent('SERVER', 'error', 'Failed to parse message', { 
        error: error.message, 
        rawData: data 
      });
    }
  }

  /**
   * Handle ready message from server
   */
  handleReadyMessage(message) {
    this.sessionData.sessionId = message.session_id;
    this.sessionData.connectionId = message.connection_id;
    this.sessionData.startTime = new Date().toISOString();
    
    this.logDebugEvent('SESSION', 'info', 'Session ready', {
      sessionId: this.sessionData.sessionId,
      connectionId: this.sessionData.connectionId
    });

    this.updateStageStatus('STT', 'ready');
    this.updateStageStatus('LLM', 'ready');
    this.updateStageStatus('TTS', 'ready');
    this.updatePipelineStatus('ready', 'Ready for audio input');
  }

  /**
   * Handle partial transcript
   */
  handlePartialTranscript(message, timestamp) {
    this.sessionData.stats.partials++;
    
    if (!this.sessionData.timing.firstPartial && this.sessionData.timing.streamStart) {
      this.sessionData.timing.firstPartial = timestamp - this.sessionData.timing.streamStart;
      this.updateStageData('STT', 'timing', `${this.sessionData.timing.firstPartial.toFixed(1)}ms`);
    }

    this.updateStageStatus('STT', 'processing');
    this.updateStageData('STT', 'data', `Partial: "${message.text}"`);
    this.updateResponsePanel('transcript', message.text, 'partial');
    
    this.logDebugEvent('STT', 'info', 'Partial transcript', { 
      text: message.text,
      confidence: message.confidence 
    });
  }

  /**
   * Handle final transcript
   */
  handleFinalTranscript(message, timestamp) {
    this.sessionData.stats.finals++;
    
    if (!this.sessionData.timing.firstFinal && this.sessionData.timing.streamStart) {
      this.sessionData.timing.firstFinal = timestamp - this.sessionData.timing.streamStart;
      this.updateStageData('STT', 'timing', `${this.sessionData.timing.firstFinal.toFixed(1)}ms`);
    }

    this.updateStageStatus('STT', 'complete');
    this.updateStageStatus('LLM', 'processing');
    this.updateStageData('STT', 'data', `Final: "${message.text}"`);
    this.updateResponsePanel('transcript', message.text, 'final');
    this.updatePipelineStatus('processing', 'Processing transcript with LLM...');
    
    this.logDebugEvent('STT', 'info', 'Final transcript', { 
      text: message.text,
      confidence: message.confidence 
    });
  }

  /**
   * Handle assistant response
   */
  handleAssistantResponse(message, timestamp) {
    this.sessionData.stats.responses++;
    
    if (!this.sessionData.timing.firstAssistant && this.sessionData.timing.streamStart) {
      this.sessionData.timing.firstAssistant = timestamp - this.sessionData.timing.streamStart;
      this.updateStageData('LLM', 'timing', `${this.sessionData.timing.firstAssistant.toFixed(1)}ms`);
    }

    this.updateStageStatus('LLM', 'complete');
    this.updateStageStatus('TTS', 'processing');
    this.updateStageData('LLM', 'data', `Response: "${message.text}"`);
    this.updateResponsePanel('assistant', message.text, 'complete');
    this.updatePipelineStatus('processing', 'Generating speech...');
    
    this.logDebugEvent('LLM', 'info', 'Assistant response', { 
      text: message.text,
      tokens: message.tokens 
    });
  }

  /**
   * Handle TTS start
   */
  handleTTSStart(message, timestamp) {
    if (!this.sessionData.timing.ttsStart && this.sessionData.timing.streamStart) {
      this.sessionData.timing.ttsStart = timestamp - this.sessionData.timing.streamStart;
      this.updateStageData('TTS', 'timing', `${this.sessionData.timing.ttsStart.toFixed(1)}ms`);
    }

    this.updateStageStatus('TTS', 'processing');
    this.updateStageData('TTS', 'data', 'Generating speech...');
    
    this.logDebugEvent('TTS', 'info', 'TTS started', message);
  }

  /**
   * Handle TTS audio
   */
  handleTTSAudio(message) {
    if (message.audio && this.elements.responseAudio) {
      try {
        const audioBlob = this.base64ToBlob(message.audio, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        this.elements.responseAudio.src = audioUrl;
        
        // Auto-play if enabled
        if (this.config.autoplay) {
          this.elements.responseAudio.play().catch(e => {
            this.logDebugEvent('TTS', 'warning', 'Auto-play failed', { error: e.message });
          });
        }
        
        this.logDebugEvent('TTS', 'info', 'Audio received', { 
          size: message.audio.length,
          autoplay: this.config.autoplay 
        });
        
      } catch (error) {
        this.logDebugEvent('TTS', 'error', 'Failed to process audio', { error: error.message });
      }
    }
  }

  /**
   * Handle TTS complete
   */
  handleTTSComplete(message, timestamp) {
    if (!this.sessionData.timing.ttsComplete && this.sessionData.timing.streamStart) {
      this.sessionData.timing.ttsComplete = timestamp - this.sessionData.timing.streamStart;
      this.updateStageData('TTS', 'timing', `${this.sessionData.timing.ttsComplete.toFixed(1)}ms`);
    }

    this.updateStageStatus('TTS', 'complete');
    this.updateStageData('TTS', 'data', 'Speech generation complete');
    this.updatePipelineStatus('complete', 'Pipeline completed successfully');
    
    // Calculate total processing time
    const totalTime = this.sessionData.timing.ttsComplete;
    this.updateStageData('TTS', 'total', `Total: ${totalTime.toFixed(1)}ms`);
    
    this.logDebugEvent('TTS', 'info', 'TTS completed', { 
      processingTime: message.processing_ms,
      totalTime: totalTime 
    });
  }

  /**
   * Handle error messages
   */
  handleError(message) {
    this.sessionData.stats.errors++;
    this.updatePipelineStatus('error', `Error: ${message.message}`);
    this.logDebugEvent('ERROR', 'error', message.message, message);
  }

  /**
   * Handle debug events from server
   */
  handleDebugEvent(message) {
    this.logDebugEvent('DEBUG', 'debug', message.event, message.data);
  }

  /**
   * Handle file selection
   */
  async handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      this.updateStatus('processing');
      this.logDebugEvent('FILE', 'info', 'Processing audio file', { 
        name: file.name, 
        size: file.size, 
        type: file.type 
      });

      // Update file label
      if (this.elements.fileLabel) {
        this.elements.fileLabel.textContent = file.name;
      }

      // Decode audio file
      this.audioBuffer = await this.decodeAudioFile(file);
      
      this.logDebugEvent('FILE', 'info', 'Audio file decoded', { 
        samples: this.audioBuffer.length,
        duration: (this.audioBuffer.length / 16000).toFixed(2) + 's'
      });

      this.updateUI();

    } catch (error) {
      this.logDebugEvent('FILE', 'error', 'Failed to process audio file', { error: error.message });
      this.updateStatus('error');
    }
  }

  /**
   * Decode audio file to 16kHz mono PCM
   */
  async decodeAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    
    // Handle raw PCM files
    if (file.name.toLowerCase().endsWith('.pcm')) {
      return new Int16Array(arrayBuffer);
    }

    // Decode audio files using Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Resample to 16kHz mono
    const sourceRate = audioBuffer.sampleRate;
    const targetRate = 16000;
    const sourceData = audioBuffer.getChannelData(0);
    
    if (sourceRate === targetRate) {
      // No resampling needed, just convert to Int16
      const int16Array = new Int16Array(sourceData.length);
      for (let i = 0; i < sourceData.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, sourceData[i] * 32767));
      }
      return int16Array;
    }

    // Simple linear resampling
    const ratio = sourceRate / targetRate;
    const targetLength = Math.floor(sourceData.length / ratio);
    const targetData = new Int16Array(targetLength);
    
    for (let i = 0; i < targetLength; i++) {
      const sourceIndex = i * ratio;
      const value = sourceData[Math.floor(sourceIndex)];
      targetData[i] = Math.max(-32768, Math.min(32767, value * 32767));
    }
    
    return targetData;
  }

  /**
   * Start streaming audio to server
   */
  startStreaming() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.logDebugEvent('STREAM', 'error', 'Not connected to server');
      return;
    }

    if (!this.audioBuffer) {
      this.logDebugEvent('STREAM', 'error', 'No audio file loaded');
      return;
    }

    if (this.isStreaming) {
      this.logDebugEvent('STREAM', 'warning', 'Already streaming');
      return;
    }

    this.isStreaming = true;
    this.streamOffset = 0;
    this.sessionData.stats.framesSent = 0;
    this.sessionData.stats.bytesSent = 0;
    this.sessionData.timing.streamStart = performance.now();

    // Reset stage statuses
    this.updateStageStatus('STT', 'waiting');
    this.updateStageStatus('LLM', 'waiting');
    this.updateStageStatus('TTS', 'waiting');
    this.updatePipelineStatus('streaming', 'Streaming audio data...');

    this.logDebugEvent('STREAM', 'info', 'Started streaming', {
      totalSamples: this.audioBuffer.length,
      frameSize: this.config.frameSize,
      pacing: this.config.pacing
    });

    this.scheduleNextFrame();
    this.updateUI();
  }

  /**
   * Schedule the next audio frame to be sent
   */
  scheduleNextFrame() {
    if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const samplesPerFrame = Math.floor(16000 * this.config.frameSize / 1000);
    
    if (this.streamOffset >= this.audioBuffer.length) {
      this.completeStreaming();
      return;
    }

    const endOffset = Math.min(this.audioBuffer.length, this.streamOffset + samplesPerFrame);
    const frame = this.audioBuffer.subarray(this.streamOffset, endOffset);
    
    try {
      this.ws.send(frame.buffer);
      this.sessionData.stats.framesSent++;
      this.sessionData.stats.bytesSent += frame.length * 2;
      
      this.streamOffset = endOffset;
      
      // Schedule next frame
      if (this.config.pacing === 'fast') {
        this.frameTimer = setTimeout(() => this.scheduleNextFrame(), 1);
      } else {
        this.frameTimer = setTimeout(() => this.scheduleNextFrame(), this.config.frameSize);
      }
      
    } catch (error) {
      this.logDebugEvent('STREAM', 'error', 'Failed to send frame', { error: error.message });
      this.abortStreaming();
    }
  }

  /**
   * Complete streaming
   */
  completeStreaming() {
    this.isStreaming = false;
    if (this.frameTimer) {
      clearTimeout(this.frameTimer);
      this.frameTimer = null;
    }

    const streamDuration = performance.now() - this.sessionData.timing.streamStart;
    const audioDuration = (this.audioBuffer.length / 16000) * 1000;
    
    this.logDebugEvent('STREAM', 'info', 'Streaming completed', {
      framesSent: this.sessionData.stats.framesSent,
      bytesSent: this.sessionData.stats.bytesSent,
      streamDuration: streamDuration.toFixed(1) + 'ms',
      audioDuration: audioDuration.toFixed(1) + 'ms'
    });

    this.updatePipelineStatus('processing', 'Audio stream completed, processing...');
    this.updateUI();
  }

  /**
   * Abort streaming
   */
  abortStreaming() {
    if (!this.isStreaming) return;

    this.isStreaming = false;
    if (this.frameTimer) {
      clearTimeout(this.frameTimer);
      this.frameTimer = null;
    }

    this.logDebugEvent('STREAM', 'warning', 'Streaming aborted');
    this.updatePipelineStatus('aborted', 'Streaming aborted');
    this.updateUI();
  }

  /**
   * Reset session data
   */
  resetSession() {
    this.sessionData = {
      sessionId: null,
      connectionId: null,
      startTime: null,
      stats: {
        framesSent: 0,
        bytesSent: 0,
        partials: 0,
        finals: 0,
        responses: 0,
        errors: 0
      },
      timing: {
        firstPartial: null,
        firstFinal: null,
        firstAssistant: null,
        ttsStart: null,
        ttsComplete: null,
        streamStart: null
      }
    };

    this.debugEvents = [];
    this.streamOffset = 0;
    this.isStreaming = false;

    // Reset UI
    this.updatePipelineStatus('idle', 'Idle');
    this.clearStageData();
    this.clearResponsePanels();
    this.updateDebugEventsDisplay();
  }

  /**
   * Update connection status
   */
  updateStatus(status) {
    if (this.elements.wsStatus) {
      this.elements.wsStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      this.elements.wsStatus.className = `status-indicator ${status}`;
    }
  }

  /**
   * Update pipeline status
   */
  updatePipelineStatus(status, message) {
    if (this.elements.pipelineStatus) {
      this.elements.pipelineStatus.textContent = message || status;
      this.elements.pipelineStatus.className = `pipeline-status ${status}`;
    }

    // Update progress bar based on status
    if (this.elements.pipelineProgress) {
      let progress = 0;
      switch (status) {
        case 'streaming': progress = 25; break;
        case 'processing': progress = 50; break;
        case 'complete': progress = 100; break;
        case 'error': progress = 0; break;
        default: progress = 0;
      }
      this.elements.pipelineProgress.style.width = `${progress}%`;
      this.elements.pipelineProgress.className = `progress-bar ${status}`;
    }
  }

  /**
   * Update stage status
   */
  updateStageStatus(stage, status) {
    const stageElement = this.elements[`stage${stage}Status`];
    if (stageElement) {
      stageElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      stageElement.className = `stage-status ${status}`;
    }
  }

  /**
   * Update stage data
   */
  updateStageData(stage, type, data) {
    const dataElement = this.elements[`stage${stage}${type.charAt(0).toUpperCase() + type.slice(1)}`];
    if (dataElement) {
      dataElement.textContent = data;
    }
  }

  /**
   * Update response panel
   */
  updateResponsePanel(type, content, status) {
    const element = this.elements[`response${type.charAt(0).toUpperCase() + type.slice(1)}`];
    if (element) {
      element.textContent = content;
      element.className = `response-content ${status}`;
    }
  }

  /**
   * Clear stage data
   */
  clearStageData() {
    ['STT', 'LLM', 'TTS'].forEach(stage => {
      this.updateStageStatus(stage, 'idle');
      this.updateStageData(stage, 'timing', '-');
      this.updateStageData(stage, 'data', '');
    });
  }

  /**
   * Clear response panels
   */
  clearResponsePanels() {
    if (this.elements.responseTranscript) this.elements.responseTranscript.textContent = '';
    if (this.elements.responseAssistant) this.elements.responseAssistant.textContent = '';
    if (this.elements.responseAudio) this.elements.responseAudio.src = '';
  }

  /**
   * Play response audio
   */
  playResponseAudio() {
    if (this.elements.responseAudio && this.elements.responseAudio.src) {
      this.elements.responseAudio.play().catch(error => {
        this.logDebugEvent('AUDIO', 'error', 'Failed to play audio', { error: error.message });
      });
    }
  }

  /**
   * Log debug event
   */
  logDebugEvent(stage, level, message, data = {}) {
    const timestamp = new Date();
    const event = {
      timestamp: timestamp.toISOString(),
      relativeTime: this.sessionData.timing.streamStart ? 
        (performance.now() - this.sessionData.timing.streamStart).toFixed(1) + 'ms' : 
        '0ms',
      stage,
      level,
      message,
      data
    };

    this.debugEvents.push(event);
    this.updateDebugEventsDisplay();
    
    // Log to console for development
    console.log(`[${event.relativeTime}] ${stage}:${level.toUpperCase()} - ${message}`, data);
  }

  /**
   * Update debug events display
   */
  updateDebugEventsDisplay() {
    if (!this.elements.debugEventsLog) return;

    const filteredEvents = this.getFilteredDebugEvents();
    const logHtml = filteredEvents.map(event => {
      const timeStr = event.relativeTime.padStart(8, ' ');
      const stageStr = event.stage.padEnd(10, ' ');
      return `
        <div class="log-entry ${event.level}">
          <span class="log-timestamp">${timeStr}</span>
          <span class="log-level ${event.level}">${event.level.toUpperCase()}</span>
          <span class="log-stage">${stageStr}</span>
          <span class="log-message">${event.message}</span>
        </div>
      `;
    }).join('');

    this.elements.debugEventsLog.innerHTML = logHtml;
    this.elements.debugEventsLog.scrollTop = this.elements.debugEventsLog.scrollHeight;
  }

  /**
   * Get filtered debug events based on current filter settings
   */
  getFilteredDebugEvents() {
    const showInfo = this.elements.showInfoEvents?.checked || false;
    const showWarning = this.elements.showWarningEvents?.checked || false;
    const showError = this.elements.showErrorEvents?.checked || false;
    const showDebug = this.elements.showDebugEvents?.checked || false;

    return this.debugEvents.filter(event => {
      switch (event.level) {
        case 'info': return showInfo;
        case 'warning': return showWarning;
        case 'error': return showError;
        case 'debug': return showDebug;
        default: return true;
      }
    });
  }

  /**
   * Filter debug events display
   */
  filterDebugEvents() {
    this.updateDebugEventsDisplay();
  }

  /**
   * Export logs
   */
  exportLogs() {
    const exportData = {
      session: this.sessionData,
      config: this.config,
      events: this.debugEvents,
      exportTime: new Date().toISOString()
    };

    const format = this.elements.exportFormat?.value || 'json';
    const filename = `beautyai-debug-logs-${Date.now()}`;

    switch (format) {
      case 'json':
        this.downloadJson(exportData, `${filename}.json`);
        break;
      case 'csv':
        this.downloadCsv(this.debugEvents, `${filename}.csv`);
        break;
      case 'markdown':
        this.downloadMarkdown(exportData, `${filename}.md`);
        break;
    }

    this.logDebugEvent('EXPORT', 'info', 'Logs exported', { format, filename });
  }

  /**
   * Export session data
   */
  exportSession() {
    const sessionData = {
      ...this.sessionData,
      config: this.config,
      exportTime: new Date().toISOString()
    };

    this.downloadJson(sessionData, `beautyai-session-${Date.now()}.json`);
    this.logDebugEvent('EXPORT', 'info', 'Session data exported');
  }

  /**
   * Export debug data
   */
  exportDebugData() {
    const debugData = {
      events: this.debugEvents.filter(e => e.level === 'debug'),
      session: this.sessionData,
      exportTime: new Date().toISOString()
    };

    this.downloadJson(debugData, `beautyai-debug-data-${Date.now()}.json`);
    this.logDebugEvent('EXPORT', 'info', 'Debug data exported');
  }

  /**
   * Download JSON data
   */
  downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    this.downloadBlob(blob, filename);
  }

  /**
   * Download CSV data
   */
  downloadCsv(events, filename) {
    const headers = ['Timestamp', 'Relative Time', 'Stage', 'Level', 'Message', 'Data'];
    const rows = events.map(event => [
      event.timestamp,
      event.relativeTime,
      event.stage,
      event.level,
      event.message,
      JSON.stringify(event.data)
    ]);

    const csvContent = [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    this.downloadBlob(blob, filename);
  }

  /**
   * Download Markdown report
   */
  downloadMarkdown(data, filename) {
    const markdown = this.generateMarkdownReport(data);
    const blob = new Blob([markdown], { type: 'text/markdown' });
    this.downloadBlob(blob, filename);
  }

  /**
   * Generate Markdown report
   */
  generateMarkdownReport(data) {
    const { session, config, events } = data;
    
    let markdown = `# BeautyAI Debug Report\n\n`;
    markdown += `Generated: ${new Date().toISOString()}\n\n`;
    
    markdown += `## Session Information\n\n`;
    markdown += `- Session ID: ${session.sessionId || 'N/A'}\n`;
    markdown += `- Connection ID: ${session.connectionId || 'N/A'}\n`;
    markdown += `- Start Time: ${session.startTime || 'N/A'}\n\n`;
    
    markdown += `## Configuration\n\n`;
    markdown += `- Language: ${config.language}\n`;
    markdown += `- Frame Size: ${config.frameSize}ms\n`;
    markdown += `- Pacing: ${config.pacing}\n`;
    markdown += `- Autoplay: ${config.autoplay}\n\n`;
    
    markdown += `## Statistics\n\n`;
    markdown += `- Frames Sent: ${session.stats.framesSent}\n`;
    markdown += `- Bytes Sent: ${session.stats.bytesSent}\n`;
    markdown += `- Partials: ${session.stats.partials}\n`;
    markdown += `- Finals: ${session.stats.finals}\n`;
    markdown += `- Responses: ${session.stats.responses}\n`;
    markdown += `- Errors: ${session.stats.errors}\n\n`;
    
    markdown += `## Timing\n\n`;
    markdown += `- First Partial: ${session.timing.firstPartial ? session.timing.firstPartial.toFixed(1) + 'ms' : 'N/A'}\n`;
    markdown += `- First Final: ${session.timing.firstFinal ? session.timing.firstFinal.toFixed(1) + 'ms' : 'N/A'}\n`;
    markdown += `- First Assistant: ${session.timing.firstAssistant ? session.timing.firstAssistant.toFixed(1) + 'ms' : 'N/A'}\n`;
    markdown += `- TTS Start: ${session.timing.ttsStart ? session.timing.ttsStart.toFixed(1) + 'ms' : 'N/A'}\n`;
    markdown += `- TTS Complete: ${session.timing.ttsComplete ? session.timing.ttsComplete.toFixed(1) + 'ms' : 'N/A'}\n\n`;
    
    markdown += `## Events\n\n`;
    markdown += `| Time | Stage | Level | Message |\n`;
    markdown += `|------|-------|-------|----------|\n`;
    
    events.forEach(event => {
      markdown += `| ${event.relativeTime} | ${event.stage} | ${event.level.toUpperCase()} | ${event.message} |\n`;
    });
    
    return markdown;
  }

  /**
   * Download blob as file
   */
  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * Convert base64 to blob
   */
  base64ToBlob(base64, mimeType) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: mimeType });
  }

  /**
   * Update UI state based on current status
   */
  updateUI() {
    const isConnected = this.ws && this.ws.readyState === WebSocket.OPEN;
    const hasAudio = this.audioBuffer !== null;
    const isStreaming = this.isStreaming;

    // Update button states
    if (this.elements.connectBtn) {
      this.elements.connectBtn.disabled = isConnected;
    }
    if (this.elements.disconnectBtn) {
      this.elements.disconnectBtn.disabled = !isConnected;
    }
    if (this.elements.streamBtn) {
      this.elements.streamBtn.disabled = !isConnected || !hasAudio || isStreaming;
    }
    if (this.elements.abortBtn) {
      this.elements.abortBtn.disabled = !isStreaming;
    }

    // Update export buttons
    const hasEvents = this.debugEvents.length > 0;
    if (this.elements.exportLogsBtn) {
      this.elements.exportLogsBtn.disabled = !hasEvents;
    }
    if (this.elements.exportSessionBtn) {
      this.elements.exportSessionBtn.disabled = !this.sessionData.sessionId;
    }
    if (this.elements.exportDebugBtn) {
      this.elements.exportDebugBtn.disabled = !hasEvents;
    }

    // Update session statistics
    this.updateSessionStats();
  }

  /**
   * Update session statistics
   */
  updateSessionStats() {
    if (this.elements.sessionCount) {
      this.elements.sessionCount.textContent = this.sessionData.sessionId ? '1' : '0';
    }

    if (this.elements.avgResponseTime && this.sessionData.timing.ttsComplete) {
      this.elements.avgResponseTime.textContent = `${this.sessionData.timing.ttsComplete.toFixed(1)}ms`;
    }

    if (this.elements.successRate) {
      const total = this.sessionData.stats.responses + this.sessionData.stats.errors;
      if (total > 0) {
        const rate = (this.sessionData.stats.responses / total * 100).toFixed(1);
        this.elements.successRate.textContent = `${rate}%`;
      }
    }
  }
}

// Export for use in HTML
window.DebugWebSocketTester = DebugWebSocketTester;