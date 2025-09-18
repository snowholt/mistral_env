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
    this.originalFileData = null;  // Store original file data for WebM files
    this.fileFormat = null;        // Track file format ('webm', 'pcm', 'decoded')
    this.streamOffset = 0;
    this.frameTimer = null;
    this.isStreaming = false;
    
    // DOM elements cache
    this.elements = {};
    this.config = {};
  }

  /**
   * Handle voice_response messages from the backend
   */
  handleVoiceResponse(message, timestamp) {
    try {
      // Update stats
      this.sessionData.stats.responses++;

      // Update transcript and assistant panels
      if (message.transcription) {
        this.updateResponsePanel('transcript', message.transcription, 'final');
      }
      if (message.response_text) {
        this.updateResponsePanel('assistant', message.response_text, 'complete');
      }

      // Handle audio if present
      if (message.audio_base64 && this.elements.responseAudio) {
        try {
          const audioBlob = this.base64ToBlob(message.audio_base64, 'audio/wav');
          const audioUrl = URL.createObjectURL(audioBlob);
          this.elements.responseAudio.src = audioUrl;
          if (this.config.autoplay) {
            this.elements.responseAudio.play().catch(e => {
              this.logDebugEvent('AUDIO', 'warning', 'Auto-play failed', { error: e.message });
            });
          }
        } catch (err) {
          this.logDebugEvent('AUDIO', 'error', 'Failed to decode response audio', { error: err.message });
        }
      }

      // Pipeline complete
      this.updateStageStatus('LLM', 'complete');
      this.updateStageStatus('TTS', 'complete');
      this.updatePipelineStatus('complete', 'Pipeline completed');
      this.logDebugEvent('SERVER', 'info', 'Voice response received', { message });

    } catch (err) {
      this.logDebugEvent('SERVER', 'error', 'handleVoiceResponse failed', { error: err.message });
    }
  }

  /**
   * Initialize the debug interface
   */
  initialize() {
    this.cacheElements();
    this.bindEvents();
    this.updateStatus('disconnected');
    this.initializeWebSocketSelector();
    this.updateWebSocketInfo();
    this.initializeDashboard();
    this.updateUI();
    this.logDebugEvent('SYSTEM', 'info', 'Debug WebSocket Tester initialized');
  }

  /**
   * Update WebSocket URL with current configuration parameters
   */
  updateWebSocketUrl() {
    if (!this.elements.websocketUrl) return;
    
    // Get current language and voice type settings from the actual form elements
    const language = this.elements.languageSelect?.value || this.elements.configLanguage?.value || 'ar';
    const voiceType = this.elements.voiceTypeSelect?.value || 'female';
    const debugMode = this.elements.debugModeToggle?.checked ? 'true' : 'false';
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const isProduction = window.location.hostname !== 'localhost';
    const host = isProduction ? 'dev.gmai.sa' : 'localhost:8000';
    
    // Construct URL with required parameters - use debug_mode, not debug
    const url = `${protocol}//${host}/api/v1/ws/simple-voice-chat?language=${encodeURIComponent(language)}&voice_type=${voiceType}&debug_mode=${debugMode}`;
    
    this.elements.websocketUrl.value = url;
    this.updateWebSocketInfo();
    
    this.logDebugEvent('CONFIG', 'info', 'WebSocket URL updated', { 
      url, 
      language, 
      voiceType, 
      debugMode: debugMode === 'true' 
    });
  }

  /**
   * Initialize WebSocket selector with default values
   */
  initializeWebSocketSelector() {
    if (!this.elements.websocketUrl) return;
    
    // Use the proper URL construction method that includes parameters
    this.updateWebSocketUrl();
  }

  /**
   * Cache DOM elements for performance
   */
  cacheElements() {
    const elementIds = [
      'wsStatus', 'configLanguage', 'configFrameSize', 'configPacing', 'configAutoplay',
      'websocketUrl', 'protocolBadge', 'endpointDescription', 'connectBtn', 'disconnectBtn', 'streamBtn', 'abortBtn',
      'audioFile', 'fileLabel', 'selectFileBtn', 'fileUploadZone', 'removeFileBtn', 
      'fileName', 'fileSize', 'fileType', 'fileInfo', 'uploadProgress', 'progressFill', 'progressText',
      'pipelineStatus', 'pipelineProgress', 'stageSTT', 'stageLLM', 'stageTTS',
      'stageSTTStatus', 'stageSTTTiming', 'stageSTTData',
      'stageLLMStatus', 'stageLLMTiming', 'stageLLMData',
      'stageTTSStatus', 'stageTTSTiming', 'stageTTSData',
      'responseTranscript', 'responseAssistant', 'responseAudio', 'responsePlayBtn',
      'debugEventsLog', 'showInfoEvents', 'showWarningEvents', 'showErrorEvents', 'showDebugEvents',
      'exportLogsBtn', 'exportSessionBtn', 'exportDebugBtn', 'exportFormat',
      'sessionCount', 'avgResponseTime', 'successRate',
      // Add the actual form element IDs from the HTML
      'languageSelect', 'voiceTypeSelect', 'debugModeToggle', 'sendAudioBtn'
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

    // WebSocket URL selector
    if (this.elements.websocketUrl) {
      this.elements.websocketUrl.addEventListener('change', () => this.updateWebSocketInfo());
    }

    // Modern file upload handling
    if (this.elements.selectFileBtn || this.elements.fileUploadZone) {
      const clickHandler = () => {
        if (this.elements.audioFile) {
          this.elements.audioFile.click();
        }
      };
      
      if (this.elements.selectFileBtn) {
        this.elements.selectFileBtn.addEventListener('click', clickHandler);
      }
      if (this.elements.fileUploadZone) {
        this.elements.fileUploadZone.addEventListener('click', clickHandler);
      }
    }

    // File input change
    if (this.elements.audioFile) {
      this.elements.audioFile.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    // Remove file button
    if (this.elements.removeFileBtn) {
      this.elements.removeFileBtn.addEventListener('click', () => this.removeFile());
    }

    // Drag and drop support
    if (this.elements.fileUploadZone) {
      this.setupDragAndDrop();
    }

    // Language and voice type changes update WebSocket URL
    if (this.elements.languageSelect) {
      this.elements.languageSelect.addEventListener('change', () => {
        this.updateWebSocketUrl();
        this.updateWebSocketInfo();
      });
    }
    
    if (this.elements.voiceTypeSelect) {
      this.elements.voiceTypeSelect.addEventListener('change', () => {
        this.updateWebSocketUrl();
        this.updateWebSocketInfo();
      });
    }
    
    if (this.elements.debugModeToggle) {
      this.elements.debugModeToggle.addEventListener('change', () => {
        this.updateWebSocketUrl();
        this.updateWebSocketInfo();
      });
    }

    // Legacy support for configLanguage if it exists
    if (this.elements.configLanguage) {
      this.elements.configLanguage.addEventListener('change', () => {
        this.updateWebSocketUrl();
        this.updateWebSocketInfo();
      });
    }

    // Streaming controls
    if (this.elements.sendAudioBtn) {
      this.elements.sendAudioBtn.addEventListener('click', () => this.startStreaming());
    }
    // Legacy support
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
      this.elements.configLanguage.addEventListener('change', () => {
        this.updateWebSocketUrl();
        this.updateWebSocketInfo();
      });
    }
  }

  /**
   * Setup drag and drop functionality for file upload
   */
  setupDragAndDrop() {
    const zone = this.elements.fileUploadZone;
    if (!zone) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      zone.addEventListener(eventName, this.preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      zone.addEventListener(eventName, () => zone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      zone.addEventListener(eventName, () => zone.classList.remove('drag-over'), false);
    });

    zone.addEventListener('drop', (e) => this.handleDrop(e), false);
  }

  /**
   * Prevent default drag behaviors
   */
  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  /**
   * Handle file drop
   */
  handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.elements.audioFile.files = files;
      this.handleFileSelect({ target: { files: files } });
    }
  }

  /**
   * Remove the selected file
   */
  removeFile() {
    if (this.elements.audioFile) {
      this.elements.audioFile.value = '';
    }
    this.audioBuffer = null;
    
    // Hide file preview
    if (this.elements.fileInfo) {
      this.elements.fileInfo.style.display = 'none';
    }
    
    // Reset file label
    if (this.elements.fileLabel) {
      this.elements.fileLabel.textContent = 'Drop audio files here';
    }
    
    // Reset dashboard stages
    this.updateStageStatus('Upload', 'ready');
    this.updateStageStatus('STT', 'waiting');
    this.updateStageStatus('LLM', 'waiting');
    this.updateStageStatus('TTS', 'waiting');
    
    this.logDebugEvent('FILE', 'info', 'File removed');
    this.updateUI();
  }

  /**
   * Update WebSocket info display when selection changes
   */
  updateWebSocketInfo() {
    if (!this.elements.websocketUrl) return;
    
    const selectedUrl = this.elements.websocketUrl.value;
    const isSecure = selectedUrl.startsWith('wss://');
    
    // Update protocol badge
    if (this.elements.protocolBadge) {
      this.elements.protocolBadge.textContent = isSecure ? 'WSS' : 'WS';
      this.elements.protocolBadge.className = `protocol-badge ${isSecure ? 'wss' : 'ws'}`;
    }
    
    // Update description
    if (this.elements.endpointDescription) {
      let description = '';
      if (selectedUrl.includes('simple-voice-chat')) {
        description = isSecure ? 'Secure WebSocket for simple voice chat' : 'Local WebSocket for simple voice chat';
      } else if (selectedUrl.includes('streaming-voice')) {
        description = isSecure ? 'Secure WebSocket for streaming voice' : 'Local WebSocket for streaming voice';
      }
      this.elements.endpointDescription.textContent = description;
    }
  }

  /**
   * Update configuration from UI
   */
  updateConfig() {
    this.config = {
      language: this.elements.languageSelect?.value || this.elements.configLanguage?.value || 'ar',
      voiceType: this.elements.voiceTypeSelect?.value || 'female',
      debugMode: this.elements.debugModeToggle?.checked || false,
      frameSize: parseInt(this.elements.configFrameSize?.value || '20'),
      pacing: this.elements.configPacing?.value || 'realtime',
      autoplay: this.elements.configAutoplay?.checked || false,
      endpointUrl: this.elements.websocketUrl?.value || this.getDefaultEndpointUrl()
    };
  }

  /**
   * Get default endpoint URL based on current configuration
   */
  getDefaultEndpointUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const isProduction = window.location.hostname !== 'localhost';
    const host = isProduction ? 'dev.gmai.sa' : 'localhost:8000';
    const language = this.elements.languageSelect?.value || this.elements.configLanguage?.value || 'ar';
    const voiceType = this.elements.voiceTypeSelect?.value || 'female';
    const debugMode = this.elements.debugModeToggle?.checked ? 'true' : 'false';
    
    return `${protocol}//${host}/api/v1/ws/simple-voice-chat?language=${encodeURIComponent(language)}&voice_type=${voiceType}&debug_mode=${debugMode}`;
  }

  /**
   * Update WebSocket info display when selection changes
   */
  updateWebSocketInfo() {
    if (!this.elements.websocketUrl) return;
    
    const selectedUrl = this.elements.websocketUrl.value;
    const isSecure = selectedUrl.startsWith('wss://');
    
    // Update protocol badge
    if (this.elements.protocolBadge) {
      this.elements.protocolBadge.textContent = isSecure ? 'WSS' : 'WS';
      this.elements.protocolBadge.className = `protocol-badge ${isSecure ? 'wss' : 'ws'}`;
    }
    
    // Update description
    if (this.elements.endpointDescription) {
      let description = '';
      if (selectedUrl.includes('simple-voice-chat')) {
        description = isSecure ? 'Secure WebSocket for simple voice chat' : 'Local WebSocket for simple voice chat';
      } else if (selectedUrl.includes('streaming-voice')) {
        description = isSecure ? 'Secure WebSocket for streaming voice' : 'Local WebSocket for streaming voice';
      }
      this.elements.endpointDescription.textContent = description;
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
      // Use the URL from the selector, which should have proper parameters
      let wsUrl = this.config.endpointUrl;
      
      // If the URL doesn't have parameters, use the default URL constructor
      if (!wsUrl.includes('language=') || !wsUrl.includes('voice_type=')) {
        wsUrl = this.getDefaultEndpointUrl();
        // Update the selector to show the proper URL
        if (this.elements.websocketUrl) {
          this.elements.websocketUrl.value = wsUrl;
        }
      }
      
      // Ensure we're using debug_mode parameter format
      if (wsUrl.includes('debug=')) {
        wsUrl = wsUrl.replace(/debug=([^&]+)/, 'debug_mode=$1');
        if (this.elements.websocketUrl) {
          this.elements.websocketUrl.value = wsUrl;
        }
      }
      
      this.logDebugEvent('CONNECTION', 'info', 'Connecting to WebSocket', { url: wsUrl });
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        this.updateStatus('connected');
        this.logDebugEvent('CONNECTION', 'info', 'WebSocket connected', { url: wsUrl });
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
        this.logDebugEvent('CONNECTION', 'error', 'WebSocket error', { error: error });
        this.sessionData.stats.errors++;
      };

      // Robust onmessage handler: support string, Blob, and ArrayBuffer payloads
      this.ws.onmessage = (event) => {
        try {
          // If the server sent a string (JSON), handle directly
          if (typeof event.data === 'string') {
            this.handleServerMessage(event.data);
            return;
          }

          // If it's a Blob (binary), try to convert to text first (many servers send JSON as Blob)
          if (event.data instanceof Blob) {
            event.data.text().then(text => {
              this.handleServerMessage(text);
            }).catch(() => {
              // Can't interpret as text - ignore binary payloads here
              this.logDebugEvent('SERVER', 'debug', 'Received binary message (Blob) - ignored by debug client');
            });
            return;
          }

          // If it's an ArrayBuffer, attempt to decode as UTF-8 JSON text
          if (event.data instanceof ArrayBuffer) {
            try {
              const text = new TextDecoder().decode(new Uint8Array(event.data));
              this.handleServerMessage(text);
            } catch (err) {
              this.logDebugEvent('SERVER', 'debug', 'Received binary message (ArrayBuffer) - ignored by debug client');
            }
            return;
          }

          // Fallback: try to stringify and pass through
          this.handleServerMessage(String(event.data));

        } catch (err) {
          this.logDebugEvent('SERVER', 'error', 'onmessage handler failed', { error: err.message });
        }
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
        // Backwards-compatible: server may send 'connection_established' or 'ready'
        case 'ready':
          this.handleReadyMessage(message);
          break;

        case 'connection_established':
          // Map server's connection_established to client-ready handling
          this.handleReadyMessage(message);
          break;

        case 'processing_started':
          this.logDebugEvent('SERVER', 'info', 'Server started processing audio', { message });
          this.updatePipelineStatus('processing', 'Server started processing audio...');
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

        case 'voice_response':
          // Newer backend message type: contains audio_base64, transcription, response_text
          this.handleVoiceResponse(message, now);
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
    
    // Update dashboard with session ID
    this.updatePipelineMetrics({ 
      sessionId: this.sessionData.sessionId 
    });
    
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
    
    // Enhanced error handling with specific guidance
    let userGuidance = '';
    const errorCode = message.error_code || 'UNKNOWN_ERROR';
    
    switch (errorCode) {
      case 'WEBM_FORMAT_MISMATCH':
        userGuidance = '💡 **Tip**: This usually happens when using the PCM debug tool with WebM files. The tool now automatically handles this by sending original WebM data instead of converted PCM.';
        break;
      case 'AUDIO_FORMAT_ERROR':
        userGuidance = '💡 **Tip**: Try uploading a different audio file format (WebM, WAV, MP3) or check if the file is corrupted.';
        break;
      case 'WEBM_DECODER_ERROR':
        userGuidance = '💡 **Tip**: The WebM file may be incomplete or corrupted. Try re-uploading the original file.';
        break;
      case 'AUDIO_PROCESSING_ERROR':
        userGuidance = '💡 **Tip**: Check your audio file format and ensure it\'s a valid audio file.';
        break;
      default:
        userGuidance = '💡 **Tip**: Please check the technical details below and try uploading a different audio file.';
    }
    
    // Log detailed error information
    this.logDebugEvent('ERROR', 'error', message.message, {
      error_code: errorCode,
      audio_format_detected: message.audio_format_detected,
      technical_details: message.technical_details,
      user_guidance: userGuidance,
      retry_suggested: message.retry_suggested
    });
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
      this.updateStageStatus('Upload', 'processing');
      this.logDebugEvent('FILE', 'info', 'Processing audio file', { 
        name: file.name, 
        size: file.size, 
        type: file.type 
      });

      // Show file preview with modern UI
      this.showFilePreview(file);

      // Store original file for WebM files, decode for PCM files
      if (file.name.toLowerCase().endsWith('.webm') || file.type === 'video/webm') {
        // For WebM files, keep the original file data to preserve container format
        this.originalFileData = await file.arrayBuffer();
        
        // Also decode for display purposes but don't use for transmission
        this.audioBuffer = await this.decodeAudioFile(file);
        this.fileFormat = 'webm';
        
        this.logDebugEvent('FILE', 'info', 'WebM file loaded (original format preserved)', { 
          originalSize: this.originalFileData.byteLength,
          decodedSamples: this.audioBuffer.length,
          duration: (this.audioBuffer.length / 16000).toFixed(2) + 's'
        });
      } else {
        // For other formats, decode to PCM
        this.audioBuffer = await this.decodeAudioFile(file);
        this.originalFileData = null;
        this.fileFormat = file.name.toLowerCase().endsWith('.pcm') ? 'pcm' : 'decoded';
        
        this.logDebugEvent('FILE', 'info', 'Audio file decoded to PCM', { 
          samples: this.audioBuffer.length,
          duration: (this.audioBuffer.length / 16000).toFixed(2) + 's'
        });
      }

      this.updateStatus('ready');
      this.updateStageStatus('Upload', 'complete');
      this.updateStageStatus('STT', 'ready');
      this.updateUI();

    } catch (error) {
      this.logDebugEvent('FILE', 'error', 'Failed to process audio file', { error: error.message });
      this.updateStatus('error');
      this.updateStageStatus('Upload', 'error');
    }
  }

  /**
   * Show file preview in modern UI
   */
  showFilePreview(file) {
    // Update file details
    if (this.elements.fileName) {
      this.elements.fileName.textContent = file.name;
    }
    if (this.elements.fileSize) {
      this.elements.fileSize.textContent = this.formatFileSize(file.size);
    }
    if (this.elements.fileType) {
      this.elements.fileType.textContent = file.type || 'Unknown';
    }

    // Show file preview panel
    if (this.elements.fileInfo) {
      this.elements.fileInfo.style.display = 'block';
    }

    // Update main label
    if (this.elements.fileLabel) {
      this.elements.fileLabel.textContent = 'File selected';
    }
  }

  /**
   * Format file size for display
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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

    if (!this.audioBuffer && !this.originalFileData) {
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

    // Handle different file formats appropriately
    if (this.fileFormat === 'webm' && this.originalFileData) {
      // Send original WebM file data directly for WebM files
      this.logDebugEvent('STREAM', 'info', 'Streaming WebM file data', {
        fileSize: this.originalFileData.byteLength,
        format: 'webm',
        method: 'single_send'
      });
      
      try {
        // Split into multiple chunks to simulate MediaRecorder chunked stream
        const desiredChunks = 40; // Ensure server's accumulation threshold (>=30) is reached
        const totalBytes = this.originalFileData.byteLength;
        const chunkSize = Math.max(1024 * 4, Math.ceil(totalBytes / desiredChunks));
        const view = new Uint8Array(this.originalFileData);
        let offset = 0;
        let chunksSent = 0;

        const sendChunk = () => {
          if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.logDebugEvent('STREAM', 'error', 'WebSocket closed while streaming WebM');
            this.isStreaming = false;
            return;
          }

          const end = Math.min(totalBytes, offset + chunkSize);
          const slice = view.subarray(offset, end);
          try {
            this.ws.send(slice.buffer);
            chunksSent++;
            this.sessionData.stats.framesSent++;
            this.sessionData.stats.bytesSent += slice.length;
          } catch (err) {
            this.logDebugEvent('STREAM', 'error', 'Failed to send WebM chunk', { error: err.message });
            this.isStreaming = false;
            return;
          }

          offset = end;
          if (offset < totalBytes) {
            // Schedule next chunk very quickly
            setTimeout(sendChunk, 2);
          } else {
            // All chunks sent
            this.logDebugEvent('STREAM', 'info', 'All WebM chunks sent', { chunksSent, totalBytes });
            // Give server a small moment to acknowledge and start processing
            setTimeout(() => this.completeStreaming(), 200);
          }
        };

        // Initialize counters
        this.sessionData.stats.framesSent = 0;
        this.sessionData.stats.bytesSent = 0;
        sendChunk();
      } catch (error) {
        this.logDebugEvent('STREAM', 'error', 'Failed to send WebM data', { error: error.message });
        this.isStreaming = false;
        this.updateUI();
      }
    } else {
      // Send PCM data in frames for other formats
      this.logDebugEvent('STREAM', 'info', 'Started PCM streaming', {
        totalSamples: this.audioBuffer.length,
        frameSize: this.config.frameSize,
        pacing: this.config.pacing,
        format: this.fileFormat
      });

      this.scheduleNextFrame();
    }
    
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
   * Update stage status in the dashboard table
   */
  updateStageStatus(stage, status) {
    const stageMap = {
      'Upload': 'upload',
      'STT': 'stt', 
      'LLM': 'llm',
      'TTS': 'tts'
    };
    
    const stageKey = stageMap[stage] || stage.toLowerCase();
    
    // Update status badge
    const statusElement = document.getElementById(`${stageKey}Status`);
    if (statusElement) {
      statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      statusElement.className = `status-badge ${status}`;
    }
    
    // Update stage row class
    const rowElement = document.getElementById(`${stageKey}StageRow`);
    if (rowElement) {
      // Remove all status classes
      rowElement.classList.remove('active', 'processing', 'complete', 'error');
      
      // Add appropriate class based on status
      if (status === 'processing') {
        rowElement.classList.add('processing');
      } else if (status === 'complete') {
        rowElement.classList.add('complete');
      } else if (status === 'error') {
        rowElement.classList.add('error');
      } else if (status === 'ready' || status === 'streaming') {
        rowElement.classList.add('active');
      }
    }
    
    // Update progress bar
    const progressElement = document.getElementById(`${stageKey}Progress`);
    if (progressElement) {
      progressElement.className = 'progress-bar';
      
      if (status === 'processing') {
        progressElement.classList.add('processing');
        progressElement.style.width = '75%';
      } else if (status === 'complete') {
        progressElement.classList.add('complete');
        progressElement.style.width = '100%';
      } else if (status === 'error') {
        progressElement.classList.add('error');
        progressElement.style.width = '100%';
      } else {
        progressElement.style.width = '0%';
      }
    }
    
    // Update details text
    const detailsElement = document.getElementById(`${stageKey}Details`);
    if (detailsElement) {
      const statusMessages = {
        'ready': 'Ready to process',
        'waiting': 'Waiting for previous stage',
        'processing': 'Processing...',
        'complete': 'Completed successfully',
        'error': 'Processing failed',
        'idle': 'Not started',
        'streaming': 'Receiving data...'
      };
      
      detailsElement.textContent = statusMessages[status] || status;
    }
  }

  /**
   * Update stage timing in the dashboard
   */
  updateStageData(stage, type, data) {
    const stageMap = {
      'STT': 'stt',
      'LLM': 'llm', 
      'TTS': 'tts'
    };
    
    const stageKey = stageMap[stage] || stage.toLowerCase();
    
    if (type === 'timing') {
      const timingElement = document.getElementById(`${stageKey}Timing`);
      
      if (timingElement) {
        if (data === '' || data === null || data === undefined) {
          timingElement.textContent = '-';
        } else {
          timingElement.textContent = data;
        }
      }
    } else if (type === 'data') {
      // Update details text with stage data  
      const detailsElement = document.getElementById(`${stageKey}Details`);
      if (detailsElement && data) {
        // Extract meaningful text from data for details
        let detailText = data;
        if (data.includes('Partial:')) {
          detailText = 'Receiving partial results...';
        } else if (data.includes('Final:')) {
          detailText = 'Transcription complete';
        } else if (data.includes('Response:')) {
          detailText = 'Generated response';
        } else if (data.includes('Generating speech')) {
          detailText = 'Generating speech...';
        } else if (data.includes('complete')) {
          detailText = 'Processing complete';
        }
        detailsElement.textContent = detailText;
      }
    } else if (type === 'total') {
      // Update total time in pipeline metrics
      this.updatePipelineMetrics({ totalTime: data });
    }
  }

  /**
   * Update pipeline metrics in the overview section
   */
  updatePipelineMetrics(metrics) {
    if (metrics.totalTime !== undefined) {
      const totalTimeElement = document.getElementById('totalTime');
      if (totalTimeElement) {
        totalTimeElement.textContent = metrics.totalTime || '-';
      }
    }
    
    if (metrics.sessionId !== undefined) {
      const sessionIdElement = document.getElementById('sessionId');
      if (sessionIdElement) {
        sessionIdElement.textContent = metrics.sessionId || '-';
      }
    }
    
    if (metrics.bottleneckStage !== undefined) {
      const bottleneckElement = document.getElementById('bottleneckStage');
      if (bottleneckElement) {
        bottleneckElement.textContent = metrics.bottleneckStage || '-';
      }
    }
    
    if (metrics.performanceGrade !== undefined) {
      const gradeElement = document.getElementById('gradeValue');
      if (gradeElement) {
        gradeElement.textContent = metrics.performanceGrade || '-';
      }
    }
  }

  /**
   * Initialize dashboard with default values
   */
  initializeDashboard() {
    // Set initial stage statuses
    this.updateStageStatus('Upload', 'ready');
    this.updateStageStatus('STT', 'waiting');
    this.updateStageStatus('LLM', 'waiting');
    this.updateStageStatus('TTS', 'waiting');
    
    // Set initial metrics
    this.updatePipelineMetrics({
      totalTime: '-',
      sessionId: '-',
      bottleneckStage: '-',
      performanceGrade: '-'
    });
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
      this.updateStageStatus(stage, 'waiting');
      this.updateStageData(stage, 'timing', '-');
    });
    
    // Reset upload stage
    this.updateStageStatus('Upload', 'ready');
    
    // Clear metrics
    this.updatePipelineMetrics({
      totalTime: '-',
      bottleneckStage: '-'
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

    // Auto-save to reports/logs/ if possible (via fetch to server endpoint)
    this.autoSaveDebugData(debugData);
    
    // Also download locally
    this.downloadJson(debugData, `beautyai-voice-file-debug-data-${Date.now()}.json`);
    this.logDebugEvent('EXPORT', 'info', 'Debug data exported and auto-saved');
  }

  /**
   * Attempt to auto-save debug data to server
   */
  async autoSaveDebugData(debugData) {
    try {
      const response = await fetch('/api/debug/save-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filename: 'beautyai-voice-file-debug-data.json',
          data: debugData
        })
      });
      
      if (response.ok) {
        this.logDebugEvent('EXPORT', 'info', 'Debug data auto-saved to server reports/logs/');
      } else {
        this.logDebugEvent('EXPORT', 'warning', 'Auto-save to server failed, only local download available');
      }
    } catch (error) {
      this.logDebugEvent('EXPORT', 'debug', 'Auto-save endpoint not available, using local download only');
    }
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
    
    // Update both streaming button IDs
    const streamingDisabled = !isConnected || !hasAudio || isStreaming;
    if (this.elements.sendAudioBtn) {
      this.elements.sendAudioBtn.disabled = streamingDisabled;
    }
    if (this.elements.streamBtn) {
      this.elements.streamBtn.disabled = streamingDisabled;
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