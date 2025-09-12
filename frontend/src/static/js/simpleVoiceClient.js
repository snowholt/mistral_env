/**
 * Simple Voice Client
 * Manages WebSocket connection, audio recording with MediaRecorder API, 
 * VAD integration, mic disable during TTS, and audio playback queue management.
 */

class SimpleVoiceClient {
    constructor(options = {}) {
        // Configuration
        this.config = {
            containerId: options.containerId || 'voice-container',
            websocketUrl: options.websocketUrl,
            sessionId: options.sessionId,
            
            // Audio settings
            sampleRate: 16000,
            channelCount: 1,
            mimeType: 'audio/webm;codecs=opus',
            
            // VAD settings
            vadConfig: {
                englishThreshold: 0.5,
                arabicThreshold: 0.45,
                minSpeechDuration: 300,
                maxSilenceDuration: 1500,
                adaptiveThreshold: true
            },
            
            // Connection settings
            reconnectAttempts: 3,
            reconnectDelay: 1000,
            connectionTimeout: 5000,
            
            // Performance settings
            maxRecordingDuration: 30000, // 30 seconds
            audioChunkSize: 1024
        };
        
        // State
        this.state = {
            connected: false,
            recording: false,
            processing: false,
            speaking: false,
            currentLanguage: 'english',
            autoSpeak: true,
            volume: 0.8
        };
        
        // WebSocket connection
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.connectionTimeout = null;
        
        // Audio components
        this.mediaRecorder = null;
        this.audioStream = null;
        this.recordedChunks = [];
        this.vad = null;
        this.audioQueue = [];
        this.currentAudio = null;
        
        // DOM elements
        this.container = null;
        this.elements = {};
        
        // Performance tracking
        this.metrics = {
            connectionTime: 0,
            lastResponseTime: 0,
            totalRequests: 0,
            successfulRequests: 0,
            averageLatency: 0
        };
        
        // Conversation history
        this.conversation = [];
        
        console.log('SimpleVoiceClient initialized with config:', this.config);
    }
    
    /**
     * Initialize the voice client
     */
    async initialize() {
        try {
            console.log('Initializing SimpleVoiceClient...');
            
            // Get DOM references
            this.container = document.getElementById(this.config.containerId);
            if (!this.container) {
                throw new Error(`Container element not found: ${this.config.containerId}`);
            }
            
            this.bindDOMElements();
            this.setupEventListeners();
            
            // Initialize VAD
            await this.initializeVAD();
            
            // Connect to WebSocket
            await this.connect();
            
            console.log('SimpleVoiceClient initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize SimpleVoiceClient:', error);
            this.showError('Failed to initialize voice client: ' + error.message);
            throw error;
        }
    }
    
    /**
     * Bind DOM elements
     */
    bindDOMElements() {
        this.elements = {
            // Connection status
            connectionStatus: document.getElementById('connection-status'),
            statusDot: this.container.querySelector('.status-dot'),
            statusText: this.container.querySelector('.status-text'),
            
            // Voice controls
            micButton: document.getElementById('mic-button'),
            recordingIndicator: document.getElementById('recording-indicator'),
            vadStatus: document.getElementById('vad-status'),
            vadState: document.getElementById('vad-state'),
            audioLevel: document.getElementById('audio-level'),
            
            // Processing status
            processingStatus: document.getElementById('processing-status'),
            stepListening: document.getElementById('step-listening'),
            stepTranscribing: document.getElementById('step-transcribing'),
            stepThinking: document.getElementById('step-thinking'),
            stepSpeaking: document.getElementById('step-speaking'),
            
            // Waveform
            waveformCanvas: document.getElementById('waveform-canvas'),
            
            // Conversation
            conversationMessages: document.getElementById('conversation-messages'),
            clearConversation: document.getElementById('clear-conversation'),
            exportConversation: document.getElementById('export-conversation'),
            
            // Controls
            languageToggle: document.getElementById('language-toggle'),
            autoSpeak: document.getElementById('auto-speak'),
            volumeSlider: document.getElementById('volume-slider'),
            volumeValue: document.getElementById('volume-value'),
            vadSensitivity: document.getElementById('vad-sensitivity'),
            vadSensitivityValue: document.getElementById('vad-sensitivity-value'),
            
            // Theme
            themeBtn: document.getElementById('theme-btn'),
            
            // Error display
            errorDisplay: document.getElementById('error-display'),
            errorMessage: document.getElementById('error-message'),
            errorClose: document.getElementById('error-close'),
            
            // TTS audio
            ttsAudio: document.getElementById('tts-audio'),
            
            // Metrics
            metricsPanel: document.getElementById('metrics-panel'),
            metricLatency: document.getElementById('metric-latency'),
            metricVadAccuracy: document.getElementById('metric-vad-accuracy'),
            metricAudioQuality: document.getElementById('metric-audio-quality'),
            metricResponseTime: document.getElementById('metric-response-time')
        };
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Microphone button
        this.elements.micButton.addEventListener('click', () => this.toggleRecording());
        
        // Language toggle
        this.elements.languageToggle.addEventListener('change', (e) => {
            this.setLanguage(e.target.checked ? 'arabic' : 'english');
        });
        
        // Auto-speak toggle
        this.elements.autoSpeak.addEventListener('change', (e) => {
            this.state.autoSpeak = e.target.checked;
        });
        
        // Volume control
        this.elements.volumeSlider.addEventListener('input', (e) => {
            const volume = parseInt(e.target.value) / 100;
            this.state.volume = volume;
            this.elements.volumeValue.textContent = e.target.value;
            if (this.elements.ttsAudio) {
                this.elements.ttsAudio.volume = volume;
            }
        });
        
        // VAD sensitivity
        this.elements.vadSensitivity.addEventListener('input', (e) => {
            const sensitivity = parseFloat(e.target.value);
            this.elements.vadSensitivityValue.textContent = sensitivity;
            if (this.vad) {
                this.vad.setSensitivity(sensitivity);
            }
        });
        
        // Theme toggle
        this.elements.themeBtn.addEventListener('click', () => this.toggleTheme());
        
        // Conversation controls
        this.elements.clearConversation.addEventListener('click', () => this.clearConversation());
        this.elements.exportConversation.addEventListener('click', () => this.exportConversation());
        
        // Error close
        this.elements.errorClose.addEventListener('click', () => this.hideError());
        
        // TTS audio events
        this.elements.ttsAudio.addEventListener('loadstart', () => this.onTTSStart());
        this.elements.ttsAudio.addEventListener('ended', () => this.onTTSEnd());
        this.elements.ttsAudio.addEventListener('error', (e) => this.onTTSError(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                this.startRecording();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                this.stopRecording();
            }
        });
        
        // Page visibility
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.state.recording) {
                this.stopRecording();
            }
        });
        
        // Before unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }
    
    /**
     * Initialize VAD system
     */
    async initializeVAD() {
        try {
            this.vad = new ImprovedVAD({
                ...this.config.vadConfig,
                onSpeechStart: () => this.onVADSpeechStart(),
                onSpeechEnd: () => this.onVADSpeechEnd(),
                onVolumeChange: (energy, threshold) => this.onVADVolumeChange(energy, threshold),
                onStateChange: (state) => this.onVADStateChange(state)
            });
            
            await this.vad.initialize();
            
            console.log('VAD initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize VAD:', error);
            throw new Error('Microphone access required for voice functionality');
        }
    }
    
    /**
     * Connect to WebSocket
     */
    async connect() {
        return new Promise((resolve, reject) => {
            console.log(`Connecting to WebSocket: ${this.config.websocketUrl}`);
            
            this.updateConnectionStatus('connecting', 'Connecting...');
            
            try {
                this.websocket = new WebSocket(this.config.websocketUrl);
                this.websocket.binaryType = 'arraybuffer';
                
                // Connection timeout
                this.connectionTimeout = setTimeout(() => {
                    if (this.websocket.readyState === WebSocket.CONNECTING) {
                        this.websocket.close();
                        reject(new Error('Connection timeout'));
                    }
                }, this.config.connectionTimeout);
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    clearTimeout(this.connectionTimeout);
                    this.state.connected = true;
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected', 'Connected');
                    this.enableControls();
                    resolve();
                };
                
                this.websocket.onmessage = (event) => this.handleWebSocketMessage(event);
                
                this.websocket.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    this.state.connected = false;
                    this.updateConnectionStatus('disconnected', 'Disconnected');
                    this.disableControls();
                    
                    if (!event.wasClean && this.reconnectAttempts < this.config.reconnectAttempts) {
                        this.attemptReconnect();
                    }
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    clearTimeout(this.connectionTimeout);
                    
                    if (this.websocket.readyState === WebSocket.CONNECTING) {
                        reject(new Error('Failed to connect to voice service'));
                    }
                };
                
            } catch (error) {
                clearTimeout(this.connectionTimeout);
                reject(error);
            }
        });
    }
    
    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            console.log('Received message:', message.type);
            
            this.metrics.totalRequests++;
            
            switch (message.type) {
                case 'transcription':
                    this.handleTranscription(message);
                    break;
                    
                case 'response':
                    this.handleResponse(message);
                    break;
                    
                case 'audio_response':
                    this.handleAudioResponse(message);
                    break;
                    
                case 'error':
                    this.handleError(message);
                    break;
                    
                case 'status':
                    this.handleStatus(message);
                    break;
                    
                default:
                    console.warn('Unknown message type:', message.type);
            }
            
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }
    
    /**
     * Start recording
     */
    async startRecording() {
        if (this.state.recording || this.state.speaking || !this.state.connected) {
            return;
        }
        
        try {
            console.log('Starting recording...');
            
            this.state.recording = true;
            this.recordedChunks = [];
            
            // Get audio stream
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channelCount,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false
                }
            });
            
            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: this.config.mimeType
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => this.processRecording();
            
            // Start recording and VAD
            this.mediaRecorder.start();
            this.vad.startListening();
            
            this.updateUIState('recording');
            this.startWaveformVisualization();
            
            // Auto-stop after max duration
            setTimeout(() => {
                if (this.state.recording) {
                    this.stopRecording();
                }
            }, this.config.maxRecordingDuration);
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showError('Failed to start recording: ' + error.message);
            this.state.recording = false;
        }
    }
    
    /**
     * Stop recording
     */
    stopRecording() {
        if (!this.state.recording) return;
        
        console.log('Stopping recording...');
        
        this.state.recording = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.vad) {
            this.vad.stopListening();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
        
        this.updateUIState('processing');
        this.stopWaveformVisualization();
    }
    
    /**
     * Toggle recording
     */
    toggleRecording() {
        if (this.state.recording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }
    
    /**
     * Process recorded audio
     */
    async processRecording() {
        if (this.recordedChunks.length === 0) {
            console.warn('No audio recorded');
            this.updateUIState('idle');
            return;
        }
        
        try {
            console.log('Processing recorded audio...');
            
            // Create blob from recorded chunks
            const audioBlob = new Blob(this.recordedChunks, { type: this.config.mimeType });
            
            // Convert to PCM for backend
            const pcmData = await this.convertToPCM(audioBlob);
            
            // Send to backend
            this.sendAudioData(pcmData);
            
        } catch (error) {
            console.error('Failed to process recording:', error);
            this.showError('Failed to process audio: ' + error.message);
            this.updateUIState('idle');
        }
    }
    
    /**
     * Convert audio blob to PCM
     */
    async convertToPCM(audioBlob) {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Get mono channel data
        const channelData = audioBuffer.getChannelData(0);
        
        // Convert to Int16 PCM
        const pcmData = AudioUtils.floatToInt16Buffer(channelData);
        
        await audioContext.close();
        return pcmData;
    }
    
    /**
     * Send audio data to backend
     */
    sendAudioData(pcmData) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            this.showError('WebSocket connection not available');
            this.updateUIState('idle');
            return;
        }
        
        const message = {
            type: 'audio_input',
            audio_data: Array.from(new Uint8Array(pcmData)),
            language: this.state.currentLanguage,
            session_id: this.config.sessionId,
            timestamp: Date.now()
        };
        
        console.log('Sending audio data to backend...');
        this.websocket.send(JSON.stringify(message));
        
        this.metrics.lastResponseTime = Date.now();
    }
    
    /**
     * Handle transcription response
     */
    handleTranscription(message) {
        console.log('Received transcription:', message.text);
        
        this.addConversationMessage('user', message.text, message.language);
        this.updateProcessingStep('thinking');
    }
    
    /**
     * Handle text response
     */
    handleResponse(message) {
        console.log('Received response:', message.text);
        
        this.addConversationMessage('assistant', message.text, message.language);
        this.updateProcessingStep('speaking');
        
        // Calculate response time
        const responseTime = Date.now() - this.metrics.lastResponseTime;
        this.metrics.averageLatency = (this.metrics.averageLatency + responseTime) / 2;
        this.updateMetrics();
        
        this.metrics.successfulRequests++;
    }
    
    /**
     * Handle audio response
     */
    handleAudioResponse(message) {
        if (!this.state.autoSpeak) {
            this.updateUIState('idle');
            return;
        }
        
        try {
            // Decode base64 audio
            const audioData = atob(message.audio_data);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            
            // Create blob and play
            const audioBlob = new Blob([audioArray], { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            this.playAudio(audioUrl);
            
        } catch (error) {
            console.error('Failed to process audio response:', error);
            this.updateUIState('idle');
        }
    }
    
    /**
     * Play audio response
     */
    playAudio(audioUrl) {
        this.state.speaking = true;
        this.elements.ttsAudio.src = audioUrl;
        this.elements.ttsAudio.volume = this.state.volume;
        this.elements.ttsAudio.play();
    }
    
    /**
     * Handle error messages
     */
    handleError(message) {
        console.error('Backend error:', message.error);
        this.showError(message.error);
        this.updateUIState('idle');
    }
    
    /**
     * Handle status messages
     */
    handleStatus(message) {
        console.log('Status update:', message.status);
        
        if (message.step) {
            this.updateProcessingStep(message.step);
        }
    }
    
    /**
     * VAD event handlers
     */
    onVADSpeechStart() {
        console.log('VAD: Speech detected');
        if (this.state.recording) {
            this.elements.vadState.textContent = 'Speaking';
            this.elements.recordingIndicator.classList.add('speaking');
        }
    }
    
    onVADSpeechEnd() {
        console.log('VAD: Speech ended');
        if (this.state.recording) {
            this.elements.vadState.textContent = 'Processing';
            this.elements.recordingIndicator.classList.remove('speaking');
            
            // Auto-stop recording after speech ends
            setTimeout(() => {
                if (this.state.recording) {
                    this.stopRecording();
                }
            }, 500);
        }
    }
    
    onVADVolumeChange(energy, threshold) {
        // Update audio level visualization
        const level = Math.min(energy / threshold, 1);
        this.elements.audioLevel.style.width = `${level * 100}%`;
        
        // Update waveform
        this.updateWaveform(energy);
    }
    
    onVADStateChange(state) {
        this.elements.vadState.textContent = state;
    }
    
    /**
     * TTS event handlers
     */
    onTTSStart() {
        console.log('TTS playback started');
        this.state.speaking = true;
        this.updateUIState('speaking');
    }
    
    onTTSEnd() {
        console.log('TTS playback ended');
        this.state.speaking = false;
        this.updateUIState('idle');
        
        // Enable microphone again
        this.elements.micButton.disabled = false;
    }
    
    onTTSError(error) {
        console.error('TTS playback error:', error);
        this.state.speaking = false;
        this.updateUIState('idle');
        this.showError('Audio playback failed');
    }
    
    /**
     * UI state management
     */
    updateUIState(state) {
        console.log(`UI state: ${state}`);
        
        // Remove all state classes
        this.container.classList.remove('recording', 'processing', 'speaking', 'idle');
        
        // Add current state class
        this.container.classList.add(state);
        
        // Update button state
        switch (state) {
            case 'recording':
                this.elements.micButton.innerHTML = '<i class="fas fa-stop"></i>';
                this.elements.micButton.disabled = false;
                break;
                
            case 'processing':
                this.elements.micButton.innerHTML = '<i class="fas fa-cog fa-spin"></i>';
                this.elements.micButton.disabled = true;
                break;
                
            case 'speaking':
                this.elements.micButton.innerHTML = '<i class="fas fa-volume-high"></i>';
                this.elements.micButton.disabled = true;
                break;
                
            default: // idle
                this.elements.micButton.innerHTML = '<i class="fas fa-microphone"></i>';
                this.elements.micButton.disabled = false;
        }
        
        // Update processing steps
        this.updateProcessingStep(state);
    }
    
    /**
     * Update processing step indicator
     */
    updateProcessingStep(step) {
        // Clear all active steps
        const steps = ['listening', 'transcribing', 'thinking', 'speaking'];
        steps.forEach(s => {
            const element = this.elements[`step${s.charAt(0).toUpperCase() + s.slice(1)}`];
            if (element) {
                element.classList.remove('active', 'completed');
            }
        });
        
        // Set active step
        const stepElement = this.elements[`step${step.charAt(0).toUpperCase() + step.slice(1)}`];
        if (stepElement) {
            stepElement.classList.add('active');
        }
        
        // Mark completed steps
        const stepOrder = ['listening', 'transcribing', 'thinking', 'speaking'];
        const currentIndex = stepOrder.indexOf(step);
        for (let i = 0; i < currentIndex; i++) {
            const element = this.elements[`step${stepOrder[i].charAt(0).toUpperCase() + stepOrder[i].slice(1)}`];
            if (element) {
                element.classList.add('completed');
            }
        }
    }
    
    /**
     * Connection status management
     */
    updateConnectionStatus(status, text) {
        this.elements.statusDot.className = `status-dot ${status}`;
        this.elements.statusText.textContent = text;
    }
    
    /**
     * Enable/disable controls
     */
    enableControls() {
        this.elements.micButton.disabled = false;
        this.elements.languageToggle.disabled = false;
        this.elements.vadSensitivity.disabled = false;
    }
    
    disableControls() {
        this.elements.micButton.disabled = true;
        this.elements.languageToggle.disabled = true;
        this.elements.vadSensitivity.disabled = true;
    }
    
    /**
     * Language management
     */
    setLanguage(language) {
        console.log(`Setting language to: ${language}`);
        this.state.currentLanguage = language;
        
        if (this.vad) {
            this.vad.setLanguage(language);
        }
        
        // Update UI language
        this.updateUILanguage(language);
    }
    
    updateUILanguage(language) {
        const isArabic = language === 'arabic';
        
        // Toggle welcome message
        const welcomeEn = this.container.querySelector('.welcome-en');
        const welcomeAr = this.container.querySelector('.welcome-ar');
        
        if (welcomeEn && welcomeAr) {
            welcomeEn.style.display = isArabic ? 'none' : 'inline';
            welcomeAr.style.display = isArabic ? 'inline' : 'none';
        }
        
        // Update document direction for Arabic
        document.documentElement.setAttribute('dir', isArabic ? 'rtl' : 'ltr');
        document.documentElement.setAttribute('lang', isArabic ? 'ar' : 'en');
    }
    
    /**
     * Conversation management
     */
    addConversationMessage(role, text, language) {
        const message = {
            role,
            text,
            language,
            timestamp: new Date().toISOString()
        };
        
        this.conversation.push(message);
        
        // Create message element
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${text}</p>
                <span class="message-time">${timeStr}</span>
            </div>
        `;
        
        // Add to conversation
        this.elements.conversationMessages.appendChild(messageElement);
        
        // Scroll to bottom
        this.elements.conversationMessages.scrollTop = this.elements.conversationMessages.scrollHeight;
        
        // Limit conversation length
        while (this.conversation.length > 50) {
            this.conversation.shift();
            const firstMessage = this.elements.conversationMessages.querySelector('.message');
            if (firstMessage && !firstMessage.classList.contains('welcome-message')) {
                firstMessage.remove();
            }
        }
    }
    
    clearConversation() {
        this.conversation = [];
        
        // Keep welcome message, remove others
        const messages = this.elements.conversationMessages.querySelectorAll('.message:not(.welcome-message)');
        messages.forEach(msg => msg.remove());
    }
    
    exportConversation() {
        const data = {
            sessionId: this.config.sessionId,
            timestamp: new Date().toISOString(),
            conversation: this.conversation,
            metrics: this.metrics
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation_${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    /**
     * Waveform visualization
     */
    startWaveformVisualization() {
        if (!this.elements.waveformCanvas) return;
        
        this.waveformData = new Array(200).fill(0);
        this.visualizing = true;
        this.drawWaveform();
    }
    
    stopWaveformVisualization() {
        this.visualizing = false;
    }
    
    updateWaveform(energy) {
        if (!this.waveformData) return;
        
        this.waveformData.push(energy);
        if (this.waveformData.length > 200) {
            this.waveformData.shift();
        }
    }
    
    drawWaveform() {
        if (!this.visualizing || !this.elements.waveformCanvas) return;
        
        const canvas = this.elements.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        if (!this.waveformData || this.waveformData.length === 0) {
            requestAnimationFrame(() => this.drawWaveform());
            return;
        }
        
        // Draw waveform
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary-color') || '#007bff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const stepX = width / this.waveformData.length;
        
        for (let i = 0; i < this.waveformData.length; i++) {
            const x = i * stepX;
            const y = height / 2 - (this.waveformData[i] * height / 4);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        requestAnimationFrame(() => this.drawWaveform());
    }
    
    /**
     * Theme management
     */
    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        
        const icon = this.elements.themeBtn.querySelector('i');
        icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        
        // Store preference
        localStorage.setItem('theme', newTheme);
    }
    
    /**
     * Error management
     */
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorDisplay.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }
    
    hideError() {
        this.elements.errorDisplay.style.display = 'none';
    }
    
    /**
     * Metrics management
     */
    updateMetrics() {
        if (!this.elements.metricsPanel) return;
        
        this.elements.metricLatency.textContent = `${Math.round(this.metrics.averageLatency)}ms`;
        this.elements.metricResponseTime.textContent = `${Math.round(this.metrics.averageLatency / 1000)}s`;
        
        const successRate = this.metrics.totalRequests > 0 
            ? (this.metrics.successfulRequests / this.metrics.totalRequests * 100)
            : 0;
        this.elements.metricVadAccuracy.textContent = `${Math.round(successRate)}%`;
    }
    
    /**
     * Reconnection logic
     */
    async attemptReconnect() {
        this.reconnectAttempts++;
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.config.reconnectAttempts}...`);
        
        this.updateConnectionStatus('connecting', `Reconnecting... (${this.reconnectAttempts})`);
        
        await new Promise(resolve => setTimeout(resolve, this.config.reconnectDelay));
        
        try {
            await this.connect();
        } catch (error) {
            console.error('Reconnection failed:', error);
            
            if (this.reconnectAttempts >= this.config.reconnectAttempts) {
                this.updateConnectionStatus('error', 'Connection failed');
                this.showError('Unable to connect to voice service. Please refresh the page.');
            }
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        console.log('Cleaning up SimpleVoiceClient...');
        
        // Stop recording
        if (this.state.recording) {
            this.stopRecording();
        }
        
        // Stop audio
        if (this.elements.ttsAudio) {
            this.elements.ttsAudio.pause();
        }
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
        }
        
        // Cleanup VAD
        if (this.vad) {
            this.vad.cleanup();
        }
        
        // Stop visualization
        this.visualizing = false;
        
        console.log('Cleanup completed');
    }
}

// Export for global use
window.SimpleVoiceClient = SimpleVoiceClient;