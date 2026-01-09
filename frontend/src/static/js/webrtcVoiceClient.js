/**
 * WebRTC Voice Client (Phase D - WebRTC Migration)
 * Browser-based WebRTC voice-to-voice communication client
 * 
 * Features:
 * - RTCPeerConnection management with SDP/ICE negotiation
 * - Local audio track handling with WebRTC audio enhancements
 * - Remote audio stream playback
 * - 10-second client-side utterance limit enforcement
 * - VAD integration for audio-level monitoring
 * - Graceful fallback to WebSocket mode
 * 
 * Created: October 15, 2025
 */

import WEBRTC_CONFIG, { isWebRTCSupported, getVoiceMode } from '../config/webrtc.config.js';

class WebRTCVoiceClient {
    constructor(options = {}) {
        // Configuration
        this.config = {
            containerId: options.containerId || 'voice-container',
            signalingUrl: options.signalingUrl || WEBRTC_CONFIG.signalingUrl,
            sessionId: options.sessionId || this.generateSessionId(),
            language: options.language || 'ar', // Default to Arabic (BeautyAI specialization)

            // WebRTC settings from config
            maxUtteranceSec: WEBRTC_CONFIG.maxUtteranceSec,
            iceServers: WEBRTC_CONFIG.iceServers,
            rtcConfiguration: {
                ...WEBRTC_CONFIG.rtcConfiguration,
                iceServers: WEBRTC_CONFIG.iceServers
            },
            audioConstraints: WEBRTC_CONFIG.audioConstraints,

            // VAD settings
            vadEnabled: WEBRTC_CONFIG.vadEnabled,
            vadAudioLevelThreshold: WEBRTC_CONFIG.vadAudioLevelThreshold,

            // Debug settings
            debug: WEBRTC_CONFIG.debug,
            logIceEvents: WEBRTC_CONFIG.logIceEvents,
            logSignaling: WEBRTC_CONFIG.logSignaling,

            // Callbacks
            onConnected: options.onConnected || (() => { }),
            onDisconnected: options.onDisconnected || (() => { }),
            onError: options.onError || ((error) => console.error('[WebRTC] Error:', error)),
            onTranscript: options.onTranscript || ((text) => console.log('[WebRTC] Transcript:', text)),
            onAudioReceived: options.onAudioReceived || (() => { }),
            onUtteranceLimitExceeded: options.onUtteranceLimitExceeded || (() => console.warn('[WebRTC] Utterance limit exceeded'))
        };

        // State
        this.state = {
            initialized: false,
            connected: false,
            recording: false,
            speaking: false,
            currentLanguage: this.config.language,
            connectionState: 'new',
            iceConnectionState: 'new',
            iceGatheringState: 'new'
        };

        // ICE Candidate Queue (Fix for race condition)
        this.iceCandidateQueue = [];

        // WebRTC components
        this.peerConnection = null;
        this.peerId = null;
        this.localStream = null;
        this.remoteStream = null;
        this.dataChannel = null;

        // Audio components
        this.localAudioTrack = null;
        this.remoteAudioElement = null;
        this.audioContext = null;
        this.audioAnalyser = null;

        // Utterance timer
        this.utteranceTimer = null;
        this.utteranceStartTime = null;
        this.utteranceDuration = 0;

        // VAD integration
        this.vad = null;
        this.audioLevel = 0;

        // Metrics
        this.metrics = {
            connectionEstablished: null,
            firstAudioReceived: null,
            totalPacketsSent: 0,
            totalPacketsReceived: 0,
            totalBytesSent: 0,
            totalBytesReceived: 0,
            averageLatency: 0
        };

        // DOM elements
        this.container = null;
        this.elements = {};

        console.log('[WebRTC] WebRTCVoiceClient initialized with config:', {
            signalingUrl: this.config.signalingUrl,
            sessionId: this.config.sessionId,
            maxUtteranceSec: this.config.maxUtteranceSec,
            language: this.config.language
        });
    }

    /**
     * Initialize the WebRTC voice client
     */
    async initialize() {
        try {
            console.log('[WebRTC] 🚀 Initializing WebRTC Voice Client...');

            // Check WebRTC support
            if (!isWebRTCSupported()) {
                throw new Error('WebRTC is not supported in this browser. Please use a modern browser (Chrome 90+, Firefox 88+, Safari 15+).');
            }
            console.log('[WebRTC] ✅ WebRTC support confirmed');

            // Get DOM references
            this.container = document.getElementById(this.config.containerId);
            if (!this.container) {
                throw new Error(`Container element not found: ${this.config.containerId}`);
            }
            console.log('[WebRTC] ✅ Container element found');

            // Bind DOM elements
            this.bindDOMElements();
            console.log('[WebRTC] ✅ DOM elements bound');

            // Setup event listeners
            this.setupEventListeners();
            console.log('[WebRTC] ✅ Event listeners set up');

            // Initialize audio context for analysis
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioAnalyser = this.audioContext.createAnalyser();
            this.audioAnalyser.fftSize = 256;
            console.log('[WebRTC] ✅ Audio context initialized');

            // Create hidden audio element for remote playback
            this.createRemoteAudioElement();
            console.log('[WebRTC] ✅ Remote audio element created');

            // Initialize VAD if enabled
            if (this.config.vadEnabled) {
                await this.initializeVAD();
                console.log('[WebRTC] ✅ VAD initialized');
            }

            this.state.initialized = true;
            console.log('[WebRTC] 🎉 WebRTC Voice Client initialized successfully!');

            return true;

        } catch (error) {
            console.error('[WebRTC] 💥 Failed to initialize WebRTC Voice Client:', error);
            this.config.onError(error);
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

            // WebRTC-specific controls
            webrtcConnectBtn: document.getElementById('webrtc-connect-btn'),
            webrtcDisconnectBtn: document.getElementById('webrtc-disconnect-btn'),

            // Voice controls
            micButton: document.getElementById('mic-button'),
            recordingIndicator: document.getElementById('recording-indicator'),

            // Processing status
            processingStatus: document.getElementById('processing-status'),

            // Conversation
            conversationMessages: document.getElementById('conversation-messages'),

            // Controls
            languageToggle: document.getElementById('language-toggle'),
            autoSpeak: document.getElementById('auto-speak'),
            volumeSlider: document.getElementById('volume-slider'),

            // Error display
            errorDisplay: document.getElementById('error-display'),
            errorMessage: document.getElementById('error-message'),

            // Metrics
            metricsPanel: document.getElementById('metrics-panel'),
            metricLatency: document.getElementById('metric-latency'),
            metricConnectionState: document.getElementById('metric-connection-state'),
            metricIceState: document.getElementById('metric-ice-state')
        };

        console.log('[WebRTC] DOM elements bound');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Microphone button
        if (this.elements.micButton) {
            this.elements.micButton.addEventListener('click', () => this.toggleRecording());
        }

        // Language toggle
        if (this.elements.languageToggle) {
            this.elements.languageToggle.addEventListener('change', (e) => {
                this.setLanguage(e.target.checked ? 'ar' : 'en');
            });
        }

        // Volume control
        if (this.elements.volumeSlider && this.remoteAudioElement) {
            this.elements.volumeSlider.addEventListener('input', (e) => {
                const volume = parseInt(e.target.value) / 100;
                this.remoteAudioElement.volume = volume;
            });
        }

        // WebRTC-specific connection controls
        if (this.elements.webrtcConnectBtn) {
            this.elements.webrtcConnectBtn.addEventListener('click', () => this.connect());
        }
        if (this.elements.webrtcDisconnectBtn) {
            this.elements.webrtcDisconnectBtn.addEventListener('click', () => this.disconnect());
        }

        console.log('[WebRTC] Event listeners configured');
    }

    /**
     * Create hidden audio element for remote audio playback
     */
    createRemoteAudioElement() {
        this.remoteAudioElement = document.createElement('audio');
        this.remoteAudioElement.id = 'webrtc-remote-audio';
        this.remoteAudioElement.autoplay = true;
        this.remoteAudioElement.volume = this.elements.volumeSlider ?
            parseInt(this.elements.volumeSlider.value) / 100 : 0.8;

        // Hide but keep in DOM for autoplay to work
        this.remoteAudioElement.style.display = 'none';
        document.body.appendChild(this.remoteAudioElement);

        console.log('[WebRTC] Remote audio element created');
    }

    /**
     * Initialize VAD for client-side audio monitoring
     */
    async initializeVAD() {
        try {
            // Check if ImprovedVAD is available
            if (typeof ImprovedVAD === 'undefined') {
                console.warn('[WebRTC] ImprovedVAD not available, skipping VAD initialization');
                return;
            }

            this.vad = new ImprovedVAD({
                onVolumeChange: (level) => {
                    this.audioLevel = level;
                    this.updateVADVisuals(level);
                },
                onStateChange: (state) => {
                    console.log('[WebRTC] VAD state changed:', state);
                    // Optionally send VAD state via data channel for server diagnostics
                    if (this.dataChannel && this.dataChannel.readyState === 'open') {
                        this.sendDataChannelMessage({
                            type: 'vad_state',
                            state: state,
                            timestamp: Date.now()
                        });
                    }
                }
            });

            console.log('[WebRTC] VAD integration configured');

        } catch (error) {
            console.warn('[WebRTC] Failed to initialize VAD:', error);
            // Continue without VAD - not critical
        }
    }

    /**
     * Connect to signaling server and establish WebRTC connection
     */
    async connect() {
        try {
            console.log('[WebRTC] 🔗 Connecting to signaling server...');
            this.updateConnectionStatus('connecting', 'Connecting...');

            // Request microphone access with WebRTC enhancements
            console.log('[WebRTC] 🎤 Requesting microphone access with constraints:', this.config.audioConstraints);
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: this.config.audioConstraints,
                video: false
            });
            console.log('[WebRTC] ✅ Microphone access granted');

            // Get local audio track
            this.localAudioTrack = this.localStream.getAudioTracks()[0];
            console.log('[WebRTC] Audio track obtained:', this.localAudioTrack.label);

            // Create RTCPeerConnection
            console.log('[WebRTC] 🌐 Creating RTCPeerConnection...');
            this.createPeerConnection();

            // Add local track to peer connection
            this.localStream.getTracks().forEach(track => {
                this.peerConnection.addTrack(track, this.localStream);
                console.log('[WebRTC] Added local track:', track.kind);
            });

            // Create data channel for optional diagnostics
            this.createDataChannel();

            // Create SDP offer
            console.log('[WebRTC] 📝 Creating SDP offer...');
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: false
            });

            // Set local description
            await this.peerConnection.setLocalDescription(offer);
            console.log('[WebRTC] ✅ Local description set');

            if (this.config.logSignaling) {
                console.log('[WebRTC] SDP Offer:', offer.sdp);
            }

            // Send offer to signaling server
            console.log('[WebRTC] 📤 Sending SDP offer to server...');
            const response = await this.sendSignalingMessage('/offer', {
                sdp: offer.sdp,
                type: offer.type,
                language: this.state.currentLanguage,
                session_metadata: {
                    user_agent: navigator.userAgent,
                    client_version: '1.0.0-phase-d'
                }
            });

            console.log('[WebRTC] ✅ Received SDP answer from server');
            this.peerId = response.peer_id || response.session_id || response.sessionId;
            if (!this.peerId) {
                throw new Error('Signaling response missing peer_id/session_id');
            }
            console.log('[WebRTC] Peer ID:', this.peerId);

            // Process queued ICE candidates
            if (this.iceCandidateQueue && this.iceCandidateQueue.length > 0) {
                console.log(`[WebRTC] 🧊 Processing ${this.iceCandidateQueue.length} queued ICE candidates`);
                for (const candidateParams of this.iceCandidateQueue) {
                    try {
                        // Ensure peer_id is set in the queued params
                        candidateParams.peer_id = this.peerId;
                        candidateParams.session_id = this.peerId;

                        await this.sendSignalingMessage('/ice', candidateParams);
                    } catch (error) {
                        console.error('[WebRTC] Failed to send queued ICE candidate:', error);
                    }
                }
                this.iceCandidateQueue = []; // Clear queue
            }

            if (this.config.logSignaling) {
                console.log('[WebRTC] SDP Answer:', response.sdp);
            }

            // Set remote description
            await this.peerConnection.setRemoteDescription({
                type: 'answer',
                sdp: response.sdp
            });
            console.log('[WebRTC] ✅ Remote description set');

            this.state.connected = true;
            this.metrics.connectionEstablished = Date.now();
            this.updateConnectionStatus('connected', 'Connected via WebRTC');
            this.config.onConnected();

            console.log('[WebRTC] 🎉 WebRTC connection established successfully!');

        } catch (error) {
            console.error('[WebRTC] 💥 Connection failed:', error);
            this.updateConnectionStatus('error', 'Connection failed');
            this.config.onError(error);
            throw error;
        }
    }

    /**
     * Disconnect WebRTC connection
     */
    async disconnect() {
        try {
            console.log('[WebRTC] 🔌 Disconnecting...');

            // Stop utterance timer if running
            this.stopUtteranceTimer();

            // Stop local stream
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => {
                    track.stop();
                    console.log('[WebRTC] Stopped local track:', track.kind);
                });
                this.localStream = null;
            }

            // Close data channel
            if (this.dataChannel) {
                this.dataChannel.close();
                this.dataChannel = null;
            }

            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                console.log('[WebRTC] Peer connection closed');
            }

            // Notify server of cleanup
            if (this.peerId) {
                try {
                    await this.sendSignalingMessage(`/${this.peerId}`, null, 'DELETE');
                    console.log('[WebRTC] Server cleanup requested');
                } catch (error) {
                    console.warn('[WebRTC] Server cleanup failed:', error);
                }
            }

            // Reset state
            this.state.connected = false;
            this.state.recording = false;
            this.peerConnection = null;
            this.peerId = null;

            this.updateConnectionStatus('disconnected', 'Disconnected');
            this.config.onDisconnected();

            console.log('[WebRTC] ✅ Disconnected successfully');

        } catch (error) {
            console.error('[WebRTC] Error during disconnect:', error);
        }
    }

    /**
     * Create RTCPeerConnection with event handlers
     */
    createPeerConnection() {
        this.peerConnection = new RTCPeerConnection(this.config.rtcConfiguration);

        // Connection state change
        this.peerConnection.onconnectionstatechange = () => {
            this.state.connectionState = this.peerConnection.connectionState;
            console.log('[WebRTC] Connection state:', this.state.connectionState);
            this.updateMetrics();

            if (this.state.connectionState === 'failed' || this.state.connectionState === 'disconnected') {
                this.config.onError(new Error(`Connection ${this.state.connectionState}`));
            }
        };

        // ICE connection state change
        this.peerConnection.oniceconnectionstatechange = () => {
            this.state.iceConnectionState = this.peerConnection.iceConnectionState;
            console.log('[WebRTC] ICE connection state:', this.state.iceConnectionState);

            if (this.config.logIceEvents) {
                console.log('[WebRTC] ICE state details:', {
                    iceConnectionState: this.peerConnection.iceConnectionState,
                    iceGatheringState: this.peerConnection.iceGatheringState
                });
            }
        };

        // ICE gathering state change
        this.peerConnection.onicegatheringstatechange = () => {
            this.state.iceGatheringState = this.peerConnection.iceGatheringState;
            console.log('[WebRTC] ICE gathering state:', this.state.iceGatheringState);
        };

        // ICE candidate
        this.peerConnection.onicecandidate = async (event) => {
            if (event.candidate) {
                if (this.config.logIceEvents) {
                    console.log('[WebRTC] New ICE candidate:', event.candidate.candidate);
                }

                const iceParams = {
                    peer_id: this.peerId,
                    session_id: this.peerId,
                    candidate: event.candidate.candidate,
                    sdp_mid: event.candidate.sdpMid,
                    sdp_m_line_index: event.candidate.sdpMLineIndex
                };

                // Check if we have a peer ID yet
                if (!this.peerId) {
                    console.log('[WebRTC] 🧊 Queuing ICE candidate (no peer_id yet)');
                    this.iceCandidateQueue.push(iceParams);
                    return;
                }

                // Send ICE candidate to signaling server
                try {
                    await this.sendSignalingMessage('/ice', iceParams);

                    if (this.config.logIceEvents) {
                        console.log('[WebRTC] ✅ ICE candidate sent to server');
                    }
                } catch (error) {
                    console.error('[WebRTC] Failed to send ICE candidate:', error);
                }
            } else {
                console.log('[WebRTC] ICE gathering complete');
            }
        };

        // Track received (remote audio)
        this.peerConnection.ontrack = (event) => {
            console.log('[WebRTC] 🎵 Remote track received:', event.track.kind);

            if (event.track.kind === 'audio') {
                this.remoteStream = event.streams[0];
                this.remoteAudioElement.srcObject = this.remoteStream;

                if (!this.metrics.firstAudioReceived) {
                    this.metrics.firstAudioReceived = Date.now();
                    const latency = this.metrics.firstAudioReceived - this.metrics.connectionEstablished;
                    console.log('[WebRTC] First audio received, latency:', latency, 'ms');
                }

                this.config.onAudioReceived();

                // Track ended handler
                event.track.onended = () => {
                    console.log('[WebRTC] Remote track ended');
                    this.state.speaking = false;
                };
            }
        };

        console.log('[WebRTC] RTCPeerConnection created with ICE servers:', this.config.iceServers);
    }

    /**
     * Create data channel for optional diagnostics
     */
    createDataChannel() {
        try {
            this.dataChannel = this.peerConnection.createDataChannel('diagnostics', {
                ordered: true
            });

            this.dataChannel.onopen = () => {
                console.log('[WebRTC] Data channel opened');
            };

            this.dataChannel.onclose = () => {
                console.log('[WebRTC] Data channel closed');
            };

            this.dataChannel.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('[WebRTC] Data channel message:', message);

                    // Handle server-side VAD state or other diagnostics
                    if (message.type === 'vad_state') {
                        console.log('[WebRTC] Server VAD state:', message.state);
                    }
                } catch (error) {
                    console.warn('[WebRTC] Failed to parse data channel message:', error);
                }
            };

            console.log('[WebRTC] Data channel created');

        } catch (error) {
            console.warn('[WebRTC] Failed to create data channel:', error);
            // Continue without data channel - not critical
        }
    }

    /**
     * Send message via data channel
     */
    sendDataChannelMessage(message) {
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            try {
                this.dataChannel.send(JSON.stringify(message));
            } catch (error) {
                console.warn('[WebRTC] Failed to send data channel message:', error);
            }
        }
    }

    /**
     * Toggle recording
     */
    async toggleRecording() {
        if (!this.state.connected) {
            console.warn('[WebRTC] Cannot record: not connected');
            this.showError('Please connect first');
            return;
        }

        if (this.state.recording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    /**
     * Start recording
     */
    async startRecording() {
        try {
            console.log('[WebRTC] 🎙️ Starting recording...');

            this.state.recording = true;
            this.updateRecordingIndicator(true);

            // Start utterance timer
            this.startUtteranceTimer();

            // Enable local audio track
            if (this.localAudioTrack) {
                this.localAudioTrack.enabled = true;
            }

            // Start VAD if available
            if (this.vad && typeof this.vad.startListening === 'function') {
                this.vad.startListening();
            }

            console.log('[WebRTC] ✅ Recording started');

        } catch (error) {
            console.error('[WebRTC] Failed to start recording:', error);
            this.state.recording = false;
            this.updateRecordingIndicator(false);
            this.config.onError(error);
        }
    }

    /**
     * Stop recording
     */
    async stopRecording() {
        try {
            console.log('[WebRTC] 🛑 Stopping recording...');

            this.state.recording = false;
            this.updateRecordingIndicator(false);

            // Stop utterance timer
            this.stopUtteranceTimer();

            // Disable local audio track (keeps connection open, just mutes mic)
            if (this.localAudioTrack) {
                this.localAudioTrack.enabled = false;
            }

            // Stop VAD if available
            if (this.vad && typeof this.vad.stopListening === 'function') {
                this.vad.stopListening();
            }

            console.log('[WebRTC] ✅ Recording stopped');

        } catch (error) {
            console.error('[WebRTC] Failed to stop recording:', error);
            this.config.onError(error);
        }
    }

    /**
     * Start utterance timer (10-second limit enforcement)
     */
    startUtteranceTimer() {
        this.stopUtteranceTimer(); // Clear any existing timer

        this.utteranceStartTime = Date.now();
        this.utteranceDuration = 0;

        // Update duration every 100ms
        this.utteranceTimer = setInterval(() => {
            this.utteranceDuration = (Date.now() - this.utteranceStartTime) / 1000;

            // Check if limit exceeded
            if (this.utteranceDuration >= this.config.maxUtteranceSec) {
                console.warn('[WebRTC] ⏱️ Utterance limit exceeded:', this.utteranceDuration, 'seconds');
                this.config.onUtteranceLimitExceeded();
                this.stopRecording(); // Automatically stop recording
            }
        }, 100);

        console.log('[WebRTC] Utterance timer started, limit:', this.config.maxUtteranceSec, 'seconds');
    }

    /**
     * Stop utterance timer
     */
    stopUtteranceTimer() {
        if (this.utteranceTimer) {
            clearInterval(this.utteranceTimer);
            this.utteranceTimer = null;
            console.log('[WebRTC] Utterance timer stopped, duration:', this.utteranceDuration, 'seconds');
        }
    }

    /**
     * Set language
     */
    setLanguage(language) {
        this.state.currentLanguage = language;
        console.log('[WebRTC] Language set to:', language);

        // Send language update via data channel if available
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            this.sendDataChannelMessage({
                type: 'language_change',
                language: language,
                timestamp: Date.now()
            });
        }
    }

    /**
     * Send signaling message to server
     */
    async sendSignalingMessage(endpoint, data, method = 'POST') {
        const url = `${this.config.signalingUrl}${endpoint}`;

        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (data && method !== 'DELETE') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Signaling error: ${response.status} ${errorText}`);
        }

        if (method === 'DELETE') {
            return null;
        }

        return await response.json();
    }

    /**
     * Update connection status UI
     */
    updateConnectionStatus(state, text) {
        if (this.elements.statusDot) {
            this.elements.statusDot.className = `status-dot ${state}`;
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = text;
        }

        console.log('[WebRTC] Status:', state, '-', text);
    }

    /**
     * Update recording indicator UI
     */
    updateRecordingIndicator(recording) {
        if (this.elements.micButton) {
            this.elements.micButton.classList.toggle('recording', recording);
        }
        if (this.elements.recordingIndicator) {
            this.elements.recordingIndicator.style.display = recording ? 'block' : 'none';
        }
    }

    /**
     * Update VAD visuals
     */
    updateVADVisuals(level) {
        // Update audio level display if available
        if (this.elements.audioLevel) {
            const percentage = Math.min(100, Math.max(0, (level + 100) * 2)); // Convert dBFS to percentage
            this.elements.audioLevel.style.width = `${percentage}%`;
        }
    }

    /**
     * Update metrics display
     */
    updateMetrics() {
        if (this.elements.metricConnectionState) {
            this.elements.metricConnectionState.textContent = this.state.connectionState;
        }
        if (this.elements.metricIceState) {
            this.elements.metricIceState.textContent = this.state.iceConnectionState;
        }

        // Calculate average latency if available
        if (this.metrics.firstAudioReceived && this.metrics.connectionEstablished) {
            const latency = this.metrics.firstAudioReceived - this.metrics.connectionEstablished;
            if (this.elements.metricLatency) {
                this.elements.metricLatency.textContent = `${latency}ms`;
            }
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        if (this.elements.errorDisplay && this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
            this.elements.errorDisplay.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.hideError();
            }, 5000);
        }

        console.error('[WebRTC] Error:', message);
    }

    /**
     * Hide error message
     */
    hideError() {
        if (this.elements.errorDisplay) {
            this.elements.errorDisplay.style.display = 'none';
        }
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return `webrtc_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get connection statistics
     */
    async getConnectionStats() {
        if (!this.peerConnection) {
            return null;
        }

        try {
            const stats = await this.peerConnection.getStats();
            const result = {
                connection: {
                    state: this.state.connectionState,
                    iceState: this.state.iceConnectionState,
                    gatheringState: this.state.iceGatheringState
                },
                audio: {
                    packetsSent: 0,
                    packetsReceived: 0,
                    bytesSent: 0,
                    bytesReceived: 0,
                    packetsLost: 0,
                    jitter: 0
                }
            };

            stats.forEach(stat => {
                if (stat.type === 'outbound-rtp' && stat.kind === 'audio') {
                    result.audio.packetsSent = stat.packetsSent || 0;
                    result.audio.bytesSent = stat.bytesSent || 0;
                } else if (stat.type === 'inbound-rtp' && stat.kind === 'audio') {
                    result.audio.packetsReceived = stat.packetsReceived || 0;
                    result.audio.bytesReceived = stat.bytesReceived || 0;
                    result.audio.packetsLost = stat.packetsLost || 0;
                    result.audio.jitter = stat.jitter || 0;
                }
            });

            return result;

        } catch (error) {
            console.error('[WebRTC] Failed to get connection stats:', error);
            return null;
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        console.log('[WebRTC] 🧹 Cleaning up resources...');

        this.disconnect();

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.remoteAudioElement) {
            this.remoteAudioElement.remove();
            this.remoteAudioElement = null;
        }

        if (this.vad) {
            this.vad = null;
        }

        console.log('[WebRTC] ✅ Cleanup complete');
    }
}

// Export for use in other modules
export default WebRTCVoiceClient;

// Also expose globally for non-module usage
if (typeof window !== 'undefined') {
    window.WebRTCVoiceClient = WebRTCVoiceClient;
}
