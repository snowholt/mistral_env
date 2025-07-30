/**
 * Enhanced Automatic Voice Conversation Manager
 * ============================================
 * 
 * Advanced voice conversation system with automatic speech detection,
 * hands-free operation, and seamless integration with existing WebSocket infrastructure.
 */

class EnhancedVoiceConversationManager {
    constructor(beautyAIChat, options = {}) {
        this.beautyAIChat = beautyAIChat;
        
        // Configuration
        this.config = {
            // Default settings
            defaultToSimpleMode: options.defaultToSimpleMode !== false,
            autoStart: options.autoStart !== false,
            handsFreeModeEnabled: options.handsFreeModeEnabled !== false,
            
            // Voice activity detection settings
            vadSettings: {
                energyThreshold: options.energyThreshold || 0.015,
                silenceTimeout: options.silenceTimeout || 1200,
                speechTimeout: options.speechTimeout || 300,
                minSpeechDuration: options.minSpeechDuration || 400,
                noiseGateThreshold: options.noiseGateThreshold || 0.025,
                useAdvancedVAD: options.useAdvancedVAD !== false
            },
            
            // Speech recognition settings
            asrSettings: {
                language: options.language || 'ar-SA',
                autoDetectLanguage: options.autoDetectLanguage || true,
                continuous: true,
                interimResults: true,
                confidenceThreshold: options.confidenceThreshold || 0.4,
                autoRestart: true,
                restartDelay: 800,
                maxRestartAttempts: 3
            },
            
            // UI settings
            showVisualFeedback: options.showVisualFeedback !== false,
            showTranscriptionPreview: options.showTranscriptionPreview !== false,
            autoScrollTranscript: options.autoScrollTranscript !== false
        };
        
        // State management
        this.isActive = false;
        this.isListening = false;
        this.isProcessing = false;
        this.isSpeaking = false;
        this.handsFreeModeActive = false;
        
        // Components
        this.vad = null;
        this.asr = null;
        this.audioStream = null;
        this.audioContext = null;
        
        // Session management
        this.sessionId = null;
        this.conversationHistory = [];
        this.messageCount = 0;
        this.startTime = null;
        
        // Timers and intervals
        this.conversationTimer = null;
        this.transcriptionDebounceTimer = null;
        this.visualizerUpdateInterval = null;
        
        // Cache for UI elements
        this.uiElements = {};
        
        console.log('🎙️ Enhanced Voice Conversation Manager initialized:', this.config);
    }
    
    /**
     * Initialize the enhanced voice conversation system
     */
    async initialize() {
        try {
            console.log('🚀 Initializing Enhanced Voice Conversation System...');
            
            // Cache UI elements
            this.cacheUIElements();
            
            // Initialize components
            await this.initializeComponents();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Configure default mode
            if (this.config.defaultToSimpleMode) {
                this.setDefaultToSimpleMode();
            }
            
            console.log('✅ Enhanced Voice Conversation System initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Enhanced Voice Conversation System:', error);
            throw error;
        }
    }
    
    /**
     * Cache UI elements for performance
     */
    cacheUIElements() {
        this.uiElements = {
            // Voice conversation overlay elements
            overlay: this.beautyAIChat.voiceConversationOverlay,
            statusText: this.beautyAIChat.voiceStatusText,
            statusSubtext: this.beautyAIChat.voiceStatusSubtext,
            statusIcon: this.beautyAIChat.voiceStatusIcon,
            visualizer: this.beautyAIChat.voiceVisualizer,
            
            // Control buttons
            startBtn: this.beautyAIChat.voiceStartBtn,
            stopBtn: this.beautyAIChat.voiceStopBtn,
            endBtn: this.beautyAIChat.voiceEndBtn,
            
            // Mode toggle
            modeToggle: this.beautyAIChat.voiceModeToggle,
            
            // Transcript elements
            transcript: this.beautyAIChat.voiceTranscript,
            transcriptMessages: this.beautyAIChat.voiceTranscriptMessages,
            
            // Stats
            messageCount: this.beautyAIChat.voiceMessageCount,
            conversationTime: this.beautyAIChat.voiceConversationTime,
            connectionStatus: this.beautyAIChat.voiceConnectionStatus
        };
    }
    
    /**
     * Initialize VAD and ASR components
     */
    async initializeComponents() {
        // Initialize Voice Activity Detector
        this.vad = new VoiceActivityDetector({
            ...this.config.vadSettings,
            onSpeechStart: (data) => this.handleSpeechStart(data),
            onSpeechEnd: (data) => this.handleSpeechEnd(data),
            onVoiceActivity: (data) => this.handleVoiceActivity(data),
            onError: (error) => this.handleVADError(error)
        });
        
        // Initialize Automatic Speech Recognition
        this.asr = new AutomaticSpeechRecognition({
            ...this.config.asrSettings,
            onFinalTranscript: (data) => this.handleFinalTranscript(data),
            onInterimTranscript: (data) => this.handleInterimTranscript(data),
            onStart: () => this.handleASRStart(),
            onEnd: () => this.handleASREnd(),
            onError: (error) => this.handleASRError(error),
            onLanguageDetected: (data) => this.handleLanguageDetected(data)
        });
        
        console.log('🎤 Voice Activity Detector and Speech Recognition initialized');
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Override default voice conversation button behavior
        if (this.uiElements.startBtn) {
            this.uiElements.startBtn.addEventListener('click', () => this.startHandsFreeConversation());
        }
        
        if (this.uiElements.stopBtn) {
            this.uiElements.stopBtn.addEventListener('click', () => this.stopListening());
        }
        
        if (this.uiElements.endBtn) {
            this.uiElements.endBtn.addEventListener('click', () => this.endConversation());
        }
        
        // Mode toggle handling
        if (this.uiElements.modeToggle) {
            this.uiElements.modeToggle.addEventListener('change', (e) => {
                this.handleModeToggle(e.target.checked);
            });
        }
        
        console.log('📡 Event listeners set up for enhanced voice conversation');
    }
    
    /**
     * Set default to simple mode
     */
    setDefaultToSimpleMode() {
        if (this.uiElements.modeToggle) {
            this.uiElements.modeToggle.checked = false; // Unchecked = Simple mode
            this.beautyAIChat.isSimpleVoiceMode = true;
            this.beautyAIChat.updateVoiceModeUI();
            console.log('🎯 Default mode set to Simple Voice Mode');
        }
    }
    
    /**
     * Start hands-free voice conversation
     */
    async startHandsFreeConversation() {
        try {
            if (this.isActive) {
                console.warn('Enhanced voice conversation already active');
                return;
            }
            
            console.log('🎙️ Starting hands-free voice conversation...');
            
            this.isActive = true;
            this.sessionId = this.generateSessionId();
            this.conversationHistory = [];
            this.messageCount = 0;
            this.startTime = Date.now();
            
            // Update UI
            this.updateUI({
                status: 'initializing',
                text: 'Initializing hands-free conversation...',
                subtext: 'Setting up microphone and speech recognition...',
                icon: 'fas fa-cog fa-spin'
            });
            
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                }
            });
            
            console.log('🎤 Microphone access granted');
            
            // Start Voice Activity Detection
            await this.vad.start(this.audioStream);
            
            // Start Speech Recognition
            await this.asr.start();
            
            // Start visualizer updates
            this.startVisualizer();
            
            // Start conversation timer
            this.startConversationTimer();
            
            // Update UI to ready state
            this.updateUI({
                status: 'ready',
                text: 'Ready for hands-free conversation',
                subtext: 'Start speaking - I\'ll automatically detect when you begin and end',
                icon: 'fas fa-microphone'
            });
            
            // Enable hands-free mode
            this.handsFreeModeActive = true;
            
            // Hide manual control buttons
            this.hideManualControls();
            
            console.log('✅ Hands-free voice conversation started successfully');
            
        } catch (error) {
            console.error('❌ Failed to start hands-free conversation:', error);
            this.handleError(error);
        }
    }
    
    /**
     * Handle speech start detection
     */
    handleSpeechStart(data) {
        if (!this.isActive || this.isProcessing) return;
        
        console.log('🎤 Speech started (automatic detection)');
        
        this.isSpeaking = true;
        this.isListening = true;
        
        // Update UI
        this.updateUI({
            status: 'listening',
            text: 'Listening...',
            subtext: 'Speak naturally - I\'ll detect when you finish',
            icon: 'fas fa-microphone'
        });
        
        // Update button states
        this.updateButtonStates('listening');
    }
    
    /**
     * Handle speech end detection
     */
    handleSpeechEnd(data) {
        if (!this.isActive || !this.isSpeaking) return;
        
        console.log('🔇 Speech ended (automatic detection)');
        
        this.isSpeaking = false;
        this.isListening = false;
        
        // Update UI
        this.updateUI({
            status: 'processing',
            text: 'Processing your message...',
            subtext: 'Generating response...',
            icon: 'fas fa-cog fa-spin'
        });
        
        // The final transcript will be handled by handleFinalTranscript
    }
    
    /**
     * Handle voice activity updates
     */
    handleVoiceActivity(data) {
        if (!this.isActive) return;
        
        // Update visualizer with voice activity data
        this.updateVisualizer(data);
    }
    
    /**
     * Handle final transcript from ASR
     */
    async handleFinalTranscript(data) {
        if (!this.isActive || !data.transcript) return;
        
        console.log('📝 Final transcript received:', data.transcript);
        
        try {
            this.isProcessing = true;
            
            // Clear any debounce timer
            if (this.transcriptionDebounceTimer) {
                clearTimeout(this.transcriptionDebounceTimer);
            }
            
            // Add user message to transcript
            this.addToTranscript('user', data.transcript, data);
            
            // Send to AI for processing
            await this.processUserMessage(data.transcript, data);
            
        } catch (error) {
            console.error('❌ Error processing transcript:', error);
            this.handleError(error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Handle interim transcript updates
     */
    handleInterimTranscript(data) {
        if (!this.isActive || !this.config.showTranscriptionPreview) return;
        
        // Show interim results in UI
        this.updateUI({
            subtext: `Hearing: "${data.transcript}"`
        });
    }
    
    /**
     * Process user message through AI
     */
    async processUserMessage(transcript, metadata = {}) {
        try {
            console.log('🤖 Processing user message through AI...');
            
            // Determine which method to use based on mode
            const isSimpleMode = this.beautyAIChat.isSimpleVoiceMode;
            
            if (isSimpleMode && this.beautyAIChat.simpleWsVoiceManager && this.beautyAIChat.simpleWsVoiceManager.isReady()) {
                // Use simple WebSocket for fast response
                await this.processViaSimpleWebSocket(transcript, metadata);
            } else if (!isSimpleMode && this.beautyAIChat.wsVoiceManager && this.beautyAIChat.wsVoiceManager.isReady()) {
                // Use advanced WebSocket
                await this.processViaAdvancedWebSocket(transcript, metadata);
            } else {
                // Fallback to REST API
                await this.processViaRestAPI(transcript, metadata);
            }
            
        } catch (error) {
            console.error('❌ Error processing user message:', error);
            this.handleError(error);
        }
    }
    
    /**
     * Process via Simple WebSocket (fast response)
     */
    async processViaSimpleWebSocket(transcript, metadata) {
        try {
            // Convert transcript to audio blob for WebSocket
            const audioBlob = await this.synthesizeAudioFromTranscript(transcript);
            
            // Send via simple WebSocket
            await this.beautyAIChat.simpleWsVoiceManager.sendAudioData(audioBlob);
            
            console.log('📤 Message sent via Simple WebSocket');
            
        } catch (error) {
            console.error('❌ Simple WebSocket processing failed:', error);
            // Fallback to REST API
            await this.processViaRestAPI(transcript, metadata);
        }
    }
    
    /**
     * Process via Advanced WebSocket
     */
    async processViaAdvancedWebSocket(transcript, metadata) {
        try {
            // Convert transcript to audio blob for WebSocket
            const audioBlob = await this.synthesizeAudioFromTranscript(transcript);
            
            // Send via advanced WebSocket
            await this.beautyAIChat.wsVoiceManager.sendAudioData(audioBlob);
            
            console.log('📤 Message sent via Advanced WebSocket');
            
        } catch (error) {
            console.error('❌ Advanced WebSocket processing failed:', error);
            // Fallback to REST API
            await this.processViaRestAPI(transcript, metadata);
        }
    }
    
    /**
     * Process via REST API (fallback)
     */
    async processViaRestAPI(transcript, metadata) {
        try {
            console.log('📡 Processing via REST API...');
            
            // Create a text-based request to the chat API instead of audio
            const formData = new FormData();
            formData.append('message', transcript);
            formData.append('model_name', this.beautyAIChat.currentModel);
            formData.append('session_id', this.sessionId);
            
            if (this.conversationHistory.length > 0) {
                formData.append('chat_history', JSON.stringify(this.conversationHistory));
            }
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Add AI response to transcript
                this.addToTranscript('assistant', data.final_content || data.response, {
                    tokens: data.tokens_generated,
                    time: data.generation_time_ms,
                    speed: data.tokens_per_second
                });
                
                // Update conversation history
                this.conversationHistory.push(
                    { role: 'user', content: transcript },
                    { role: 'assistant', content: data.final_content || data.response }
                );
                
                // Text-to-speech for response (if needed)
                if (this.config.enableTTS) {
                    await this.synthesizeResponse(data.final_content || data.response);
                }
                
                // Return to ready state
                this.returnToReadyState();
                
            } else {
                throw new Error(data.error || 'API request failed');
            }
            
        } catch (error) {
            console.error('❌ REST API processing failed:', error);
            throw error;
        }
    }
    
    /**
     * Synthesize audio from transcript (for WebSocket compatibility)
     */
    async synthesizeAudioFromTranscript(transcript) {
        // This is a simplified version - in practice, you might want to 
        // record actual audio or use Web Speech API synthesis
        const blob = new Blob([transcript], { type: 'text/plain' });
        return blob;
    }
    
    /**
     * Add message to transcript
     */
    addToTranscript(role, content, metadata = {}) {
        if (!this.uiElements.transcriptMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `voice-transcript-message ${role}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const confidenceText = metadata.confidence ? ` (${Math.round(metadata.confidence * 100)}%)` : '';
        
        messageDiv.innerHTML = `
            <div class="transcript-header">
                <span class="transcript-role">${role === 'user' ? 'You' : 'AI'}</span>
                <span class="transcript-time">${timestamp}${confidenceText}</span>
            </div>
            <div class="transcript-content">${content}</div>
        `;
        
        this.uiElements.transcriptMessages.appendChild(messageDiv);
        
        // Auto-scroll if enabled
        if (this.config.autoScrollTranscript) {
            messageDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Update message count
        this.messageCount++;
        this.updateStats();
    }
    
    /**
     * Return to ready state after processing
     */
    returnToReadyState() {
        this.updateUI({
            status: 'ready',
            text: 'Ready for next message',
            subtext: 'Continue speaking whenever you\'re ready',
            icon: 'fas fa-microphone'
        });
        
        this.updateButtonStates('ready');
    }
    
    /**
     * Update UI elements
     */
    updateUI(updates) {
        if (updates.text && this.uiElements.statusText) {
            this.uiElements.statusText.textContent = updates.text;
        }
        
        if (updates.subtext && this.uiElements.statusSubtext) {
            this.uiElements.statusSubtext.textContent = updates.subtext;
        }
        
        if (updates.icon && this.uiElements.statusIcon) {
            this.uiElements.statusIcon.className = updates.icon;
        }
        
        if (updates.status && this.uiElements.overlay) {
            this.uiElements.overlay.setAttribute('data-status', updates.status);
        }
    }
    
    /**
     * Update button states
     */
    updateButtonStates(state) {
        const buttons = [this.uiElements.startBtn, this.uiElements.stopBtn, this.uiElements.endBtn];
        
        buttons.forEach(btn => {
            if (btn) {
                btn.setAttribute('data-state', state);
            }
        });
    }
    
    /**
     * Hide manual control buttons in hands-free mode
     */
    hideManualControls() {
        if (this.uiElements.startBtn) {
            this.uiElements.startBtn.style.display = 'none';
        }
        
        if (this.uiElements.stopBtn) {
            this.uiElements.stopBtn.style.display = 'none';
        }
        
        // Show only end conversation button
        if (this.uiElements.endBtn) {
            this.uiElements.endBtn.style.display = 'block';
            this.uiElements.endBtn.textContent = 'End Conversation';
        }
    }
    
    /**
     * Start conversation timer
     */
    startConversationTimer() {
        if (this.conversationTimer) {
            clearInterval(this.conversationTimer);
        }
        
        this.conversationTimer = setInterval(() => {
            this.updateStats();
        }, 1000);
    }
    
    /**
     * Update conversation statistics
     */
    updateStats() {
        if (this.uiElements.messageCount) {
            this.uiElements.messageCount.textContent = this.messageCount;
        }
        
        if (this.uiElements.conversationTime && this.startTime) {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            this.uiElements.conversationTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        if (this.uiElements.connectionStatus) {
            const status = this.isActive ? 'Connected' : 'Disconnected';
            this.uiElements.connectionStatus.textContent = status;
        }
    }
    
    /**
     * Start visualizer updates
     */
    startVisualizer() {
        if (!this.config.showVisualFeedback) return;
        
        if (this.visualizerUpdateInterval) {
            clearInterval(this.visualizerUpdateInterval);
        }
        
        this.visualizerUpdateInterval = setInterval(() => {
            if (this.vad && this.vad.isActive) {
                const levels = this.vad.getAudioLevels();
                this.updateVisualizer({ energy: levels.energy, peak: levels.peak });
            }
        }, 50); // Update 20 times per second
    }
    
    /**
     * Update audio visualizer
     */
    updateVisualizer(data) {
        if (!this.uiElements.visualizer || !this.config.showVisualFeedback) return;
        
        const bars = this.uiElements.visualizer.querySelectorAll('.voice-visualizer-bar');
        const energy = data.energy || 0;
        const peak = data.peak || 0;
        
        bars.forEach((bar, index) => {
            const height = Math.min(100, (energy + peak) * 100 * (1 + index * 0.1));
            bar.style.height = `${height}%`;
            
            // Add activity class when speech is detected
            if (data.detected && this.isSpeaking) {
                bar.classList.add('active');
            } else {
                bar.classList.remove('active');
            }
        });
    }
    
    /**
     * Handle mode toggle
     */
    handleModeToggle(isAdvancedMode) {
        this.beautyAIChat.isSimpleVoiceMode = !isAdvancedMode;
        this.beautyAIChat.updateVoiceModeUI();
        
        console.log(`🔄 Voice mode switched to: ${isAdvancedMode ? 'Advanced' : 'Simple'}`);
        
        // Update UI text
        const modeText = isAdvancedMode ? 'Advanced Voice Mode' : 'Simple Voice Mode';
        this.updateUI({
            subtext: `${modeText} - Continue speaking whenever you're ready`
        });
    }
    
    /**
     * Stop listening (temporary pause)
     */
    stopListening() {
        console.log('⏸️ Stopping listening (temporary pause)');
        
        this.isListening = false;
        this.isSpeaking = false;
        
        if (this.vad) {
            this.vad.stop();
        }
        
        if (this.asr) {
            this.asr.stop();
        }
        
        this.updateUI({
            status: 'paused',
            text: 'Conversation paused',
            subtext: 'Click "Start" to resume listening',
            icon: 'fas fa-pause'
        });
        
        // Show manual controls again
        this.showManualControls();
    }
    
    /**
     * Show manual controls
     */
    showManualControls() {
        if (this.uiElements.startBtn) {
            this.uiElements.startBtn.style.display = 'block';
            this.uiElements.startBtn.textContent = 'Resume';
        }
        
        if (this.uiElements.stopBtn) {
            this.uiElements.stopBtn.style.display = 'block';
        }
    }
    
    /**
     * End conversation completely
     */
    async endConversation() {
        try {
            console.log('🛑 Ending enhanced voice conversation');
            
            this.isActive = false;
            this.isListening = false;
            this.isSpeaking = false;
            this.isProcessing = false;
            this.handsFreeModeActive = false;
            
            // Stop all components
            if (this.vad) {
                this.vad.stop();
            }
            
            if (this.asr) {
                this.asr.stop();
            }
            
            // Stop audio stream
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
                this.audioStream = null;
            }
            
            // Clear timers
            if (this.conversationTimer) {
                clearInterval(this.conversationTimer);
                this.conversationTimer = null;
            }
            
            if (this.visualizerUpdateInterval) {
                clearInterval(this.visualizerUpdateInterval);
                this.visualizerUpdateInterval = null;
            }
            
            if (this.transcriptionDebounceTimer) {
                clearTimeout(this.transcriptionDebounceTimer);
                this.transcriptionDebounceTimer = null;
            }
            
            // Reset UI
            this.resetUI();
            
            // Call original end conversation method
            if (this.beautyAIChat.endVoiceConversation) {
                this.beautyAIChat.endVoiceConversation();
            }
            
            console.log('✅ Enhanced voice conversation ended');
            
        } catch (error) {
            console.error('❌ Error ending conversation:', error);
        }
    }
    
    /**
     * Reset UI to initial state
     */
    resetUI() {
        this.updateUI({
            status: 'idle',
            text: 'Voice conversation ended',
            subtext: 'Click "Start Voice Conversation" to begin again',
            icon: 'fas fa-microphone-slash'
        });
        
        // Clear transcript
        if (this.uiElements.transcriptMessages) {
            this.uiElements.transcriptMessages.innerHTML = '';
        }
        
        // Reset stats
        this.messageCount = 0;
        this.startTime = null;
        this.updateStats();
        
        // Show all controls again
        this.showManualControls();
    }
    
    /**
     * Handle errors
     */
    handleError(error) {
        console.error('🚨 Enhanced voice conversation error:', error);
        
        this.updateUI({
            status: 'error',
            text: 'Error occurred',
            subtext: error.message || 'An unexpected error occurred',
            icon: 'fas fa-exclamation-triangle'
        });
        
        // Try to recover after a delay
        setTimeout(() => {
            if (this.isActive) {
                this.returnToReadyState();
            }
        }, 3000);
    }
    
    /**
     * Handle VAD errors
     */
    handleVADError(error) {
        console.error('🎤 VAD Error:', error);
        // Handle VAD-specific errors
    }
    
    /**
     * Handle ASR errors
     */
    handleASRError(error) {
        console.error('🗣️ ASR Error:', error);
        // Handle ASR-specific errors
    }
    
    /**
     * Handle ASR start
     */
    handleASRStart() {
        console.log('🎙️ Speech recognition started');
    }
    
    /**
     * Handle ASR end
     */
    handleASREnd() {
        console.log('🔇 Speech recognition ended');
    }
    
    /**
     * Handle language detection
     */
    handleLanguageDetected(data) {
        console.log('🌍 Language detected:', data.language);
        
        // Update UI to show detected language
        this.updateUI({
            subtext: `Language detected: ${data.language} - Continue speaking`
        });
    }
    
    /**
     * Generate session ID
     */
    generateSessionId() {
        return 'enhanced_voice_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        // Update component configurations
        if (this.vad) {
            this.vad.updateSensitivity(newConfig.vadSettings || {});
        }
        
        if (this.asr) {
            this.asr.updateConfig(newConfig.asrSettings || {});
        }
        
        console.log('🎛️ Enhanced voice conversation configuration updated');
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            isActive: this.isActive,
            isListening: this.isListening,
            isSpeaking: this.isSpeaking,
            isProcessing: this.isProcessing,
            handsFreeModeActive: this.handsFreeModeActive,
            sessionId: this.sessionId,
            messageCount: this.messageCount,
            conversationDuration: this.startTime ? Date.now() - this.startTime : 0,
            vadStatus: this.vad ? this.vad.getStatus() : null,
            asrStatus: this.asr ? this.asr.getStatus() : null
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedVoiceConversationManager;
} else {
    window.EnhancedVoiceConversationManager = EnhancedVoiceConversationManager;
}
