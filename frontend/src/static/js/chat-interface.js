/**
 * BeautyAI Modern Chat Interface
 * Supports both text and voice chat with advanced settings
 */

class BeautyAIChat {
    constructor() {
        // State
        this.currentMode = 'text';
        this.isConnected = false;
        this.isRecording = false;
        this.isProcessing = false;
        this.conversationStarted = false;
        
        // WebSocket for voice chat
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioContext = null;
        this.analyser = null;
        this.silenceTimeout = null;
        this.silenceDetectionActive = false;
        
        // Auto detection settings
        this.autoStartEnabled = false;
        this.autoStopEnabled = false;
        this.silenceThreshold = 3000;
        
        // Real-time VAD settings (server-side VAD enabled by default)
        this.vadEnabled = true;  // Enable VAD by default - server handles VAD processing
        this.streamingMode = true;  // Use streaming mode for real-time interaction
        this.chunkSize = 100;  // Audio chunk size for streaming (100ms chunks for real-time)
        this.vadFeedback = true;  // Show real-time VAD feedback from server
        this.useServerSideVAD = true;  // Use server-side VAD instead of client-side silence detection
        
        // Audio feedback prevention
        this.isPlayingAIAudio = false;  // Track when AI is speaking to prevent feedback
        this.audioFeedbackTimeout = null;  // Timeout for feedback prevention
        
        // VAD state management to prevent infinite loops
        this.vadState = {
            lastSpeechEnd: 0,  // Timestamp of last speech_end event
            lastVoiceResponse: 0,  // Timestamp of last voice_response event
            cooldownPeriod: 2000,  // 2 second cooldown before allowing new speech detection
            isInCooldown: false,  // Flag to prevent immediate re-detection
            consecutiveSpeechEnds: 0,  // Count consecutive speech_end events
            maxConsecutiveEnds: 3  // Max consecutive ends before forcing longer cooldown
        };
        
        // Response deduplication to prevent audio playback loops
        this.responseState = {
            lastResponseId: null,  // Track last response ID to prevent duplicates
            processedResponses: new Set(),  // Set of processed response IDs
            currentlyPlaying: false,  // Flag to prevent overlapping audio playback
            audioQueue: [],  // Queue for managing multiple audio responses
            maxProcessedResponses: 50  // Limit set size to prevent memory leaks
        };
        
        // Overlay state
        this.overlayConnected = false;
        this.overlayRecording = false;
        this.overlayWebSocket = null;
        this.overlayMediaRecorder = null;
        this.overlayAudioChunks = [];
        this.overlayAnalyser = null;
        this.overlaySilenceTimeout = null;
        this.overlaySilenceDetectionActive = false;
        this.overlayAutoStartEnabled = false;
        this.overlayAutoStopEnabled = false;
        this.overlaySilenceThresholdValue = 3000;
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateVoiceOptions();
        this.updateOverlayVoiceOptions();
        this.setupAudioContext();
        this.resetVADState(); // Initialize VAD state
    }

    resetVADState() {
        /**
         * Reset VAD state to prevent infinite loops and manage speech detection cooldowns.
         */
        this.vadState = {
            lastSpeechEnd: 0,
            lastVoiceResponse: 0,
            cooldownPeriod: 2000,
            isInCooldown: false,
            consecutiveSpeechEnds: 0,
            maxConsecutiveEnds: 3
        };
        
        // Reset response state to prevent audio playback loops
        this.responseState = {
            lastResponseId: null,
            processedResponses: new Set(),
            currentlyPlaying: false,
            audioQueue: [],
            maxProcessedResponses: 50
        };
        
        console.log('VAD state and response deduplication reset for clean conversation start');
    }

    isDuplicateResponse(data) {
        /**
         * Check if this voice response is a duplicate to prevent audio playback loops.
         */
        if (!data || !data.success || !data.audio_base64) {
            return false; // Not a valid audio response
        }
        
        // Create unique identifier for this response (using audio data + timestamp)
        const responseId = data.audio_base64.substring(0, 50) + '_' + (data.timestamp || Date.now());
        
        // Check if we've already processed this response
        if (this.responseState.processedResponses.has(responseId) || this.responseState.lastResponseId === responseId) {
            console.warn('🚫 Duplicate voice response detected - preventing playback loop:', responseId.substring(0, 20) + '...');
            return true;
        }
        
        // Add to processed responses
        this.responseState.processedResponses.add(responseId);
        this.responseState.lastResponseId = responseId;
        
        // Limit set size to prevent memory leaks
        if (this.responseState.processedResponses.size > this.responseState.maxProcessedResponses) {
            const firstItem = this.responseState.processedResponses.values().next().value;
            this.responseState.processedResponses.delete(firstItem);
        }
        
        console.log('✅ New unique voice response accepted:', responseId.substring(0, 20) + '...');
        return false;
    }
    
    async playAudioResponse(audioBlob, isOverlay = false) {
        /**
         * Play audio response with queue management to prevent overlapping playback.
         */
        if (this.responseState.currentlyPlaying) {
            console.log('🎵 Audio already playing - queuing response');
            this.responseState.audioQueue.push({ audioBlob, isOverlay });
            return;
        }
        
        this.responseState.currentlyPlaying = true;
        this.isPlayingAIAudio = true;
        
        try {
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            // Set up event handlers
            audio.onended = () => {
                console.log('🔊 Audio playback completed');
                URL.revokeObjectURL(audioUrl);
                this.isPlayingAIAudio = false;
                this.responseState.currentlyPlaying = false;
                
                // Process next item in queue
                if (this.responseState.audioQueue.length > 0) {
                    const nextItem = this.responseState.audioQueue.shift();
                    setTimeout(() => this.playAudioResponse(nextItem.audioBlob, nextItem.isOverlay), 100);
                } else {
                    // Auto-restart logic here (if enabled)
                    this.handleAudioPlaybackEnd(isOverlay);
                }
            };
            
            audio.onerror = () => {
                console.error('❌ Error playing AI audio');
                URL.revokeObjectURL(audioUrl);
                this.isPlayingAIAudio = false;
                this.responseState.currentlyPlaying = false;
                
                // Process next item in queue even on error
                if (this.responseState.audioQueue.length > 0) {
                    const nextItem = this.responseState.audioQueue.shift();
                    setTimeout(() => this.playAudioResponse(nextItem.audioBlob, nextItem.isOverlay), 100);
                }
            };
            
            // Update status and play
            if (isOverlay) {
                this.showOverlayVoiceStatus('🔊 Playing AI response...');
            } else {
                this.showVoiceStatus('🔊 Playing AI response...');
            }
            
            console.log('🎵 Starting audio playback...');
            await audio.play();
            
        } catch (error) {
            console.error('❌ Failed to play audio response:', error);
            this.isPlayingAIAudio = false;
            this.responseState.currentlyPlaying = false;
        }
    }
    
    handleAudioPlaybackEnd(isOverlay = false) {
        /**
         * Handle auto-restart logic after audio playback ends.
         * FIXED: Better state management with controlled auto-restart
         */
        this.isPlayingAIAudio = false;
        const autoStartEnabled = isOverlay ? this.overlayAutoStartEnabled : this.autoStartEnabled;
        const isConnected = isOverlay ? this.overlayConnected : this.isConnected;
        const isRecording = isOverlay ? this.overlayRecording : this.isRecording;
        
        // IMPORTANT: Update UI to show ready state but implement controlled auto-restart
        if (autoStartEnabled && isConnected && !isRecording && this.useServerSideVAD) {
            console.log('✅ AI finished speaking. Preparing for auto-restart with proper delay...');
            
            if (isOverlay) {
                this.updateOverlayConnectionStatus('connected', 'AI finished. Ready for your next input...');
                this.overlayVoiceToggle.classList.remove('recording');
                this.overlayVoiceToggle.querySelector('span').textContent = 'Auto-restart pending...';
            } else {
                this.updateConnectionStatus('connected', 'AI finished. Ready for your next input...');
                this.voiceToggle.classList.remove('recording');
                this.voiceToggle.textContent = '🎤';
            }
            
            // Controlled auto-restart with extended delay and comprehensive checks
            setTimeout(() => {
                const now = Date.now();
                const currentIsRecording = isOverlay ? this.overlayRecording : this.isRecording;
                const minRestartDelay = 3000; // Extended to 3 seconds for better stability
                
                const canRestart = !currentIsRecording && 
                                 !this.isPlayingAIAudio && 
                                 !this.vadState.isInCooldown &&
                                 (now - this.vadState.lastVoiceResponse) > minRestartDelay &&
                                 (now - this.vadState.lastSpeechEnd) > minRestartDelay;
                
                if (canRestart) {
                    console.log('🔄 Auto-restarting recording after safe delay and state validation');
                    if (isOverlay) {
                        this.startOverlayRecording();
                    } else {
                        this.startRecording();
                    }
                } else {
                    console.log('🚫 Auto-restart blocked by state management - user must manually start next turn');
                    // Update UI to indicate manual start required
                    if (isOverlay) {
                        this.overlayVoiceToggle.querySelector('span').textContent = 'Click to Start';
                        this.updateOverlayConnectionStatus('connected', 'Ready. Click mic to start next turn.');
                    } else {
                        this.updateConnectionStatus('connected', 'Ready. Click mic to start next turn.');
                    }
                }
            }, 3000); // Extended delay for better stability
        } else {
            // Manual mode or no auto-restart - update UI to indicate ready state
            console.log('✅ AI finished speaking. Manual restart required.');
            if (isOverlay) {
                this.updateOverlayConnectionStatus('connected', 'Ready. Click mic to speak.');
                this.overlayVoiceToggle.classList.remove('recording');
                this.overlayVoiceToggle.querySelector('span').textContent = 'Click to Start';
            } else {
                this.updateConnectionStatus('connected', 'Ready. Click mic to speak.');
                this.voiceToggle.classList.remove('recording');
                this.voiceToggle.textContent = '🎤';
            }
        }
        
        // Reset VAD state to clean slate for next conversation turn
        this.resetVADState();
        
        // Release any orphaned audio resources
        this.muteAllAudioElements(false);
    }

    initializeElements() {
        // Mode buttons
        this.textModeBtn = document.getElementById('textModeBtn');
        this.voiceModeBtn = document.getElementById('voiceModeBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        
        // Main containers
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.settingsPanel = document.getElementById('settingsPanel');
        
        // Input elements
        this.messageInput = document.getElementById('messageInput');
        this.voiceToggle = document.getElementById('voiceToggle');
        this.startConversationBtn = document.getElementById('voiceConversationBtn');
        this.sendBtn = document.getElementById('sendBtn');
        this.voiceStatus = document.getElementById('voiceStatus');
        
        // Status
        this.connectionStatus = document.getElementById('connectionStatus');
        
        // Settings elements
        this.modelSelect = document.getElementById('modelSelect');
        this.thinkingMode = document.getElementById('thinkingMode');
        this.temperature = document.getElementById('temperature');
        this.topP = document.getElementById('topP');
        this.topK = document.getElementById('topK');
        this.maxTokens = document.getElementById('maxTokens');
        this.repetitionPenalty = document.getElementById('repetitionPenalty');
        this.disableFilter = document.getElementById('disableFilter');
        this.filterStrictness = document.getElementById('filterStrictness');
        this.language = document.getElementById('language');
        this.voice = document.getElementById('voice');
        this.autoStart = document.getElementById('autoStart');
        this.autoStop = document.getElementById('autoStop');
        this.silenceThresholdSlider = document.getElementById('silenceThreshold');
        
        // Value displays
        this.tempValue = document.getElementById('tempValue');
        this.topPValue = document.getElementById('topPValue');
        this.topKValue = document.getElementById('topKValue');
        this.maxTokensValue = document.getElementById('maxTokensValue');
        this.repPenaltyValue = document.getElementById('repPenaltyValue');
        this.silenceValueDisplay = document.getElementById('silenceValue');
        
        // Voice Overlay elements
        this.floatingVoiceBtn = document.getElementById('floatingVoiceBtn');
        this.voiceOverlay = document.getElementById('voiceOverlay');
        this.closeVoiceOverlay = document.getElementById('closeVoiceOverlay');
        this.overlayLanguage = document.getElementById('overlayLanguage');
        this.overlayVoice = document.getElementById('overlayVoice');
        this.overlayThinkingMode = document.getElementById('overlayThinkingMode');
        this.overlayAutoStart = document.getElementById('overlayAutoStart');
        this.overlayAutoStop = document.getElementById('overlayAutoStop');
        this.overlaySilenceThreshold = document.getElementById('overlaySilenceThreshold');
        this.overlaySilenceValue = document.getElementById('overlaySilenceValue');
        this.overlayConnectionStatus = document.getElementById('overlayConnectionStatus');
        this.overlayVoiceToggle = document.getElementById('overlayVoiceToggle');
        this.overlayVoiceStatus = document.getElementById('overlayVoiceStatus');
        this.overlayConversation = document.getElementById('overlayConversation');
    }

    setupEventListeners() {
        // Mode switching (if elements exist)
        if (this.textModeBtn) {
            this.textModeBtn.addEventListener('click', () => this.switchMode('text'));
        }
        if (this.voiceModeBtn) {
            this.voiceModeBtn.addEventListener('click', () => this.switchMode('voice'));
        }
        
        // Settings panel
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        
        // Text input
        this.messageInput.addEventListener('input', () => this.autoResize());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.handleSend());
        
        // Voice conversation button
        if (this.startConversationBtn) {
            this.startConversationBtn.addEventListener('click', () => this.openVoiceOverlay());
        }
        
        // Voice controls (legacy)
        if (this.voiceToggle) {
            this.voiceToggle.addEventListener('mousedown', () => this.handleVoiceToggle(true));
            this.voiceToggle.addEventListener('mouseup', () => this.handleVoiceToggle(false));
            this.voiceToggle.addEventListener('mouseleave', () => this.handleVoiceToggle(false));
        }
        
        // Settings controls
        this.setupSettingsListeners();
        
        // Language change
        this.language.addEventListener('change', () => this.updateVoiceOptions());
        
        // Auto detection settings
        this.autoStart.addEventListener('change', (e) => {
            this.autoStartEnabled = e.target.checked;
        });
        
        this.autoStop.addEventListener('change', (e) => {
            this.autoStopEnabled = e.target.checked;
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.settingsPanel.classList.remove('open');
            }
            // Space for voice recording in voice mode
            if (e.code === 'Space' && this.currentMode === 'voice' && !e.repeat) {
                if (document.activeElement !== this.messageInput) {
                    e.preventDefault();
                    this.handleVoiceToggle(true);
                }
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && this.currentMode === 'voice') {
                if (document.activeElement !== this.messageInput) {
                    e.preventDefault();
                    this.handleVoiceToggle(false);
                }
            }
        });
        
        // Voice Overlay event listeners
        this.setupOverlayListeners();
    }

    setupOverlayListeners() {
        // Floating button to open overlay (if it exists)
        if (this.floatingVoiceBtn) {
            this.floatingVoiceBtn.addEventListener('click', () => this.openVoiceOverlay());
        }
        
        // Close overlay
        if (this.closeVoiceOverlay) {
            this.closeVoiceOverlay.addEventListener('click', () => this.closeVoiceOverlayMethod());
        }
        
        // Close overlay on background click
        if (this.voiceOverlay) {
            this.voiceOverlay.addEventListener('click', (e) => {
                if (e.target === this.voiceOverlay) {
                    this.closeVoiceOverlayMethod();
                }
            });
        }
        
        // Overlay voice toggle
        if (this.overlayVoiceToggle) {
            this.overlayVoiceToggle.addEventListener('mousedown', () => this.handleOverlayVoiceToggle(true));
            this.overlayVoiceToggle.addEventListener('mouseup', () => this.handleOverlayVoiceToggle(false));
            this.overlayVoiceToggle.addEventListener('mouseleave', () => this.handleOverlayVoiceToggle(false));
        }
        
        // Overlay settings
        if (this.overlayLanguage) {
            this.overlayLanguage.addEventListener('change', () => this.updateOverlayVoiceOptions());
        }
        if (this.overlayAutoStart) {
            this.overlayAutoStart.addEventListener('change', (e) => {
                this.overlayAutoStartEnabled = e.target.checked;
            });
        }
        if (this.overlayAutoStop) {
            this.overlayAutoStop.addEventListener('change', (e) => {
                this.overlayAutoStopEnabled = e.target.checked;
            });
        }
        if (this.overlaySilenceThreshold) {
            this.overlaySilenceThreshold.addEventListener('input', (e) => {
                this.overlaySilenceThresholdValue = parseFloat(e.target.value) * 1000;
                if (this.overlaySilenceValue) {
                    this.overlaySilenceValue.textContent = `${e.target.value}s`;
                }
            });
        }
        
        // Keyboard shortcuts for overlay
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.voiceOverlay.classList.contains('hidden')) {
                this.closeVoiceOverlayMethod();
            }
            // Space for voice recording in overlay
            if (e.code === 'Space' && !this.voiceOverlay.classList.contains('hidden') && !e.repeat) {
                e.preventDefault();
                this.handleOverlayVoiceToggle(true);
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && !this.voiceOverlay.classList.contains('hidden')) {
                e.preventDefault();
                this.handleOverlayVoiceToggle(false);
            }
        });
    }

    setupSettingsListeners() {
        // Range sliders with value updates
        this.temperature.addEventListener('input', (e) => {
            this.tempValue.textContent = e.target.value;
        });
        
        this.topP.addEventListener('input', (e) => {
            this.topPValue.textContent = e.target.value;
        });
        
        this.topK.addEventListener('input', (e) => {
            this.topKValue.textContent = e.target.value;
        });
        
        this.maxTokens.addEventListener('input', (e) => {
            this.maxTokensValue.textContent = e.target.value;
        });
        
        this.repetitionPenalty.addEventListener('input', (e) => {
            this.repPenaltyValue.textContent = e.target.value;
        });
        
        this.silenceThresholdSlider.addEventListener('input', (e) => {
            this.silenceThreshold = parseFloat(e.target.value) * 1000;
            this.silenceValueDisplay.textContent = `${e.target.value}s`;
        });
        
        // Filter settings
        this.disableFilter.addEventListener('change', (e) => {
            this.filterStrictness.disabled = e.target.checked;
            if (e.target.checked) {
                this.filterStrictness.value = 'disabled';
            }
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update mode buttons (if they exist)
        if (mode === 'text') {
            if (this.textModeBtn) this.textModeBtn.classList.add('active');
            if (this.voiceModeBtn) this.voiceModeBtn.classList.remove('active');
            if (this.voiceToggle) this.voiceToggle.style.display = 'none';
            this.updateConnectionStatus('disconnected', 'Text Chat Mode');
            
            // Disconnect voice if connected
            if (this.isConnected) {
                this.disconnectVoice();
            }
        } else {
            if (this.voiceModeBtn) this.voiceModeBtn.classList.add('active');
            if (this.textModeBtn) this.textModeBtn.classList.remove('active');
            if (this.voiceToggle) this.voiceToggle.style.display = 'flex';
            this.updateConnectionStatus('disconnected', 'Voice Chat Mode - Click to connect');
            
            // Auto-connect for voice mode
            this.connectVoice();
        }
    }

    toggleSettings() {
        this.settingsPanel.classList.toggle('open');
    }

    autoResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSend();
        }
    }

    async handleSend() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        
        this.startConversation();
        
        if (this.currentMode === 'text') {
            await this.sendTextMessage(message);
        } else {
            // In voice mode, convert text to speech and send
            await this.sendTextMessage(message);
        }
    }

    handleVoiceToggle(isPressed) {
        if (this.currentMode !== 'voice') return;
        
        // With server-side VAD, use toggle mode (click to start/stop) instead of push-to-talk
        if (this.useServerSideVAD && this.vadEnabled) {
            if (isPressed && !this.isRecording && this.isConnected) {
                // Start continuous recording with VAD
                this.startRecording();
            } else if (isPressed && this.isRecording) {
                // Stop continuous recording
                this.stopRecording();
            }
        } else {
            // Traditional push-to-talk mode
            if (isPressed && !this.isRecording && this.isConnected) {
                this.startRecording();
            } else if (!isPressed && this.isRecording) {
                this.stopRecording();
            }
        }
    }

    startConversation() {
        if (!this.conversationStarted) {
            this.conversationStarted = true;
            this.welcomeScreen.classList.add('hidden');
            this.messagesContainer.classList.remove('hidden');
        }
    }

    async sendTextMessage(message) {
        this.isProcessing = true;
        this.sendBtn.disabled = true;
        this.messageInput.value = '';
        this.autoResize();
        
        // Add user message
        this.addMessage('user', message);
        
        // Show typing indicator
        const typingId = this.addMessage('assistant', '', true);
        
        try {
            // Prepare request
            const payload = {
                model_name: this.modelSelect.value,
                message: message,
                enable_thinking: this.thinkingMode.value === 'true',  // Send as boolean
                disable_content_filter: this.disableFilter.checked,
                content_filter_strictness: this.filterStrictness.value,
                
                // Direct parameters (preferred by backend)
                max_new_tokens: parseInt(this.maxTokens.value),
                temperature: parseFloat(this.temperature.value),
                top_p: parseFloat(this.topP.value),
                top_k: parseInt(this.topK.value),
                repetition_penalty: parseFloat(this.repetitionPenalty.value),
                
                // Also include in generation_config for compatibility
                generation_config: {
                    max_tokens: parseInt(this.maxTokens.value),
                    temperature: parseFloat(this.temperature.value),
                    top_p: parseFloat(this.topP.value),
                    top_k: parseInt(this.topK.value),
                    repetition_penalty: parseFloat(this.repetitionPenalty.value)
                }
            };
            
            // Send request
            // Use dynamic API URL based on current host
            const apiHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
                ? 'http://localhost:8000' 
                : 'https://api.gmai.sa';
            const apiUrl = `${apiHost}/inference/chat`;
            console.log('💬 Sending chat request to:', apiUrl);
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            // Remove typing indicator
            this.removeMessage(typingId);
            
            if (result.success) {
                // Add AI response
                this.addMessage('assistant', result.response || result.final_content, false, {
                    thinking: result.thinking_content,
                    stats: {
                        tokens: result.tokens_generated,
                        time: result.generation_time_ms,
                        speed: result.tokens_per_second
                    }
                });
            } else {
                this.addMessage('assistant', `❌ Error: ${result.error}`);
            }
            
        } catch (error) {
            console.error('Text chat error:', error);
            this.removeMessage(typingId);
            this.addMessage('assistant', `❌ Connection error: ${error.message}`);
        } finally {
            this.isProcessing = false;
            this.sendBtn.disabled = false;
            this.messageInput.focus();
        }
    }

    async setupAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (error) {
            console.error('Failed to create audio context:', error);
        }
    }

    updateVoiceOptions() {
        const language = this.language.value;
        const voiceSelect = this.voice;
        
        // Clear current options
        voiceSelect.innerHTML = '';
        
        if (language === 'ar' || language === 'auto') {
            voiceSelect.innerHTML += '<option value="ar-SA-ZariyahNeural">Arabic Female (Zariyah)</option>';
            voiceSelect.innerHTML += '<option value="ar-SA-HamedNeural">Arabic Male (Hamed)</option>';
        }
        
        if (language === 'en' || language === 'auto') {
            voiceSelect.innerHTML += '<option value="en-US-JennyNeural">English Female (Jenny)</option>';
            voiceSelect.innerHTML += '<option value="en-US-GuyNeural">English Male (Guy)</option>';
        }
        
        // Default selection
        if (language === 'auto') {
            voiceSelect.value = 'ar-SA-ZariyahNeural';
        }
    }

    async connectVoice() {
        if (this.isConnected) return;
        
        try {
            this.updateConnectionStatus('processing', 'Connecting...');
            
            // Get configuration values  
            const language = this.language.value === 'auto' ? 'ar' : this.language.value; // Default auto to Arabic
            const voiceType = this.getVoiceType(this.voice.value);
            
            // Build WebSocket URL with query parameters
            // Use dynamic URL based on current host
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
                ? 'localhost:8000' 
                : 'api.gmai.sa';
            const wsUrl = `${protocol}//${host}/api/v1/ws/simple-voice-chat?language=${language}&voice_type=${voiceType}`;
            console.log('🔗 Connecting to WebSocket:', wsUrl);
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                // Status will be updated when we receive connection_established message with VAD info
                this.updateConnectionStatus('connected', 'Connected - Waiting for VAD capabilities...');
                console.log('Voice WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                this.handleVoiceMessage(event);
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Disconnected');
                if (this.isRecording) {
                    this.stopRecording();
                }
                console.log('Voice WebSocket disconnected');
            };
            
            this.ws.onerror = (error) => {
                console.error('Voice WebSocket error:', error);
                this.updateConnectionStatus('disconnected', 'Connection failed');
            };
            
        } catch (error) {
            console.error('Failed to connect voice:', error);
            this.updateConnectionStatus('disconnected', 'Connection failed');
        }
    }

    disconnectVoice() {
        if (this.ws) {
            this.ws.close();
        }
        this.isConnected = false;
        if (this.isRecording) {
            this.stopRecording();
        }
    }

    async handleVoiceMessage(event) {
        try {
            // Parse JSON response from backend
            const data = JSON.parse(event.data);
            
            if (data.type === 'connection_established') {
                // Connection established - check for VAD capabilities
                console.log('Voice connection established:', data);
                this.vadEnabled = data.vad_enabled !== false;  // Default to true unless explicitly disabled
                this.useServerSideVAD = this.vadEnabled;
                console.log(`VAD enabled: ${this.vadEnabled}, Server-side VAD: ${this.useServerSideVAD}`);
                
                // Reset VAD state for clean start
                this.resetVADState();
                
                // Update UI based on VAD capabilities
                this.updateVoiceButtonText();
                
                // Update connection status with VAD info
                if (this.useServerSideVAD) {
                    this.updateConnectionStatus('connected', 'Connected with AI Voice Detection - Click mic to start');
                } else {
                    this.updateConnectionStatus('connected', 'Connected - Hold space or mic button to talk');
                }
                
            } else if (data.type === 'vad_update') {
                // Real-time VAD feedback
                this.handleVADUpdate(data);
                
            } else if (data.type === 'speech_start') {
                // Speech started
                this.showVoiceStatus('🎤 Listening...');
                console.log('Speech started');
                
            } else if (data.type === 'speech_end') {
                // Speech ended - STOP recording to prevent feedback loop
                console.log('Speech ended - applying VAD state management');
                
                const now = Date.now();
                this.vadState.lastSpeechEnd = now;
                this.vadState.consecutiveSpeechEnds++;
                
                // Check for consecutive speech_end events (indicates loop)
                if (this.vadState.consecutiveSpeechEnds > this.vadState.maxConsecutiveEnds) {
                    console.warn('� Multiple consecutive speech_end events detected - extending cooldown');
                    this.vadState.cooldownPeriod = 5000; // Extend to 5 seconds
                    this.vadState.consecutiveSpeechEnds = 0; // Reset counter
                }
                
                // Enter cooldown state
                this.vadState.isInCooldown = true;
                setTimeout(() => {
                    this.vadState.isInCooldown = false;
                    this.vadState.cooldownPeriod = 2000; // Reset to normal cooldown
                    console.log('VAD cooldown period ended');
                }, this.vadState.cooldownPeriod);
                
                this.showVoiceStatus('🔄 Processing your speech...');
                console.log('Entered processing state - microphone disabled until AI responds');
                
                // CRITICAL: Stop recording immediately when speech ends
                if (this.isRecording) {
                    console.log('🛑 Stopping recording due to speech_end detection');
                    this.stopRecording();
                }
                
            } else if (data.type === 'turn_processing_started') {
                // Turn processing started
                this.showVoiceStatus('⚡ Processing complete turn...');
                console.log('Turn processing started');
                
            } else if (data.type === 'voice_response') {
                // Handle voice response from AI with deduplication and VAD state management
                console.log('Voice response received - checking for duplicates and applying state management');
                
                // Update VAD state
                this.vadState.lastVoiceResponse = Date.now();
                this.vadState.consecutiveSpeechEnds = 0; // Reset consecutive counter on successful response
                
                // IMPORTANT: Stop recording immediately when we get a response to prevent feedback loop
                if (this.isRecording) {
                    console.log('🛑 Stopping recording due to voice response to prevent feedback loop');
                    this.stopRecording();
                }
                
                // Check for duplicate responses to prevent audio playback loops
                if (this.isDuplicateResponse(data)) {
                    console.warn('🚫 Skipping duplicate voice response to prevent audio loop');
                    return;
                }
                
                if (data.success && data.audio_base64) {
                    // Convert base64 to audio blob
                    const audioBlob = this.base64ToBlob(data.audio_base64, 'audio/wav');
                    
                    // Add to conversation with transcription and response text (only once)
                    this.addMessage('assistant', data.response_text, false, { 
                        transcription: data.transcription,
                        stats: {
                            time: data.response_time_ms,
                            tokens: data.response_text ? data.response_text.split(' ').length : 0,
                            speed: data.response_time_ms > 0 ? (data.response_text.split(' ').length / (data.response_time_ms / 1000)) : 0
                        },
                        processingMode: data.processing_mode || 'realtime_vad'
                    });
                    
                    // Play audio using the new queue management system
                    await this.playAudioResponse(audioBlob, false);
                    
                } else {
                    this.addMessage('assistant', `❌ Voice Error: ${data.message || 'Unknown error'}`);
                }
                
            } else if (data.type === 'processing_started') {
                this.showVoiceStatus(data.message);
            } else if (data.type === 'error') {
                console.error('Voice server error:', data.message);
                this.addMessage('assistant', `❌ Voice Error: ${data.message}`);
            }
        } catch (error) {
            console.error('Error handling voice message:', error);
            this.addMessage('assistant', `❌ Error processing voice response: ${error.message}`);
        }
    }
    
    handleVADUpdate(data) {
        /**
         * Handle real-time VAD updates from the server for improved user feedback.
         * This replaces client-side silence detection with server-driven feedback.
         */
        if (!this.vadFeedback || !this.useServerSideVAD) return;
        
        const state = data.state || {};
        const isSpeaking = state.is_speaking;
        const silenceDuration = state.silence_duration_ms || 0;
        const bufferedChunks = state.buffered_chunks || 0;
        
        // Update voice status with real-time feedback from server
        if (isSpeaking) {
            this.showVoiceStatus(`🎤 Speaking detected... (${bufferedChunks} chunks buffered)`);
        } else if (silenceDuration > 0) {
            const silenceSeconds = (silenceDuration / 1000).toFixed(1);
            if (silenceDuration > 300) { // Show silence feedback after 300ms
                this.showVoiceStatus(`🔇 Silence: ${silenceSeconds}s (${bufferedChunks} chunks ready)`);
            }
        }
        
        // Update visual indicators based on server VAD state
        this.updateVADIndicators(state);
        
        console.log(`VAD Update: Speaking=${isSpeaking}, Silence=${silenceDuration}ms, Chunks=${bufferedChunks}`);
    }
    
    updateVoiceButtonText() {
        /**
         * Update voice button text based on VAD capabilities.
         */
        if (this.useServerSideVAD && this.vadEnabled) {
            // Server-side VAD mode - click to start/stop
            if (this.voiceToggle) {
                this.voiceToggle.title = 'Click to start/stop listening with AI voice detection';
            }
            if (this.overlayVoiceToggle) {
                const span = this.overlayVoiceToggle.querySelector('span');
                if (span) span.textContent = 'Click to Listen';
                this.overlayVoiceToggle.title = 'Click to start/stop listening with AI voice detection';
            }
        } else {
            // Traditional push-to-talk mode
            if (this.voiceToggle) {
                this.voiceToggle.title = 'Hold to talk (traditional mode)';
            }
            if (this.overlayVoiceToggle) {
                const span = this.overlayVoiceToggle.querySelector('span');
                if (span) span.textContent = 'Hold to Talk';
                this.overlayVoiceToggle.title = 'Hold to talk (traditional mode)';
            }
        }
    }

    updateVADIndicators(state) {
        /**
         * Update visual indicators based on VAD state.
         */
        // Update voice button appearance based on VAD state
        if (this.voiceToggle) {
            if (state.is_speaking) {
                this.voiceToggle.classList.add('speaking');
                this.voiceToggle.classList.remove('silent');
            } else {
                this.voiceToggle.classList.add('silent');
                this.voiceToggle.classList.remove('speaking');
            }
        }
        
        // Update overlay indicators if in overlay mode
        if (this.overlayVoiceToggle && !this.voiceOverlay.classList.contains('hidden')) {
            if (state.is_speaking) {
                this.overlayVoiceToggle.classList.add('speaking');
                this.overlayVoiceToggle.classList.remove('silent');
            } else {
                this.overlayVoiceToggle.classList.add('silent');
                this.overlayVoiceToggle.classList.remove('speaking');
            }
        }
    }

    async startRecording() {
        if (!this.isConnected || this.isRecording) return;
        
        // Prevent recording while AI is speaking to avoid feedback
        if (this.isPlayingAIAudio) {
            console.log('🚫 Cannot start recording while AI is speaking (preventing feedback)');
            return;
        }
        
        // VAD state management: Check cooldown period to prevent infinite loops
        const now = Date.now();
        if (this.vadState.isInCooldown) {
            const remainingCooldown = this.vadState.cooldownPeriod - (now - this.vadState.lastSpeechEnd);
            console.log(`🚫 Cannot start recording during VAD cooldown (${remainingCooldown}ms remaining)`);
            return;
        }
        
        // Additional check: If recent speech_end or voice_response, wait longer
        const timeSinceLastSpeechEnd = now - this.vadState.lastSpeechEnd;
        const timeSinceLastVoiceResponse = now - this.vadState.lastVoiceResponse;
        const minDelayAfterEvents = 1500; // Minimum 1.5 seconds after speech events
        
        if (timeSinceLastSpeechEnd < minDelayAfterEvents || timeSinceLastVoiceResponse < minDelayAfterEvents) {
            console.log(`🚫 Cannot start recording too soon after speech events (need ${minDelayAfterEvents}ms delay)`);
            return;
        }
        
        try {
            // Advanced audio configuration to prevent feedback loops
            const audioConstraints = {
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: false,  // Disable echo cancellation for better real-time control
                    noiseSuppression: true,
                    autoGainControl: false,   // Disable auto gain to prevent feedback amplification
                    googEchoCancellation: false,  // Google-specific echo cancellation off
                    googAutoGainControl: false,   // Google-specific auto gain off
                    googNoiseSuppression: true,   // Keep noise suppression
                    googTypingNoiseDetection: false
                }
            };
            
            console.log('🎤 Requesting microphone access with feedback prevention settings...');
            const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);
            
            // Setup MediaRecorder
            const options = { mimeType: 'audio/webm;codecs=opus' };
            
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                    options.mimeType = 'audio/ogg;codecs=opus';
                } else {
                    options.mimeType = 'audio/webm';
                }
            }
            
            this.mediaRecorder = new MediaRecorder(stream, options);
            this.audioChunks = [];
            
            // Always use streaming mode with server-side VAD when available
            if (this.useServerSideVAD && this.vadEnabled) {
                // Server-side VAD mode - send small chunks continuously
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        // Send chunk immediately for server-side VAD processing
                        this.sendAudioChunk(event.data);
                    }
                };
                
                // Start recording with small time slices for streaming
                this.mediaRecorder.start(this.chunkSize); // Use configurable chunk size (100ms default)
                
                this.mediaRecorder.onstop = () => {
                    // Clean up stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
            } else {
                // Fallback: Traditional mode - collect all chunks
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };
                
                this.mediaRecorder.onstop = () => {
                    this.sendVoiceData();
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.mediaRecorder.start(100);
            }
            
            // Only setup local silence detection if server-side VAD is not available
            if (this.autoStopEnabled && !this.useServerSideVAD && this.audioContext) {
                this.setupSilenceDetection(stream);
            }
            
            this.isRecording = true;
            
            // Update UI
            this.voiceToggle.classList.add('recording');
            this.voiceToggle.textContent = '🔴';
            
            if (this.useServerSideVAD && this.vadEnabled) {
                this.showVoiceStatus('🎤 Listening with VAD... (speak naturally)');
                console.log('Recording started with server-side VAD streaming mode');
            } else {
                this.showVoiceStatus('🎤 Recording... (release to send)');
                console.log('Recording started in traditional mode');
            }
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showVoiceStatus('❌ Microphone access denied');
        }
    }
    
    sendAudioChunk(audioChunk) {
        /**
         * Send individual audio chunk for real-time processing.
         */
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not ready for audio chunk');
            return;
        }
        
        this.startConversation();
        
        // Send the chunk immediately
        this.ws.send(audioChunk);
        
        console.log('Audio chunk sent:', audioChunk.size, 'bytes');
    }

    setupSilenceDetection(stream) {
        if (!this.audioContext) return;
        
        try {
            const source = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
            
            this.silenceDetectionActive = true;
            this.detectSilence();
        } catch (error) {
            console.error('Failed to setup silence detection:', error);
        }
    }

    detectSilence() {
        if (!this.silenceDetectionActive || !this.analyser) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const isSilent = average < 10;
        
        if (isSilent) {
            if (!this.silenceTimeout) {
                this.silenceTimeout = setTimeout(() => {
                    if (this.isRecording && this.autoStopEnabled) {
                        this.stopRecording();
                    }
                }, this.silenceThreshold);
            }
        } else {
            if (this.silenceTimeout) {
                clearTimeout(this.silenceTimeout);
                this.silenceTimeout = null;
            }
        }
        
        if (this.isRecording) {
            setTimeout(() => this.detectSilence(), 100);
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        this.silenceDetectionActive = false;
        
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
        }
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        // Update UI
        this.voiceToggle.classList.remove('recording');
        this.voiceToggle.textContent = '🎤';
        
        if (this.useServerSideVAD && this.vadEnabled) {
            this.showVoiceStatus('⏹️ Stopped listening (server will process buffered audio)');
            console.log('Server-side VAD recording stopped');
        } else {
            this.showVoiceStatus('📤 Processing audio...');
            console.log('Traditional recording stopped');
        }
    }

    sendVoiceData() {
        if (this.audioChunks.length === 0 || !this.ws) return;
        
        this.startConversation();
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Send audio data directly (no config needed - already sent via query params)
        this.ws.send(audioBlob);
        
        // Add voice message to conversation
        this.addMessage('user', '🎤 Voice message sent');
        
        this.hideVoiceStatus();
        console.log('Voice data sent, size:', audioBlob.size, 'bytes');
    }

    addMessage(sender, content, isTyping = false, extras = {}) {
        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.id = messageId;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : 'AI';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isTyping) {
            contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        } else if (content) {
            contentDiv.textContent = content;
            
            // Add thinking content if available
            if (extras.thinking) {
                const thinkingDiv = document.createElement('details');
                thinkingDiv.className = 'message-thinking';
                thinkingDiv.innerHTML = `
                    <summary>🤔 Thinking Process</summary>
                    <div>${extras.thinking}</div>
                `;
                contentDiv.appendChild(thinkingDiv);
            }
            
            // Add stats if available
            if (extras.stats) {
                const statsDiv = document.createElement('div');
                statsDiv.style.fontSize = '12px';
                statsDiv.style.color = '#6b7280';
                statsDiv.style.marginTop = '8px';
                statsDiv.textContent = `${extras.stats.tokens} tokens, ${Math.round(extras.stats.time)}ms, ${extras.stats.speed.toFixed(1)} tok/s`;
                contentDiv.appendChild(statsDiv);
            }
        }
        
        // Add audio if available
        if (extras.audioUrl) {
            const audioDiv = document.createElement('div');
            audioDiv.className = 'message-audio';
            audioDiv.innerHTML = `<audio controls><source src="${extras.audioUrl}" type="audio/webm"></audio>`;
            contentDiv.appendChild(audioDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        
        return messageId;
    }

    removeMessage(messageId) {
        const element = document.getElementById(messageId);
        if (element) {
            element.remove();
        }
    }

    updateConnectionStatus(status, message) {
        const statusElement = this.connectionStatus;
        statusElement.className = `status-indicator ${status}`;
        statusElement.innerHTML = `<span>●</span><span>${message}</span>`;
    }

    showVoiceStatus(message) {
        this.voiceStatus.textContent = message;
        this.voiceStatus.classList.add('show');
    }

    hideVoiceStatus() {
        this.voiceStatus.classList.remove('show');
    }

    // Voice Overlay Methods
    openVoiceOverlay() {
        this.voiceOverlay.classList.remove('hidden');
        this.initializeOverlaySettings();
        this.clearOverlayConversation();
        
        // Auto-connect voice when overlay opens
        if (!this.overlayConnected) {
            this.connectOverlayVoice();
        }
    }

    closeVoiceOverlayMethod() {
        this.voiceOverlay.classList.add('hidden');
        
        // Disconnect voice and cleanup
        if (this.overlayConnected) {
            this.disconnectOverlayVoice();
        }
        
        // Stop any ongoing recording
        if (this.overlayRecording) {
            this.stopOverlayRecording();
        }
    }

    initializeOverlaySettings() {
        // Set default values
        this.overlayAutoStartEnabled = this.overlayAutoStart.checked;
        this.overlayAutoStopEnabled = this.overlayAutoStop.checked;
        this.overlaySilenceThresholdValue = parseFloat(this.overlaySilenceThreshold.value) * 1000;
        this.overlaySilenceValue.textContent = `${this.overlaySilenceThreshold.value}s`;
        
        // Initialize connection status
        this.overlayConnected = false;
        this.overlayRecording = false;
        this.overlayAudioChunks = [];
        this.overlayWebSocket = null;
        
        this.updateOverlayConnectionStatus('disconnected', 'Disconnected');
    }

    clearOverlayConversation() {
        this.overlayConversation.innerHTML = `
            <div class="conversation-placeholder">
                <p>Start your voice conversation!</p>
                <p>Hold the microphone button and speak.</p>
            </div>
        `;
    }

    updateOverlayVoiceOptions() {
        const language = this.overlayLanguage.value;
        const voiceSelect = this.overlayVoice;
        
        // Clear current options
        voiceSelect.innerHTML = '';
        
        if (language === 'ar' || language === 'auto') {
            voiceSelect.innerHTML += '<option value="ar-SA-ZariyahNeural">Arabic Female (Zariyah)</option>';
            voiceSelect.innerHTML += '<option value="ar-SA-HamedNeural">Arabic Male (Hamed)</option>';
        }
        
        if (language === 'en' || language === 'auto') {
            voiceSelect.innerHTML += '<option value="en-US-JennyNeural">English Female (Jenny)</option>';
            voiceSelect.innerHTML += '<option value="en-US-GuyNeural">English Male (Guy)</option>';
        }
    }

    async connectOverlayVoice() {
        if (this.overlayConnected) return;
        
        try {
            this.updateOverlayConnectionStatus('processing', 'Connecting...');
            
            // Get configuration values
            const language = this.overlayLanguage.value === 'auto' ? 'ar' : this.overlayLanguage.value; // Default auto to English
            const voiceType = this.getVoiceType(this.overlayVoice.value);
            
            // Build WebSocket URL with query parameters
            // Use dynamic URL based on current host
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
                ? 'localhost:8000' 
                : 'api.gmai.sa';
            const wsUrl = `${protocol}//${host}/api/v1/ws/simple-voice-chat?language=${language}&voice_type=${voiceType}`;
            console.log('🎤 Connecting to Overlay WebSocket:', wsUrl);
            this.overlayWebSocket = new WebSocket(wsUrl);
            
            this.overlayWebSocket.onopen = () => {
                this.overlayConnected = true;
                // Status will be updated when we receive connection_established message with VAD info
                this.updateOverlayConnectionStatus('connected', 'Connected - Waiting for VAD capabilities...');
                console.log('Overlay Voice WebSocket connected');
            };
            
            this.overlayWebSocket.onmessage = (event) => {
                this.handleOverlayVoiceMessage(event);
            };
            
            this.overlayWebSocket.onclose = () => {
                this.overlayConnected = false;
                this.updateOverlayConnectionStatus('disconnected', 'Disconnected');
                if (this.overlayRecording) {
                    this.stopOverlayRecording();
                }
                console.log('Overlay Voice WebSocket disconnected');
            };
            
            this.overlayWebSocket.onerror = (error) => {
                console.error('Overlay Voice WebSocket error:', error);
                this.updateOverlayConnectionStatus('disconnected', 'Connection failed');
            };
            
        } catch (error) {
            console.error('Failed to connect overlay voice:', error);
            this.updateOverlayConnectionStatus('disconnected', 'Connection failed');
        }
    }

    disconnectOverlayVoice() {
        if (this.overlayWebSocket) {
            this.overlayWebSocket.close();
        }
        this.overlayConnected = false;
        if (this.overlayRecording) {
            this.stopOverlayRecording();
        }
    }

    async handleOverlayVoiceMessage(event) {
        try {
            // Parse JSON response from backend
            const data = JSON.parse(event.data);
            
            if (data.type === 'connection_established') {
                // Connection established - check for VAD capabilities
                console.log('Overlay voice connection established:', data);
                
                // Reset VAD state for clean overlay start
                this.resetVADState();
                
                // Update overlay connection status with VAD info
                if (this.useServerSideVAD) {
                    this.updateOverlayConnectionStatus('connected', 'Connected with AI Voice Detection - Click mic to start');
                } else {
                    this.updateOverlayConnectionStatus('connected', 'Connected - Hold space or mic button to talk');
                }
                
            } else if (data.type === 'vad_update') {
                // Real-time VAD feedback for overlay
                this.handleOverlayVADUpdate(data);
                
            } else if (data.type === 'speech_start') {
                // Speech started
                this.showOverlayVoiceStatus('🎤 Listening...');
                console.log('Overlay speech started');
                
            } else if (data.type === 'speech_end') {
                // Speech ended - STOP recording immediately to prevent feedback loop (overlay)
                console.log('Overlay speech ended - stopping recording and entering processing state');
                
                // CRITICAL: Stop recording immediately when speech ends
                if (this.overlayRecording) {
                    console.log('🛑 Stopping overlay recording due to speech_end detection');
                    this.stopOverlayRecording();
                }
                
                const now = Date.now();
                this.vadState.lastSpeechEnd = now;
                this.vadState.consecutiveSpeechEnds++;
                
                // Check for consecutive speech_end events (indicates loop)
                if (this.vadState.consecutiveSpeechEnds > this.vadState.maxConsecutiveEnds) {
                    console.warn('🚨 Overlay: Multiple consecutive speech_end events detected - extending cooldown');
                    this.vadState.cooldownPeriod = 5000; // Extend to 5 seconds
                    this.vadState.consecutiveSpeechEnds = 0; // Reset counter
                }
                
                // Enter cooldown state to prevent immediate restart
                this.vadState.isInCooldown = true;
                setTimeout(() => {
                    this.vadState.isInCooldown = false;
                    this.vadState.cooldownPeriod = 2000; // Reset to normal cooldown
                    console.log('Overlay VAD cooldown period ended');
                }, this.vadState.cooldownPeriod);
                
                this.showOverlayVoiceStatus('🔄 Processing your speech...');
                console.log('Overlay entered processing state - microphone disabled until AI responds');
                
            } else if (data.type === 'turn_processing_started') {
                // Turn processing started
                this.showOverlayVoiceStatus('⚡ Processing complete turn...');
                console.log('Overlay turn processing started');
                
            } else if (data.type === 'voice_response') {
                // Handle voice response from AI with deduplication and VAD state management (overlay)
                console.log('Overlay voice response received - checking for duplicates and applying state management');
                
                // Update VAD state
                this.vadState.lastVoiceResponse = Date.now();
                this.vadState.consecutiveSpeechEnds = 0; // Reset consecutive counter on successful response
                
                // IMPORTANT: Stop recording immediately when we get a response to prevent feedback loop
                if (this.overlayRecording) {
                    console.log('🛑 Stopping overlay recording due to voice response to prevent feedback loop');
                    this.stopOverlayRecording();
                }
                
                // Check for duplicate responses to prevent audio playback loops
                if (this.isDuplicateResponse(data)) {
                    console.warn('🚫 Skipping duplicate overlay voice response to prevent audio loop');
                    return;
                }
                
                if (data.success && data.audio_base64) {
                    // Convert base64 to audio blob
                    const audioBlob = this.base64ToBlob(data.audio_base64, 'audio/wav');
                    
                    // Add to conversation with transcription and response text (only once)
                    this.addOverlayMessage('assistant', data.response_text, { 
                        transcription: data.transcription,
                        responseTime: data.response_time_ms,
                        processingMode: data.processing_mode || 'realtime_vad'
                    });
                    
                    // Play audio using the new queue management system
                    await this.playAudioResponse(audioBlob, true);
                    
                } else {
                    this.addOverlayMessage('assistant', `❌ Voice Error: ${data.message || 'Unknown error'}`);
                }
                
            } else if (data.type === 'processing_started') {
                this.showOverlayVoiceStatus(data.message);
            } else if (data.type === 'error') {
                console.error('Overlay voice server error:', data.message);
                this.addOverlayMessage('assistant', `❌ Voice Error: ${data.message}`);
            }
        } catch (error) {
            console.error('Error handling overlay voice message:', error);
            this.addOverlayMessage('assistant', `❌ Error processing voice response: ${error.message}`);
        }
    }
    
    handleOverlayVADUpdate(data) {
        /**
         * Handle real-time VAD updates for overlay mode.
         */
        if (!this.vadFeedback) return;
        
        const state = data.state || {};
        const isSpeaking = state.is_speaking;
        const silenceDuration = state.silence_duration_ms || 0;
        const bufferedChunks = state.buffered_chunks || 0;
        
        // Update voice status with real-time feedback
        if (isSpeaking) {
            this.showOverlayVoiceStatus(`🎤 Speaking... (${bufferedChunks} chunks)`);
        } else if (silenceDuration > 0) {
            const silenceSeconds = (silenceDuration / 1000).toFixed(1);
            this.showOverlayVoiceStatus(`🔇 Silence: ${silenceSeconds}s (${bufferedChunks} chunks buffered)`);
        }
        
        // Update visual indicators
        this.updateVADIndicators(state);
    }

    handleOverlayVoiceToggle(isPressed) {
        // Use same VAD logic for overlay as main interface
        if (this.useServerSideVAD && this.vadEnabled) {
            if (isPressed && !this.overlayRecording && this.overlayConnected) {
                // Start continuous recording with VAD
                this.startOverlayRecording();
            } else if (isPressed && this.overlayRecording) {
                // Stop continuous recording
                this.stopOverlayRecording();
            }
        } else {
            // Traditional push-to-talk mode
            if (isPressed && !this.overlayRecording && this.overlayConnected) {
                this.startOverlayRecording();
            } else if (!isPressed && this.overlayRecording) {
                this.stopOverlayRecording();
            }
        }
    }

    async startOverlayRecording() {
        if (!this.overlayConnected || this.overlayRecording) return;
        
        // Prevent recording while AI is speaking to avoid feedback
        if (this.isPlayingAIAudio) {
            console.log('🚫 Cannot start overlay recording while AI is speaking (preventing feedback)');
            return;
        }
        
        // VAD state management: Check cooldown period to prevent infinite loops (overlay)
        const now = Date.now();
        if (this.vadState.isInCooldown) {
            const remainingCooldown = this.vadState.cooldownPeriod - (now - this.vadState.lastSpeechEnd);
            console.log(`🚫 Cannot start overlay recording during VAD cooldown (${remainingCooldown}ms remaining)`);
            return;
        }
        
        // Additional check: If recent speech_end or voice_response, wait longer (overlay)
        const timeSinceLastSpeechEnd = now - this.vadState.lastSpeechEnd;
        const timeSinceLastVoiceResponse = now - this.vadState.lastVoiceResponse;
        const minDelayAfterEvents = 1500; // Minimum 1.5 seconds after speech events
        
        if (timeSinceLastSpeechEnd < minDelayAfterEvents || timeSinceLastVoiceResponse < minDelayAfterEvents) {
            console.log(`🚫 Cannot start overlay recording too soon after speech events (need ${minDelayAfterEvents}ms delay)`);
            return;
        }
        
        try {
            // Advanced audio configuration to prevent feedback loops (overlay)
            const audioConstraints = {
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: false,  // Disable echo cancellation for better real-time control
                    noiseSuppression: true,
                    autoGainControl: false,   // Disable auto gain to prevent feedback amplification
                    googEchoCancellation: false,  // Google-specific echo cancellation off
                    googAutoGainControl: false,   // Google-specific auto gain off
                    googNoiseSuppression: true,   // Keep noise suppression
                    googTypingNoiseDetection: false
                }
            };
            
            console.log('🎤 Requesting overlay microphone access with feedback prevention settings...');
            const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);
            
            // Setup MediaRecorder
            const options = { mimeType: 'audio/webm;codecs=opus' };
            
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                    options.mimeType = 'audio/ogg;codecs=opus';
                } else {
                    options.mimeType = 'audio/webm';
                }
            }
            
            this.overlayMediaRecorder = new MediaRecorder(stream, options);
            this.overlayAudioChunks = [];
            
            // Use server-side VAD for overlay too
            if (this.useServerSideVAD && this.vadEnabled) {
                // Server-side VAD mode - send small chunks continuously
                this.overlayMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        // Send chunk immediately for server-side VAD processing
                        this.sendOverlayAudioChunk(event.data);
                    }
                };
                
                // Start recording with small time slices for streaming
                this.overlayMediaRecorder.start(this.chunkSize);
                
                this.overlayMediaRecorder.onstop = () => {
                    // Clean up stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
            } else {
                // Traditional mode - collect all chunks
                this.overlayMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.overlayAudioChunks.push(event.data);
                    }
                };
                
                this.overlayMediaRecorder.onstop = () => {
                    this.sendOverlayVoiceData();
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.overlayMediaRecorder.start(100);
            }
            
            // Only setup local silence detection if server-side VAD is not available
            if (this.overlayAutoStopEnabled && !this.useServerSideVAD && this.audioContext) {
                this.setupOverlaySilenceDetection(stream);
            }
            
            this.overlayRecording = true;
            
            // Update UI
            this.overlayVoiceToggle.classList.add('recording');
            this.overlayVoiceToggle.querySelector('span').textContent = 'Recording...';
            
            if (this.useServerSideVAD && this.vadEnabled) {
                this.showOverlayVoiceStatus('🎤 Listening with VAD... (speak naturally)');
                console.log('Overlay recording started with server-side VAD streaming mode');
            } else {
                this.showOverlayVoiceStatus('🎤 Recording... (release to send)');
                console.log('Overlay recording started in traditional mode');
            }
            
        } catch (error) {
            console.error('Failed to start overlay recording:', error);
            this.showOverlayVoiceStatus('❌ Microphone access denied');
        }
    }

    setupOverlaySilenceDetection(stream) {
        if (!this.audioContext) return;
        
        try {
            const source = this.audioContext.createMediaStreamSource(stream);
            this.overlayAnalyser = this.audioContext.createAnalyser();
            this.overlayAnalyser.fftSize = 256;
            source.connect(this.overlayAnalyser);
            
            this.overlaySilenceDetectionActive = true;
            this.detectOverlaySilence();
        } catch (error) {
            console.error('Failed to setup overlay silence detection:', error);
        }
    }

    detectOverlaySilence() {
        if (!this.overlaySilenceDetectionActive || !this.overlayAnalyser) return;
        
        const bufferLength = this.overlayAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.overlayAnalyser.getByteFrequencyData(dataArray);
        
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const isSilent = average < 10;
        
        if (isSilent) {
            if (!this.overlaySilenceTimeout) {
                this.overlaySilenceTimeout = setTimeout(() => {
                    if (this.overlayRecording && this.overlayAutoStopEnabled) {
                        this.stopOverlayRecording();
                    }
                }, this.overlaySilenceThresholdValue);
            }
        } else {
            if (this.overlaySilenceTimeout) {
                clearTimeout(this.overlaySilenceTimeout);
                this.overlaySilenceTimeout = null;
            }
        }
        
        if (this.overlayRecording) {
            setTimeout(() => this.detectOverlaySilence(), 100);
        }
    }

    stopOverlayRecording() {
        if (!this.overlayRecording) return;
        
        this.overlayRecording = false;
        this.overlaySilenceDetectionActive = false;
        
        if (this.overlaySilenceTimeout) {
            clearTimeout(this.overlaySilenceTimeout);
            this.overlaySilenceTimeout = null;
        }
        
        if (this.overlayMediaRecorder && this.overlayMediaRecorder.state !== 'inactive') {
            this.overlayMediaRecorder.stop();
        }
        
        // Update UI - use correct button text based on VAD mode
        this.overlayVoiceToggle.classList.remove('recording');
        const span = this.overlayVoiceToggle.querySelector('span');
        if (span) {
            if (this.useServerSideVAD && this.vadEnabled) {
                span.textContent = 'Click to Listen';
            } else {
                span.textContent = 'Hold to Talk';
            }
        }
        this.showOverlayVoiceStatus('📤 Processing audio...');
        
        console.log('Overlay recording stopped');
    }

    sendOverlayAudioChunk(audioChunk) {
        /**
         * Send individual audio chunk for real-time VAD processing (overlay mode).
         */
        if (!this.overlayWebSocket || this.overlayWebSocket.readyState !== WebSocket.OPEN) {
            console.warn('Overlay WebSocket not ready for audio chunk');
            return;
        }
        
        // Send the chunk immediately
        this.overlayWebSocket.send(audioChunk);
        
        console.log('Overlay audio chunk sent:', audioChunk.size, 'bytes');
    }

    sendOverlayVoiceData() {
        if (this.overlayAudioChunks.length === 0 || !this.overlayWebSocket) return;
        
        const audioBlob = new Blob(this.overlayAudioChunks, { type: 'audio/webm' });
        
        // Send audio data directly (no config needed - already sent via query params)
        this.overlayWebSocket.send(audioBlob);
        
        // Add voice message to conversation
        this.addOverlayMessage('user', '🎤 Voice message sent');
        
        this.hideOverlayVoiceStatus();
        console.log('Overlay voice data sent, size:', audioBlob.size, 'bytes');
    }

    addOverlayMessage(sender, content, extras = {}) {
        // Remove placeholder if it exists
        const placeholder = this.overlayConversation.querySelector('.conversation-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `voice-message ${sender}`;
        
        if (content) {
            const contentP = document.createElement('p');
            contentP.textContent = content;
            messageDiv.appendChild(contentP);
        }
        
        // Add audio if available
        if (extras.audioUrl) {
            const audioDiv = document.createElement('div');
            audioDiv.className = 'voice-audio';
            audioDiv.innerHTML = `<audio controls><source src="${extras.audioUrl}" type="audio/webm"></audio>`;
            messageDiv.appendChild(audioDiv);
        }
        
        this.overlayConversation.appendChild(messageDiv);
        this.overlayConversation.scrollTop = this.overlayConversation.scrollHeight;
    }

    updateOverlayConnectionStatus(status, message) {
        const statusElement = this.overlayConnectionStatus;
        statusElement.className = `status-indicator ${status}`;
        statusElement.innerHTML = `<span>●</span><span>${message}</span>`;
    }

    showOverlayVoiceStatus(message) {
        this.overlayVoiceStatus.textContent = message;
        this.overlayVoiceStatus.classList.add('show');
    }

    hideOverlayVoiceStatus() {
        this.overlayVoiceStatus.classList.remove('show');
    }

    getVoiceType(voiceValue) {
        // Extract voice type from voice value
        if (voiceValue.includes('Zariyah') || voiceValue.includes('Jenny')) {
            return 'female';
        } else if (voiceValue.includes('Hamed') || voiceValue.includes('Guy')) {
            return 'male';
        }
        return 'female'; // default
    }

    base64ToBlob(base64Data, contentType = '') {
        // Convert base64 string to blob
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: contentType });
    }
    
    muteAllAudioElements(mute) {
        /**
         * Mute or unmute all audio elements to prevent feedback during AI playback.
         */
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            if (mute) {
                audio.muted = true;
            } else {
                audio.muted = false;
            }
        });
    }
}

// Global functions for example prompts
window.setPrompt = function(prompt) {
    const chat = window.beautyAIChat;
    chat.messageInput.value = prompt;
    chat.autoResize();
    chat.messageInput.focus();
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.beautyAIChat = new BeautyAIChat();
    console.log('BeautyAI Modern Chat Interface initialized');
});
