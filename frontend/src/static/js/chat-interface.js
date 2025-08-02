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
        this.autoStartEnabled = true;
        this.autoStopEnabled = true;
        this.silenceThreshold = 3000;
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateVoiceOptions();
        this.setupAudioContext();
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
    }

    setupEventListeners() {
        // Mode switching
        this.textModeBtn.addEventListener('click', () => this.switchMode('text'));
        this.voiceModeBtn.addEventListener('click', () => this.switchMode('voice'));
        
        // Settings panel
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        
        // Text input
        this.messageInput.addEventListener('input', () => this.autoResize());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.handleSend());
        
        // Voice controls
        this.voiceToggle.addEventListener('mousedown', () => this.handleVoiceToggle(true));
        this.voiceToggle.addEventListener('mouseup', () => this.handleVoiceToggle(false));
        this.voiceToggle.addEventListener('mouseleave', () => this.handleVoiceToggle(false));
        
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
        
        // Update mode buttons
        if (mode === 'text') {
            this.textModeBtn.classList.add('active');
            this.voiceModeBtn.classList.remove('active');
            this.voiceToggle.style.display = 'none';
            this.updateConnectionStatus('disconnected', 'Text Chat Mode');
            
            // Disconnect voice if connected
            if (this.isConnected) {
                this.disconnectVoice();
            }
        } else {
            this.voiceModeBtn.classList.add('active');
            this.textModeBtn.classList.remove('active');
            this.voiceToggle.style.display = 'flex';
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
        
        if (isPressed && !this.isRecording && this.isConnected) {
            this.startRecording();
        } else if (!isPressed && this.isRecording) {
            this.stopRecording();
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
                thinking_mode: this.thinkingMode.value === 'true',
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
            const response = await fetch('/api/chat', {
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
            
            const wsUrl = "ws://localhost:8000/ws/simple-voice";
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus('connected', 'Connected - Hold space or mic button to talk');
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
            if (event.data instanceof Blob) {
                // Audio response
                const audioUrl = URL.createObjectURL(event.data);
                const audio = new Audio(audioUrl);
                
                // Add to conversation
                this.addMessage('assistant', null, false, { audioUrl });
                
                // Play audio
                audio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    
                    // Auto-start recording if enabled
                    if (this.autoStartEnabled && this.isConnected && !this.isRecording) {
                        setTimeout(() => {
                            this.startRecording();
                        }, 500);
                    }
                };
                
                await audio.play();
                
            } else {
                // Text message
                const data = JSON.parse(event.data);
                
                if (data.type === 'transcript') {
                    this.showVoiceStatus(`Transcript: ${data.text}`);
                } else if (data.type === 'response') {
                    this.addMessage('assistant', data.text);
                } else if (data.type === 'error') {
                    console.error('Voice server error:', data.message);
                    this.addMessage('assistant', `❌ Voice Error: ${data.message}`);
                }
            }
        } catch (error) {
            console.error('Error handling voice message:', error);
        }
    }

    async startRecording() {
        if (!this.isConnected || this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
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
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.sendVoiceData();
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Setup silence detection
            if (this.autoStopEnabled && this.audioContext) {
                this.setupSilenceDetection(stream);
            }
            
            this.mediaRecorder.start(100);
            this.isRecording = true;
            
            // Update UI
            this.voiceToggle.classList.add('recording');
            this.voiceToggle.textContent = '🔴';
            this.showVoiceStatus('🎤 Recording... (release to send)');
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showVoiceStatus('❌ Microphone access denied');
        }
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
        this.showVoiceStatus('📤 Processing audio...');
        
        console.log('Recording stopped');
    }

    sendVoiceData() {
        if (this.audioChunks.length === 0 || !this.ws) return;
        
        this.startConversation();
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Send configuration
        const config = {
            type: 'config',
            language: this.language.value,
            voice: this.voice.value,
            thinking_mode: this.thinkingMode.value
        };
        
        this.ws.send(JSON.stringify(config));
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
