/**
 * BeautyAI Chat Interface - Text and Voice Chat
 * Supports both text chat and real-time voice-to-voice conversation
 */

class BeautyAIChat {
    constructor() {
        // WebSocket for voice chat
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isConnected = false;
        this.isRecording = false;
        
        // Auto detection settings
        this.autoStartEnabled = true;
        this.autoStopEnabled = true;
        this.silenceTimeout = null;
        this.silenceThreshold = 3000; // 3 seconds
        this.audioContext = null;
        this.analyser = null;
        this.silenceDetectionActive = false;
        
        // Chat mode
        this.currentMode = 'text'; // 'text' or 'voice'
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupAudioContext();
    }

    initializeElements() {
        // Mode toggle buttons
        this.textChatBtn = document.getElementById('textChatBtn');
        this.voiceChatBtn = document.getElementById('voiceChatBtn');
        
        // Text chat elements
        this.textChatSection = document.getElementById('textChatSection');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // Voice chat elements
        this.voiceChatSection = document.getElementById('voiceChatSection');
        this.connectBtn = document.getElementById('connectBtn');
        this.recordBtn = document.getElementById('recordBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        // Controls
        this.languageSelect = document.getElementById('language');
        this.voiceSelect = document.getElementById('voice');
        this.thinkingSelect = document.getElementById('thinking');
        
        // Auto detection settings
        this.autoStartCheckbox = document.getElementById('autoStart');
        this.autoStopCheckbox = document.getElementById('autoStop');
        this.silenceSlider = document.getElementById('silenceThreshold');
        this.silenceValue = document.getElementById('silenceValue');
        
        // Display elements
        this.statusElement = document.getElementById('status');
        this.transcriptElement = document.getElementById('transcript');
        this.conversationElement = document.getElementById('conversation');
    }

    setupEventListeners() {
        // Mode toggle
        this.textChatBtn.addEventListener('click', () => this.switchMode('text'));
        this.voiceChatBtn.addEventListener('click', () => this.switchMode('voice'));
        
        // Text chat events
        this.sendBtn.addEventListener('click', () => this.sendTextMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTextMessage();
            }
        });
        
        // Voice chat events
        this.connectBtn.addEventListener('click', () => this.toggleConnection());
        this.recordBtn.addEventListener('mousedown', () => this.startRecording());
        this.recordBtn.addEventListener('mouseup', () => this.stopRecording());
        this.recordBtn.addEventListener('mouseleave', () => this.stopRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        
        // Auto detection settings
        this.autoStartCheckbox.addEventListener('change', (e) => {
            this.autoStartEnabled = e.target.checked;
        });
        
        this.autoStopCheckbox.addEventListener('change', (e) => {
            this.autoStopEnabled = e.target.checked;
        });
        
        this.silenceSlider.addEventListener('input', (e) => {
            this.silenceThreshold = parseFloat(e.target.value) * 1000;
            this.silenceValue.textContent = `${e.target.value}s`;
        });

        // Voice selection auto-update
        this.languageSelect.addEventListener('change', () => this.updateVoiceOptions());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat && this.currentMode === 'voice') {
                e.preventDefault();
                if (this.isConnected && !this.isRecording) {
                    this.startRecording();
                }
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && this.currentMode === 'voice') {
                e.preventDefault();
                if (this.isRecording) {
                    this.stopRecording();
                }
            }
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        if (mode === 'text') {
            this.textChatBtn.classList.add('active');
            this.voiceChatBtn.classList.remove('active');
            this.textChatSection.classList.remove('hidden');
            this.voiceChatSection.classList.add('hidden');
            this.updateStatus('Text chat mode - Type your message below', 'connected');
            
            // Disconnect voice if connected
            if (this.isConnected) {
                this.disconnect();
            }
        } else {
            this.voiceChatBtn.classList.add('active');
            this.textChatBtn.classList.remove('active');
            this.voiceChatSection.classList.remove('hidden');
            this.textChatSection.classList.add('hidden');
            this.updateStatus('Voice chat mode - Click Connect to start', 'disconnected');
        }
    }

    async sendTextMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Disable send button
        this.sendBtn.disabled = true;
        this.sendBtn.textContent = 'Sending...';
        
        // Add user message to conversation
        this.addMessageToConversation('user', message);
        
        // Clear input
        this.messageInput.value = '';
        
        try {
            // Prepare request payload
            const payload = {
                model_name: 'qwen3-unsloth-q4ks', // Default model
                message: message,
                thinking_mode: this.thinkingSelect.value === 'think',
                disable_content_filter: true,
                generation_config: {
                    max_tokens: 2048,
                    temperature: 0.3,
                    top_p: 0.95,
                    repetition_penalty: 1.1
                }
            };
            
            // Send to backend API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Add AI response to conversation
                this.addMessageToConversation('ai', result.response || result.final_content);
            } else {
                // Add error message
                this.addMessageToConversation('ai', `Error: ${result.error}`);
            }
            
        } catch (error) {
            console.error('Text chat error:', error);
            this.addMessageToConversation('ai', `Connection error: ${error.message}`);
        } finally {
            // Re-enable send button
            this.sendBtn.disabled = false;
            this.sendBtn.textContent = 'Send';
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
        const language = this.languageSelect.value;
        const voiceSelect = this.voiceSelect;
        
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
        
        // If auto, show both
        if (language === 'auto') {
            // Default to Arabic female for auto-detect
            voiceSelect.value = 'ar-SA-ZariyahNeural';
        }
    }

    toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            this.connect();
        }
    }

    connect() {
        try {
            // Connect directly to backend WebSocket
            const wsUrl = "ws://localhost:8000/ws/simple-voice";
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('Connected - Ready to chat!', 'connected');
                this.connectBtn.textContent = 'Disconnect';
                this.connectBtn.className = 'voice-button stop-btn';
                this.recordBtn.disabled = false;
                
                console.log('WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('Disconnected', 'disconnected');
                this.connectBtn.textContent = 'Connect';
                this.connectBtn.className = 'voice-button record-btn';
                this.recordBtn.disabled = true;
                
                // Stop recording if active
                if (this.isRecording) {
                    this.stopRecording();
                }
                
                console.log('WebSocket disconnected');
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('Connection error', 'disconnected');
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this.updateStatus('Failed to connect', 'disconnected');
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
        
        // Stop recording if active
        if (this.isRecording) {
            this.stopRecording();
        }
    }

    async handleWebSocketMessage(event) {
        try {
            if (event.data instanceof Blob) {
                // Audio response - play it
                const audioUrl = URL.createObjectURL(event.data);
                const audio = new Audio(audioUrl);
                
                // Add to conversation
                this.addMessageToConversation('ai', null, audioUrl);
                
                // Play the audio
                audio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    
                    // Auto-start recording if enabled
                    if (this.autoStartEnabled && this.isConnected && !this.isRecording) {
                        setTimeout(() => {
                            this.startRecording();
                        }, 500); // Small delay before starting
                    }
                };
                
                await audio.play();
                
            } else {
                // Text message
                const data = JSON.parse(event.data);
                
                if (data.type === 'transcript') {
                    this.transcriptElement.textContent = data.text || '';
                } else if (data.type === 'response') {
                    // Add text response to conversation
                    this.addMessageToConversation('ai', data.text);
                } else if (data.type === 'error') {
                    console.error('Server error:', data.message);
                    this.updateStatus(`Error: ${data.message}`, 'disconnected');
                } else if (data.type === 'status') {
                    console.log('Server status:', data.message);
                }
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
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
            
            // Setup for WebM recording (browser compatible)
            const options = {
                mimeType: 'audio/webm;codecs=opus'
            };
            
            // Fallback to other formats if WebM not supported
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                    options.mimeType = 'audio/ogg;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options.mimeType = 'audio/mp4';
                } else {
                    options.mimeType = 'audio/webm'; // Default fallback
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
                this.sendAudioData();
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Setup silence detection if enabled
            if (this.autoStopEnabled && this.audioContext) {
                this.setupSilenceDetection(stream);
            }
            
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            
            this.updateStatus('🎤 Recording... (release to send)', 'recording');
            this.recordBtn.textContent = '🔴 Recording...';
            this.recordBtn.classList.add('recording');
            this.stopBtn.classList.remove('hidden');
            this.stopBtn.disabled = false;
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.updateStatus('Microphone access denied', 'disconnected');
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
        
        // Calculate average volume
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const isSilent = average < 10; // Adjust threshold as needed
        
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
        
        // Continue monitoring
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
        
        this.updateStatus('Processing...', 'connected');
        this.recordBtn.textContent = '🎤 Hold to Talk';
        this.recordBtn.classList.remove('recording');
        this.stopBtn.classList.add('hidden');
        this.stopBtn.disabled = true;
        
        console.log('Recording stopped');
    }

    sendAudioData() {
        if (this.audioChunks.length === 0 || !this.ws) return;
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Send parameters first
        const params = {
            type: 'config',
            language: this.languageSelect.value,
            voice: this.voiceSelect.value,
            thinking_mode: this.thinkingSelect.value
        };
        
        this.ws.send(JSON.stringify(params));
        
        // Then send audio data
        this.ws.send(audioBlob);
        
        // Add user message to conversation
        this.addMessageToConversation('user', 'Voice message sent');
        
        console.log('Audio sent, size:', audioBlob.size, 'bytes');
    }

    addMessageToConversation(sender, text, audioUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        let content = `<strong>${sender === 'user' ? 'You' : 'BeautyAI'}:</strong>`;
        
        if (text) {
            content += ` ${text}`;
        }
        
        if (audioUrl) {
            content += `<div class="message-audio"><audio controls><source src="${audioUrl}" type="audio/webm"></audio></div>`;
        }
        
        messageDiv.innerHTML = content;
        this.conversationElement.appendChild(messageDiv);
        
        // Scroll to bottom
        this.conversationElement.scrollTop = this.conversationElement.scrollHeight;
    }

    updateStatus(message, type) {
        this.statusElement.textContent = message;
        this.statusElement.className = `status ${type}`;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.beautyAIChat = new BeautyAIChat();
    console.log('BeautyAI Chat initialized with text and voice support');
});
