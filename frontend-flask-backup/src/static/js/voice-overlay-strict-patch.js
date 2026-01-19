/**
 * STRICT Voice Chat Fix - Overlay Only
 * This replaces the voice overlay functionality with a race-condition-free implementation
 */

// Patch the existing BeautyAIChat class with strict voice state management
(function() {
    'use strict';
    
    // Wait for the main class to be available
    document.addEventListener('DOMContentLoaded', function() {
        if (window.beautyAIChat) {
            patchVoiceOverlay(window.beautyAIChat);
        }
    });
    
    function patchVoiceOverlay(chatInstance) {
        console.log('🔧 Applying STRICT voice overlay patch to prevent infinite loops');
        
        // STRICT STATE MANAGEMENT
        chatInstance.voiceState = 'IDLE'; // IDLE, RECORDING, PROCESSING, SPEAKING
        chatInstance.lastProcessedTurnId = null;
        chatInstance.strictAudioQueue = [];
        chatInstance.isStrictPlayingAudio = false;
        
        // Override the problematic methods with strict implementations
        
        // STRICT: Force stop all recording when not in RECORDING state
        chatInstance.forceStopAllRecording = function() {
            if (this.overlayMediaRecorder && this.overlayMediaRecorder.state === 'recording') {
                console.log('🛑 FORCE STOPPING overlay recording');
                this.overlayMediaRecorder.stop();
            }
            this.overlayRecording = false;
        };
        
        // STRICT: Set voice state with forced recording control
        chatInstance.setVoiceState = function(newState) {
            const oldState = this.voiceState;
            console.log(`🔄 VOICE STATE: ${oldState} → ${newState}`);
            this.voiceState = newState;
            
            // CRITICAL: Stop recording if not in RECORDING state
            if (newState !== 'RECORDING') {
                this.forceStopAllRecording();
            }
            
            // Update UI
            this.updateOverlayUIForState(newState);
        };
        
        // STRICT: Update UI based on state
        chatInstance.updateOverlayUIForState = function(state) {
            const statusElement = this.overlayConnectionStatus;
            const toggleButton = this.overlayVoiceToggle;
            const buttonSpan = toggleButton ? toggleButton.querySelector('span') : null;

            if (!statusElement || !toggleButton || !buttonSpan) return;

            switch (state) {
                case 'IDLE':
                    statusElement.className = 'status-indicator connected';
                    statusElement.innerHTML = '<span>●</span><span>Ready - Click mic to speak</span>';
                    toggleButton.classList.remove('recording');
                    buttonSpan.textContent = 'Click to Listen';
                    this.showOverlayVoiceStatus('');
                    break;
                case 'RECORDING':
                    statusElement.className = 'status-indicator connected';
                    statusElement.innerHTML = '<span>●</span><span>Listening...</span>';
                    toggleButton.classList.add('recording');
                    buttonSpan.textContent = 'Recording...';
                    this.showOverlayVoiceStatus('🎤 Listening... Speak now');
                    break;
                case 'PROCESSING':
                    statusElement.className = 'status-indicator processing';
                    statusElement.innerHTML = '<span>●</span><span>Processing...</span>';
                    toggleButton.classList.remove('recording');
                    buttonSpan.textContent = 'Processing';
                    this.showOverlayVoiceStatus('🔄 Processing your speech...');
                    break;
                case 'SPEAKING':
                    statusElement.className = 'status-indicator connected';
                    statusElement.innerHTML = '<span>●</span><span>AI is Responding</span>';
                    toggleButton.classList.remove('recording');
                    buttonSpan.textContent = 'AI Speaking';
                    this.showOverlayVoiceStatus('🔊 Playing AI response...');
                    break;
            }
        };
        
        // STRICT: Duplicate detection using turn_id only
        chatInstance.isStrictDuplicate = function(data) {
            if (!data || !data.turn_id) {
                console.warn('🚫 Response missing turn_id - rejecting');
                return true;
            }
            
            if (this.lastProcessedTurnId === data.turn_id) {
                console.warn(`🚫 DUPLICATE TURN_ID: ${data.turn_id} - BLOCKING`);
                return true;
            }
            
            this.lastProcessedTurnId = data.turn_id;
            console.log(`✅ UNIQUE TURN: ${data.turn_id}`);
            return false;
        };
        
        // STRICT: Audio chunk sending only when in RECORDING state
        chatInstance.sendOverlayAudioChunk = function(audioChunk) {
            if (this.voiceState !== 'RECORDING') {
                console.log('🚫 BLOCKED chunk - state:', this.voiceState);
                return;
            }

            if (!this.overlayWebSocket || this.overlayWebSocket.readyState !== WebSocket.OPEN) {
                return;
            }

            if (audioChunk.size < 10) {
                return;
            }

            this.overlayWebSocket.send(audioChunk);
            console.log('📤 Chunk sent:', audioChunk.size, 'bytes');
        };
        
        // STRICT: Audio playback with state control
        chatInstance.strictPlayAudio = async function(audioBlob) {
            if (this.isStrictPlayingAudio) {
                console.log('🎵 Queueing audio');
                this.strictAudioQueue.push(audioBlob);
                return;
            }

            this.isStrictPlayingAudio = true;
            this.setVoiceState('SPEAKING');

            try {
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);

                audio.onended = () => {
                    console.log('🔊 Audio ended');
                    URL.revokeObjectURL(audioUrl);
                    this.isStrictPlayingAudio = false;

                    if (this.strictAudioQueue.length > 0) {
                        const next = this.strictAudioQueue.shift();
                        this.strictPlayAudio(next);
                    } else {
                        this.handleStrictAudioEnd();
                    }
                };

                audio.onerror = () => {
                    console.error('❌ Audio error');
                    URL.revokeObjectURL(audioUrl);
                    this.isStrictPlayingAudio = false;
                    this.setVoiceState('IDLE');
                };

                console.log('🎵 Playing audio...');
                await audio.play();

            } catch (error) {
                console.error('❌ Audio play failed:', error);
                this.isStrictPlayingAudio = false;
                this.setVoiceState('IDLE');
            }
        };
        
        // STRICT: Handle audio end
        chatInstance.handleStrictAudioEnd = function() {
            console.log('✅ All audio finished');
            this.setVoiceState('IDLE');

            if (this.overlayAutoStartEnabled && this.overlayConnected) {
                console.log('🔄 Auto-restart after 1 second');
                setTimeout(() => {
                    if (this.voiceState === 'IDLE' && !this.isStrictPlayingAudio) {
                        this.startOverlayRecording();
                    }
                }, 1000);
            }
        };
        
        // STRICT: Override voice toggle
        chatInstance.handleOverlayVoiceToggle = function() {
            if (!this.overlayConnected) return;

            if (this.voiceState === 'IDLE') {
                this.startOverlayRecording();
            } else if (this.voiceState === 'RECORDING') {
                this.stopOverlayRecording();
            }
        };
        
        // STRICT: Override recording start
        chatInstance.startOverlayRecording = async function() {
            if (this.voiceState !== 'IDLE') return;
            if (this.isStrictPlayingAudio) {
                console.warn('❌ Cannot record while AI speaking');
                return;
            }

            this.setVoiceState('RECORDING');

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });

                const options = { mimeType: 'audio/webm;codecs=opus' };
                this.overlayMediaRecorder = new MediaRecorder(stream, options);

                this.overlayMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.sendOverlayAudioChunk(event.data);
                    }
                };

                this.overlayMediaRecorder.onstop = () => {
                    stream.getTracks().forEach(track => track.stop());
                };

                this.overlayMediaRecorder.start(100);
                console.log('🎙️ Recording started');

            } catch (error) {
                console.error('❌ Recording failed:', error);
                this.setVoiceState('IDLE');
            }
        };
        
        // STRICT: Override recording stop
        chatInstance.stopOverlayRecording = function() {
            if (this.voiceState !== 'RECORDING') return;

            if (this.overlayMediaRecorder && this.overlayMediaRecorder.state !== 'inactive') {
                this.overlayMediaRecorder.stop();
            }

            this.setVoiceState('PROCESSING');
            console.log('🛑 Recording stopped');
        };
        
        // STRICT: Override message handler
        const originalHandleOverlayVoiceMessage = chatInstance.handleOverlayVoiceMessage;
        chatInstance.handleOverlayVoiceMessage = async function(event) {
            try {
                const data = JSON.parse(event.data);
                console.log('📨 Received:', data.type);

                switch (data.type) {
                    case 'connection_established':
                        this.setVoiceState('IDLE');
                        break;

                    case 'speech_end':
                        if (this.voiceState === 'RECORDING') {
                            this.stopOverlayRecording();
                        }
                        break;

                    case 'voice_response':
                        // CRITICAL: Check for duplicates
                        if (this.isStrictDuplicate(data)) return;

                        if (data.success && data.audio_base64) {
                            const audioBlob = this.base64ToBlob(data.audio_base64, 'audio/wav');
                            this.addOverlayMessage('assistant', data.response_text, { transcription: data.transcription });
                            await this.strictPlayAudio(audioBlob);
                        } else {
                            this.addOverlayMessage('assistant', `❌ Error: ${data.message || 'Unknown error'}`);
                            this.setVoiceState('IDLE');
                        }
                        break;

                    default:
                        // Handle other message types with original handler if they exist
                        if (originalHandleOverlayVoiceMessage) {
                            originalHandleOverlayVoiceMessage.call(this, event);
                        }
                        break;
                }

            } catch (error) {
                console.error('❌ Message handling error:', error);
            }
        };
        
        // Initialize strict state
        chatInstance.setVoiceState('IDLE');
        
        console.log('✅ STRICT voice overlay patch applied successfully');
        console.log('🎯 This should eliminate the infinite Arabic phrase loop');
    }
})();
