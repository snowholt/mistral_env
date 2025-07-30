/**
 * WebSocket Voice Conversation Manager
 * ===================================
 * 
 * This module handles real-time voice conversation using WebSocket connection
 * instead of REST API calls for better performance and user experience.
 */

class WebSocketVoiceManager {
    constructor(beautyAIChat) {
        this.beautyAIChat = beautyAIChat;
        this.websocket = null;
        this.isConnected = false;
        this.isConnecting = false;
        this.connectionRetries = 0;
        this.maxRetries = 3;
        this.reconnectDelay = 1000;
        
        // WebSocket settings - Use WSS for HTTPS pages, WS for HTTP
        this.wsUrl = this.getWebSocketUrl();
        this.connectionTimeout = 10000; // 10 seconds
        this.heartbeatInterval = 30000; // 30 seconds
        this.heartbeatTimer = null;
        
        // Conversation state
        this.sessionId = null;
        this.conversationHistory = [];
        this.pendingAudioChunks = [];
        
        // Event handlers storage
        this.eventHandlers = {
            onConnectionEstablished: [],
            onVoiceResponse: [],
            onError: [],
            onConnectionLost: []
        };
    }

    /**
     * Validate WebSocket URL and SSL configuration
     */
    validateWebSocketConnection() {
        const wsUrl = this.getWebSocketUrl();
        const isSecure = wsUrl.startsWith('wss://');
        const currentProtocol = window.location.protocol;
        
        console.log(`🔍 WebSocket validation:`);
        console.log(`  - URL: ${wsUrl}`);
        console.log(`  - Is Secure (WSS): ${isSecure}`);
        console.log(`  - Page Protocol: ${currentProtocol}`);
        
        // Check for mixed content issues
        if (currentProtocol === 'https:' && !isSecure) {
            console.warn('⚠️ Mixed content warning: HTTPS page trying to connect to WS (not WSS)');
            return {
                valid: false,
                error: 'Mixed content: HTTPS page requires WSS connection',
                suggestion: 'Update WebSocket URL to use wss:// instead of ws://'
            };
        }
        
        // Check for production environment
        const hostname = window.location.hostname;
        if ((hostname.includes('gmai.sa') || hostname.includes('api.gmai.sa')) && !isSecure) {
            console.warn('⚠️ Production environment should use WSS');
            return {
                valid: false,
                error: 'Production environment must use secure WebSocket (WSS)',
                suggestion: 'Configure nginx to proxy WSS connections to WebSocket server'
            };
        }
        
        return {
            valid: true,
            url: wsUrl,
            secure: isSecure
        };
    }

    /**
     * Get the appropriate WebSocket URL based on current page protocol
     * Ensures WSS is used for production environments (api.gmai.sa)
     */
    getWebSocketUrl() {
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        
        console.log(`🔍 Determining WebSocket URL - hostname: ${hostname}, protocol: ${protocol}`);
        
        // For localhost development - always use WS
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            console.log('🏠 Using localhost development WebSocket');
            return 'ws://localhost:8000/api/v1/ws/voice-conversation';
        }
        
        // For production environments - FORCE WSS for security
        if (hostname === 'dev.gmai.sa' || hostname === 'api.gmai.sa' || hostname.includes('gmai.sa')) {
            console.log('🔒 Using production WSS WebSocket for api.gmai.sa');
            return 'wss://api.gmai.sa/api/v1/ws/voice-conversation';
        }
        
        // Default fallback - detect protocol but prefer WSS for production
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
        const defaultUrl = `${wsProtocol}//api.gmai.sa/api/v1/ws/voice-conversation`;
        
        console.log(`🌐 Default WebSocket URL: ${defaultUrl}`);
        return defaultUrl;
    }

    /**
     * Initialize WebSocket connection for voice conversation
     */
    async connect(options = {}) {
        if (this.isConnected || this.isConnecting) {
            console.log('WebSocket already connected or connecting');
            return true;
        }

        // Validate connection first
        const validation = this.validateWebSocketConnection();
        if (!validation.valid) {
            console.error('❌ WebSocket validation failed:', validation.error);
            this.triggerEvent('onError', {
                error: validation.error,
                details: validation.suggestion,
                type: 'validation'
            });
            return false;
        }

        this.isConnecting = true;
        
        try {
            // Build WebSocket URL with parameters
            const params = new URLSearchParams({
                input_language: options.inputLanguage || 'auto',
                output_language: options.outputLanguage || 'auto',
                speaker_voice: options.speakerVoice || 'female',
                preset: options.preset || 'qwen_optimized',
                session_id: options.sessionId || this.generateSessionId(),
                chat_model_name: options.chatModel || 'qwen3-unsloth-q4ks',
                stt_model_name: options.sttModel || 'whisper-turbo-arabic',
                tts_model_name: options.ttsModel || 'edge-tts',
                emotion: options.emotion || 'neutral',
                speech_speed: options.speechSpeed || '1.0'
            });

            const wsUrlWithParams = `${this.wsUrl}?${params.toString()}`;
            console.log('🔗 Connecting to WebSocket:', wsUrlWithParams);

            // Create WebSocket connection
            this.websocket = new WebSocket(wsUrlWithParams);
            
            // Set up event handlers
            this.setupWebSocketHandlers();
            
            // Wait for connection with timeout
            return await this.waitForConnection();
            
        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.isConnecting = false;
            this.handleConnectionError(error);
            return false;
        }
    }

    /**
     * Set up WebSocket event handlers
     */
    setupWebSocketHandlers() {
        this.websocket.onopen = () => {
            console.log('✅ WebSocket connection opened');
        };

        this.websocket.onmessage = (event) => {
            this.handleWebSocketMessage(event);
        };

        this.websocket.onclose = (event) => {
            console.log('🔌 WebSocket connection closed:', event.code, event.reason);
            this.handleConnectionClosed(event);
        };

        this.websocket.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.handleConnectionError(error);
        };
    }

    /**
     * Wait for WebSocket connection to be established
     */
    async waitForConnection() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Connection timeout'));
            }, this.connectionTimeout);

            const checkConnection = () => {
                if (this.websocket.readyState === WebSocket.OPEN) {
                    clearTimeout(timeout);
                    // Wait for connection confirmation message
                    const messageHandler = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'connection_established') {
                                this.websocket.removeEventListener('message', messageHandler);
                                this.handleConnectionEstablished(data);
                                resolve(true);
                            }
                        } catch (e) {
                            console.warn('Failed to parse connection message:', e);
                        }
                    };
                    
                    this.websocket.addEventListener('message', messageHandler);
                } else if (this.websocket.readyState === WebSocket.CLOSED) {
                    clearTimeout(timeout);
                    reject(new Error('Connection failed'));
                } else {
                    // Still connecting, check again
                    setTimeout(checkConnection, 100);
                }
            };

            checkConnection();
        });
    }

    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('📨 WebSocket message received:', data.type);

            switch (data.type) {
                case 'connection_established':
                    this.handleConnectionEstablished(data);
                    break;
                
                case 'voice_response':
                    this.handleVoiceResponse(data);
                    break;
                
                case 'processing_started':
                    this.handleProcessingStarted(data);
                    break;
                
                case 'error':
                    this.handleServerError(data);
                    break;
                
                case 'pong':
                    this.handlePong(data);
                    break;
                
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    /**
     * Handle connection established
     */
    handleConnectionEstablished(data) {
        console.log('🎉 WebSocket voice conversation connected!');
        this.isConnected = true;
        this.isConnecting = false;
        this.connectionRetries = 0;
        
        // Store session info
        this.sessionId = data.session_id;
        this.connectionId = data.connection_id;
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Notify listeners
        this.triggerEvent('onConnectionEstablished', {
            sessionId: this.sessionId,
            connectionId: this.connectionId,
            message: data.message
        });
    }

    /**
     * Handle voice response from server
     */
    handleVoiceResponse(data) {
        console.log('🎤 Voice response received');
        
        // Update conversation history
        if (data.transcription) {
            this.conversationHistory.push({
                role: 'user',
                content: data.transcription,
                timestamp: Date.now()
            });
        }
        
        if (data.response_text) {
            this.conversationHistory.push({
                role: 'assistant',
                content: data.response_text,
                timestamp: Date.now()
            });
        }
        
        // Notify listeners with complete response data
        this.triggerEvent('onVoiceResponse', {
            transcription: data.transcription,
            responseText: data.response_text,
            audioBase64: data.audio_base64,
            audioFormat: data.audio_format || 'wav',
            processingTime: data.processing_time_ms,
            sessionId: data.session_id,
            success: data.success
        });
    }

    /**
     * Handle processing started notification
     */
    handleProcessingStarted(data) {
        console.log('⚙️ Processing started on server');
        // Could show processing indicator in UI
    }

    /**
     * Handle server errors
     */
    handleServerError(data) {
        console.error('❌ Server error:', data.error);
        this.triggerEvent('onError', {
            error: data.error,
            code: data.code,
            details: data.details
        });
    }

    /**
     * Handle pong response
     */
    handlePong(data) {
        console.log('🏓 Pong received');
        // Could measure latency here if needed
    }

    /**
     * Send audio data to server
     */
    async sendAudioData(audioBlob) {
        if (!this.isConnected) {
            throw new Error('WebSocket not connected');
        }

        try {
            console.log(`📤 Sending audio data: ${audioBlob.size} bytes`);
            
            // Send binary audio data directly
            this.websocket.send(audioBlob);
            
            return true;
        } catch (error) {
            console.error('Failed to send audio data:', error);
            throw error;
        }
    }

    /**
     * Send control message
     */
    sendControlMessage(type, data = {}) {
        if (!this.isConnected) {
            console.warn('Cannot send control message: WebSocket not connected');
            return false;
        }

        try {
            const message = {
                type: type,
                timestamp: Date.now(),
                ...data
            };
            
            this.websocket.send(JSON.stringify(message));
            return true;
        } catch (error) {
            console.error('Failed to send control message:', error);
            return false;
        }
    }

    /**
     * Send ping to keep connection alive
     */
    sendPing() {
        return this.sendControlMessage('ping', {
            test: true
        });
    }

    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }
        
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this.sendPing();
            }
        }, this.heartbeatInterval);
    }

    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    /**
     * Handle connection closed
     */
    handleConnectionClosed(event) {
        this.isConnected = false;
        this.isConnecting = false;
        this.stopHeartbeat();
        
        console.log(`🔌 WebSocket connection closed: code=${event.code}, reason="${event.reason}", wasClean=${event.wasClean}`);
        
        // Determine close reason for better user feedback
        let closeReason = event.reason || 'Unknown reason';
        let shouldReconnect = false;
        
        switch (event.code) {
            case 1000: // Normal closure
                closeReason = 'Connection closed normally';
                break;
            case 1001: // Going away
                closeReason = 'Server going away';
                shouldReconnect = true;
                break;
            case 1002: // Protocol error
                closeReason = 'Protocol error';
                break;
            case 1003: // Unsupported data
                closeReason = 'Unsupported data type';
                break;
            case 1006: // Abnormal closure
                closeReason = 'Connection lost unexpectedly';
                shouldReconnect = true;
                break;
            case 1011: // Server error
                closeReason = 'Server error';
                shouldReconnect = true;
                break;
            case 1015: // TLS handshake failure
                closeReason = 'TLS/SSL handshake failed - check WSS configuration';
                break;
            default:
                if (event.code >= 3000) {
                    closeReason = 'Application-specific error';
                    shouldReconnect = true;
                }
        }
        
        // Trigger connection lost event
        this.triggerEvent('onConnectionLost', {
            code: event.code,
            reason: closeReason,
            wasClean: event.wasClean,
            shouldReconnect: shouldReconnect
        });
        
        // Auto-reconnect if appropriate and retries available
        if ((shouldReconnect || !event.wasClean) && this.connectionRetries < this.maxRetries) {
            console.log(`🔄 Attempting reconnection (${this.connectionRetries + 1}/${this.maxRetries}) - ${closeReason}`);
            this.connectionRetries++;
            
            setTimeout(() => {
                this.reconnect();
            }, this.reconnectDelay * this.connectionRetries);
        } else if (this.connectionRetries >= this.maxRetries) {
            console.warn(`⚠️ Max reconnection attempts (${this.maxRetries}) reached`);
            this.triggerEvent('onError', {
                error: 'Max reconnection attempts reached',
                details: 'Unable to maintain WebSocket connection',
                type: 'connection_exhausted'
            });
        }
    }

    /**
     * Handle connection errors
     */
    handleConnectionError(error) {
        this.isConnected = false;
        this.isConnecting = false;
        this.stopHeartbeat();
        
        console.error('🚨 WebSocket connection error:', error);
        
        // Detailed error handling
        let errorMessage = 'Connection error';
        let errorDetails = '';
        
        if (error.type === 'error' && this.websocket) {
            if (this.websocket.readyState === WebSocket.CLOSED) {
                errorMessage = 'Connection failed to establish';
                errorDetails = 'Server may be down or WebSocket endpoint unavailable';
            } else {
                errorMessage = 'WebSocket error occurred';
                errorDetails = 'Check network connection and server status';
            }
        } else if (error.message) {
            errorMessage = error.message;
            if (error.message.includes('timeout')) {
                errorDetails = 'Connection timeout - server may be slow or unreachable';
            } else if (error.message.includes('refused')) {
                errorDetails = 'Connection refused - check if server is running';
            } else if (error.message.includes('forbidden')) {
                errorDetails = 'Access forbidden - check authentication or CORS settings';
            }
        }
        
        console.log(`🔍 Error details: ${errorMessage} - ${errorDetails}`);
        
        this.triggerEvent('onError', {
            error: errorMessage,
            details: errorDetails,
            type: 'connection',
            canRetry: this.connectionRetries < this.maxRetries
        });
    }

    /**
     * Reconnect to WebSocket
     */
    async reconnect() {
        if (this.isConnected || this.isConnecting) {
            return;
        }
        
        console.log('🔄 Reconnecting to WebSocket...');
        
        // Use last known good configuration
        return await this.connect({
            sessionId: this.sessionId // Keep same session if possible
        });
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        console.log('🔌 Disconnecting WebSocket');
        
        this.stopHeartbeat();
        
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnect');
            this.websocket = null;
        }
        
        this.isConnected = false;
        this.isConnecting = false;
        this.sessionId = null;
        this.connectionId = null;
    }

    /**
     * Check if WebSocket is connected and ready
     */
    isReady() {
        return this.isConnected && this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }

    /**
     * Get current connection status
     */
    getConnectionStatus() {
        if (!this.websocket) {
            return 'disconnected';
        }
        
        switch (this.websocket.readyState) {
            case WebSocket.CONNECTING:
                return 'connecting';
            case WebSocket.OPEN:
                return this.isConnected ? 'connected' : 'authenticating';
            case WebSocket.CLOSING:
                return 'closing';
            case WebSocket.CLOSED:
                return 'closed';
            default:
                return 'unknown';
        }
    }

    /**
     * Add event listener
     */
    addEventListener(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }

    /**
     * Remove event listener
     */
    removeEventListener(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }

    /**
     * Trigger event handlers
     */
    triggerEvent(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return 'ws_session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Get conversation history
     */
    getConversationHistory() {
        return [...this.conversationHistory];
    }

    /**
     * Clear conversation history
     */
    clearConversationHistory() {
        this.conversationHistory = [];
    }

    /**
     * Get current session info
     */
    getSessionInfo() {
        return {
            sessionId: this.sessionId,
            connectionId: this.connectionId,
            isConnected: this.isConnected,
            status: this.getConnectionStatus(),
            messageCount: this.conversationHistory.length
        };
    }

    /**
     * Update voice settings (for future messages)
     */
    updateVoiceSettings(settings) {
        // Send update to server
        return this.sendControlMessage('update_settings', {
            voice_settings: settings
        });
    }
}

// Implement WebSocket for voice-to-voice conversation
const voiceManager = new WebSocketVoiceManager();

// Connect to WebSocket on page load
window.onload = async () => {
    const options = {
        inputLanguage: 'ar',
        outputLanguage: 'ar',
        speakerVoice: 'female'
    };
    const connected = await voiceManager.connect(options);
    if (connected) {
        console.log('Connected to WebSocket for voice conversation');
    } else {
        console.error('Failed to connect to WebSocket');
    }
};

// Function to send audio data
async function sendAudio(audioBlob) {
    try {
        await voiceManager.sendAudioData(audioBlob);
        console.log('Audio data sent successfully');
    } catch (error) {
        console.error('Error sending audio data:', error);
    }
}

// Add event listeners for voice responses
voiceManager.addEventListener('onVoiceResponse', (data) => {
    console.log('Voice response received:', data);
    // Handle the response (e.g., play audio)
});

voiceManager.addEventListener('onError', (error) => {
    console.error('Error from WebSocket:', error);
});

// Example usage: Call sendAudio with an audio Blob when ready
// sendAudio(audioBlob); // Uncomment and use actual audio blob

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketVoiceManager;
} else {
    window.WebSocketVoiceManager = WebSocketVoiceManager;
}
