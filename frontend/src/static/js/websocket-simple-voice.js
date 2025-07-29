/**
 * Simple WebSocket Voice Manager
 * ==============================
 * 
 * Lightweight WebSocket manager for simple voice conversations.
 * Optimized for speed (<2 seconds response time) with minimal parameters.
 */

class SimpleWebSocketVoiceManager {
    constructor(beautyAIChat) {
        this.beautyAIChat = beautyAIChat;
        this.websocket = null;
        this.isConnected = false;
        this.isConnecting = false;
        this.connectionRetries = 0;
        this.maxRetries = 3;
        this.reconnectDelay = 1000;
        
        // Simple WebSocket settings
        this.wsUrl = this.getWebSocketUrl();
        this.connectionTimeout = 10000; // 10 seconds
        this.heartbeatInterval = 30000; // 30 seconds
        this.heartbeatTimer = null;
        
        // Simple conversation state
        this.sessionId = null;
        this.connectionId = null;
        this.messageCount = 0;
        
        // Event handlers storage
        this.eventHandlers = {
            onConnectionEstablished: [],
            onVoiceResponse: [],
            onProcessingStarted: [],
            onError: [],
            onConnectionLost: []
        };
    }

    /**
     * Get the appropriate WebSocket URL for simple voice chat
     */
    getWebSocketUrl() {
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        
        console.log(`🔍 Simple Voice - Determining WebSocket URL - hostname: ${hostname}, protocol: ${protocol}`);
        
        // For localhost development
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            console.log('🏠 Using localhost development WebSocket for simple voice');
            return 'ws://localhost:8000/api/v1/ws/simple-voice-chat';
        }
        
        // For production environments - Use WSS
        if (hostname === 'dev.gmai.sa' || hostname === 'api.gmai.sa' || hostname.includes('gmai.sa')) {
            console.log('🔒 Using production WSS WebSocket for simple voice at api.gmai.sa');
            return 'wss://api.gmai.sa/api/v1/ws/simple-voice-chat';
        }
        
        // Default fallback
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
        const defaultUrl = `${wsProtocol}//api.gmai.sa/api/v1/ws/simple-voice-chat`;
        
        console.log(`🌐 Default Simple Voice WebSocket URL: ${defaultUrl}`);
        return defaultUrl;
    }

    /**
     * Validate WebSocket connection
     */
    validateWebSocketConnection() {
        const wsUrl = this.getWebSocketUrl();
        const isSecure = wsUrl.startsWith('wss://');
        const currentProtocol = window.location.protocol;
        
        console.log(`🔍 Simple Voice WebSocket validation:`);
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
        
        return {
            valid: true,
            url: wsUrl,
            secure: isSecure
        };
    }

    /**
     * Initialize simple WebSocket connection
     */
    async connect(options = {}) {
        if (this.isConnected || this.isConnecting) {
            console.log('Simple Voice WebSocket already connected or connecting');
            return true;
        }

        // Validate connection first
        const validation = this.validateWebSocketConnection();
        if (!validation.valid) {
            console.error('❌ Simple Voice WebSocket validation failed:', validation.error);
            this.triggerEvent('onError', {
                error: validation.error,
                details: validation.suggestion,
                type: 'validation'
            });
            return false;
        }

        this.isConnecting = true;
        
        try {
            // Build WebSocket URL with simple parameters only
            const params = new URLSearchParams({
                language: options.language || 'ar',  // ar or en only
                voice_type: options.voice_type || 'female',  // male or female
                session_id: options.sessionId || this.generateSessionId()
            });

            const wsUrlWithParams = `${this.wsUrl}?${params.toString()}`;
            console.log('🔗 Connecting to Simple Voice WebSocket:', wsUrlWithParams);

            // Create WebSocket connection
            this.websocket = new WebSocket(wsUrlWithParams);
            
            // Set up event handlers
            this.setupWebSocketHandlers();
            
            // Wait for connection with timeout
            return await this.waitForConnection();
            
        } catch (error) {
            console.error('❌ Simple Voice WebSocket connection failed:', error);
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
            console.log('✅ Simple Voice WebSocket connection opened');
        };

        this.websocket.onmessage = (event) => {
            this.handleWebSocketMessage(event);
        };

        this.websocket.onclose = (event) => {
            console.log('🔌 Simple Voice WebSocket connection closed:', event.code, event.reason);
            this.handleConnectionClosed(event);
        };

        this.websocket.onerror = (error) => {
            console.error('❌ Simple Voice WebSocket error:', error);
            this.handleConnectionError(error);
        };
    }

    /**
     * Wait for WebSocket connection to be established
     */
    async waitForConnection() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Simple Voice connection timeout'));
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
                            console.warn('Failed to parse simple voice connection message:', e);
                        }
                    };
                    
                    this.websocket.addEventListener('message', messageHandler);
                } else if (this.websocket.readyState === WebSocket.CLOSED) {
                    clearTimeout(timeout);
                    reject(new Error('Simple Voice connection failed'));
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
            console.log('📨 Simple Voice WebSocket message received:', data.type);

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
                    console.log('Unknown simple voice message type:', data.type);
            }
        } catch (error) {
            console.error('Failed to parse Simple Voice WebSocket message:', error);
        }
    }

    /**
     * Handle connection established
     */
    handleConnectionEstablished(data) {
        console.log('🎉 Simple Voice WebSocket connected! Target response time: <2 seconds');
        this.isConnected = true;
        this.isConnecting = false;
        this.connectionRetries = 0;
        
        // Store session info
        this.sessionId = data.session_id;
        this.connectionId = data.connection_id;
        this.messageCount = 0;
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Notify listeners
        this.triggerEvent('onConnectionEstablished', {
            sessionId: this.sessionId,
            connectionId: this.connectionId,
            message: data.message,
            config: data.config,
            mode: 'simple',
            targetResponseTime: '< 2 seconds'
        });
    }

    /**
     * Handle voice response from server
     */
    handleVoiceResponse(data) {
        console.log('🎤 Simple Voice response received in', data.response_time_ms, 'ms');
        
        this.messageCount++;
        
        // Notify listeners with complete response data
        this.triggerEvent('onVoiceResponse', {
            transcription: data.transcription,
            responseText: data.response_text,
            audioBase64: data.audio_base64,
            language: data.language,
            voiceType: data.voice_type,
            responseTimeMs: data.response_time_ms,
            sessionId: data.session_id,
            messageCount: data.message_count,
            success: data.success,
            mode: 'simple'
        });
    }

    /**
     * Handle processing started notification
     */
    handleProcessingStarted(data) {
        console.log('⚙️ Simple Voice processing started on server');
        this.triggerEvent('onProcessingStarted', {
            message: data.message,
            timestamp: data.timestamp
        });
    }

    /**
     * Handle server errors
     */
    handleServerError(data) {
        console.error('❌ Simple Voice server error:', data.error);
        this.triggerEvent('onError', {
            error: data.message,
            code: data.error_code,
            retrySupported: data.retry_suggested,
            responseTimeMs: data.response_time_ms,
            type: 'server'
        });
    }

    /**
     * Handle pong response
     */
    handlePong(data) {
        console.log('🏓 Simple Voice pong received');
    }

    /**
     * Send audio data to server
     */
    async sendAudioData(audioBlob) {
        if (!this.isConnected) {
            throw new Error('Simple Voice WebSocket not connected');
        }

        try {
            console.log(`📤 Sending audio data to Simple Voice: ${audioBlob.size} bytes`);
            
            // Send binary audio data directly
            this.websocket.send(audioBlob);
            
            return true;
        } catch (error) {
            console.error('Failed to send audio data to Simple Voice:', error);
            throw error;
        }
    }

    /**
     * Send control message
     */
    sendControlMessage(type, data = {}) {
        if (!this.isConnected) {
            console.warn('Cannot send control message: Simple Voice WebSocket not connected');
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
            console.error('Failed to send Simple Voice control message:', error);
            return false;
        }
    }

    /**
     * Send ping to keep connection alive
     */
    sendPing() {
        return this.sendControlMessage('ping');
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
        
        console.log(`🔌 Simple Voice WebSocket closed: code=${event.code}, reason="${event.reason}"`);
        
        let closeReason = event.reason || 'Unknown reason';
        let shouldReconnect = false;
        
        switch (event.code) {
            case 1000: // Normal closure
                closeReason = 'Simple voice conversation ended normally';
                break;
            case 1001: // Going away
                closeReason = 'Simple voice server going away';
                shouldReconnect = true;
                break;
            case 1006: // Abnormal closure
                closeReason = 'Simple voice connection lost unexpectedly';
                shouldReconnect = true;
                break;
            case 1011: // Server error
                closeReason = 'Simple voice server error';
                shouldReconnect = true;
                break;
            default:
                shouldReconnect = !event.wasClean;
        }
        
        this.triggerEvent('onConnectionLost', {
            code: event.code,
            reason: closeReason,
            wasClean: event.wasClean,
            shouldReconnect: shouldReconnect
        });
        
        // Auto-reconnect if appropriate
        if (shouldReconnect && this.connectionRetries < this.maxRetries) {
            console.log(`🔄 Reconnecting Simple Voice (${this.connectionRetries + 1}/${this.maxRetries})`);
            this.connectionRetries++;
            
            setTimeout(() => {
                this.reconnect();
            }, this.reconnectDelay * this.connectionRetries);
        }
    }

    /**
     * Handle connection errors
     */
    handleConnectionError(error) {
        this.isConnected = false;
        this.isConnecting = false;
        this.stopHeartbeat();
        
        console.error('🚨 Simple Voice WebSocket error:', error);
        
        let errorMessage = 'Simple voice connection error';
        let errorDetails = 'Check network connection and server status';
        
        if (error.message) {
            errorMessage = error.message;
            if (error.message.includes('timeout')) {
                errorDetails = 'Simple voice connection timeout - server may be slow';
            } else if (error.message.includes('refused')) {
                errorDetails = 'Simple voice connection refused - check server status';
            }
        }
        
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
        
        console.log('🔄 Reconnecting Simple Voice WebSocket...');
        return await this.connect({
            sessionId: this.sessionId
        });
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        console.log('🔌 Disconnecting Simple Voice WebSocket');
        
        this.stopHeartbeat();
        
        if (this.websocket) {
            this.websocket.close(1000, 'Simple voice client disconnect');
            this.websocket = null;
        }
        
        this.isConnected = false;
        this.isConnecting = false;
        this.sessionId = null;
        this.connectionId = null;
        this.messageCount = 0;
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
                    console.error(`Error in Simple Voice event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return 'simple_voice_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
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
            messageCount: this.messageCount,
            mode: 'simple'
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SimpleWebSocketVoiceManager;
} else {
    window.SimpleWebSocketVoiceManager = SimpleWebSocketVoiceManager;
}
