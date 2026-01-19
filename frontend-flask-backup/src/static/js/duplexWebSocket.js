/**
 * Duplex WebSocket Client for Full Duplex Voice Streaming
 * 
 * Handles bidirectional audio streaming with binary frame protocol:
 * - Uplink: Microphone audio chunks
 * - Downlink: TTS audio chunks  
 * - Control: JSON messages for state management
 * - Meta: Device info and configuration
 */

class DuplexWebSocket {
  constructor(options = {}) {
    this.url = options.url;
    this.ws = null;
    this.isConnected = false;
    
    // Message handling
    this.onTextMessage = options.onTextMessage || (() => {});
    this.onTTSChunk = options.onTTSChunk || (() => {});
    this.onControlMessage = options.onControlMessage || (() => {});
    this.onConnectionChange = options.onConnectionChange || (() => {});
    this.onError = options.onError || (() => {});
    
    // Protocol constants
    this.MessageType = {
      MIC_CHUNK: 0x01,
      TTS_CHUNK: 0x02,
      CONTROL: 0x03,
      META: 0x04
    };
    
    this.MessageFlags = {
      START: 0x01,
      END: 0x02,
      URGENT: 0x04,
      COMPRESSED: 0x08
    };
    
    // Sequence tracking
    this.micSequence = 0;
    this.expectedTTSSequence = 0;
    this.ttsChunkBuffer = new Map(); // Buffer for out-of-order chunks
    
    // Reconnection
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
    this.reconnectDelayMs = options.reconnectDelayMs || 1000;
    this.autoReconnect = options.autoReconnect !== false;
    
    // Metrics
    this.metrics = {
      micChunksSent: 0,
      ttsChunksReceived: 0,
      reconnections: 0,
      bytesReceived: 0,
      bytesSent: 0
    };
    
    this.debug = !!options.debug;
    this._log('Duplex WebSocket client created');
  }
  
  /**
   * Connect to WebSocket server
   */
  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return; // Already connected
    }
    
    return new Promise((resolve, reject) => {
      try {
        this._log('Connecting to:', this.url);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';
        
        this.ws.onopen = () => {
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this._log('Connected');
          this.onConnectionChange(true);
          resolve();
        };
        
        this.ws.onclose = (event) => {
          this.isConnected = false;
          this._log('Disconnected:', event.code, event.reason);
          this.onConnectionChange(false);
          
          if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this._scheduleReconnect();
          }
        };
        
        this.ws.onerror = (error) => {
          this._log('WebSocket error:', error);
          this.onError(error);
          reject(error);
        };
        
        this.ws.onmessage = (event) => {
          this._handleMessage(event);
        };
        
      } catch (error) {
        this._log('Connection failed:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Disconnect from server
   */
  disconnect() {
    this.autoReconnect = false;
    
    if (this.ws) {
      this.ws.close(1000, 'client_disconnect');
      this.ws = null;
    }
    
    this.isConnected = false;
    this._log('Disconnected by client');
  }
  
  /**
   * Send microphone audio chunk
   */
  sendMicChunk(audioData, flags = 0) {
    if (!this.isConnected || !this.ws) {
      this._log('Cannot send mic chunk - not connected');
      return false;
    }
    
    try {
      const frame = this._packBinaryFrame(
        this.MessageType.MIC_CHUNK,
        this.micSequence++,
        flags,
        audioData
      );
      
      this.ws.send(frame);
      this.metrics.micChunksSent++;
      this.metrics.bytesSent += frame.byteLength;
      
      return true;
      
    } catch (error) {
      this._log('Failed to send mic chunk:', error);
      this.onError(error);
      return false;
    }
  }
  
  /**
   * Send JSON control message
   */
  sendControlMessage(message) {
    if (!this.isConnected || !this.ws) {
      this._log('Cannot send control message - not connected');
      return false;
    }
    
    try {
      if (typeof message === 'object') {
        message = JSON.stringify(message);
      }
      
      this.ws.send(message);
      this._log('Sent control message:', message);
      
      return true;
      
    } catch (error) {
      this._log('Failed to send control message:', error);
      this.onError(error);
      return false;
    }
  }
  
  /**
   * Send device metadata
   */
  sendMetadata(metadata) {
    if (!this.isConnected || !this.ws) {
      this._log('Cannot send metadata - not connected');
      return false;
    }
    
    try {
      const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
      const frame = this._packBinaryFrame(
        this.MessageType.META,
        0, // Metadata doesn't use sequence numbers
        0, // No flags
        metadataBytes
      );
      
      this.ws.send(frame);
      this._log('Sent metadata:', metadata);
      
      return true;
      
    } catch (error) {
      this._log('Failed to send metadata:', error);
      this.onError(error);
      return false;
    }
  }
  
  /**
   * Handle incoming WebSocket message
   */
  _handleMessage(event) {
    this.metrics.bytesReceived += event.data.byteLength || event.data.length;
    
    if (typeof event.data === 'string') {
      // JSON text message
      try {
        const data = JSON.parse(event.data);
        this.onTextMessage(data);
      } catch (e) {
        this._log('Failed to parse JSON message:', event.data);
      }
    } else {
      // Binary message - unpack frame
      try {
        const { messageType, sequenceNumber, flags, timestamp, payload } = 
          this._unpackBinaryFrame(event.data);
        
        this._handleBinaryMessage(messageType, sequenceNumber, flags, timestamp, payload);
        
      } catch (error) {
        this._log('Failed to process binary message:', error);
      }
    }
  }
  
  /**
   * Handle binary message based on type
   */
  _handleBinaryMessage(messageType, sequenceNumber, flags, timestamp, payload) {
    switch (messageType) {
      case this.MessageType.TTS_CHUNK:
        this._handleTTSChunk(sequenceNumber, flags, timestamp, payload);
        break;
        
      case this.MessageType.CONTROL:
        try {
          const controlMessage = JSON.parse(new TextDecoder().decode(payload));
          this.onControlMessage(controlMessage);
        } catch (e) {
          this._log('Failed to decode control message');
        }
        break;
        
      case this.MessageType.META:
        try {
          const metadata = JSON.parse(new TextDecoder().decode(payload));
          this._log('Received metadata:', metadata);
        } catch (e) {
          this._log('Failed to decode metadata');
        }
        break;
        
      default:
        this._log('Unknown binary message type:', messageType);
    }
  }
  
  /**
   * Handle TTS audio chunk with sequencing
   */
  _handleTTSChunk(sequenceNumber, flags, timestamp, payload) {
    this.metrics.ttsChunksReceived++;
    
    const isStart = (flags & this.MessageFlags.START) !== 0;
    const isEnd = (flags & this.MessageFlags.END) !== 0;
    
    if (isStart) {
      this.expectedTTSSequence = sequenceNumber;
      this.ttsChunkBuffer.clear();
      this._log('TTS stream started, sequence:', sequenceNumber);
    }
    
    // Handle chunk ordering
    if (sequenceNumber < this.expectedTTSSequence) {
      this._log('Ignoring duplicate/late TTS chunk:', sequenceNumber);
      return;
    }
    
    if (sequenceNumber > this.expectedTTSSequence) {
      // Buffer out-of-order chunk
      this.ttsChunkBuffer.set(sequenceNumber, { flags, timestamp, payload });
      this._log('Buffered out-of-order TTS chunk:', sequenceNumber);
      return;
    }
    
    // Process chunk in order
    this._processTTSChunk(sequenceNumber, flags, timestamp, payload);
    
    // Process any buffered chunks that are now in order
    this.expectedTTSSequence++;
    while (this.ttsChunkBuffer.has(this.expectedTTSSequence)) {
      const buffered = this.ttsChunkBuffer.get(this.expectedTTSSequence);
      this._processTTSChunk(
        this.expectedTTSSequence, 
        buffered.flags, 
        buffered.timestamp, 
        buffered.payload
      );
      this.ttsChunkBuffer.delete(this.expectedTTSSequence);
      this.expectedTTSSequence++;
    }
  }
  
  /**
   * Process TTS chunk in sequence
   */
  _processTTSChunk(sequenceNumber, flags, timestamp, payload) {
    const isEnd = (flags & this.MessageFlags.END) !== 0;
    
    if (isEnd && payload.byteLength === 0) {
      this._log('TTS stream ended');
      this.onTTSChunk(null, sequenceNumber, flags); // Signal end
    } else {
      this.onTTSChunk(payload, sequenceNumber, flags);
    }
  }
  
  /**
   * Schedule reconnection attempt
   */
  _scheduleReconnect() {
    this.reconnectAttempts++;
    this.metrics.reconnections++;
    
    const delay = this.reconnectDelayMs * Math.pow(2, this.reconnectAttempts - 1);
    this._log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (!this.isConnected) {
        this.connect().catch(error => {
          this._log('Reconnection failed:', error);
        });
      }
    }, delay);
  }
  
  /**
   * Pack binary frame with header
   */
  _packBinaryFrame(messageType, sequenceNumber, flags, payload) {
    const timestamp = Date.now() & 0xFFFFFFFF; // 32-bit timestamp
    const headerSize = 8;
    const frame = new ArrayBuffer(headerSize + payload.byteLength);
    const view = new DataView(frame);
    
    // Pack header: Type(1) + Seq(2) + Flags(1) + Timestamp(4)
    view.setUint8(0, messageType);
    view.setUint16(1, sequenceNumber & 0xFFFF, true); // Little endian
    view.setUint8(3, flags);
    view.setUint32(4, timestamp, true); // Little endian
    
    // Copy payload
    if (payload.byteLength > 0) {
      const payloadView = new Uint8Array(frame, headerSize);
      const sourceView = new Uint8Array(payload);
      payloadView.set(sourceView);
    }
    
    return frame;
  }
  
  /**
   * Unpack binary frame header
   */
  _unpackBinaryFrame(data) {
    if (data.byteLength < 8) {
      throw new Error('Frame too short');
    }
    
    const view = new DataView(data);
    
    const messageType = view.getUint8(0);
    const sequenceNumber = view.getUint16(1, true); // Little endian
    const flags = view.getUint8(3);
    const timestamp = view.getUint32(4, true); // Little endian
    
    const payload = data.slice(8);
    
    return { messageType, sequenceNumber, flags, timestamp, payload };
  }
  
  /**
   * Get connection metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      bufferedTTSChunks: this.ttsChunkBuffer.size,
    };
  }
  
  /**
   * Reset metrics
   */
  resetMetrics() {
    this.metrics = {
      micChunksSent: 0,
      ttsChunksReceived: 0,
      reconnections: 0,
      bytesReceived: 0,
      bytesSent: 0
    };
  }
  
  _log(message, ...args) {
    if (this.debug) {
      console.log('[DuplexWebSocket]', message, ...args);
    }
  }
}

// Make available globally
window.DuplexWebSocket = DuplexWebSocket;