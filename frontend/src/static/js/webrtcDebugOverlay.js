/**
 * WebRTC Debug Overlay (Phase E - Integration & Observability)
 * 
 * Provides real-time WebRTC connection statistics and debugging information
 * when ?debug=1 query parameter is present in URL.
 * 
 * Features:
 * - Real-time connection state display
 * - ICE candidate pair information
 * - Audio statistics (bitrate, packets, jitter, packet loss)
 * - Data channel status
 * - Performance metrics
 * 
 * Usage:
 *   const debugOverlay = new WebRTCDebugOverlay(webrtcClient);
 *   debugOverlay.show();
 * 
 * Created: October 15, 2025
 */

class WebRTCDebugOverlay {
    constructor(webrtcClient) {
        this.client = webrtcClient;
        this.overlay = null;
        this.statsInterval = null;
        this.updateIntervalMs = 1000; // Update every second
        this.visible = false;
        
        // Check if debug mode is enabled via query parameter
        this.debugEnabled = this.checkDebugMode();
    }
    
    /**
     * Check if debug mode is enabled via URL query parameter
     */
    checkDebugMode() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('debug') === '1';
    }
    
    /**
     * Initialize and show debug overlay
     */
    show() {
        if (!this.debugEnabled) {
            console.log('[WebRTC Debug] Debug mode not enabled (add ?debug=1 to URL)');
            return;
        }
        
        if (this.visible) {
            return; // Already visible
        }
        
        this.createOverlay();
        this.startStatsCollection();
        this.visible = true;
        
        console.log('[WebRTC Debug] Debug overlay enabled');
    }
    
    /**
     * Hide debug overlay
     */
    hide() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
        
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
        
        this.visible = false;
    }
    
    /**
     * Toggle debug overlay visibility
     */
    toggle() {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }
    
    /**
     * Create debug overlay DOM structure
     */
    createOverlay() {
        // Remove existing overlay if any
        const existing = document.getElementById('webrtc-debug-overlay');
        if (existing) {
            existing.remove();
        }
        
        this.overlay = document.createElement('div');
        this.overlay.id = 'webrtc-debug-overlay';
        this.overlay.className = 'webrtc-debug-overlay';
        this.overlay.innerHTML = `
            <div class="debug-header">
                <h3>🔍 WebRTC Debug Panel</h3>
                <button class="debug-close-btn" onclick="window.webrtcDebugOverlay?.hide()">×</button>
            </div>
            <div class="debug-content">
                <div class="debug-section">
                    <h4>Connection State</h4>
                    <div id="debug-connection-state">
                        <div class="debug-item">
                            <span class="debug-label">Connection:</span>
                            <span class="debug-value" id="debug-conn-state">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">ICE Connection:</span>
                            <span class="debug-value" id="debug-ice-conn-state">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">ICE Gathering:</span>
                            <span class="debug-value" id="debug-ice-gather-state">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Signaling:</span>
                            <span class="debug-value" id="debug-signaling-state">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="debug-section">
                    <h4>Audio Statistics</h4>
                    <div id="debug-audio-stats">
                        <div class="debug-item">
                            <span class="debug-label">Bitrate (out):</span>
                            <span class="debug-value" id="debug-bitrate-out">- kbps</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Bitrate (in):</span>
                            <span class="debug-value" id="debug-bitrate-in">- kbps</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Packets Sent:</span>
                            <span class="debug-value" id="debug-packets-sent">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Packets Received:</span>
                            <span class="debug-value" id="debug-packets-received">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Packets Lost:</span>
                            <span class="debug-value" id="debug-packets-lost">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Jitter:</span>
                            <span class="debug-value" id="debug-jitter">- ms</span>
                        </div>
                    </div>
                </div>
                
                <div class="debug-section">
                    <h4>ICE Candidates</h4>
                    <div id="debug-ice-candidates">
                        <div class="debug-item">
                            <span class="debug-label">Local:</span>
                            <span class="debug-value" id="debug-local-candidates">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Remote:</span>
                            <span class="debug-value" id="debug-remote-candidates">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Selected Pair:</span>
                            <span class="debug-value" id="debug-selected-pair">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="debug-section">
                    <h4>Performance</h4>
                    <div id="debug-performance">
                        <div class="debug-item">
                            <span class="debug-label">Round Trip Time:</span>
                            <span class="debug-value" id="debug-rtt">- ms</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Audio Level:</span>
                            <span class="debug-value" id="debug-audio-level">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Frames Decoded:</span>
                            <span class="debug-value" id="debug-frames-decoded">-</span>
                        </div>
                        <div class="debug-item">
                            <span class="debug-label">Data Channel:</span>
                            <span class="debug-value" id="debug-data-channel">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="debug-section">
                    <h4>Quick Actions</h4>
                    <div class="debug-actions">
                        <button onclick="window.open('chrome://webrtc-internals', '_blank')">
                            chrome://webrtc-internals
                        </button>
                        <button onclick="window.webrtcDebugOverlay?.exportStats()">
                            Export Stats
                        </button>
                        <button onclick="window.webrtcDebugOverlay?.clearStats()">
                            Clear Stats
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Add styles
        this.addStyles();
        
        // Append to body
        document.body.appendChild(this.overlay);
        
        // Make draggable
        this.makeDraggable();
    }
    
    /**
     * Add CSS styles for debug overlay
     */
    addStyles() {
        const styleId = 'webrtc-debug-styles';
        if (document.getElementById(styleId)) {
            return; // Already added
        }
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .webrtc-debug-overlay {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 400px;
                max-height: 80vh;
                background: rgba(0, 0, 0, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 8px;
                color: #fff;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                z-index: 10000;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                overflow-y: auto;
            }
            
            .debug-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                background: #4CAF50;
                color: #000;
                border-radius: 6px 6px 0 0;
                cursor: move;
            }
            
            .debug-header h3 {
                margin: 0;
                font-size: 14px;
                font-weight: bold;
            }
            
            .debug-close-btn {
                background: none;
                border: none;
                color: #000;
                font-size: 24px;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
                line-height: 30px;
                text-align: center;
            }
            
            .debug-close-btn:hover {
                background: rgba(0, 0, 0, 0.1);
                border-radius: 4px;
            }
            
            .debug-content {
                padding: 12px;
            }
            
            .debug-section {
                margin-bottom: 16px;
                padding-bottom: 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .debug-section:last-child {
                border-bottom: none;
            }
            
            .debug-section h4 {
                margin: 0 0 8px 0;
                font-size: 13px;
                color: #4CAF50;
                text-transform: uppercase;
            }
            
            .debug-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
                padding: 2px 0;
            }
            
            .debug-label {
                color: #aaa;
            }
            
            .debug-value {
                color: #fff;
                font-weight: bold;
            }
            
            .debug-value.status-good {
                color: #4CAF50;
            }
            
            .debug-value.status-warning {
                color: #ff9800;
            }
            
            .debug-value.status-error {
                color: #f44336;
            }
            
            .debug-actions {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .debug-actions button {
                padding: 8px 12px;
                background: #4CAF50;
                color: #000;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
            }
            
            .debug-actions button:hover {
                background: #45a049;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * Make overlay draggable
     */
    makeDraggable() {
        const header = this.overlay.querySelector('.debug-header');
        let isDragging = false;
        let currentX;
        let currentY;
        let initialX;
        let initialY;
        
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            initialX = e.clientX - this.overlay.offsetLeft;
            initialY = e.clientY - this.overlay.offsetTop;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                e.preventDefault();
                currentX = e.clientX - initialX;
                currentY = e.clientY - initialY;
                
                this.overlay.style.left = currentX + 'px';
                this.overlay.style.top = currentY + 'px';
                this.overlay.style.right = 'auto';
            }
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }
    
    /**
     * Start collecting and updating statistics
     */
    startStatsCollection() {
        this.statsInterval = setInterval(async () => {
            await this.updateStats();
        }, this.updateIntervalMs);
    }
    
    /**
     * Update statistics display
     */
    async updateStats() {
        if (!this.client || !this.client.peerConnection) {
            return;
        }
        
        try {
            // Get connection stats
            const stats = await this.client.getConnectionStats();
            
            if (!stats) {
                return;
            }
            
            // Update connection state
            this.updateElement('debug-conn-state', stats.connection?.state || '-', 
                              this.getStateClass(stats.connection?.state));
            this.updateElement('debug-ice-conn-state', stats.connection?.iceState || '-',
                              this.getStateClass(stats.connection?.iceState));
            this.updateElement('debug-ice-gather-state', stats.connection?.gatheringState || '-');
            this.updateElement('debug-signaling-state', 
                              this.client.peerConnection.signalingState || '-');
            
            // Update audio stats
            if (stats.audio) {
                this.updateElement('debug-bitrate-out', 
                                 this.formatBitrate(stats.audio.bitrateOut));
                this.updateElement('debug-bitrate-in', 
                                 this.formatBitrate(stats.audio.bitrateIn));
                this.updateElement('debug-packets-sent', 
                                 this.formatNumber(stats.audio.packetsSent));
                this.updateElement('debug-packets-received', 
                                 this.formatNumber(stats.audio.packetsReceived));
                this.updateElement('debug-packets-lost', 
                                 this.formatNumber(stats.audio.packetsLost),
                                 stats.audio.packetsLost > 0 ? 'status-warning' : 'status-good');
                this.updateElement('debug-jitter', 
                                 this.formatJitter(stats.audio.jitter));
            }
            
            // Update ICE candidates
            if (stats.ice) {
                this.updateElement('debug-local-candidates', 
                                 this.formatNumber(stats.ice.localCandidates));
                this.updateElement('debug-remote-candidates', 
                                 this.formatNumber(stats.ice.remoteCandidates));
                this.updateElement('debug-selected-pair', 
                                 stats.ice.selectedPair || 'None',
                                 stats.ice.selectedPair ? 'status-good' : 'status-warning');
            }
            
            // Update performance
            if (stats.performance) {
                this.updateElement('debug-rtt', 
                                 this.formatRTT(stats.performance.rtt));
                this.updateElement('debug-audio-level', 
                                 this.formatAudioLevel(stats.performance.audioLevel));
                this.updateElement('debug-frames-decoded', 
                                 this.formatNumber(stats.performance.framesDecoded));
            }
            
            // Update data channel status
            const dataChannelState = this.client.dataChannel?.readyState || 'closed';
            this.updateElement('debug-data-channel', dataChannelState,
                              dataChannelState === 'open' ? 'status-good' : 'status-warning');
            
        } catch (error) {
            console.error('[WebRTC Debug] Error updating stats:', error);
        }
    }
    
    /**
     * Helper: Update element content and class
     */
    updateElement(id, value, className = '') {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            element.className = 'debug-value ' + className;
        }
    }
    
    /**
     * Get CSS class based on connection state
     */
    getStateClass(state) {
        const goodStates = ['connected', 'completed', 'stable'];
        const warningStates = ['checking', 'connecting', 'have-local-offer', 'have-remote-offer'];
        
        if (goodStates.includes(state)) return 'status-good';
        if (warningStates.includes(state)) return 'status-warning';
        return 'status-error';
    }
    
    /**
     * Format bitrate (bits/s to kbps)
     */
    formatBitrate(bitrate) {
        if (!bitrate) return '0 kbps';
        return (bitrate / 1000).toFixed(1) + ' kbps';
    }
    
    /**
     * Format jitter (seconds to ms)
     */
    formatJitter(jitter) {
        if (!jitter) return '0 ms';
        return (jitter * 1000).toFixed(1) + ' ms';
    }
    
    /**
     * Format RTT (seconds to ms)
     */
    formatRTT(rtt) {
        if (!rtt) return '-';
        return (rtt * 1000).toFixed(0) + ' ms';
    }
    
    /**
     * Format audio level (0-1 to percentage)
     */
    formatAudioLevel(level) {
        if (level === undefined || level === null) return '-';
        return (level * 100).toFixed(0) + '%';
    }
    
    /**
     * Format number with commas
     */
    formatNumber(num) {
        if (!num) return '0';
        return num.toLocaleString();
    }
    
    /**
     * Export statistics to JSON file
     */
    async exportStats() {
        try {
            const stats = await this.client.getConnectionStats();
            const exportData = {
                timestamp: new Date().toISOString(),
                stats: stats,
                clientState: this.client.state,
                metrics: this.client.metrics
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                                 { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `webrtc-stats-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            console.log('[WebRTC Debug] Stats exported');
        } catch (error) {
            console.error('[WebRTC Debug] Export failed:', error);
        }
    }
    
    /**
     * Clear collected statistics
     */
    clearStats() {
        if (this.client && this.client.metrics) {
            this.client.metrics = {
                connectionEstablished: null,
                firstAudioReceived: null,
                totalPacketsSent: 0,
                totalPacketsReceived: 0,
                totalBytesSent: 0,
                totalBytesReceived: 0,
                averageLatency: 0
            };
        }
        console.log('[WebRTC Debug] Stats cleared');
    }
}

// Export for use in other modules
export default WebRTCDebugOverlay;

// Global reference for inline onclick handlers
if (typeof window !== 'undefined') {
    window.WebRTCDebugOverlay = WebRTCDebugOverlay;
}
