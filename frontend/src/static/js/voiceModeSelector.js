/**
 * Voice Mode Selector (Phase D - WebRTC Migration)
 * Handles automatic mode selection between WebRTC and WebSocket
 * 
 * Features:
 * - Feature flag detection from environment
 * - Browser WebRTC capability detection
 * - Automatic fallback to WebSocket
 * - Manual mode override support
 * - UI integration for mode toggle
 * 
 * Created: October 15, 2025
 */

import WEBRTC_CONFIG, { isWebRTCSupported, getVoiceMode } from '../config/webrtc.config.js';
import WebRTCVoiceClient from './webrtcVoiceClient.js';

class VoiceModeSelector {
    constructor(options = {}) {
        this.config = {
            containerId: options.containerId || 'voice-container',
            defaultMode: options.defaultMode || 'auto', // 'auto', 'webrtc', 'websocket'
            allowManualOverride: options.allowManualOverride !== false,
            onModeChange: options.onModeChange || ((mode) => console.log('[Mode] Voice mode changed to:', mode))
        };
        
        this.state = {
            selectedMode: null,
            activeMode: null,
            webrtcAvailable: false,
            initialized: false
        };
        
        this.activeClient = null;
        this.container = null;
        this.elements = {};
        
        console.log('[Mode] VoiceModeSelector initialized with config:', this.config);
    }
    
    /**
     * Initialize mode selector
     */
    async initialize() {
        try {
            console.log('[Mode] Initializing voice mode selector...');
            
            // Check WebRTC availability
            this.state.webrtcAvailable = WEBRTC_CONFIG.enabled && isWebRTCSupported();
            console.log('[Mode] WebRTC available:', this.state.webrtcAvailable);
            
            // Determine initial mode
            this.state.selectedMode = this.determineMode();
            console.log('[Mode] Selected mode:', this.state.selectedMode);
            
            // Get DOM references
            this.container = document.getElementById(this.config.containerId);
            if (!this.container) {
                throw new Error(`Container element not found: ${this.config.containerId}`);
            }
            
            // Setup mode toggle UI if manual override allowed
            if (this.config.allowManualOverride) {
                this.setupModeToggleUI();
            }
            
            // Initialize the selected client
            await this.initializeClient();
            
            this.state.initialized = true;
            console.log('[Mode] Voice mode selector initialized successfully');
            
            return true;
            
        } catch (error) {
            console.error('[Mode] Failed to initialize mode selector:', error);
            throw error;
        }
    }
    
    /**
     * Determine appropriate voice mode based on configuration and capabilities
     */
    determineMode() {
        // If manual mode override is set, respect it
        if (this.config.defaultMode === 'webrtc') {
            if (this.state.webrtcAvailable) {
                return 'webrtc';
            } else {
                console.warn('[Mode] WebRTC requested but not available, falling back to WebSocket');
                return 'websocket';
            }
        } else if (this.config.defaultMode === 'websocket') {
            return 'websocket';
        }
        
        // Auto mode: use getVoiceMode from config
        return getVoiceMode();
    }
    
    /**
     * Setup mode toggle UI
     */
    setupModeToggleUI() {
        // Check if mode toggle already exists
        let modeToggleContainer = document.getElementById('voice-mode-toggle');
        
        if (!modeToggleContainer) {
            // Create mode toggle UI
            modeToggleContainer = document.createElement('div');
            modeToggleContainer.id = 'voice-mode-toggle';
            modeToggleContainer.className = 'mode-toggle-container';
            modeToggleContainer.innerHTML = `
                <div class="mode-toggle">
                    <label class="mode-label">
                        <input type="checkbox" id="mode-toggle-checkbox" ${this.state.selectedMode === 'webrtc' ? 'checked' : ''}>
                        <span class="mode-slider"></span>
                        <span class="mode-text">
                            <span class="mode-websocket">WebSocket</span>
                            <span class="mode-webrtc">WebRTC</span>
                        </span>
                    </label>
                    ${!this.state.webrtcAvailable ? '<small class="mode-warning">WebRTC not available</small>' : ''}
                </div>
            `;
            
            // Insert at top of container
            this.container.insertBefore(modeToggleContainer, this.container.firstChild);
        }
        
        // Bind toggle event
        const checkbox = document.getElementById('mode-toggle-checkbox');
        if (checkbox) {
            // Disable if WebRTC not available
            if (!this.state.webrtcAvailable) {
                checkbox.disabled = true;
                checkbox.checked = false;
            }
            
            checkbox.addEventListener('change', (e) => {
                const newMode = e.target.checked ? 'webrtc' : 'websocket';
                this.switchMode(newMode);
            });
        }
        
        console.log('[Mode] Mode toggle UI setup complete');
    }
    
    /**
     * Initialize the appropriate voice client based on selected mode
     */
    async initializeClient() {
        console.log('[Mode] Initializing client for mode:', this.state.selectedMode);
        
        // Cleanup existing client if any
        if (this.activeClient) {
            await this.cleanupClient();
        }
        
        try {
            if (this.state.selectedMode === 'webrtc') {
                // Initialize WebRTC client
                console.log('[Mode] Loading WebRTC client...');
                this.activeClient = new WebRTCVoiceClient({
                    containerId: this.config.containerId,
                    onError: (error) => {
                        console.error('[Mode] WebRTC client error:', error);
                        // Fallback to WebSocket on critical errors
                        if (!this.state.webrtcAvailable) {
                            console.warn('[Mode] WebRTC failed, falling back to WebSocket');
                            this.switchMode('websocket');
                        }
                    }
                });
                
                await this.activeClient.initialize();
                console.log('[Mode] ✅ WebRTC client initialized');
                
            } else {
                // Initialize WebSocket client (SimpleVoiceClient)
                console.log('[Mode] Loading WebSocket client...');
                
                // Check if SimpleVoiceClient is available
                if (typeof SimpleVoiceClient !== 'undefined') {
                    this.activeClient = new SimpleVoiceClient({
                        containerId: this.config.containerId
                    });
                    
                    await this.activeClient.initialize();
                    console.log('[Mode] ✅ WebSocket client initialized');
                } else {
                    throw new Error('SimpleVoiceClient not loaded');
                }
            }
            
            this.state.activeMode = this.state.selectedMode;
            this.config.onModeChange(this.state.activeMode);
            
            console.log('[Mode] Client initialized successfully');
            
        } catch (error) {
            console.error('[Mode] Failed to initialize client:', error);
            
            // If WebRTC failed and we haven't tried WebSocket yet, fallback
            if (this.state.selectedMode === 'webrtc') {
                console.warn('[Mode] WebRTC initialization failed, falling back to WebSocket');
                this.state.selectedMode = 'websocket';
                await this.initializeClient();
            } else {
                throw error;
            }
        }
    }
    
    /**
     * Switch between WebRTC and WebSocket modes
     */
    async switchMode(newMode) {
        if (newMode === this.state.activeMode) {
            console.log('[Mode] Already in mode:', newMode);
            return;
        }
        
        console.log('[Mode] Switching from', this.state.activeMode, 'to', newMode);
        
        // Validate new mode
        if (newMode === 'webrtc' && !this.state.webrtcAvailable) {
            console.error('[Mode] Cannot switch to WebRTC: not available');
            return;
        }
        
        try {
            // Update state
            this.state.selectedMode = newMode;
            
            // Reinitialize with new mode
            await this.initializeClient();
            
            console.log('[Mode] ✅ Mode switch complete:', newMode);
            
        } catch (error) {
            console.error('[Mode] Failed to switch mode:', error);
            // Revert to previous mode
            this.state.selectedMode = this.state.activeMode;
            
            // Update UI
            const checkbox = document.getElementById('mode-toggle-checkbox');
            if (checkbox) {
                checkbox.checked = (this.state.activeMode === 'webrtc');
            }
        }
    }
    
    /**
     * Cleanup active client
     */
    async cleanupClient() {
        if (!this.activeClient) return;
        
        console.log('[Mode] Cleaning up active client...');
        
        try {
            // Disconnect if connected
            if (typeof this.activeClient.disconnect === 'function') {
                await this.activeClient.disconnect();
            }
            
            // Cleanup resources if method available
            if (typeof this.activeClient.cleanup === 'function') {
                this.activeClient.cleanup();
            }
            
            this.activeClient = null;
            console.log('[Mode] Client cleanup complete');
            
        } catch (error) {
            console.error('[Mode] Error during client cleanup:', error);
        }
    }
    
    /**
     * Get active client instance
     */
    getActiveClient() {
        return this.activeClient;
    }
    
    /**
     * Get current mode
     */
    getCurrentMode() {
        return this.state.activeMode;
    }
    
    /**
     * Check if WebRTC is available
     */
    isWebRTCAvailable() {
        return this.state.webrtcAvailable;
    }
    
    /**
     * Get mode capabilities
     */
    getCapabilities() {
        return {
            webrtc: {
                available: this.state.webrtcAvailable,
                supported: isWebRTCSupported(),
                enabled: WEBRTC_CONFIG.enabled
            },
            websocket: {
                available: true,
                supported: true,
                enabled: true
            },
            currentMode: this.state.activeMode,
            selectedMode: this.state.selectedMode
        };
    }
}

// Export for use in other modules
export default VoiceModeSelector;

// Also expose globally for non-module usage
if (typeof window !== 'undefined') {
    window.VoiceModeSelector = VoiceModeSelector;
}
