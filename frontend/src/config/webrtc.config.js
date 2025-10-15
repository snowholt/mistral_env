/**
 * WebRTC Voice Configuration (Phase A - WebRTC Migration)
 * Frontend configuration for WebRTC voice features
 */

// Environment variables with defaults
const WEBRTC_CONFIG = {
    // Feature Toggle
    enabled: process.env.REACT_APP_VOICE_WEBRTC_ENABLED === 'true' || false,
    
    // Utterance Limits
    maxUtteranceSec: parseInt(process.env.REACT_APP_VOICE_WEBRTC_MAX_UTTERANCE_SEC || '10', 10),
    
    // Signaling Server
    signalingUrl: process.env.REACT_APP_VOICE_WEBRTC_SIGNALING_URL || 
                  `${window.location.protocol}//${window.location.host}/api/v1/webrtc/voice`,
    
    // ICE Configuration
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
        // Additional STUN/TURN servers can be added via environment
    ],
    
    // RTC Configuration
    rtcConfiguration: {
        iceTransportPolicy: 'all', // 'all' or 'relay'
        bundlePolicy: 'balanced',
        rtcpMuxPolicy: 'require',
    },
    
    // Audio Constraints
    audioConstraints: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 48000,
        channelCount: 1, // mono
    },
    
    // Client-side VAD (optional, for UI feedback)
    vadEnabled: true,
    vadAudioLevelThreshold: -50, // dBFS
    
    // Debug Options
    debug: process.env.REACT_APP_VOICE_WEBRTC_DEBUG === 'true' || false,
    logIceEvents: false,
    logSignaling: false,
};

// Validation
if (WEBRTC_CONFIG.maxUtteranceSec < 1 || WEBRTC_CONFIG.maxUtteranceSec > 60) {
    console.warn('[WebRTC Config] Invalid maxUtteranceSec, using default: 10');
    WEBRTC_CONFIG.maxUtteranceSec = 10;
}

// Export configuration
export default WEBRTC_CONFIG;

// Export helper to check if WebRTC is available
export const isWebRTCSupported = () => {
    return !!(
        window.RTCPeerConnection &&
        navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia
    );
};

// Export helper to get mode (WebRTC vs WebSocket fallback)
export const getVoiceMode = () => {
    if (!WEBRTC_CONFIG.enabled) {
        return 'websocket';
    }
    if (!isWebRTCSupported()) {
        console.warn('[WebRTC] Browser does not support WebRTC, falling back to WebSocket');
        return 'websocket';
    }
    return 'webrtc';
};
