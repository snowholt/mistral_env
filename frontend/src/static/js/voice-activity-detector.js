/**
 * Voice Activity Detection (VAD) System
 * =====================================
 * 
 * Advanced voice activity detection using Web Audio API and smart algorithms
 * for automatic speech start/end detection in voice conversations.
 */

class VoiceActivityDetector {
    constructor(options = {}) {
        // Configuration
        this.config = {
            sampleRate: options.sampleRate || 16000,
            fftSize: options.fftSize || 1024,
            smoothingTimeConstant: options.smoothingTimeConstant || 0.8,
            minDecibels: options.minDecibels || -100,
            maxDecibels: options.maxDecibels || -30,
            
            // VAD thresholds
            energyThreshold: options.energyThreshold || 0.01,
            silenceThreshold: options.silenceThreshold || 0.005,
            speechTimeout: options.speechTimeout || 1500, // ms
            silenceTimeout: options.silenceTimeout || 800, // ms
            minSpeechDuration: options.minSpeechDuration || 300, // ms
            
            // Advanced detection
            useAdvancedVAD: options.useAdvancedVAD !== false,
            noiseGateThreshold: options.noiseGateThreshold || 0.02,
            spectralCentroidThreshold: options.spectralCentroidThreshold || 2500
        };
        
        // State
        this.isActive = false;
        this.isSpeaking = false;
        this.speechStartTime = null;
        this.lastVoiceActivity = null;
        this.silenceStartTime = null;
        
        // Audio analysis
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.frequencyData = null;
        
        // Timers
        this.speechTimer = null;
        this.silenceTimer = null;
        this.analysisInterval = null;
        
        // Callbacks
        this.onSpeechStart = options.onSpeechStart || (() => {});
        this.onSpeechEnd = options.onSpeechEnd || (() => {});
        this.onSilenceStart = options.onSilenceStart || (() => {});
        this.onVoiceActivity = options.onVoiceActivity || (() => {});
        this.onError = options.onError || (() => {});
        
        // History for smoothing
        this.energyHistory = [];
        this.maxHistorySize = 20;
        
        console.log('🎤 VoiceActivityDetector initialized with config:', this.config);
    }
    
    /**
     * Start voice activity detection
     */
    async start(stream) {
        try {
            console.log('🎤 Starting Voice Activity Detection...');
            
            if (this.isActive) {
                console.warn('VAD already active');
                return;
            }
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            
            // Create analyser node
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = this.config.smoothingTimeConstant;
            this.analyser.minDecibels = this.config.minDecibels;
            this.analyser.maxDecibels = this.config.maxDecibels;
            
            // Connect microphone
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);
            
            // Prepare data arrays
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            this.frequencyData = new Uint8Array(bufferLength);
            
            // Reset state
            this.isActive = true;
            this.isSpeaking = false;
            this.speechStartTime = null;
            this.lastVoiceActivity = null;
            this.silenceStartTime = null;
            this.energyHistory = [];
            
            // Start analysis loop
            this.startAnalysis();
            
            console.log('✅ Voice Activity Detection started successfully');
            
        } catch (error) {
            console.error('❌ Failed to start Voice Activity Detection:', error);
            this.onError(error);
            throw error;
        }
    }
    
    /**
     * Stop voice activity detection
     */
    stop() {
        console.log('🛑 Stopping Voice Activity Detection...');
        
        this.isActive = false;
        this.isSpeaking = false;
        
        // Clear timers
        if (this.speechTimer) {
            clearTimeout(this.speechTimer);
            this.speechTimer = null;
        }
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }
        
        // Disconnect audio nodes
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }
        
        // Close audio context
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().then(() => {
                console.log('🔇 Audio context closed');
            }).catch(err => {
                console.warn('Warning closing audio context:', err);
            });
            this.audioContext = null;
        }
        
        console.log('✅ Voice Activity Detection stopped');
    }
    
    /**
     * Start analysis loop
     */
    startAnalysis() {
        const analyze = () => {
            if (!this.isActive) return;
            
            // Get time and frequency domain data
            this.analyser.getByteTimeDomainData(this.dataArray);
            this.analyser.getByteFrequencyData(this.frequencyData);
            
            // Calculate energy levels
            const energy = this.calculateEnergy();
            const isVoiceDetected = this.detectVoiceActivity(energy);
            
            // Handle voice activity changes
            this.handleVoiceActivity(isVoiceDetected, energy);
            
            // Continue analysis
            requestAnimationFrame(analyze);
        };
        
        analyze();
    }
    
    /**
     * Calculate audio energy level
     */
    calculateEnergy() {
        let sum = 0;
        let peak = 0;
        
        // Calculate RMS energy from time domain data
        for (let i = 0; i < this.dataArray.length; i++) {
            const amplitude = (this.dataArray[i] - 128) / 128;
            sum += amplitude * amplitude;
            peak = Math.max(peak, Math.abs(amplitude));
        }
        
        const rms = Math.sqrt(sum / this.dataArray.length);
        
        // Add to history for smoothing
        this.energyHistory.push(rms);
        if (this.energyHistory.length > this.maxHistorySize) {
            this.energyHistory.shift();
        }
        
        // Calculate smoothed energy
        const smoothedEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
        
        return {
            rms: rms,
            smoothed: smoothedEnergy,
            peak: peak,
            raw: rms
        };
    }
    
    /**
     * Detect voice activity using multiple algorithms
     */
    detectVoiceActivity(energy) {
        // Basic energy-based detection
        const energyBasedVoice = energy.smoothed > this.config.energyThreshold;
        
        if (!this.config.useAdvancedVAD) {
            return energyBasedVoice;
        }
        
        // Advanced detection with spectral analysis
        const spectralFeatures = this.calculateSpectralFeatures();
        const spectralBasedVoice = this.isVoiceLikeSpectrum(spectralFeatures);
        
        // Noise gate - reject very low energy signals
        const passesNoiseGate = energy.peak > this.config.noiseGateThreshold;
        
        // Combine multiple indicators
        const voiceScore = (
            (energyBasedVoice ? 1 : 0) +
            (spectralBasedVoice ? 1 : 0) +
            (passesNoiseGate ? 1 : 0)
        ) / 3;
        
        // Require at least 2 out of 3 indicators
        return voiceScore >= 0.67;
    }
    
    /**
     * Calculate spectral features for advanced VAD
     */
    calculateSpectralFeatures() {
        let spectralCentroid = 0;
        let spectralEnergy = 0;
        let spectralFlatness = 0;
        
        // Calculate spectral centroid (brightness indicator)
        let weightedSum = 0;
        let magnitudeSum = 0;
        
        for (let i = 1; i < this.frequencyData.length; i++) {
            const magnitude = this.frequencyData[i];
            const frequency = (i * this.audioContext.sampleRate) / (2 * this.frequencyData.length);
            
            weightedSum += frequency * magnitude;
            magnitudeSum += magnitude;
            spectralEnergy += magnitude * magnitude;
        }
        
        if (magnitudeSum > 0) {
            spectralCentroid = weightedSum / magnitudeSum;
        }
        
        // Calculate spectral flatness (harmonicity indicator)
        let geometricMean = 1;
        let arithmeticMean = 0;
        let validBins = 0;
        
        for (let i = 1; i < this.frequencyData.length; i++) {
            const magnitude = this.frequencyData[i];
            if (magnitude > 0) {
                geometricMean *= Math.pow(magnitude, 1 / this.frequencyData.length);
                arithmeticMean += magnitude;
                validBins++;
            }
        }
        
        if (validBins > 0) {
            arithmeticMean /= validBins;
            spectralFlatness = geometricMean / arithmeticMean;
        }
        
        return {
            centroid: spectralCentroid,
            energy: spectralEnergy,
            flatness: spectralFlatness
        };
    }
    
    /**
     * Determine if spectrum looks like voice
     */
    isVoiceLikeSpectrum(features) {
        // Voice typically has:
        // - Spectral centroid in speech range (300-3000 Hz for formants)
        // - Low spectral flatness (harmonic structure)
        // - Sufficient spectral energy
        
        const isInVoiceRange = features.centroid > 300 && features.centroid < 4000;
        const isHarmonic = features.flatness < 0.5; // Lower = more harmonic
        const hasSufficientEnergy = features.energy > 100;
        
        return isInVoiceRange && isHarmonic && hasSufficientEnergy;
    }
    
    /**
     * Handle voice activity state changes
     */
    handleVoiceActivity(isVoiceDetected, energy) {
        const now = Date.now();
        
        // Notify of voice activity level
        this.onVoiceActivity({
            detected: isVoiceDetected,
            energy: energy,
            isSpeaking: this.isSpeaking,
            timestamp: now
        });
        
        if (isVoiceDetected) {
            this.lastVoiceActivity = now;
            this.silenceStartTime = null;
            
            // Clear silence timer
            if (this.silenceTimer) {
                clearTimeout(this.silenceTimer);
                this.silenceTimer = null;
            }
            
            // Start speech detection
            if (!this.isSpeaking) {
                if (!this.speechStartTime) {
                    this.speechStartTime = now;
                }
                
                // Start speech timer
                if (this.speechTimer) {
                    clearTimeout(this.speechTimer);
                }
                
                this.speechTimer = setTimeout(() => {
                    const speechDuration = now - this.speechStartTime;
                    if (speechDuration >= this.config.minSpeechDuration && !this.isSpeaking) {
                        this.isSpeaking = true;
                        console.log('🎤 Speech started (VAD)');
                        this.onSpeechStart({
                            timestamp: now,
                            energy: energy.smoothed
                        });
                    }
                }, this.config.minSpeechDuration);
            }
            
        } else {
            // No voice detected
            if (!this.silenceStartTime && this.lastVoiceActivity) {
                this.silenceStartTime = now;
                this.onSilenceStart({
                    timestamp: now,
                    lastVoiceActivity: this.lastVoiceActivity
                });
            }
            
            // Clear speech timer
            if (this.speechTimer) {
                clearTimeout(this.speechTimer);
                this.speechTimer = null;
                this.speechStartTime = null;
            }
            
            // Handle speech end
            if (this.isSpeaking) {
                const silenceDuration = now - (this.silenceStartTime || now);
                
                if (!this.silenceTimer) {
                    this.silenceTimer = setTimeout(() => {
                        if (this.isSpeaking) {
                            this.isSpeaking = false;
                            console.log('🔇 Speech ended (VAD)');
                            this.onSpeechEnd({
                                timestamp: now,
                                silenceDuration: silenceDuration
                            });
                        }
                        this.silenceTimer = null;
                        this.silenceStartTime = null;
                    }, this.config.silenceTimeout);
                }
            }
        }
    }
    
    /**
     * Update VAD sensitivity settings
     */
    updateSensitivity(settings) {
        if (settings.energyThreshold !== undefined) {
            this.config.energyThreshold = settings.energyThreshold;
        }
        
        if (settings.silenceTimeout !== undefined) {
            this.config.silenceTimeout = settings.silenceTimeout;
        }
        
        if (settings.speechTimeout !== undefined) {
            this.config.speechTimeout = settings.speechTimeout;
        }
        
        if (settings.noiseGateThreshold !== undefined) {
            this.config.noiseGateThreshold = settings.noiseGateThreshold;
        }
        
        console.log('🎛️ VAD sensitivity updated:', settings);
    }
    
    /**
     * Get current VAD status
     */
    getStatus() {
        return {
            isActive: this.isActive,
            isSpeaking: this.isSpeaking,
            lastActivity: this.lastVoiceActivity,
            config: { ...this.config }
        };
    }
    
    /**
     * Get current audio levels for visualization
     */
    getAudioLevels() {
        if (!this.isActive || !this.dataArray) {
            return { energy: 0, peak: 0, frequency: [] };
        }
        
        const energy = this.calculateEnergy();
        
        // Get frequency data for visualization
        const frequencyBars = [];
        const step = Math.floor(this.frequencyData.length / 8);
        
        for (let i = 0; i < 8; i++) {
            const startIdx = i * step;
            const endIdx = Math.min(startIdx + step, this.frequencyData.length);
            let sum = 0;
            let count = 0;
            
            for (let j = startIdx; j < endIdx; j++) {
                sum += this.frequencyData[j];
                count++;
            }
            
            frequencyBars.push(count > 0 ? sum / count : 0);
        }
        
        return {
            energy: energy.smoothed,
            peak: energy.peak,
            rms: energy.rms,
            frequency: frequencyBars
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceActivityDetector;
} else {
    window.VoiceActivityDetector = VoiceActivityDetector;
}
