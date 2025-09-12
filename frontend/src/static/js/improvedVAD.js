/**
 * Improved Voice Activity Detection (VAD) Module
 * Provides client-side VAD for immediate feedback, energy-based speech detection,
 * configurable thresholds, and visual feedback integration.
 */

class ImprovedVAD {
    constructor(options = {}) {
        // Configuration
        this.config = {
            // Audio processing settings
            sampleRate: options.sampleRate || 16000,
            fftSize: options.fftSize || 256,
            bufferSize: options.bufferSize || 1024,
            
            // VAD thresholds (language-specific)
            thresholds: {
                english: options.englishThreshold || 0.5,
                arabic: options.arabicThreshold || 0.45
            },
            
            // Timing parameters
            minSpeechDuration: options.minSpeechDuration || 300,  // ms
            maxSilenceDuration: options.maxSilenceDuration || 1500, // ms
            preSpeechBuffer: options.preSpeechBuffer || 200,     // ms
            
            // Adaptive threshold settings
            adaptiveThreshold: options.adaptiveThreshold !== false,
            adaptationRate: options.adaptationRate || 0.1,
            noiseFloor: options.noiseFloor || 0.01,
            
            // Callbacks
            onSpeechStart: options.onSpeechStart || (() => {}),
            onSpeechEnd: options.onSpeechEnd || (() => {}),
            onVolumeChange: options.onVolumeChange || (() => {}),
            onStateChange: options.onStateChange || (() => {})
        };
        
        // State variables
        this.isListening = false;
        this.isSpeaking = false;
        this.currentLanguage = 'english';
        this.currentThreshold = this.config.thresholds.english;
        
        // Audio processing
        this.audioContext = null;
        this.mediaStream = null;
        this.analyser = null;
        this.microphone = null;
        this.audioWorklet = null;
        
        // VAD state tracking
        this.speechStartTime = null;
        this.lastSpeechTime = null;
        this.silenceStartTime = null;
        this.preSpeechBuffer = [];
        
        // Energy calculation
        this.energyHistory = [];
        this.energyHistorySize = 50;
        this.noiseLevel = 0;
        this.currentEnergy = 0;
        
        // Visual feedback
        this.visualCallbacks = [];
        
        // Performance metrics
        this.metrics = {
            speechDetections: 0,
            falsePositives: 0,
            avgSpeechDuration: 0,
            avgSilenceDuration: 0
        };
        
        console.log('ImprovedVAD initialized with config:', this.config);
    }
    
    /**
     * Initialize the VAD system
     */
    async initialize() {
        try {
            console.log('Initializing ImprovedVAD...');
            
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false
                }
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            
            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = 0.3;
            
            // Connect microphone to analyser
            this.microphone = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.microphone.connect(this.analyser);
            
            // Initialize noise level estimation
            await this.calibrateNoiseLevel();
            
            console.log('ImprovedVAD initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Failed to initialize ImprovedVAD:', error);
            throw error;
        }
    }
    
    /**
     * Start VAD processing
     */
    startListening() {
        if (this.isListening) return;
        
        console.log('Starting VAD listening...');
        this.isListening = true;
        this.resetState();
        this.processAudio();
        
        this.config.onStateChange('listening');
    }
    
    /**
     * Stop VAD processing
     */
    stopListening() {
        if (!this.isListening) return;
        
        console.log('Stopping VAD listening...');
        this.isListening = false;
        
        if (this.isSpeaking) {
            this.endSpeech();
        }
        
        this.config.onStateChange('idle');
    }
    
    /**
     * Set the current language for language-specific thresholds
     */
    setLanguage(language) {
        console.log(`Setting VAD language to: ${language}`);
        this.currentLanguage = language;
        this.currentThreshold = this.config.thresholds[language] || this.config.thresholds.english;
        
        // Adjust threshold based on language
        if (this.config.adaptiveThreshold) {
            this.adaptThreshold();
        }
    }
    
    /**
     * Update VAD sensitivity
     */
    setSensitivity(sensitivity) {
        const language = this.currentLanguage;
        this.config.thresholds[language] = sensitivity;
        this.currentThreshold = sensitivity;
        console.log(`Updated VAD sensitivity for ${language}: ${sensitivity}`);
    }
    
    /**
     * Main audio processing loop
     */
    processAudio() {
        if (!this.isListening || !this.analyser) return;
        
        // Get audio data
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        
        // Calculate energy
        const energy = this.calculateEnergy(dataArray);
        this.currentEnergy = energy;
        
        // Update energy history
        this.updateEnergyHistory(energy);
        
        // Adaptive threshold adjustment
        if (this.config.adaptiveThreshold) {
            this.adaptThreshold();
        }
        
        // VAD decision
        const isSpeech = this.detectSpeech(energy);
        
        // Handle state transitions
        this.handleVADState(isSpeech);
        
        // Visual feedback
        this.config.onVolumeChange(energy, this.currentThreshold);
        
        // Continue processing
        if (this.isListening) {
            requestAnimationFrame(() => this.processAudio());
        }
    }
    
    /**
     * Calculate audio energy from frequency data
     */
    calculateEnergy(dataArray) {
        let sum = 0;
        let peak = 0;
        
        // Focus on speech frequencies (300Hz - 3400Hz)
        const minBin = Math.floor(300 * dataArray.length / (this.config.sampleRate / 2));
        const maxBin = Math.floor(3400 * dataArray.length / (this.config.sampleRate / 2));
        
        for (let i = minBin; i < maxBin; i++) {
            const value = dataArray[i] / 255.0;
            sum += value * value;
            peak = Math.max(peak, value);
        }
        
        const rms = Math.sqrt(sum / (maxBin - minBin));
        
        // Combine RMS and peak for better speech detection
        return (rms * 0.7) + (peak * 0.3);
    }
    
    /**
     * Update energy history for adaptive threshold
     */
    updateEnergyHistory(energy) {
        this.energyHistory.push(energy);
        
        if (this.energyHistory.length > this.energyHistorySize) {
            this.energyHistory.shift();
        }
        
        // Update noise level (average of lower energy values)
        if (this.energyHistory.length >= 10) {
            const sortedEnergies = [...this.energyHistory].sort((a, b) => a - b);
            const lowerHalf = sortedEnergies.slice(0, Math.floor(sortedEnergies.length * 0.3));
            this.noiseLevel = lowerHalf.reduce((sum, e) => sum + e, 0) / lowerHalf.length;
        }
    }
    
    /**
     * Adapt threshold based on ambient noise
     */
    adaptThreshold() {
        const baseThreshold = this.config.thresholds[this.currentLanguage];
        const noiseMargin = Math.max(this.noiseLevel * 2, this.config.noiseFloor);
        const adaptedThreshold = Math.max(baseThreshold, noiseMargin);
        
        // Smooth threshold changes
        this.currentThreshold += (adaptedThreshold - this.currentThreshold) * this.config.adaptationRate;
    }
    
    /**
     * Detect speech based on energy and threshold
     */
    detectSpeech(energy) {
        // Basic energy threshold
        if (energy < this.currentThreshold) {
            return false;
        }
        
        // Additional checks for better accuracy
        if (this.energyHistory.length >= 5) {
            // Check for sustained energy over multiple frames
            const recentEnergies = this.energyHistory.slice(-5);
            const avgRecentEnergy = recentEnergies.reduce((sum, e) => sum + e, 0) / recentEnergies.length;
            
            // Require sustained energy above threshold
            return avgRecentEnergy > this.currentThreshold * 0.8;
        }
        
        return true;
    }
    
    /**
     * Handle VAD state transitions
     */
    handleVADState(isSpeech) {
        const now = Date.now();
        
        if (isSpeech) {
            // Speech detected
            if (!this.isSpeaking) {
                // Start of speech
                if (!this.speechStartTime) {
                    this.speechStartTime = now;
                } else if (now - this.speechStartTime >= this.config.minSpeechDuration) {
                    // Confirmed speech start
                    this.startSpeech();
                }
            }
            
            this.lastSpeechTime = now;
            this.silenceStartTime = null;
            
        } else {
            // Silence detected
            if (this.isSpeaking) {
                // Potential end of speech
                if (!this.silenceStartTime) {
                    this.silenceStartTime = now;
                } else if (now - this.silenceStartTime >= this.config.maxSilenceDuration) {
                    // Confirmed speech end
                    this.endSpeech();
                }
            } else {
                // Reset speech start tracking if not sustained
                this.speechStartTime = null;
            }
        }
    }
    
    /**
     * Handle speech start
     */
    startSpeech() {
        if (this.isSpeaking) return;
        
        console.log('VAD: Speech started');
        this.isSpeaking = true;
        this.metrics.speechDetections++;
        
        this.config.onSpeechStart();
        this.config.onStateChange('speaking');
    }
    
    /**
     * Handle speech end
     */
    endSpeech() {
        if (!this.isSpeaking) return;
        
        console.log('VAD: Speech ended');
        const speechDuration = Date.now() - (this.speechStartTime || Date.now());
        
        // Update metrics
        this.metrics.avgSpeechDuration = 
            (this.metrics.avgSpeechDuration + speechDuration) / 2;
        
        this.isSpeaking = false;
        this.speechStartTime = null;
        this.silenceStartTime = null;
        
        this.config.onSpeechEnd();
        this.config.onStateChange('processing');
    }
    
    /**
     * Calibrate noise level on initialization
     */
    async calibrateNoiseLevel() {
        console.log('Calibrating VAD noise level...');
        
        return new Promise((resolve) => {
            let samples = 0;
            const maxSamples = 30; // ~500ms at 60fps
            let energySum = 0;
            
            const calibrate = () => {
                if (samples >= maxSamples) {
                    this.noiseLevel = energySum / samples;
                    console.log(`VAD noise level calibrated: ${this.noiseLevel.toFixed(4)}`);
                    resolve();
                    return;
                }
                
                if (this.analyser) {
                    const bufferLength = this.analyser.frequencyBinCount;
                    const dataArray = new Uint8Array(bufferLength);
                    this.analyser.getByteFrequencyData(dataArray);
                    
                    const energy = this.calculateEnergy(dataArray);
                    energySum += energy;
                    samples++;
                }
                
                requestAnimationFrame(calibrate);
            };
            
            calibrate();
        });
    }
    
    /**
     * Reset VAD state
     */
    resetState() {
        this.isSpeaking = false;
        this.speechStartTime = null;
        this.lastSpeechTime = null;
        this.silenceStartTime = null;
        this.preSpeechBuffer = [];
        this.currentEnergy = 0;
    }
    
    /**
     * Get current VAD state
     */
    getState() {
        return {
            isListening: this.isListening,
            isSpeaking: this.isSpeaking,
            currentEnergy: this.currentEnergy,
            currentThreshold: this.currentThreshold,
            noiseLevel: this.noiseLevel,
            language: this.currentLanguage,
            metrics: { ...this.metrics }
        };
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        console.log('Cleaning up ImprovedVAD...');
        
        this.stopListening();
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
        
        this.audioContext = null;
        this.mediaStream = null;
        this.analyser = null;
        this.microphone = null;
    }
}

// Export for use in other modules
window.ImprovedVAD = ImprovedVAD;