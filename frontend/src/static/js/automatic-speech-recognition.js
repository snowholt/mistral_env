/**
 * Automatic Speech Recognition (ASR) System
 * ==========================================
 * 
 * Advanced automatic speech recognition using Web Speech API with intelligent
 * continuous recognition, error handling, and seamless integration.
 */

class AutomaticSpeechRecognition {
    constructor(options = {}) {
        // Configuration
        this.config = {
            language: options.language || 'ar-SA', // Default to Arabic
            continuous: options.continuous !== false,
            interimResults: options.interimResults !== false,
            maxAlternatives: options.maxAlternatives || 3,
            
            // Automatic behavior
            autoRestart: options.autoRestart !== false,
            restartDelay: options.restartDelay || 1000, // ms
            maxRestartAttempts: options.maxRestartAttempts || 5,
            silenceTimeout: options.silenceTimeout || 3000, // ms
            
            // Language detection
            autoDetectLanguage: options.autoDetectLanguage || false,
            supportedLanguages: options.supportedLanguages || ['ar-SA', 'en-US'],
            languageConfidenceThreshold: options.languageConfidenceThreshold || 0.7,
            
            // Quality control
            confidenceThreshold: options.confidenceThreshold || 0.3,
            minTranscriptLength: options.minTranscriptLength || 2,
            debounceDelay: options.debounceDelay || 500, // ms
        };
        
        // State
        this.isActive = false;
        this.isListening = false;
        this.currentLanguage = this.config.language;
        this.recognition = null;
        this.restartAttempts = 0;
        this.lastResult = null;
        this.lastSpeechTime = null;
        
        // Timers
        this.restartTimer = null;
        this.silenceTimer = null;
        this.debounceTimer = null;
        
        // Callbacks
        this.onTranscript = options.onTranscript || (() => {});
        this.onFinalTranscript = options.onFinalTranscript || (() => {});
        this.onInterimTranscript = options.onInterimTranscript || (() => {});
        this.onStart = options.onStart || (() => {});
        this.onEnd = options.onEnd || (() => {});
        this.onError = options.onError || (() => {});
        this.onLanguageDetected = options.onLanguageDetected || (() => {});
        this.onStatusChange = options.onStatusChange || (() => {});
        
        // Check browser support
        this.isSupported = this.checkSupport();
        
        console.log('🎙️ AutomaticSpeechRecognition initialized:', {
            supported: this.isSupported,
            language: this.currentLanguage,
            config: this.config
        });
    }
    
    /**
     * Check if Web Speech API is supported
     */
    checkSupport() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const supported = !!SpeechRecognition;
        
        if (!supported) {
            console.warn('⚠️ Web Speech API not supported in this browser');
        }
        
        return supported;
    }
    
    /**
     * Start automatic speech recognition
     */
    async start() {
        try {
            if (!this.isSupported) {
                throw new Error('Web Speech API not supported');
            }
            
            if (this.isActive) {
                console.warn('ASR already active');
                return;
            }
            
            console.log('🎙️ Starting Automatic Speech Recognition...');
            
            // Create recognition instance
            await this.createRecognition();
            
            // Start recognition
            this.recognition.start();
            this.isActive = true;
            this.restartAttempts = 0;
            
            this.notifyStatusChange('starting');
            
            console.log('✅ ASR started successfully');
            
        } catch (error) {
            console.error('❌ Failed to start ASR:', error);
            this.onError(error);
            throw error;
        }
    }
    
    /**
     * Stop automatic speech recognition
     */
    stop() {
        console.log('🛑 Stopping Automatic Speech Recognition...');
        
        this.isActive = false;
        this.isListening = false;
        this.restartAttempts = 0;
        
        // Clear timers
        this.clearTimers();
        
        // Stop recognition
        if (this.recognition) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.warn('Warning stopping recognition:', error);
            }
            this.recognition = null;
        }
        
        this.notifyStatusChange('stopped');
        console.log('✅ ASR stopped');
    }
    
    /**
     * Create speech recognition instance
     */
    async createRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        this.recognition = new SpeechRecognition();
        
        // Configure recognition
        this.recognition.continuous = this.config.continuous;
        this.recognition.interimResults = this.config.interimResults;
        this.recognition.maxAlternatives = this.config.maxAlternatives;
        this.recognition.lang = this.currentLanguage;
        
        // Event handlers
        this.recognition.onstart = () => {
            console.log('🎤 Speech recognition started');
            this.isListening = true;
            this.notifyStatusChange('listening');
            this.onStart();
        };
        
        this.recognition.onend = () => {
            console.log('🔇 Speech recognition ended');
            this.isListening = false;
            this.notifyStatusChange('ended');
            this.onEnd();
            
            // Auto-restart if still active
            if (this.isActive && this.config.autoRestart) {
                this.handleAutoRestart();
            }
        };
        
        this.recognition.onresult = (event) => {
            this.handleResults(event);
        };
        
        this.recognition.onerror = (event) => {
            this.handleError(event);
        };
        
        this.recognition.onspeechstart = () => {
            console.log('🎤 Speech detected');
            this.lastSpeechTime = Date.now();
            this.clearSilenceTimer();
        };
        
        this.recognition.onspeechend = () => {
            console.log('🔇 Speech ended');
            this.startSilenceTimer();
        };
        
        this.recognition.onnomatch = () => {
            console.log('🤷 No speech match');
        };
    }
    
    /**
     * Handle recognition results
     */
    handleResults(event) {
        const now = Date.now();
        let finalTranscript = '';
        let interimTranscript = '';
        
        // Process all results
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const transcript = result[0].transcript.trim();
            const confidence = result[0].confidence;
            
            if (result.isFinal) {
                // Final result
                if (this.isValidTranscript(transcript, confidence)) {
                    finalTranscript += transcript + ' ';
                    
                    console.log('📝 Final transcript:', transcript, `(confidence: ${confidence?.toFixed(2) || 'N/A'})`);
                    
                    // Debounce final results
                    this.debounceFinalTranscript(transcript, confidence, now);
                } else {
                    console.log('🚫 Low confidence transcript rejected:', transcript, confidence);
                }
            } else {
                // Interim result
                if (transcript.length >= this.config.minTranscriptLength) {
                    interimTranscript += transcript + ' ';
                }
            }
        }
        
        // Notify callbacks
        if (interimTranscript) {
            this.onInterimTranscript({
                transcript: interimTranscript.trim(),
                timestamp: now
            });
        }
        
        // General transcript callback
        this.onTranscript({
            final: finalTranscript.trim(),
            interim: interimTranscript.trim(),
            timestamp: now
        });
        
        // Auto language detection
        if (this.config.autoDetectLanguage && finalTranscript) {
            this.detectLanguage(finalTranscript.trim());
        }
    }
    
    /**
     * Debounce final transcript to avoid duplicates
     */
    debounceFinalTranscript(transcript, confidence, timestamp) {
        // Clear existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        // Set new timer
        this.debounceTimer = setTimeout(() => {
            // Check if this is a new unique result
            if (!this.lastResult || 
                this.lastResult.transcript !== transcript ||
                timestamp - this.lastResult.timestamp > this.config.debounceDelay) {
                
                this.lastResult = { transcript, confidence, timestamp };
                
                this.onFinalTranscript({
                    transcript: transcript,
                    confidence: confidence,
                    timestamp: timestamp,
                    language: this.currentLanguage
                });
            }
            
            this.debounceTimer = null;
        }, this.config.debounceDelay);
    }
    
    /**
     * Validate transcript quality
     */
    isValidTranscript(transcript, confidence) {
        // Check minimum length
        if (transcript.length < this.config.minTranscriptLength) {
            return false;
        }
        
        // Check confidence if available
        if (confidence !== undefined && confidence < this.config.confidenceThreshold) {
            return false;
        }
        
        // Filter out common false positives
        const falsePositives = ['uh', 'um', 'ah', 'eh', 'mm', 'hmm'];
        const words = transcript.toLowerCase().split(/\s+/);
        
        if (words.length === 1 && falsePositives.includes(words[0])) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Handle recognition errors
     */
    handleError(event) {
        const error = event.error;
        console.error('🚨 Speech recognition error:', error);
        
        this.notifyStatusChange('error', error);
        
        switch (error) {
            case 'network':
                console.log('🌐 Network error - will retry');
                break;
                
            case 'not-allowed':
            case 'service-not-allowed':
                console.error('🚫 Microphone access denied');
                this.isActive = false;
                this.onError(new Error('Microphone access denied'));
                return;
                
            case 'no-speech':
                console.log('🤫 No speech detected - continuing...');
                break;
                
            case 'audio-capture':
                console.error('🎤 Audio capture error');
                this.onError(new Error('Audio capture failed'));
                return;
                
            case 'aborted':
                console.log('🛑 Recognition aborted');
                return;
                
            default:
                console.error('❓ Unknown error:', error);
        }
        
        // Auto-restart on certain errors
        if (this.isActive && this.config.autoRestart) {
            this.handleAutoRestart();
        }
    }
    
    /**
     * Handle automatic restart
     */
    handleAutoRestart() {
        if (this.restartAttempts >= this.config.maxRestartAttempts) {
            console.error('🚫 Max restart attempts reached, stopping ASR');
            this.isActive = false;
            this.onError(new Error('Maximum restart attempts exceeded'));
            return;
        }
        
        this.restartAttempts++;
        console.log(`🔄 Auto-restarting ASR (attempt ${this.restartAttempts}/${this.config.maxRestartAttempts})`);
        
        this.notifyStatusChange('restarting', this.restartAttempts);
        
        // Delay restart
        this.restartTimer = setTimeout(async () => {
            if (this.isActive) {
                try {
                    await this.createRecognition();
                    this.recognition.start();
                    console.log('✅ ASR restarted successfully');
                } catch (error) {
                    console.error('❌ Failed to restart ASR:', error);
                    this.handleAutoRestart(); // Try again
                }
            }
            this.restartTimer = null;
        }, this.config.restartDelay);
    }
    
    /**
     * Start silence timer
     */
    startSilenceTimer() {
        this.clearSilenceTimer();
        
        this.silenceTimer = setTimeout(() => {
            console.log('🤫 Silence timeout reached');
            // Could trigger certain actions here
        }, this.config.silenceTimeout);
    }
    
    /**
     * Clear silence timer
     */
    clearSilenceTimer() {
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
    }
    
    /**
     * Clear all timers
     */
    clearTimers() {
        if (this.restartTimer) {
            clearTimeout(this.restartTimer);
            this.restartTimer = null;
        }
        
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
        
        this.clearSilenceTimer();
    }
    
    /**
     * Detect language from transcript
     */
    detectLanguage(transcript) {
        // Simple language detection based on character patterns
        const arabicPattern = /[\u0600-\u06FF\u0750-\u077F]/;
        const englishPattern = /^[a-zA-Z\s.,!?]+$/;
        
        let detectedLanguage = null;
        let confidence = 0;
        
        if (arabicPattern.test(transcript)) {
            detectedLanguage = 'ar-SA';
            confidence = 0.8;
        } else if (englishPattern.test(transcript)) {
            detectedLanguage = 'en-US';
            confidence = 0.7;
        }
        
        if (detectedLanguage && 
            detectedLanguage !== this.currentLanguage && 
            confidence >= this.config.languageConfidenceThreshold) {
            
            console.log(`🌍 Language detected: ${detectedLanguage} (confidence: ${confidence})`);
            this.switchLanguage(detectedLanguage);
        }
    }
    
    /**
     * Switch recognition language
     */
    async switchLanguage(newLanguage) {
        if (!this.config.supportedLanguages.includes(newLanguage)) {
            console.warn(`⚠️ Language ${newLanguage} not supported`);
            return;
        }
        
        console.log(`🌍 Switching language from ${this.currentLanguage} to ${newLanguage}`);
        
        const wasListening = this.isListening;
        this.currentLanguage = newLanguage;
        
        // Restart recognition with new language
        if (wasListening) {
            try {
                this.recognition.stop();
                setTimeout(async () => {
                    await this.createRecognition();
                    this.recognition.start();
                }, 100);
            } catch (error) {
                console.error('❌ Failed to switch language:', error);
            }
        }
        
        this.onLanguageDetected({
            language: newLanguage,
            previousLanguage: this.currentLanguage
        });
    }
    
    /**
     * Update ASR configuration
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        if (newConfig.language && newConfig.language !== this.currentLanguage) {
            this.switchLanguage(newConfig.language);
        }
        
        console.log('🎛️ ASR configuration updated:', newConfig);
    }
    
    /**
     * Get current ASR status
     */
    getStatus() {
        return {
            isSupported: this.isSupported,
            isActive: this.isActive,
            isListening: this.isListening,
            currentLanguage: this.currentLanguage,
            restartAttempts: this.restartAttempts,
            lastResult: this.lastResult,
            config: { ...this.config }
        };
    }
    
    /**
     * Notify status change
     */
    notifyStatusChange(status, data = null) {
        this.onStatusChange({
            status: status,
            data: data,
            timestamp: Date.now(),
            isActive: this.isActive,
            isListening: this.isListening
        });
    }
    
    /**
     * Force restart recognition
     */
    async restart() {
        console.log('🔄 Forcing ASR restart...');
        
        if (this.recognition) {
            this.recognition.stop();
        }
        
        this.restartAttempts = 0;
        
        setTimeout(async () => {
            await this.createRecognition();
            this.recognition.start();
        }, 100);
    }
    
    /**
     * Get supported languages
     */
    static getSupportedLanguages() {
        // Common supported languages
        return [
            { code: 'ar-SA', name: 'Arabic (Saudi Arabia)' },
            { code: 'en-US', name: 'English (US)' },
            { code: 'en-GB', name: 'English (UK)' },
            { code: 'es-ES', name: 'Spanish (Spain)' },
            { code: 'fr-FR', name: 'French (France)' },
            { code: 'de-DE', name: 'German (Germany)' },
            { code: 'ja-JP', name: 'Japanese (Japan)' },
            { code: 'ko-KR', name: 'Korean (South Korea)' },
            { code: 'zh-CN', name: 'Chinese (Simplified)' }
        ];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AutomaticSpeechRecognition;
} else {
    window.AutomaticSpeechRecognition = AutomaticSpeechRecognition;
}
