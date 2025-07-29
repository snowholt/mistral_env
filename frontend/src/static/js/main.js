// BeautyAI Assistant - Main JavaScript Class
class BeautyAIChat {
    constructor() {
        this.currentModel = '';
        this.loadedModels = new Set(); // Track which models are loaded
        this.isLoading = false;
        this.hasAutoLoaded = false; // Prevent multiple auto-loads
        this.chatHistory = []; // Initialize chat history
        this.sessionId = this.generateSessionId(); // Generate session ID
        
        // Initialize WebSocket voice manager
        this.wsVoiceManager = null;
        this.useWebSocketVoice = true; // Flag to enable/disable WebSocket voice
        
        this.initializeElements();
        this.loadModels();
        this.setupEventListeners();
        this.initializeWebSocketVoice();
        
        // Initialize voice mode UI - default to Advanced mode (toggle checked)
        if (this.voiceModeToggle) {
            this.voiceModeToggle.checked = true; // Checked = Advanced mode
            this.isSimpleVoiceMode = false;
            this.updateVoiceModeUI();
        }
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.loadModelBtn = document.getElementById('loadModelBtn');
        this.unloadModelBtn = document.getElementById('unloadModelBtn');
        this.refreshModelsBtn = document.getElementById('refreshModelsBtn');
        this.modelStatus = document.getElementById('modelStatus');
        
        // Audio controls
        this.recordBtn = document.getElementById('recordBtn');
        this.voiceConversationBtn = document.getElementById('voiceConversationBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.audioFileInput = document.getElementById('audioFileInput');
        this.audioVisualizer = document.getElementById('audioVisualizer');
        this.audioStatus = document.getElementById('audioStatus');
        this.audioStatusText = document.getElementById('audioStatusText');
        
        // Voice conversation elements
        this.voiceConversationOverlay = document.getElementById('voiceConversationOverlay');
        this.voiceCloseBtn = document.getElementById('voiceCloseBtn');
        this.voiceStatus = document.getElementById('voiceStatus');
        this.voiceStatusIcon = document.getElementById('voiceStatusIcon');
        this.voiceStatusIconContainer = document.getElementById('voiceStatusIconContainer');
        this.voiceStatusText = document.getElementById('voiceStatusText');
        this.voiceStatusSubtext = document.getElementById('voiceStatusSubtext');
        this.voiceVisualizer = document.getElementById('voiceVisualizer');
        this.voiceStats = document.getElementById('voiceStats');
        this.voiceMessageCount = document.getElementById('voiceMessageCount');
        this.voiceConversationTime = document.getElementById('voiceConversationTime');
        this.voiceConnectionStatus = document.getElementById('voiceConnectionStatus');
        this.voiceStartBtn = document.getElementById('voiceStartBtn');
        this.voiceStopBtn = document.getElementById('voiceStopBtn');
        this.voiceEndBtn = document.getElementById('voiceEndBtn');
        this.showTranscriptToggle = document.getElementById('showTranscriptToggle');
        this.voiceTranscript = document.getElementById('voiceTranscript');
        this.voiceTranscriptMessages = document.getElementById('voiceTranscriptMessages');
        this.autoStartNextTurn = document.getElementById('autoStartNextTurn');
        this.exportTranscriptBtn = document.getElementById('exportTranscriptBtn');
        this.clearTranscriptBtn = document.getElementById('clearTranscriptBtn');
        
        // Voice mode toggle elements
        this.voiceModeToggle = document.getElementById('voiceModeToggle');
        this.voiceQuickSettings = document.getElementById('voiceQuickSettings');
        this.voiceSimpleSettings = document.getElementById('voiceSimpleSettings');
        this.voiceAutoStart = document.getElementById('voiceAutoStart');
        this.voiceTranscriptToggle = document.getElementById('voiceTranscriptToggle');
        this.voiceTranscriptActions = document.getElementById('voiceTranscriptActions');
        this.simpleLanguageSelect = document.getElementById('simpleLanguageSelect');
        this.simpleVoiceSelect = document.getElementById('simpleVoiceSelect');
        
        // Quick settings
        this.voiceLanguageToggle = document.getElementById('voiceLanguageToggle');
        this.voiceQualityToggle = document.getElementById('voiceQualityToggle');
        this.voiceSpeedToggle = document.getElementById('voiceSpeedToggle');
        
        // Audio recording properties
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordingStartTime = null;
        this.recordingTimer = null;
        
        // Voice conversation properties
        this.isVoiceConversationActive = false;
        this.voiceSessionId = null;
        this.voiceConversationHistory = [];
        this.voiceMessageCounter = 0;
        this.voiceConversationStartTime = null;
        this.voiceConversationTimer = null;
        this.voiceMediaRecorder = null;
        this.voiceAudioChunks = [];
        this.isVoiceRecording = false;
        this.currentAudioElement = null;
        this.autoStartEnabled = false;
        this.voiceSettings = {
            language: 'auto',
            quality: 'qwen_optimized',
            speed: 1.0,
            emotion: 'neutral',
            voice: 'female'
        };
        
        // Voice mode settings
        this.isSimpleVoiceMode = false; // false = Advanced, true = Simple
        this.simpleVoiceSettings = {
            language: 'ar', // ar or en
            voice_type: 'female' // male or female
        };
        
        // Parameter controls
        this.parameterControls = {
            temperature: document.getElementById('temperature'),
            top_p: document.getElementById('top_p'),
            top_k: document.getElementById('top_k'),
            max_new_tokens: document.getElementById('max_new_tokens'),
            repetition_penalty: document.getElementById('repetition_penalty'),
            min_p: document.getElementById('min_p'),
            content_filter_strictness: document.getElementById('content_filter_strictness'),
            disable_content_filter: document.getElementById('disable_content_filter'),
            enable_thinking: document.getElementById('enable_thinking')
        };
        
        // Value displays
        this.valueDisplays = {
            temperature: document.getElementById('tempValue'),
            top_p: document.getElementById('topPValue'),
            top_k: document.getElementById('topKValue'),
            max_new_tokens: document.getElementById('maxTokensValue'),
            repetition_penalty: document.getElementById('repPenaltyValue'),
            min_p: document.getElementById('minPValue')
        };
    }
    
    setupEventListeners() {
        // Send button
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key in textarea
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });
        
        // Parameter controls
        Object.keys(this.parameterControls).forEach(param => {
            const control = this.parameterControls[param];
            if (control && control.type === 'range') {
                control.addEventListener('input', () => {
                    this.valueDisplays[param].textContent = control.value;
                });
            }
        });
        
        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
        });
        
        // Audio controls
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.voiceConversationBtn.addEventListener('click', () => this.startVoiceConversation());
        this.audioFileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Voice conversation controls
        if (this.voiceCloseBtn) {
            this.voiceCloseBtn.addEventListener('click', () => this.endVoiceConversation());
        }
        if (this.voiceStartBtn) {
            this.voiceStartBtn.addEventListener('click', () => this.startVoiceTurn());
        }
        if (this.voiceStopBtn) {
            this.voiceStopBtn.addEventListener('click', () => this.stopVoiceTurn());
        }
        if (this.voiceEndBtn) {
            this.voiceEndBtn.addEventListener('click', () => this.endVoiceConversation());
        }
        if (this.showTranscriptToggle) {
            this.showTranscriptToggle.addEventListener('change', () => this.toggleTranscriptDisplay());
        }
        if (this.autoStartNextTurn) {
            this.autoStartNextTurn.addEventListener('change', (e) => {
                this.autoStartEnabled = e.target.checked;
                console.log('Auto-start enabled:', this.autoStartEnabled);
            });
        }
        if (this.exportTranscriptBtn) {
            this.exportTranscriptBtn.addEventListener('click', () => this.exportTranscript());
        }
        if (this.clearTranscriptBtn) {
            this.clearTranscriptBtn.addEventListener('click', () => this.clearTranscript());
        }
        
        // Voice mode toggle
        if (this.voiceModeToggle) {
            this.voiceModeToggle.addEventListener('change', (e) => this.toggleVoiceMode(e.target.checked));
        }
        
        // Simple voice settings
        if (this.simpleLanguageSelect) {
            this.simpleLanguageSelect.addEventListener('change', (e) => {
                this.simpleVoiceSettings.language = e.target.value;
                console.log('Simple voice language changed to:', e.target.value);
            });
        }
        if (this.simpleVoiceSelect) {
            this.simpleVoiceSelect.addEventListener('change', (e) => {
                this.simpleVoiceSettings.voice_type = e.target.value;
                console.log('Simple voice type changed to:', e.target.value);
            });
        }
        
        // Quick settings
        if (this.voiceLanguageToggle) {
            this.voiceLanguageToggle.addEventListener('click', () => this.toggleVoiceSetting('language'));
        }
        if (this.voiceQualityToggle) {
            this.voiceQualityToggle.addEventListener('click', () => this.toggleVoiceSetting('quality'));
        }
        if (this.voiceSpeedToggle) {
            this.voiceSpeedToggle.addEventListener('click', () => this.toggleVoiceSetting('speed'));
        }
        
        // Model selection
        this.modelSelect.addEventListener('change', () => {
            const selectedModel = this.modelSelect.value;
            if (selectedModel && selectedModel !== this.currentModel) {
                console.log(`User selected model: ${selectedModel} (was: ${this.currentModel})`);
                this.currentModel = selectedModel;
            }
            this.updateModelStatus();
        });
        
        // Model control buttons
        this.loadModelBtn.addEventListener('click', () => this.loadCurrentModel());
        this.unloadModelBtn.addEventListener('click', () => this.unloadCurrentModel());
        this.refreshModelsBtn.addEventListener('click', () => {
            console.log('Manual refresh requested');
            this.loadModels();
        });
    }
    
    async loadModels() {
        try {
            console.log('Loading models list...');
            this.setModelStatus('loading', 'Loading models...');
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (data.success && data.models.length > 0) {
                this.models = data.models; // Store models data
                console.log('Models loaded:', data.models.map(m => `${m.name}: ${m.loaded ? 'loaded' : 'not loaded'}`));
                
                // Clear and repopulate dropdown
                this.modelSelect.innerHTML = '';
                
                // Find the default model first
                const defaultModel = data.models.find(m => 
                    m.is_default === true || (m.name || m.model_name) === 'qwen3-unsloth-q4ks'
                );
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name || model.model_name;
                    const status = model.loaded ? ' (Loaded)' : '';
                    const memory = model.memory_usage_mb > 0 ? ` [${Math.round(model.memory_usage_mb)}MB]` : '';
                    option.textContent = `${model.name || model.model_name}${status}${memory}`;
                    
                    // Mark as selected if this is the current model
                    if (this.currentModel && (model.name || model.model_name) === this.currentModel) {
                        option.selected = true;
                    }
                    // Or if no current model and this is the default
                    else if (!this.currentModel && defaultModel && (model.name || model.model_name) === (defaultModel.name || defaultModel.model_name)) {
                        option.selected = true;
                    }
                    
                    this.modelSelect.appendChild(option);
                });
                
                // Set current model to default if not already set
                if (!this.currentModel && defaultModel) {
                    this.currentModel = defaultModel.name || defaultModel.model_name;
                    this.modelSelect.value = this.currentModel;
                    console.log(`Set default model: ${this.currentModel}`);
                }
                
                this.updateModelStatus();
                
                // Auto-load default model if it's not loaded (only on first load)
                const autoLoadDefaultModel = data.models.find(m => 
                    m.is_default === true || (m.name || m.model_name) === 'qwen3-unsloth-q4ks'
                );
                if (autoLoadDefaultModel && !autoLoadDefaultModel.loaded && this.currentModel === (autoLoadDefaultModel.name || autoLoadDefaultModel.model_name) && !this.hasAutoLoaded) {
                    this.hasAutoLoaded = true; // Prevent multiple auto-loads
                    this.addMessage('system', `🔄 Auto-loading default model: ${this.currentModel}...`);
                    setTimeout(() => this.loadCurrentModel(), 1000);
                } else if (autoLoadDefaultModel && autoLoadDefaultModel.loaded && !this.hasAutoLoaded) {
                    this.hasAutoLoaded = true;
                    this.addMessage('system', `✅ Default model ${this.currentModel} is ready! Content filter is disabled for testing. You can ask any questions.`);
                } else if (!this.hasAutoLoaded) {
                    this.hasAutoLoaded = true;
                    this.addMessage('system', `📋 Loaded ${data.models.length} models. Current model: ${this.currentModel}. Content filter is disabled for testing.`);
                }
            } else {
                this.modelSelect.innerHTML = '<option value="">No models available</option>';
                this.setModelStatus('error', 'No models available');
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.modelSelect.innerHTML = '<option value="">Error loading models</option>';
            this.setModelStatus('error', 'Error loading models');
        }
    }

    updateModelStatus() {
        const selectedModel = this.modelSelect.value;
        if (!selectedModel) {
            this.setModelStatus('error', 'No model selected');
            return;
        }
        
        const modelData = this.models?.find(m => (m.name || m.model_name) === selectedModel);
        if (modelData) {
            if (modelData.loaded) {
                this.setModelStatus('loaded', `${selectedModel} (Loaded)`);
                this.loadModelBtn.disabled = true;
                this.unloadModelBtn.disabled = false;
            } else {
                this.setModelStatus('', `${selectedModel} (Not loaded)`);
                this.loadModelBtn.disabled = false;
                this.unloadModelBtn.disabled = true;
            }
        } else {
            this.setModelStatus('error', 'Model status unknown');
        }
    }

    setModelStatus(type, text) {
        const indicator = this.modelStatus.querySelector('.status-indicator');
        const textEl = this.modelStatus.querySelector('.status-text');
        
        indicator.className = 'status-indicator';
        if (type) indicator.classList.add(type);
        textEl.textContent = text;
    }

    async loadCurrentModel() {
        const modelName = this.modelSelect.value;
        if (!modelName) {
            this.addMessage('system', '❌ Please select a model first');
            return;
        }

        this.setModelStatus('loading', `Loading ${modelName}...`);
        this.loadModelBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/models/${modelName}/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force_reload: false })
            });

            const result = await response.json();
            
            if (result.success) {
                this.setModelStatus('loaded', `${modelName} (Loaded)`);
                this.loadModelBtn.disabled = true;
                this.unloadModelBtn.disabled = false;
                this.addMessage('system', `✅ Model ${modelName} loaded successfully!`);
                // Refresh models to update status
                await this.loadModels();
            } else {
                this.setModelStatus('error', `Failed to load ${modelName}`);
                this.loadModelBtn.disabled = false;
                this.addMessage('system', `❌ Failed to load model ${modelName}: ${result.error}`);
            }
        } catch (error) {
            this.setModelStatus('error', `Error loading ${modelName}`);
            this.loadModelBtn.disabled = false;
            this.addMessage('system', `❌ Error loading model ${modelName}: ${error.message}`);
        }
    }

    async unloadCurrentModel() {
        const modelName = this.modelSelect.value;
        if (!modelName) {
            this.addMessage('system', '❌ Please select a model first');
            return;
        }

        this.setModelStatus('loading', `Unloading ${modelName}...`);
        this.unloadModelBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/models/${modelName}/unload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            
            if (result.success) {
                this.setModelStatus('', `${modelName} (Not loaded)`);
                this.loadModelBtn.disabled = false;
                this.unloadModelBtn.disabled = true;
                this.addMessage('system', `✅ Model ${modelName} unloaded successfully!`);
                // Refresh models to update status
                await this.loadModels();
            } else {
                this.setModelStatus('error', `Failed to unload ${modelName}`);
                this.unloadModelBtn.disabled = false;
                this.addMessage('system', `❌ Failed to unload model ${modelName}: ${result.error}`);
            }
        } catch (error) {
            this.setModelStatus('error', `Error unloading ${modelName}`);
            this.unloadModelBtn.disabled = false;
            this.addMessage('system', `❌ Error unloading model ${modelName}: ${error.message}`);
        }
    }

    async loadModelIfNeeded(modelName) {
        if (!modelName) return;
        
        // Check if model is already loaded
        const modelData = this.models?.find(m => (m.name || m.model_name) === modelName);
        if (modelData && modelData.loaded) {
            console.log(`Model ${modelName} is already loaded`);
            return; // Model is already loaded
        }
        
        // Load the specific model
        console.log(`Loading model: ${modelName}`);
        this.addMessage('system', `🔄 Loading model: ${modelName}...`);
        
        try {
            const response = await fetch(`/api/models/${modelName}/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force_reload: false })
            });

            const result = await response.json();
            
            if (result.success) {
                this.addMessage('system', `✅ Model ${modelName} loaded successfully!`);
                // Refresh models to update status
                await this.loadModels();
            } else {
                this.addMessage('system', `❌ Failed to load model ${modelName}: ${result.error}`);
            }
        } catch (error) {
            this.addMessage('system', `❌ Error loading model ${modelName}: ${error.message}`);
        }
    }

    applyPreset(preset) {
        // Remove active class from all buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        document.querySelector(`[data-preset="${preset}"]`).classList.add('active');
        
        // Apply preset values
        const presets = {
            qwen_optimized: { temperature: 0.3, top_p: 0.95, top_k: 20, repetition_penalty: 1.1 },
            high_quality: { temperature: 0.1, top_p: 1.0, top_k: 50, repetition_penalty: 1.15 },
            creative_optimized: { temperature: 0.5, top_p: 1.0, top_k: 80, repetition_penalty: 1.05 },
            speed_optimized: { temperature: 0.2, top_p: 0.9, top_k: 10, repetition_penalty: 1.1 },
            balanced: { temperature: 0.4, top_p: 0.95, top_k: 40, repetition_penalty: 1.1 },
            conservative: { temperature: 0.1, top_p: 0.8, top_k: 10, repetition_penalty: 1.2 }
        };
        
        const config = presets[preset];
        if (config) {
            Object.keys(config).forEach(param => {
                if (this.parameterControls[param]) {
                    this.parameterControls[param].value = config[param];
                    if (this.valueDisplays[param]) {
                        this.valueDisplays[param].textContent = config[param];
                    }
                }
            });
        }
    }
    
    debugState() {
        console.log('=== DEBUG STATE ===');
        console.log('currentModel:', this.currentModel);
        console.log('dropdown value:', this.modelSelect.value);
        console.log('dropdown options:', Array.from(this.modelSelect.options).map(o => o.value));
        console.log('models data:', this.models?.map(m => ({name: m.name || m.model_name, loaded: m.loaded})));
        console.log('==================');
    }

    // Audio Recording Methods
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            // Enhanced browser support check
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                this.showError('❌ Audio recording is not supported in your browser');
                return;
            }

            // Check if we're on HTTPS or localhost
            const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
            if (!isSecure) {
                this.showError('🔒 Microphone access requires HTTPS. Please access via https:// or localhost');
                return;
            }

            // Check microphone permission status
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
                if (permissionStatus.state === 'denied') {
                    this.showError('🎤 Microphone access denied. Please enable in browser settings.');
                    return;
                }
            } catch (e) {
                // Permission API not supported in all browsers, continue anyway
                console.log('Permission API not supported, continuing...');
            }

            // Request microphone access with better error handling
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
            } catch (error) {
                let errorMessage = '🎤 Microphone access failed: ';
                
                switch (error.name) {
                    case 'NotAllowedError':
                        errorMessage += 'Permission denied. Please allow microphone access.';
                        break;
                    case 'NotFoundError':
                        errorMessage += 'No microphone found.';
                        break;
                    case 'NotReadableError':
                        errorMessage += 'Microphone is being used by another application.';
                        break;
                    case 'OverconstrainedError':
                        errorMessage += 'Microphone constraints cannot be satisfied.';
                        break;
                    default:
                        errorMessage += error.message || 'Unknown error';
                }
                
                this.showError(errorMessage);
                return;
            }

            // Set up MediaRecorder with better format support
            let mimeType = 'audio/webm';
            if (MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/wav';
            } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            }
            
            this.mediaRecorder = new MediaRecorder(stream, { mimeType });

            this.audioChunks = [];
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
                stream.getTracks().forEach(track => track.stop());
            };

            // Start recording
            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordingStartTime = Date.now();

            // Update UI
            this.recordBtn.classList.add('recording');
            this.recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
            this.audioVisualizer.classList.add('active');
            this.startRecordingTimer();
            this.showAudioStatus('🎤 Recording...', 'recording');

            console.log('🎤 Recording started successfully');

        } catch (error) {
            console.error('Recording error:', error);
            this.showError(`Recording failed: ${error.message}`);
            this.resetRecordingUI();
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.clearRecordingTimer();
            this.resetRecordingUI();
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            if (this.recordingStartTime) {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                this.showAudioStatus(`Recording... ${timeStr}`, 'recording');
            }
        }, 1000);
    }

    clearRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    resetRecordingUI() {
        this.recordBtn.classList.remove('recording');
        this.recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        this.audioVisualizer.classList.remove('active');
        this.hideAudioStatus();
    }

    async processRecording() {
        try {
            this.showAudioStatus('Processing audio...', 'processing');

            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { 
                type: this.mediaRecorder.mimeType 
            });

            // Convert to WAV if needed and send
            await this.sendAudioMessage(audioBlob, 'recording');

        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process recording: ' + error.message);
            this.hideAudioStatus();
        }
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/x-m4a', 'audio/x-ms-wma'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|flac|m4a|wma)$/i)) {
            this.showError('Unsupported audio format. Please use WAV, MP3, OGG, FLAC, M4A, or WMA files.');
            return;
        }

        // Check file size (limit to 25MB)
        if (file.size > 25 * 1024 * 1024) {
            this.showError('Audio file is too large. Please use files smaller than 25MB.');
            return;
        }

        try {
            this.showAudioStatus('Processing uploaded file...', 'processing');
            await this.sendAudioMessage(file, 'upload');
        } catch (error) {
            console.error('Error processing uploaded file:', error);
            this.showError('Failed to process uploaded file: ' + error.message);
            this.hideAudioStatus();
        } finally {
            // Clear the file input
            event.target.value = '';
        }
    }

    async sendAudioMessage(audioData, source = 'recording') {
        if (this.isLoading) return;

        try {
            this.setLoading(true);
            this.showAudioStatus('Transcribing and generating response...', 'processing');

            // Debug the current state
            this.debugState();

            // Ensure currentModel matches dropdown selection
            const selectedModel = this.modelSelect.value;
            if (selectedModel !== this.currentModel) {
                console.log(`Updating current model from ${this.currentModel} to ${selectedModel}`);
                this.currentModel = selectedModel;
            }

            if (!this.currentModel) {
                this.showError('Please select a model first');
                return;
            }

            // Prepare form data
            const formData = new FormData();
            
            // Determine the correct filename and content type based on audio format
            let filename = 'recording.wav';
            let contentType = 'audio/wav';
            
            if (source === 'recording' && this.mediaRecorder) {
                const mimeType = this.mediaRecorder.mimeType;
                console.log('Recording MIME type:', mimeType);
                
                if (mimeType.includes('webm')) {
                    filename = 'recording.webm';
                    contentType = 'audio/webm';
                } else if (mimeType.includes('mp4')) {
                    filename = 'recording.m4a';
                    contentType = 'audio/mp4';
                } else if (mimeType.includes('ogg')) {
                    filename = 'recording.ogg';
                    contentType = 'audio/ogg';
                } else if (mimeType.includes('wav')) {
                    filename = 'recording.wav';
                    contentType = 'audio/wav';
                }
            } else if (audioData.name) {
                // For uploaded files, use original name and detect type
                filename = audioData.name;
                if (filename.endsWith('.wav')) {
                    contentType = 'audio/wav';
                } else if (filename.endsWith('.mp3')) {
                    contentType = 'audio/mpeg';
                } else if (filename.endsWith('.ogg')) {
                    contentType = 'audio/ogg';
                } else if (filename.endsWith('.m4a')) {
                    contentType = 'audio/mp4';
                } else if (filename.endsWith('.webm')) {
                    contentType = 'audio/webm';
                }
            }
            
            console.log('Sending audio file:', filename, 'with content type:', contentType);
            formData.append('audio_file', audioData, filename);
            formData.append('model_name', this.currentModel);

            // Add optional parameters
            formData.append('whisper_model_name', 'whisper-large-v3-turbo-arabic');
            formData.append('audio_language', 'ar');

            // Add generation parameters from UI
            this.addParametersToFormData(formData);

            // Add session and history
            formData.append('session_id', this.sessionId || '');
            if (this.chatHistory.length > 0) {
                formData.append('chat_history', JSON.stringify(this.chatHistory));
            }

            console.log('Sending audio message to API...');
            
            const response = await fetch('/api/audio-chat', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log('Audio API response:', data);

            if (data.success) {
                // Show transcription in the input if available
                if (data.transcription) {
                    this.messageInput.value = data.transcription;
                    this.messageInput.style.height = 'auto';
                    this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
                }

                // Display transcription as user message
                this.addMessage('user', data.transcription || '[Audio message]', {
                    isAudio: true,
                    audioSource: source,
                    transcriptionTime: data.transcription_time_ms
                });

                // Display AI response
                const responseText = data.final_content || data.response;
                this.addMessage('assistant', responseText, {
                    tokens: data.tokens_generated,
                    time: data.generation_time_ms,
                    speed: data.tokens_per_second,
                    preset: data.preset_used,
                    thinking: data.thinking_content,
                    contentFilter: data.content_filter_applied,
                    isAudioResponse: true,
                    totalProcessingTime: data.total_processing_time_ms
                });

                // Add to chat history
                this.chatHistory.push({
                    role: 'user',
                    content: data.transcription || '[Audio message]'
                });
                this.chatHistory.push({
                    role: 'assistant', 
                    content: responseText
                });

                this.showAudioStatus(`✓ Processed (${Math.round(data.total_processing_time_ms)}ms)`, 'success');
                setTimeout(() => this.hideAudioStatus(), 3000);

            } else {
                console.error('Audio API error:', data.error);
                
                // Show transcription if available even on failure
                if (data.transcription) {
                    this.addMessage('user', `🎙️ ${data.transcription}`, {
                        isAudio: true,
                        audioSource: source,
                        transcriptionOnly: true
                    });
                }
                
                this.showError(`Audio processing failed: ${data.error}`);
                this.hideAudioStatus();
            }

        } catch (error) {
            console.error('Audio message error:', error);
            this.showError('Failed to process audio message: ' + error.message);
            this.hideAudioStatus();
        } finally {
            this.setLoading(false);
        }
    }

    addParametersToFormData(formData) {
        // Get current preset
        const activePreset = document.querySelector('.preset-btn.active');
        if (activePreset) {
            formData.append('preset', activePreset.dataset.preset);
        }

        // Core parameters
        Object.keys(this.parameterControls).forEach(param => {
            const control = this.parameterControls[param];
            if (control && control.value !== '' && control.value !== null) {
                formData.append(param, control.value);
            }
        });

        // Content filtering
        const contentFilterCheckbox = document.getElementById('disable_content_filter');
        if (contentFilterCheckbox?.checked) {
            formData.append('disable_content_filter', 'true');
        }

        const contentFilterSelect = document.getElementById('content_filter_strictness');
        if (contentFilterSelect?.value) {
            formData.append('content_filter_strictness', contentFilterSelect.value);
        }

        // Thinking mode
        const thinkingCheckbox = document.getElementById('enable_thinking');
        formData.append('thinking_mode', thinkingCheckbox?.checked ? 'true' : 'false');
    }

    showAudioStatus(text, type = 'info') {
        this.audioStatusText.textContent = text;
        this.audioStatus.classList.add('show');
        
        // Update icon based on type
        const icon = this.audioStatus.querySelector('i');
        icon.className = '';
        
        switch (type) {
            case 'recording':
                icon.className = 'fas fa-circle recording-timer';
                break;
            case 'processing':
                icon.className = 'fas fa-circle-notch fa-spin';
                break;
            case 'success':
                icon.className = 'fas fa-check-circle';
                break;
            case 'error':
                icon.className = 'fas fa-exclamation-circle';
                break;
            default:
                icon.className = 'fas fa-info-circle';
        }
    }

    hideAudioStatus() {
        this.audioStatus.classList.remove('show');
    }

    showError(message) {
        this.addMessage('system', `❌ ${message}`);
        console.error('Error:', message);
    }

    addMessage(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        let messageContent = content;
        
        // Handle thinking content if present
        if (metadata.thinking) {
            messageContent += `<div class="thinking-content">Thinking: ${metadata.thinking}</div>`;
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">${messageContent}</div>
            <div class="message-meta">
                <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
                ${new Date().toLocaleTimeString()}
                ${metadata.audioSource ? `<i class="fas fa-microphone" title="Audio message"></i>` : ''}
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    async sendMessage(audioData = null) {
        const message = this.messageInput.value.trim();
        if (!message && !audioData) return;
        
        if (!this.currentModel) {
            this.showError('Please select a model first');
            return;
        }
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Add user message
        if (message) {
            this.addMessage('user', message);
        }
        
        // Disable send button
        this.sendBtn.disabled = true;
        this.isLoading = true;
        
        try {
            // Prepare JSON payload instead of FormData
            const payload = {
                model_name: this.currentModel,
                message: message, // Send original message, backend will handle \no_think prefix
                session_id: this.sessionId || '',
                chat_history: this.chatHistory || [],
                generation_config: {
                    max_tokens: parseInt(this.parameterControls.max_new_tokens?.value || 2048),
                    temperature: parseFloat(this.parameterControls.temperature?.value || 0.3),
                    top_p: parseFloat(this.parameterControls.top_p?.value || 0.95),
                    top_k: parseInt(this.parameterControls.top_k?.value || 20),
                    repetition_penalty: parseFloat(this.parameterControls.repetition_penalty?.value || 1.1),
                    min_p: parseFloat(this.parameterControls.min_p?.value || 0.05)
                }
            };
            
            // Add content filtering options
            if (this.parameterControls.disable_content_filter?.checked) {
                payload.disable_content_filter = true;
            }
            
            if (this.parameterControls.content_filter_strictness?.value) {
                payload.content_filter_strictness = this.parameterControls.content_filter_strictness.value;
            }
            
            // Add thinking mode
            const isThinkingEnabled = this.parameterControls.enable_thinking?.checked || false;
            payload.thinking_mode = isThinkingEnabled;
            
            console.log('Sending chat payload:', payload);
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            console.log('Chat response:', data);
            
            if (data.success) {
                const responseText = data.final_content || data.response;
                this.addMessage('assistant', responseText, {
                    thinking: data.thinking_content,
                    tokens: data.tokens_generated,
                    time: data.generation_time_ms,
                    speed: data.tokens_per_second
                });
                
                // Add to chat history
                this.chatHistory.push({
                    role: 'user',
                    content: message
                });
                this.chatHistory.push({
                    role: 'assistant', 
                    content: responseText
                });
            } else {
                this.showError(data.error || 'Failed to send message');
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.showError('Failed to send message: ' + error.message);
        } finally {
            this.sendBtn.disabled = false;
            this.isLoading = false;
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.sendBtn.disabled = loading;
    }

    // Voice Conversation Methods
    
    /**
     * Initialize WebSocket voice manager
     */
    initializeWebSocketVoice() {
        if (typeof WebSocketVoiceManager === 'undefined') {
            console.warn('WebSocketVoiceManager not available, using REST API fallback');
            this.useWebSocketVoice = false;
        } else {
            this.wsVoiceManager = new WebSocketVoiceManager(this);
            this.useWebSocketVoice = true;
        }
        
        // Initialize simple WebSocket voice manager
        if (typeof SimpleWebSocketVoiceManager === 'undefined') {
            console.warn('SimpleWebSocketVoiceManager not available');
            this.useSimpleWebSocketVoice = false;
        } else {
            this.simpleWsVoiceManager = new SimpleWebSocketVoiceManager(this);
            this.useSimpleWebSocketVoice = true;
        }
        
        this.setupVoiceEventHandlers();
    }
    
    /**
     * Set up event handlers for both voice modes
     */
    setupVoiceEventHandlers() {
        // Advanced WebSocket voice event handlers
        if (this.wsVoiceManager) {
            this.wsVoiceManager.addEventListener('onConnectionEstablished', (data) => {
                console.log('🎉 Advanced WebSocket voice connection established:', data.sessionId);
                this.updateVoiceConnectionStatus('Connected (Advanced)');
                this.updateVoiceStatus('ready', 'Connected - Ready to start', 'fas fa-microphone');
                this.updateVoiceSubtext('Tap "Start Talking" to begin voice conversation');
            });
            
            this.wsVoiceManager.addEventListener('onVoiceResponse', (data) => {
                this.handleWebSocketVoiceResponse(data);
            });
            
            this.wsVoiceManager.addEventListener('onError', (data) => {
                console.error('🚨 Advanced WebSocket voice error:', data.error);
                this.handleVoiceConnectionError(data, 'Advanced');
            });
            
            this.wsVoiceManager.addEventListener('onConnectionLost', (data) => {
                console.warn('🔌 Advanced WebSocket voice connection lost:', data.reason);
                this.updateVoiceConnectionStatus('Disconnected');
                this.updateVoiceStatus('error', 'Connection lost: ' + data.reason, 'fas fa-exclamation-triangle');
                // Try to reconnect
                setTimeout(() => {
                    if (this.isVoiceConversationActive && !this.isSimpleVoiceMode) {
                        this.connectWebSocketVoice();
                    }
                }, 2000);
            });
        }
        
        // Simple WebSocket voice event handlers
        if (this.simpleWsVoiceManager) {
            this.simpleWsVoiceManager.addEventListener('onConnectionEstablished', (data) => {
                console.log('🎉 Simple WebSocket voice connection established:', data.sessionId);
                this.updateVoiceConnectionStatus('Connected (Simple)');
                this.updateVoiceStatus('ready', 'Simple Mode Ready - Ultra Fast Response', 'fas fa-bolt');
                this.updateVoiceSubtext('Target response time: <2 seconds');
            });
            
            this.simpleWsVoiceManager.addEventListener('onVoiceResponse', (data) => {
                this.handleSimpleVoiceResponse(data);
            });
            
            this.simpleWsVoiceManager.addEventListener('onProcessingStarted', (data) => {
                this.updateVoiceStatus('processing', 'Processing your message...', 'fas fa-cog fa-spin');
                this.updateVoiceSubtext('Simple mode processing...');
            });
            
            this.simpleWsVoiceManager.addEventListener('onError', (data) => {
                console.error('🚨 Simple WebSocket voice error:', data.error);
                this.handleVoiceConnectionError(data, 'Simple');
            });
            
            this.simpleWsVoiceManager.addEventListener('onConnectionLost', (data) => {
                console.warn('🔌 Simple WebSocket voice connection lost:', data.reason);
                this.updateVoiceConnectionStatus('Disconnected');
                this.updateVoiceStatus('error', 'Simple mode connection lost: ' + data.reason, 'fas fa-exclamation-triangle');
                // Try to reconnect
                setTimeout(() => {
                    if (this.isVoiceConversationActive && this.isSimpleVoiceMode) {
                        this.connectSimpleWebSocketVoice();
                    }
                }, 2000);
            });
        }
    }
    
    /**
     * Handle voice connection errors (both simple and advanced)
     */
    handleVoiceConnectionError(data, mode) {
        this.updateVoiceStatus('error', `${mode} mode error: ${data.error}`, 'fas fa-exclamation-triangle');
        this.updateVoiceConnectionStatus('Error');
        
        let errorMessage = `❌ ${mode} WebSocket error: ${data.error}`;
        if (data.details) {
            errorMessage += ` (${data.details})`;
        }
        
        // Special handling for SSL/WSS errors
        if (data.error.includes('TLS') || data.error.includes('SSL') || data.error.includes('handshake')) {
            errorMessage += '\n\n🔒 SSL/WSS Issue Detected:\n';
            errorMessage += '• Ensure you\'re accessing via HTTPS (https://dev.gmai.sa)\n';
            errorMessage += '• Check that WSS is properly configured in nginx\n';
            errorMessage += '• Verify SSL certificates are valid\n';
            const endpoint = mode === 'Simple' ? 'simple-voice-chat' : 'voice-conversation';
            errorMessage += `• WebSocket endpoint should be: wss://api.gmai.sa/ws/${endpoint}`;
            
            this.updateVoiceSubtext('SSL/WSS configuration issue - check HTTPS setup');
        } else if (data.error.includes('refused') || data.error.includes('timeout')) {
            errorMessage += '\n\n🌐 Connection Issue:\n';
            errorMessage += '• Check if BeautyAI API server is running on port 8000\n';
            errorMessage += '• Verify nginx proxy configuration for /ws/ endpoints\n';
            errorMessage += '• Ensure firewall allows WebSocket connections';
            
            this.updateVoiceSubtext('Connection issue - check server status');
        }
        
        console.error(errorMessage);
        this.addMessage('system', errorMessage);
        
        // Fallback to REST API if retries are exhausted
        if (data.type === 'connection_exhausted') {
            if (mode === 'Simple') {
                this.useSimpleWebSocketVoice = false;
            } else {
                this.useWebSocketVoice = false;
            }
            this.addMessage('system', `🔄 ${mode} mode WebSocket failed - switching to REST API fallback`);
        }
    }
    
    /**
     * Toggle between Simple and Advanced voice modes
     */
    toggleVoiceMode(isAdvanced) {
        this.isSimpleVoiceMode = !isAdvanced; // Toggle is checked for Advanced, unchecked for Simple
        
        console.log(`🔄 Switching to ${this.isSimpleVoiceMode ? 'Simple' : 'Advanced'} voice mode`);
        
        // Update UI elements
        this.updateVoiceModeUI();
        
        // If conversation is active, reconnect with new mode
        if (this.isVoiceConversationActive) {
            this.switchVoiceModeConnection();
        }
    }
    
    /**
     * Update UI based on voice mode
     */
    updateVoiceModeUI() {
        const overlay = this.voiceConversationOverlay;
        
        if (this.isSimpleVoiceMode) {
            // Simple mode - show simple settings, gray out advanced features
            overlay.classList.add('simple-mode');
            this.voiceSimpleSettings.style.display = 'block';
            this.voiceQuickSettings.style.opacity = '0.3';
            this.voiceAutoStart.style.opacity = '0.3';
            this.voiceTranscriptToggle.style.opacity = '0.3';
            this.voiceTranscriptActions.style.opacity = '0.3';
            
            // Disable advanced controls
            this.voiceQuickSettings.style.pointerEvents = 'none';
            this.voiceAutoStart.style.pointerEvents = 'none';
            this.voiceTranscriptToggle.style.pointerEvents = 'none';
            this.voiceTranscriptActions.style.pointerEvents = 'none';
            
            console.log('✨ Simple Voice Mode activated - Advanced features disabled');
        } else {
            // Advanced mode - hide simple settings, enable all features
            overlay.classList.remove('simple-mode');
            this.voiceSimpleSettings.style.display = 'none';
            this.voiceQuickSettings.style.opacity = '1';
            this.voiceAutoStart.style.opacity = '1';
            this.voiceTranscriptToggle.style.opacity = '1';
            this.voiceTranscriptActions.style.opacity = '1';
            
            // Enable advanced controls
            this.voiceQuickSettings.style.pointerEvents = 'auto';
            this.voiceAutoStart.style.pointerEvents = 'auto';
            this.voiceTranscriptToggle.style.pointerEvents = 'auto';
            this.voiceTranscriptActions.style.pointerEvents = 'auto';
            
            console.log('🚀 Advanced Voice Mode activated - All features enabled');
        }
    }
    
    /**
     * Switch voice mode connection when conversation is active
     */
    async switchVoiceModeConnection() {
        console.log('🔄 Switching voice connection mode...');
        
        // Disconnect current connection
        if (this.isSimpleVoiceMode) {
            // Switching TO simple mode - disconnect advanced
            if (this.wsVoiceManager && this.wsVoiceManager.isReady()) {
                this.wsVoiceManager.disconnect();
            }
        } else {
            // Switching TO advanced mode - disconnect simple
            if (this.simpleWsVoiceManager && this.simpleWsVoiceManager.isReady()) {
                this.simpleWsVoiceManager.disconnect();
            }
        }
        
        // Connect with new mode
        this.updateVoiceStatus('processing', 'Switching voice mode...', 'fas fa-sync fa-spin');
        this.updateVoiceSubtext('Please wait while we switch modes...');
        
        setTimeout(async () => {
            const connected = await this.connectVoiceWebSocket();
            if (!connected) {
                this.updateVoiceStatus('error', 'Failed to switch voice mode', 'fas fa-exclamation-triangle');
                this.updateVoiceSubtext('Connection failed - check network status');
            }
        }, 500);
    }
    
    /**
     * Connect to WebSocket voice service (chooses the right manager based on mode)
     */
    async connectVoiceWebSocket() {
        if (this.isSimpleVoiceMode) {
            return await this.connectSimpleWebSocketVoice();
        } else {
            return await this.connectAdvancedWebSocketVoice();
        }
    }
    
    /**
     * Connect to Simple WebSocket voice service
     */
    async connectSimpleWebSocketVoice() {
        if (!this.simpleWsVoiceManager) {
            console.error('Simple WebSocket voice manager not initialized');
            return false;
        }
        
        try {
            this.updateVoiceStatus('processing', 'Connecting to simple voice service...', 'fas fa-bolt');
            this.updateVoiceConnectionStatus('Connecting...');
            
            const connected = await this.simpleWsVoiceManager.connect({
                language: this.simpleVoiceSettings.language,
                voice_type: this.simpleVoiceSettings.voice_type,
                sessionId: this.voiceSessionId
            });
            
            if (connected) {
                console.log('✅ Simple WebSocket voice connected successfully');
                return true;
            } else {
                console.error('❌ Failed to connect Simple WebSocket voice');
                this.updateVoiceStatus('error', 'Failed to connect simple mode', 'fas fa-exclamation-triangle');
                this.updateVoiceConnectionStatus('Failed');
                return false;
            }
        } catch (error) {
            console.error('❌ Simple WebSocket voice connection error:', error);
            this.updateVoiceStatus('error', 'Simple connection error: ' + error.message, 'fas fa-exclamation-triangle');
            this.updateVoiceConnectionStatus('Error');
            return false;
        }
    }
    
    /**
     * Connect to Advanced WebSocket voice service
     */
    async connectAdvancedWebSocketVoice() {
        if (!this.wsVoiceManager) {
            console.error('Advanced WebSocket voice manager not initialized');
            return false;
        }
        
        try {
            this.updateVoiceStatus('processing', 'Connecting to advanced voice service...', 'fas fa-wifi');
            this.updateVoiceConnectionStatus('Connecting...');
            
            const connected = await this.wsVoiceManager.connect({
                inputLanguage: this.voiceSettings.language,
                outputLanguage: this.voiceSettings.language,
                speakerVoice: this.voiceSettings.voice,
                emotion: this.voiceSettings.emotion,
                speechSpeed: this.voiceSettings.speed,
                preset: this.voiceSettings.quality,
                chatModel: this.currentModel || 'qwen3-unsloth-q4ks',
                sessionId: this.voiceSessionId
            });
            
            if (connected) {
                console.log('✅ Advanced WebSocket voice connected successfully');
                return true;
            } else {
                console.error('❌ Failed to connect Advanced WebSocket voice');
                this.updateVoiceStatus('error', 'Failed to connect advanced mode', 'fas fa-exclamation-triangle');
                this.updateVoiceConnectionStatus('Failed');
                return false;
            }
        } catch (error) {
            console.error('❌ Advanced WebSocket voice connection error:', error);
            this.updateVoiceStatus('error', 'Advanced connection error: ' + error.message, 'fas fa-exclamation-triangle');
            this.updateVoiceConnectionStatus('Error');
            return false;
        }
    }
    
    /**
     * Handle simple voice response
     */
    async handleSimpleVoiceResponse(data) {
        console.log('🎤 Simple voice response received:', data);
        
        // Update transcript if enabled
        if (this.showTranscriptToggle && this.showTranscriptToggle.checked) {
            this.addTranscriptMessage('user', data.transcription);
            this.addTranscriptMessage('assistant', data.responseText, {
                responseTime: data.responseTimeMs,
                mode: 'simple'
            });
        }
        
        // Play audio response
        if (data.audioBase64) {
            this.updateVoiceStatus('speaking', 'AI is speaking...', 'fas fa-volume-up');
            this.updateVoiceSubtext(`Response in ${data.responseTimeMs}ms (Simple Mode)`);
            
            try {
                await this.playWebSocketAudioResponse(data.audioBase64, 'wav');
                
                // Auto-start next turn if enabled
                if (this.autoStartEnabled && this.autoStartNextTurn && this.autoStartNextTurn.checked) {
                    setTimeout(() => {
                        if (this.isVoiceConversationActive) {
                            this.startVoiceTurn();
                        }
                    }, 500);
                } else {
                    this.updateVoiceStatus('ready', 'Ready for next message', 'fas fa-microphone');
                    this.updateVoiceSubtext('Simple mode ready - tap "Start Talking"');
                }
            } catch (error) {
                console.error('Failed to play simple voice response:', error);
                this.updateVoiceStatus('error', 'Failed to play response', 'fas fa-exclamation-triangle');
            }
        } else {
            console.warn('No audio data in simple voice response');
            this.updateVoiceStatus('ready', 'Ready for next message', 'fas fa-microphone');
            this.updateVoiceSubtext('No audio received - try again');
        }
        
        // Update conversation stats
        this.voiceMessageCounter++;
        this.updateVoiceStats();
    }
    
    /**
     * Handle WebSocket voice response (Advanced mode)
     */
    async handleWebSocketVoiceResponse(data) {
        console.log('🎤 Processing WebSocket voice response');
        
        try {
            // Update conversation history
            this.voiceConversationHistory.push({
                role: 'user',
                content: data.transcription || 'Audio message'
            });
            this.voiceConversationHistory.push({
                role: 'assistant',
                content: data.responseText || 'Audio response'
            });

            // Add to transcript if enabled
            this.addToVoiceTranscript('user', data.transcription || 'Audio message');
            this.addToVoiceTranscript('assistant', data.responseText || 'Audio response');

            // Play audio response if available
            if (data.audioBase64) {
                await this.playWebSocketAudioResponse(data.audioBase64, data.audioFormat || 'wav');
            } else {
                console.warn('No audio data in WebSocket response');
                this.updateVoiceStatus('ready', 'Response received (no audio)', 'fas fa-check');
                this.updateVoiceSubtext('Text response only - ready for next message');
                this.resetVoiceControls();
            }

            // Update stats
            this.voiceMessageCounter++;
            this.updateVoiceStats();
            this.updateVoiceConnectionStatus('Connected');

            // Auto-start next turn if enabled
            if (this.autoStartEnabled && this.isVoiceConversationActive) {
                setTimeout(() => {
                    if (this.isVoiceConversationActive && !this.isVoiceRecording) {
                        console.log('🤖 Auto-starting next turn (WebSocket)');
                        this.startVoiceTurn();
                    }
                }, 1000);
            }

        } catch (error) {
            console.error('Error handling WebSocket voice response:', error);
            this.updateVoiceStatus('error', 'Response processing failed', 'fas fa-exclamation-triangle');
            this.resetVoiceControls();
        }
    }
    
    /**
     * Play audio response from WebSocket
     */
    async playWebSocketAudioResponse(audioBase64, format) {
        try {
            console.log('🔊 Playing WebSocket audio response');
            this.updateVoiceStatus('speaking', 'AI is speaking...', 'fas fa-volume-up');
            this.updateVoiceSubtext('Playing response...');
            this.updateVoiceConnectionStatus('Speaking');

            // Convert base64 to blob
            const audioData = atob(audioBase64);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            const audioBlob = new Blob([audioArray], { type: `audio/${format}` });
            const audioUrl = URL.createObjectURL(audioBlob);

            // Create and play audio element
            this.currentAudioElement = new Audio(audioUrl);
            
            this.currentAudioElement.onended = () => {
                console.log('🔇 WebSocket audio finished playing');
                this.updateVoiceStatus('ready', 'Ready for your next message', 'fas fa-microphone');
                this.updateVoiceSubtext('Tap "Start Talking" to continue');
                this.updateVoiceConnectionStatus('Connected');
                URL.revokeObjectURL(audioUrl);
                this.currentAudioElement = null;
                this.resetVoiceControls();
            };

            this.currentAudioElement.onerror = (error) => {
                console.error('WebSocket audio playback error:', error);
                this.updateVoiceStatus('error', 'Audio playback failed', 'fas fa-exclamation-triangle');
                this.updateVoiceSubtext('Could not play response audio');
                this.updateVoiceConnectionStatus('Error');
                URL.revokeObjectURL(audioUrl);
                this.currentAudioElement = null;
                this.resetVoiceControls();
            };

            // Add visual feedback during playback
            this.startSpeakingVisualization();

            await this.currentAudioElement.play();

        } catch (error) {
            console.error('WebSocket audio playback error:', error);
            this.updateVoiceStatus('error', 'Audio playback failed', 'fas fa-exclamation-triangle');
            this.updateVoiceSubtext('Could not play response audio');
            this.updateVoiceConnectionStatus('Error');
            this.resetVoiceControls();
        }
    }

    startVoiceConversation() {
        if (this.isVoiceConversationActive) {
            return;
        }

        console.log(`🎙️ Starting voice conversation in ${this.isSimpleVoiceMode ? 'Simple' : 'Advanced'} mode`);
        this.isVoiceConversationActive = true;
        this.voiceSessionId = this.generateSessionId();
        this.voiceConversationHistory = [];
        this.voiceMessageCounter = 0;
        this.voiceConversationStartTime = Date.now();

        // Update button state
        this.voiceConversationBtn.classList.add('active');
        
        // Show overlay with animation
        this.voiceConversationOverlay.classList.add('active');
        
        // Update UI based on current mode
        this.updateVoiceModeUI();
        
        // Initialize UI
        const modeText = this.isSimpleVoiceMode ? 'Simple Voice Mode' : 'Advanced Voice Mode';
        this.updateVoiceStatus('ready', `Initializing ${modeText}...`, 'fas fa-microphone');
        this.updateVoiceSubtext('Setting up WebSocket connection...');
        this.updateVoiceStats();
        this.startConversationTimer();
        
        // Hide regular audio controls to avoid conflicts
        this.recordBtn.disabled = true;
        this.uploadBtn.disabled = true;
        
        // Connect to the appropriate WebSocket voice service based on mode
        const useWebSocket = this.isSimpleVoiceMode ? 
            (this.useSimpleWebSocketVoice && this.simpleWsVoiceManager) : 
            (this.useWebSocketVoice && this.wsVoiceManager);
        
        if (useWebSocket) {
            this.connectVoiceWebSocket().then(connected => {
                if (!connected) {
                    console.warn(`⚠️ ${this.isSimpleVoiceMode ? 'Simple' : 'Advanced'} WebSocket connection failed, falling back to REST API`);
                    if (this.isSimpleVoiceMode) {
                        this.useSimpleWebSocketVoice = false;
                    } else {
                        this.useWebSocketVoice = false;
                    }
                    this.checkMicrophonePermissions();
                }
            });
        } else {
            // Fallback to REST API method
            console.log(`📡 Using REST API for ${this.isSimpleVoiceMode ? 'simple' : 'advanced'} voice conversation`);
            this.checkMicrophonePermissions();
        }
    }

    async checkMicrophonePermissions() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                this.updateVoiceStatus('error', 'Microphone not supported', 'fas fa-exclamation-triangle');
                this.updateVoiceSubtext('Your browser does not support microphone access');
                return;
            }

            // Check if we're on HTTPS or localhost
            const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
            if (!isSecure) {
                this.updateVoiceStatus('error', 'Secure connection required', 'fas fa-lock');
                this.updateVoiceSubtext('Please access via HTTPS for microphone access');
                return;
            }

            // Check permission status
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
                if (permissionStatus.state === 'denied') {
                    this.updateVoiceStatus('error', 'Microphone access denied', 'fas fa-microphone-slash');
                    this.updateVoiceSubtext('Please enable microphone in browser settings');
                    return;
                }
            } catch (e) {
                // Permission API not supported, continue
            }

            this.updateVoiceConnectionStatus('Ready');
        } catch (error) {
            console.error('Permission check error:', error);
            this.updateVoiceStatus('error', 'Permission check failed', 'fas fa-exclamation-triangle');
        }
    }

    endVoiceConversation() {
        console.log('🛑 Ending voice conversation');
        
        // Stop any ongoing recording
        if (this.isVoiceRecording) {
            this.stopVoiceTurn();
        }
        
        // Stop audio playback
        if (this.currentAudioElement) {
            this.currentAudioElement.pause();
            this.currentAudioElement = null;
        }
        
        // Disconnect WebSocket if using it
        if (this.isSimpleVoiceMode) {
            if (this.useSimpleWebSocketVoice && this.simpleWsVoiceManager) {
                this.simpleWsVoiceManager.disconnect();
                console.log('🔌 Simple WebSocket voice connection closed');
            }
        } else {
            if (this.useWebSocketVoice && this.wsVoiceManager) {
                this.wsVoiceManager.disconnect();
                console.log('🔌 Advanced WebSocket voice connection closed');
            }
        }
        
        // Clean up
        this.isVoiceConversationActive = false;
        this.clearConversationTimer();
        
        // Update button state
        this.voiceConversationBtn.classList.remove('active');
        
        // Hide overlay with animation
        this.voiceConversationOverlay.classList.remove('active');
        
        // Re-enable regular controls
        this.recordBtn.disabled = false;
        this.uploadBtn.disabled = false;
        
        // Show conversation summary if there were messages
        if (this.voiceMessageCounter > 0) {
            this.showConversationSummary();
        }
        
        // Reset voice settings display
        this.resetVoiceSettingsDisplay();
        
        // Reset WebSocket flag for next session
        this.useWebSocketVoice = true;
    }

    async startVoiceTurn() {
        if (this.isVoiceRecording || !this.isVoiceConversationActive) {
            return;
        }

        try {
            console.log('🎤 Starting voice turn');
            this.updateVoiceStatus('listening', 'Listening...', 'fas fa-microphone');
            this.updateVoiceSubtext('Speak now... Tap "Stop" when finished');
            this.updateVoiceConnectionStatus('Recording');
            this.voiceVisualizer.classList.add('active');
            this.voiceStartBtn.style.display = 'none';
            this.voiceStopBtn.style.display = 'flex';

            // Request microphone access with enhanced settings
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });

            // Set up MediaRecorder with best available format
            let mimeType = 'audio/webm;codecs=opus';
            if (MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/wav';
            } else if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
            }
            
            this.voiceMediaRecorder = new MediaRecorder(stream, { mimeType });
            this.voiceAudioChunks = [];

            this.voiceMediaRecorder.ondataavailable = (event) => {
                this.voiceAudioChunks.push(event.data);
            };

            this.voiceMediaRecorder.onstop = () => {
                this.processVoiceTurn();
                stream.getTracks().forEach(track => track.stop());
            };

            // Start recording
            this.voiceMediaRecorder.start();
            this.isVoiceRecording = true;

            // Add voice activity detection simulation
            this.startVoiceActivityDetection();

        } catch (error) {
            console.error('Voice recording error:', error);
            this.updateVoiceStatus('error', 'Microphone access failed', 'fas fa-exclamation-triangle');
            
            let errorMsg = 'Please allow microphone access';
            if (error.name === 'NotAllowedError') {
                errorMsg = 'Microphone permission denied';
            } else if (error.name === 'NotFoundError') {
                errorMsg = 'No microphone found';
            } else if (error.name === 'NotReadableError') {
                errorMsg = 'Microphone in use by another app';
            }
            
            this.updateVoiceSubtext(errorMsg);
            this.updateVoiceConnectionStatus('Error');
            this.resetVoiceControls();
        }
    }

    startVoiceActivityDetection() {
        // Simulate voice activity with visual feedback
        let activityLevel = 0;
        const activityInterval = setInterval(() => {
            if (!this.isVoiceRecording) {
                clearInterval(activityInterval);
                return;
            }
            
            // Simulate voice activity levels
            activityLevel = Math.random() * 0.8 + 0.2;
            this.updateVisualizerActivity(activityLevel);
        }, 100);
    }

    updateVisualizerActivity(level) {
        const bars = this.voiceVisualizer.querySelectorAll('.voice-bar');
        bars.forEach((bar, index) => {
            const height = Math.max(10, level * 60 * Math.random());
            bar.style.height = `${height}px`;
        });
    }

    stopVoiceTurn() {
        if (this.voiceMediaRecorder && this.isVoiceRecording) {
            console.log('🛑 Stopping voice turn');
            this.voiceMediaRecorder.stop();
            this.isVoiceRecording = false;
            this.voiceVisualizer.classList.remove('active');
            this.updateVoiceStatus('processing', 'Processing your message...', 'fas fa-brain');
            this.updateVoiceSubtext('Converting speech to text and generating response...');
            this.updateVoiceConnectionStatus('Processing');
        }
    }

    async processVoiceTurn() {
        try {
            // Create audio blob with optimized format
            const audioBlob = new Blob(this.voiceAudioChunks, { 
                type: this.voiceMediaRecorder.mimeType || 'audio/webm' 
            });
            console.log(`📤 Processing voice turn (${this.isSimpleVoiceMode ? 'Simple' : 'Advanced'} mode), audio size: ${audioBlob.size} bytes, type: ${this.voiceMediaRecorder.mimeType}`);

            this.updateVoiceStatus('processing', 'Processing your message...', 'fas fa-brain');
            
            if (this.isSimpleVoiceMode) {
                this.updateVoiceSubtext('Simple mode - ultra fast processing...');
                
                // Use Simple WebSocket if available and connected
                if (this.useSimpleWebSocketVoice && this.simpleWsVoiceManager && this.simpleWsVoiceManager.isReady()) {
                    console.log('📡 Sending audio via Simple WebSocket');
                    await this.simpleWsVoiceManager.sendAudioData(audioBlob);
                    return; // Simple WebSocket response will be handled by event listener
                } else {
                    console.log('🔄 Simple WebSocket not available, using REST API fallback');
                    await this.processVoiceTurnREST(audioBlob);
                }
            } else {
                this.updateVoiceSubtext('Advanced mode - processing with full features...');
                
                // Use Advanced WebSocket if available and connected
                if (this.useWebSocketVoice && this.wsVoiceManager && this.wsVoiceManager.isReady()) {
                    console.log('📡 Sending audio via Advanced WebSocket');
                    await this.wsVoiceManager.sendAudioData(audioBlob);
                    return; // Advanced WebSocket response will be handled by event listener
                } else {
                    console.log('🔄 Advanced WebSocket not available, using REST API fallback');
                    await this.processVoiceTurnREST(audioBlob);
                }
            }

        } catch (error) {
            console.error('Voice processing error:', error);
            this.updateVoiceStatus('error', 'Processing failed: ' + error.message, 'fas fa-exclamation-triangle');
            this.updateVoiceSubtext('Please check your connection and try again');
            this.updateVoiceConnectionStatus('Error');
            this.resetVoiceControls();
        }
    }

    /**
     * Process voice turn using REST API (fallback method)
     */
    async processVoiceTurnREST(audioBlob) {
        // Prepare form data
        const formData = new FormData();
        
        // Use efficient filename based on actual format
        const filename = this.voiceMediaRecorder.mimeType?.includes('webm') ? 'voice_message.webm' : 'voice_message.wav';
        formData.append('audio_file', audioBlob, filename);
        
        // Core voice-to-voice parameters (use auto for language detection)
        formData.append('session_id', this.voiceSessionId);
        formData.append('input_language', this.voiceSettings.language);
        formData.append('output_language', this.voiceSettings.language);
        formData.append('chat_model_name', this.currentModel || 'qwen3-unsloth-q4ks');
        formData.append('stt_model_name', 'whisper-large-v3-turbo-arabic');
        formData.append('tts_model_name', 'coqui-tts-arabic');
        
        // Voice output parameters
        formData.append('speaker_voice', this.voiceSettings.voice);
        formData.append('emotion', this.voiceSettings.emotion);
        formData.append('speech_speed', this.voiceSettings.speed.toString());
        formData.append('audio_output_format', 'wav');
        
        // Add missing backend-required parameters with correct defaults
        formData.append('thinking_mode', 'false'); // Backend default: false
        formData.append('content_filter_strictness', 'balanced'); // Backend default: balanced
        
        // Add conversation history
        if (this.voiceConversationHistory.length > 0) {
            formData.append('chat_history', JSON.stringify(this.voiceConversationHistory));
        }

        // Add preset and generation parameters (avoid duplicates)
        formData.append('preset', this.voiceSettings.quality);
        
        // Add content filtering and thinking mode from UI controls
        const contentFilterCheckbox = document.getElementById('disable_content_filter');
        if (contentFilterCheckbox?.checked) {
            formData.append('enable_content_filter', 'false');
        } else {
            formData.append('enable_content_filter', 'true');
        }

        const contentFilterSelect = document.getElementById('content_filter_strictness');
        if (contentFilterSelect?.value) {
            formData.set('content_filter_strictness', contentFilterSelect.value); // Override default if user selects
        }

        const thinkingCheckbox = document.getElementById('enable_thinking');
        if (thinkingCheckbox?.checked) {
            formData.set('thinking_mode', 'true'); // Override default if user enables it
        }

        // Add manual generation parameters only if not using a preset
        const activePreset = document.querySelector('.preset-btn.active');
        if (!activePreset || activePreset.dataset.preset === 'custom') {
            Object.keys(this.parameterControls).forEach(param => {
                const control = this.parameterControls[param];
                if (control && control.value !== '' && control.value !== null && param !== 'disable_content_filter' && param !== 'enable_thinking') {
                    formData.append(param, control.value);
                }
            });
        }

        this.updateVoiceStatus('processing', 'Processing your message...', 'fas fa-brain');
        this.updateVoiceSubtext('This may take a few seconds...');

        // Send to voice-to-voice API
        const response = await fetch('/inference/voice-to-voice', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            console.log('✅ Voice-to-voice response received');
            
            // Update conversation history
            this.voiceConversationHistory.push({
                role: 'user',
                content: result.transcription || 'Audio message'
            });
            this.voiceConversationHistory.push({
                role: 'assistant',
                content: result.response_text || 'Audio response'
            });

            // Add to transcript if enabled
            this.addToVoiceTranscript('user', result.transcription || 'Audio message');
            this.addToVoiceTranscript('assistant', result.response_text || 'Audio response');

            // Play audio response
            if (result.audio_data) {
                await this.playVoiceResponse(result.audio_data, result.audio_output_format || 'wav');
            } else if (result.audio_size_bytes > 0 && result.session_id) {
                // Enhanced fallback for missing audio_data
                console.warn(`🔊 Audio was generated (${result.audio_size_bytes} bytes) but not included in response`);
                console.log(`🔍 Session ID: ${result.session_id} - Attempting audio recovery...`);
                
                // Show user that we're trying to recover audio
                this.updateVoiceStatus('processing', 'Recovering audio...', 'fas fa-sync fa-spin');
                this.updateVoiceSubtext('Audio was generated, attempting to retrieve...');
                
                try {
                    // Try the existing audio-download endpoint
                    const audioResponse = await fetch(`/api/audio-download/${result.session_id}`);
                    if (audioResponse.ok) {
                        const audioBlob = await audioResponse.blob();
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        // Play audio directly
                        const audio = new Audio(audioUrl);
                        audio.onended = () => {
                            URL.revokeObjectURL(audioUrl);
                            this.updateVoiceStatus('ready', 'Ready for your next message', 'fas fa-microphone');
                            this.updateVoiceSubtext('Tap "Start Talking" to continue');
                            this.resetVoiceControls();
                        };
                        
                        audio.onerror = () => {
                            URL.revokeObjectURL(audioUrl);
                            this.handleAudioRecoveryFailure(result);
                        };
                        
                        this.updateVoiceStatus('speaking', 'AI is speaking...', 'fas fa-volume-up');
                        this.updateVoiceSubtext('Playing recovered audio...');
                        this.startSpeakingVisualization();
                        await audio.play();
                        
                        console.log('✅ Successfully played audio from alternative endpoint');
                    } else {
                        throw new Error(`Audio endpoint returned ${audioResponse.status}`);
                    }
                } catch (audioError) {
                    console.warn('Could not fetch audio from alternative endpoint:', audioError);
                    this.handleAudioRecoveryFailure(result);
                }
            } else {
                console.warn('No audio data received in response');
                this.updateVoiceStatus('ready', 'Response received (no audio)', 'fas fa-check');
                this.updateVoiceSubtext('Text response only - ready for next message');
                this.resetVoiceControls();
            }

            // Update stats
            this.voiceMessageCounter++;
            this.updateVoiceStats();
            this.updateVoiceConnectionStatus('Connected');

            // Auto-start next turn if enabled
            if (this.autoStartEnabled && this.isVoiceConversationActive) {
                setTimeout(() => {
                    if (this.isVoiceConversationActive && !this.isVoiceRecording) {
                        console.log('🤖 Auto-starting next turn');
                        this.startVoiceTurn();
                    }
                }, 1000);
            }

        } else {
            console.error('❌ Voice-to-voice failed:', result.error);
            this.updateVoiceStatus('error', result.error || 'Processing failed', 'fas fa-exclamation-triangle');
            this.updateVoiceSubtext('Please try again');
            this.updateVoiceConnectionStatus('Error');
            this.resetVoiceControls();
        }
    }

    async playVoiceResponse(audioBase64, format) {
        try {
            console.log('🔊 Playing voice response');
            this.updateVoiceStatus('speaking', 'AI is speaking...', 'fas fa-volume-up');
            this.updateVoiceSubtext('Playing response...');
            this.updateVoiceConnectionStatus('Speaking');

            // Convert base64 to blob
            const audioData = atob(audioBase64);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            const audioBlob = new Blob([audioArray], { type: `audio/${format}` });
            const audioUrl = URL.createObjectURL(audioBlob);

            // Create and play audio element
            this.currentAudioElement = new Audio(audioUrl);
            
            this.currentAudioElement.onended = () => {
                console.log('🔇 Audio finished playing');
                this.updateVoiceStatus('ready', 'Ready for your next message', 'fas fa-microphone');
                this.updateVoiceSubtext('Tap "Start Talking" to continue');
                this.updateVoiceConnectionStatus('Connected');
                URL.revokeObjectURL(audioUrl);
                this.currentAudioElement = null;
                this.resetVoiceControls();
            };

            this.currentAudioElement.onerror = (error) => {
                console.error('Audio playback error:', error);
                this.updateVoiceStatus('error', 'Audio playback failed', 'fas fa-exclamation-triangle');
                this.updateVoiceSubtext('Could not play response audio');
                this.updateVoiceConnectionStatus('Error');
                URL.revokeObjectURL(audioUrl);
                this.currentAudioElement = null;
                this.resetVoiceControls();
            };

            // Add visual feedback during playback
            this.startSpeakingVisualization();

            await this.currentAudioElement.play();

        } catch (error) {
            console.error('Audio playback error:', error);
            this.updateVoiceStatus('error', 'Audio playback failed', 'fas fa-exclamation-triangle');
            this.updateVoiceSubtext('Could not play response audio');
            this.updateVoiceConnectionStatus('Error');
            this.resetVoiceControls();
        }
    }

    startSpeakingVisualization() {
        const bars = this.voiceVisualizer.querySelectorAll('.voice-bar');
        this.voiceVisualizer.classList.add('active');
        
        const speakingInterval = setInterval(() => {
            if (!this.currentAudioElement || this.currentAudioElement.ended || this.currentAudioElement.paused) {
                clearInterval(speakingInterval);
                this.voiceVisualizer.classList.remove('active');
                return;
            }
            
            // Animate bars for speaking visualization
            bars.forEach((bar, index) => {
                const height = Math.random() * 40 + 20;
                bar.style.height = `${height}px`;
            });
        }, 150);
    }

    updateVoiceStatus(type, text, iconClass) {
        if (this.voiceStatusText) {
            this.voiceStatusText.textContent = text;
        }
        if (this.voiceStatusIcon) {
            this.voiceStatusIcon.className = iconClass;
        }
        if (this.voiceStatusIconContainer) {
            // Remove previous type classes
            this.voiceStatusIconContainer.classList.remove('listening', 'speaking', 'processing', 'error');
            // Add current type class
            if (type !== 'ready') {
                this.voiceStatusIconContainer.classList.add(type);
            }
        }
    }

    updateVoiceSubtext(text) {
        if (this.voiceStatusSubtext) {
            this.voiceStatusSubtext.textContent = text;
            this.voiceStatusSubtext.style.display = text ? 'block' : 'none';
        }
    }

    updateVoiceConnectionStatus(status) {
        if (this.voiceConnectionStatus) {
            this.voiceConnectionStatus.textContent = status;
            
            // Update color based on status
            this.voiceConnectionStatus.style.color = 
                status === 'Connected' || status === 'Ready' ? '#4caf50' :
                status === 'Error' ? '#f44336' :
                status === 'Processing' || status === 'Recording' || status === 'Speaking' ? '#ffa726' :
                'rgba(255, 255, 255, 0.8)';
        }
    }

    updateVoiceStats() {
        if (this.voiceMessageCount) {
            this.voiceMessageCount.textContent = this.voiceMessageCounter;
        }
        
        if (this.voiceConversationTime && this.voiceConversationStartTime) {
            const elapsed = Math.floor((Date.now() - this.voiceConversationStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            this.voiceConversationTime.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    resetVoiceControls() {
        this.voiceStartBtn.style.display = 'flex';
        this.voiceStopBtn.style.display = 'none';
    }

    handleAudioRecoveryFailure(result) {
        // Enhanced error handling for audio recovery failure
        console.group('🔧 VOICE-TO-VOICE AUDIO FIX NEEDED');
        console.warn('Audio was generated but cannot be played automatically');
        console.log(`Session ID: ${result.session_id}`);
        console.log(`Audio size: ${result.audio_size_bytes} bytes (${Math.round(result.audio_size_bytes/1024)}KB)`);
        console.log('This indicates the BeautyAI API needs to include audio_data in JSON response');
        console.log('Solutions:');
        console.log('1. Update BeautyAI API to include "audio_data" field in voice-to-voice response');
        console.log('2. Implement separate audio download endpoint in BeautyAI API');
        console.log('3. Check BeautyAI API logs for TTS errors');
        console.groupEnd();
        
        // Show informative message to user
        this.updateVoiceStatus('warning', 'Response received (audio issue)', 'fas fa-exclamation-triangle');
        this.updateVoiceSubtext(`Audio generated (${Math.round(result.audio_size_bytes/1024)}KB) but cannot be played - API needs update`);
        this.resetVoiceControls();
        
        // Add detailed system message
        this.addToVoiceTranscript('system', 
            `⚠️ Audio (${Math.round(result.audio_size_bytes/1024)}KB) generated but API doesn't include audio_data field for playback`);
        
        // Show notification with actionable info
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Voice Response Ready', {
                body: `Text response received. Audio generated but needs API fix for playback.`,
                icon: '/static/favicon.ico'
            });
        }
    }

    resetVoiceSettingsDisplay() {
        // Reset quick settings buttons
        document.querySelectorAll('.voice-setting-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Reset auto-start
        if (this.autoStartNextTurn) {
            this.autoStartNextTurn.checked = false;
            this.autoStartEnabled = false;
        }
    }

    toggleVoiceSetting(setting) {
        const btn = document.querySelector(`[data-setting="${setting}"]`);
        if (!btn) return;

        switch (setting) {
            case 'language':
                const languages = ['auto', 'ar', 'en'];
                const langIndex = languages.indexOf(this.voiceSettings.language);
                this.voiceSettings.language = languages[(langIndex + 1) % languages.length];
                
                const languageNames = ['Auto', 'العربية', 'English'];
                btn.querySelector('span').textContent = languageNames[languages.indexOf(this.voiceSettings.language)];
                break;
            
            case 'quality':
                const qualities = ['qwen_optimized', 'high_quality', 'creative_optimized'];
                const qualityIndex = qualities.indexOf(this.voiceSettings.quality);
                this.voiceSettings.quality = qualities[(qualityIndex + 1) % qualities.length];
                
                const qualityNames = ['Optimized', 'High Quality', 'Creative'];
                btn.querySelector('span').textContent = qualityNames[qualities.indexOf(this.voiceSettings.quality)];
                break;
            
            case 'speed':
                const speeds = [0.8, 1.0, 1.2];
                const speedIndex = speeds.indexOf(this.voiceSettings.speed);
                this.voiceSettings.speed = speeds[(speedIndex + 1) % speeds.length];
                
                const speedNames = ['Slow', 'Normal', 'Fast'];
                btn.querySelector('span').textContent = `${speedNames[speedIndex]} Speed`;
                break;
        }

        // Visual feedback
        btn.classList.add('active');
        setTimeout(() => btn.classList.remove('active'), 200);
        
        console.log('Voice settings updated:', this.voiceSettings);
    }

    // Transcript management
    exportTranscript() {
        if (this.voiceConversationHistory.length === 0) {
            alert('No conversation to export');
            return;
        }

        const transcript = this.voiceConversationHistory.map((msg, index) => {
            const role = msg.role === 'user' ? 'You' : 'AI Assistant';
            return `${index + 1}. ${role}: ${msg.content}`;
        }).join('\n\n');

        const blob = new Blob([transcript], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice_conversation_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    clearTranscript() {
        if (this.voiceTranscriptMessages) {
            this.voiceTranscriptMessages.innerHTML = '';
        }
        // Note: We don't clear voiceConversationHistory as it's needed for conversation context
    }

    startConversationTimer() {
        this.voiceConversationTimer = setInterval(() => {
            this.updateVoiceStats();
        }, 1000);
    }

    clearConversationTimer() {
        if (this.voiceConversationTimer) {
            clearInterval(this.voiceConversationTimer);
            this.voiceConversationTimer = null;
        }
    }

    toggleTranscriptDisplay() {
        if (this.showTranscriptToggle && this.voiceTranscript) {
            if (this.showTranscriptToggle.checked) {
                this.voiceTranscript.style.display = 'block';
            } else {
                this.voiceTranscript.style.display = 'none';
            }
        }
    }

    addToVoiceTranscript(role, content) {
        if (!this.voiceTranscriptMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `voice-transcript-message ${role}`;
        
        const roleDiv = document.createElement('div');
        roleDiv.className = 'voice-transcript-message-role';
        roleDiv.textContent = role === 'user' ? 'You' : 'AI Assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'voice-transcript-message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(roleDiv);
        messageDiv.appendChild(contentDiv);
        this.voiceTranscriptMessages.appendChild(messageDiv);
        
        // Auto-scroll to bottom
        this.voiceTranscript.scrollTop = this.voiceTranscript.scrollHeight;
    }

    showConversationSummary() {
        const summary = `Voice conversation completed!\n\n` +
                       `Messages exchanged: ${this.voiceMessageCounter}\n` +
                       `Duration: ${this.voiceConversationTime?.textContent || 'Unknown'}\n\n` +
                       `The conversation history has been saved.`;
        
        // Add summary to main chat if desired
        this.addMessage('system', `🎙️ ${summary}`);
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BeautyAIChat();
});
