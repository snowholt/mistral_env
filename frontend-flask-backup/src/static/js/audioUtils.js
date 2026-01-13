// Enhanced Audio Utility Functions
// Provides Float32 -> Int16 conversion, WebM to PCM conversion, 
// audio resampling, waveform visualization helpers, and audio level monitoring.

(function(global){
  
  /**
   * Convert Float32 audio data to Int16
   */
  function floatToInt16(float32){
    const out = new Int16Array(float32.length);
    for(let i=0;i<float32.length;i++){
      let s = float32[i];
      if (s > 1) s = 1; else if (s < -1) s = -1;
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  /**
   * Convert Float32 audio data to Int16 buffer
   */
  function floatToInt16Buffer(float32){
    return floatToInt16(float32).buffer;
  }

  /**
   * Convert Int16 audio data to Float32
   */
  function int16ToFloat32(int16Array) {
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32[i] = int16Array[i] / (int16Array[i] < 0 ? 0x8000 : 0x7FFF);
    }
    return float32;
  }

  /**
   * Resample audio from one sample rate to another
   * Simple linear interpolation resampling
   */
  function resampleAudio(audioData, fromSampleRate, toSampleRate) {
    if (fromSampleRate === toSampleRate) {
      return audioData;
    }

    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexInt = Math.floor(srcIndex);
      const srcIndexFrac = srcIndex - srcIndexInt;

      if (srcIndexInt >= audioData.length - 1) {
        resampled[i] = audioData[audioData.length - 1];
      } else {
        // Linear interpolation
        const sample1 = audioData[srcIndexInt];
        const sample2 = audioData[srcIndexInt + 1];
        resampled[i] = sample1 + (sample2 - sample1) * srcIndexFrac;
      }
    }

    return resampled;
  }

  /**
   * Convert WebM audio blob to PCM data
   */
  async function webmToPCM(webmBlob, targetSampleRate = 16000) {
    try {
      const arrayBuffer = await webmBlob.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000 // WebM typically uses 48kHz
      });

      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Get mono channel data
      const channelData = audioBuffer.getChannelData(0);
      
      // Resample if needed
      const resampledData = resampleAudio(
        channelData, 
        audioBuffer.sampleRate, 
        targetSampleRate
      );

      // Convert to Int16 PCM
      const pcmData = floatToInt16Buffer(resampledData);
      
      await audioContext.close();
      return pcmData;

    } catch (error) {
      console.error('Failed to convert WebM to PCM:', error);
      throw error;
    }
  }

  /**
   * Calculate RMS (Root Mean Square) energy of audio data
   */
  function calculateRMS(audioData) {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      sum += audioData[i] * audioData[i];
    }
    return Math.sqrt(sum / audioData.length);
  }

  /**
   * Calculate peak amplitude of audio data
   */
  function calculatePeak(audioData) {
    let peak = 0;
    for (let i = 0; i < audioData.length; i++) {
      const abs = Math.abs(audioData[i]);
      if (abs > peak) {
        peak = abs;
      }
    }
    return peak;
  }

  /**
   * Calculate dB level from linear amplitude
   */
  function amplitudeToDb(amplitude) {
    if (amplitude <= 0) return -Infinity;
    return 20 * Math.log10(amplitude);
  }

  /**
   * Get audio level from audio data (0-1 range)
   */
  function getAudioLevel(audioData, useRMS = true) {
    if (!audioData || audioData.length === 0) return 0;
    
    if (useRMS) {
      return calculateRMS(audioData);
    } else {
      return calculatePeak(audioData);
    }
  }

  /**
   * Apply simple high-pass filter to remove DC offset and low-frequency noise
   */
  function highPassFilter(audioData, cutoffFreq = 80, sampleRate = 16000) {
    const RC = 1.0 / (cutoffFreq * 2 * Math.PI);
    const dt = 1.0 / sampleRate;
    const alpha = RC / (RC + dt);
    
    const filtered = new Float32Array(audioData.length);
    filtered[0] = audioData[0];
    
    for (let i = 1; i < audioData.length; i++) {
      filtered[i] = alpha * (filtered[i-1] + audioData[i] - audioData[i-1]);
    }
    
    return filtered;
  }

  /**
   * Normalize audio data to prevent clipping
   */
  function normalizeAudio(audioData, targetLevel = 0.8) {
    const peak = calculatePeak(audioData);
    if (peak === 0) return audioData;
    
    const scale = targetLevel / peak;
    const normalized = new Float32Array(audioData.length);
    
    for (let i = 0; i < audioData.length; i++) {
      normalized[i] = audioData[i] * scale;
    }
    
    return normalized;
  }

  /**
   * Create waveform visualization data from audio data
   */
  function createWaveformData(audioData, targetWidth = 800, downsampleFactor = 1) {
    if (!audioData || audioData.length === 0) {
      return new Array(targetWidth).fill(0);
    }

    const samplesPerPixel = Math.ceil(audioData.length / targetWidth);
    const waveformData = [];

    for (let i = 0; i < targetWidth; i++) {
      const startIndex = i * samplesPerPixel;
      const endIndex = Math.min(startIndex + samplesPerPixel, audioData.length);
      
      let min = 0, max = 0;
      
      // Find min/max in this segment for better visualization
      for (let j = startIndex; j < endIndex; j++) {
        const sample = audioData[j];
        if (sample < min) min = sample;
        if (sample > max) max = sample;
      }
      
      // Use the larger absolute value for waveform height
      waveformData.push(Math.max(Math.abs(min), Math.abs(max)));
    }

    return waveformData;
  }

  /**
   * Draw waveform on canvas
   */
  function drawWaveform(canvas, waveformData, options = {}) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    const {
      strokeStyle = '#007bff',
      lineWidth = 2,
      backgroundColor = 'transparent',
      fillStyle = null,
      centerLine = true
    } = options;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Fill background if specified
    if (backgroundColor !== 'transparent') {
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);
    }

    if (!waveformData || waveformData.length === 0) return;

    const stepX = width / waveformData.length;
    const centerY = height / 2;

    // Draw center line
    if (centerLine) {
      ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
    }

    // Draw waveform
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;

    if (fillStyle) {
      // Draw filled waveform
      ctx.fillStyle = fillStyle;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      
      for (let i = 0; i < waveformData.length; i++) {
        const x = i * stepX;
        const amplitude = waveformData[i];
        const y = centerY - (amplitude * centerY);
        ctx.lineTo(x, y);
      }
      
      for (let i = waveformData.length - 1; i >= 0; i--) {
        const x = i * stepX;
        const amplitude = waveformData[i];
        const y = centerY + (amplitude * centerY);
        ctx.lineTo(x, y);
      }
      
      ctx.closePath();
      ctx.fill();
    } else {
      // Draw outline waveform
      ctx.beginPath();
      
      for (let i = 0; i < waveformData.length; i++) {
        const x = i * stepX;
        const amplitude = waveformData[i];
        const y1 = centerY - (amplitude * centerY);
        const y2 = centerY + (amplitude * centerY);
        
        if (i === 0) {
          ctx.moveTo(x, y1);
        } else {
          ctx.lineTo(x, y1);
        }
      }
      
      // Draw bottom half
      for (let i = waveformData.length - 1; i >= 0; i--) {
        const x = i * stepX;
        const amplitude = waveformData[i];
        const y = centerY + (amplitude * centerY);
        ctx.lineTo(x, y);
      }
      
      ctx.closePath();
      ctx.stroke();
    }
  }

  /**
   * Apply window function to audio data (Hamming window)
   */
  function applyHammingWindow(audioData) {
    const windowed = new Float32Array(audioData.length);
    const N = audioData.length - 1;
    
    for (let i = 0; i < audioData.length; i++) {
      const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / N);
      windowed[i] = audioData[i] * window;
    }
    
    return windowed;
  }

  /**
   * Detect silence in audio data
   */
  function detectSilence(audioData, threshold = 0.01, minSilenceDuration = 0.5, sampleRate = 16000) {
    const minSilenceSamples = Math.floor(minSilenceDuration * sampleRate);
    const silenceRegions = [];
    let silenceStart = -1;
    
    for (let i = 0; i < audioData.length; i++) {
      const amplitude = Math.abs(audioData[i]);
      
      if (amplitude < threshold) {
        // Silence detected
        if (silenceStart === -1) {
          silenceStart = i;
        }
      } else {
        // Audio detected
        if (silenceStart !== -1) {
          const silenceDuration = i - silenceStart;
          if (silenceDuration >= minSilenceSamples) {
            silenceRegions.push({
              start: silenceStart / sampleRate,
              end: i / sampleRate,
              duration: silenceDuration / sampleRate
            });
          }
          silenceStart = -1;
        }
      }
    }
    
    // Handle silence at the end
    if (silenceStart !== -1) {
      const silenceDuration = audioData.length - silenceStart;
      if (silenceDuration >= minSilenceSamples) {
        silenceRegions.push({
          start: silenceStart / sampleRate,
          end: audioData.length / sampleRate,
          duration: silenceDuration / sampleRate
        });
      }
    }
    
    return silenceRegions;
  }

  /**
   * Create audio buffer from Float32 data
   */
  function createAudioBuffer(audioData, sampleRate = 16000) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const buffer = audioContext.createBuffer(1, audioData.length, sampleRate);
    buffer.getChannelData(0).set(audioData);
    return { buffer, audioContext };
  }

  /**
   * Play audio buffer
   */
  function playAudioBuffer(audioBuffer, audioContext, volume = 1.0) {
    return new Promise((resolve, reject) => {
      try {
        const source = audioContext.createBufferSource();
        const gainNode = audioContext.createGain();
        
        source.buffer = audioBuffer;
        gainNode.gain.value = volume;
        
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        source.onended = resolve;
        source.start(0);
        
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Convert stereo to mono
   */
  function stereoToMono(leftChannel, rightChannel) {
    const mono = new Float32Array(leftChannel.length);
    for (let i = 0; i < leftChannel.length; i++) {
      mono[i] = (leftChannel[i] + rightChannel[i]) / 2;
    }
    return mono;
  }

  /**
   * Get audio device information
   */
  async function getAudioDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return {
        audioInputs: devices.filter(device => device.kind === 'audioinput'),
        audioOutputs: devices.filter(device => device.kind === 'audiooutput')
      };
    } catch (error) {
      console.error('Failed to get audio devices:', error);
      return { audioInputs: [], audioOutputs: [] };
    }
  }

  /**
   * Test audio input/output
   */
  async function testAudioIO(deviceId = null) {
    try {
      const constraints = {
        audio: deviceId ? { deviceId: { exact: deviceId } } : true
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Test recording for 1 second
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];
      
      return new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (event) => {
          chunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
          stream.getTracks().forEach(track => track.stop());
          const blob = new Blob(chunks, { type: 'audio/webm' });
          resolve({
            success: true,
            audioBlob: blob,
            duration: 1000
          });
        };
        
        mediaRecorder.onerror = (error) => {
          stream.getTracks().forEach(track => track.stop());
          reject(error);
        };
        
        mediaRecorder.start();
        setTimeout(() => mediaRecorder.stop(), 1000);
      });
      
    } catch (error) {
      console.error('Audio I/O test failed:', error);
      return { success: false, error: error.message };
    }
  }

  // Export all functions
  global.AudioUtils = {
    // Basic conversion functions
    floatToInt16,
    floatToInt16Buffer,
    int16ToFloat32,
    
    // Advanced audio processing
    resampleAudio,
    webmToPCM,
    
    // Audio analysis
    calculateRMS,
    calculatePeak,
    amplitudeToDb,
    getAudioLevel,
    detectSilence,
    
    // Audio processing
    highPassFilter,
    normalizeAudio,
    applyHammingWindow,
    stereoToMono,
    
    // Visualization
    createWaveformData,
    drawWaveform,
    
    // Audio playback
    createAudioBuffer,
    playAudioBuffer,
    
    // Device management
    getAudioDevices,
    testAudioIO
  };

})(window);
