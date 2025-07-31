"""
GPU-Optimized Transformers Whisper Service for BeautyAI Framework.

This service provides fast, GPU-accelerated speech recognition using the official
Hugging Face Transformers implementation of Whisper large-v3-turbo.

Key Features:
- Full GPU acceleration with CUDA/cuDNN
- 4x faster than large-v3, up to 4.5x with torch.compile
- Production-ready with proper error handling
- Arabic language optimized
- Registry-driven configuration
"""
import logging
import tempfile
import os
import time
import io
from typing import Dict, Any, Optional, BinaryIO, List, Tuple
from pathlib import Path
import numpy as np

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import librosa
import soundfile as sf
from pydub import AudioSegment

from ...base.base_service import BaseService
from ....config.voice_config_loader import get_voice_config

logger = logging.getLogger(__name__)


class TransformersWhisperService(BaseService):
    """GPU-optimized Whisper service using Hugging Face Transformers."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.loaded_model_name = None
        
        # Hardware optimization
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Performance settings
        self.low_cpu_mem_usage = True
        self.use_safetensors = True
        
        logger.info(f"TransformersWhisper initialized - Device: {self.device}, Dtype: {self.torch_dtype}")
        
        # GPU optimizations
        if self.device.startswith("cuda"):
            # Enable optimizations for GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if torch.cuda.get_device_properties(0).total_memory > 8 * 1024**3:  # >8GB
                logger.info("High-memory GPU detected - enabling optimal settings")
            
    def load_whisper_model(self, model_name: str = None) -> bool:
        """
        Load a Whisper model for transcription using Transformers.
        
        Args:
            model_name: Name of the Whisper model to load (optional, uses voice registry default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use voice configuration loader for consistent model selection
            voice_config = get_voice_config()
            
            # Get STT model configuration from voice registry
            if model_name is None:
                stt_config = voice_config.get_stt_model_config()
                actual_model_id = stt_config.model_id
                model_name = voice_config._config["default_models"]["stt"]
                logger.info(f"Using default STT model from voice registry: {model_name} -> {actual_model_id}")
            else:
                # Validate that the requested model is the one in our voice registry
                stt_config = voice_config.get_stt_model_config()
                registry_model_name = voice_config._config["default_models"]["stt"]
                if model_name != registry_model_name:
                    logger.warning(f"Requested model '{model_name}' differs from voice registry model '{registry_model_name}'. Using registry model.")
                    model_name = registry_model_name
                actual_model_id = stt_config.model_id
            
            logger.info(f"Loading Transformers Whisper model: {actual_model_id}")
            
            # Load model with GPU optimization
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                actual_model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                use_safetensors=self.use_safetensors,
                # attn_implementation="flash_attention_2",  # Uncomment if flash-attn is installed
                attn_implementation="sdpa"  # Use SDPA for better performance
            )
            self.model.to(self.device)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(actual_model_id)
            
            # Note: Using direct model interface instead of pipeline for better control
            # and compatibility with various generation parameters
            
            self.loaded_model_name = model_name
            logger.info(f"✅ Transformers Whisper model '{model_name}' ({actual_model_id}) loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Transformers Whisper model '{model_name}': {e}")
            return False

    def _prepare_audio_for_inference(self, audio_data: bytes, audio_format: str, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Prepare audio data for Whisper inference.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Format of the audio (wav, mp3, webm, etc.)
            target_sample_rate: Target sample rate for Whisper (16kHz)
            
        Returns:
            numpy array of audio samples
        """
        try:
            # Handle different audio formats
            if audio_format.lower() in ['webm', 'ogg']:
                # Use pydub for webm/ogg format
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format.lower())
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                if audio_segment.channels == 2:
                    audio_array = audio_array.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
                audio_array = audio_array / np.iinfo(np.int16).max  # Normalize to [-1, 1]
                sample_rate = audio_segment.frame_rate
            else:
                # Use librosa for other formats
                audio_array, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            
            # Resample to target sample rate if needed
            if sample_rate != target_sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sample_rate)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preparing audio: {e}")
            raise Exception(f"Audio preparation failed: {e}")

    def transcribe_audio_bytes(
        self, 
        audio_bytes: bytes, 
        audio_format: str = "wav", 
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio from bytes using the loaded Whisper model.
        
        Args:
            audio_bytes: Audio data as bytes
            audio_format: Format of the audio data (wav, mp3, webm, etc.)
            language: Language code for transcription (ar, en, or None for auto-detect)
            
        Returns:
            Transcribed text
        """
        try:
            if not self.model or not self.processor:
                raise Exception("Whisper model not loaded. Call load_whisper_model() first.")
            
            # Prepare audio for inference
            audio_array = self._prepare_audio_for_inference(audio_bytes, audio_format)
            
            # Convert to tensor
            input_features = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device, dtype=self.torch_dtype)
            
            # Perform transcription
            start_time = time.time()
            
            # Generate with the model directly
            with torch.no_grad():
                # Simple generation without complex parameters
                if language:
                    if language == "ar":
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
                    elif language == "en":
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
                    else:
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                    
                    predicted_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=256,
                        num_beams=1,
                        do_sample=False
                    )
                else:
                    # Auto-detect language
                    predicted_ids = self.model.generate(
                        input_features,
                        max_new_tokens=256,
                        num_beams=1,
                        do_sample=False
                    )
                
            # Decode the output
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            transcription_time = time.time() - start_time
            
            # Clean up the text
            transcribed_text = transcription.strip() if transcription else ""
            
            logger.info(f"Transcription completed in {transcription_time:.2f}s: '{transcribed_text[:100]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_audio_file(self, audio_file_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio from a file path.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        try:
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            audio_format = Path(audio_file_path).suffix[1:]  # Remove the dot
            return self.transcribe_audio_bytes(audio_bytes, audio_format, language)
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.loaded_model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
        }

    def cleanup(self):
        """Clean up GPU memory and resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ Whisper model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Compatibility alias for existing code
FasterWhisperTranscriptionService = TransformersWhisperService
